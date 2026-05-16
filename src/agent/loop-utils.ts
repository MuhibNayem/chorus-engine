/**
 * Loop Utilities — Shared primitives for agent loop execution.
 *
 * Extracted from loop.ts to eliminate duplication between the streaming
 * generator (runAgentLoop) and the StateGraph node-based agent (createAgent).
 */

import { z } from "zod";
import type { ChatMessage, ModelResponse, ToolCall, ToolDef, ToolStreamEvent } from "../llm/provider.js";
import { DEFAULT_RETRY_POLICY, type RetryPolicy, withRetry } from "./retry.js";
import type { AgentTool } from "./types.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Stream Timeout
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Wraps a streaming iterable with a per-chunk timeout deadline.
 * Throws if no chunk (token/done) arrives within `timeoutMs` milliseconds.
 * The deadline resets after each received chunk, so fast streams are never
 * penalized — only hung providers that stall mid-stream.
 */
export async function* withStreamTimeout<T extends ToolStreamEvent>(
  source: AsyncIterable<T>,
  timeoutMs: number,
): AsyncGenerator<T> {
  const iter = source[Symbol.asyncIterator]();
  try {
    while (true) {
      const timeoutError = new Error(`Provider stream timed out after ${timeoutMs}ms`);
      const result = await Promise.race([
        iter.next(),
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(timeoutError), timeoutMs),
        ),
      ]);
      if (result.done) break;
      yield result.value;
    }
  } finally {
    await iter.return?.(undefined);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Stream Consumption
// ═══════════════════════════════════════════════════════════════════════════════

export type StreamConsumptionEvent =
  | { type: "token"; text: string }
  | { type: "thinking"; text: string }
  | { type: "stream-done"; response: ModelResponse; inputTokens: number; outputTokens: number }
  | { type: "stream-error"; message: string; fatal: boolean };

/**
 * Consumes the LLM provider stream with automatic retry for transient failures.
 *
 * Design:
 *   • Retryable errors that occur BEFORE any tokens are yielded → silent retry
 *     with exponential backoff. The consumer sees no partial output.
 *   • Errors that occur AFTER tokens have been yielded → fatal. We cannot retry
 *     without duplicating already-emitted tokens.
 *   • Truncated streams (end without `done`) → retry if no tokens emitted,
 *     otherwise fatal.
 *   • Non-retryable errors → fatal immediately.
 */
export async function* consumeStream(
  provider: { streamWithTools(input: { model: string; messages: ChatMessage[]; systemPrompt?: string; tools: ToolDef[] }): AsyncIterable<ToolStreamEvent> },
  model: string,
  messages: ChatMessage[],
  systemPrompt: string,
  toolDefs: ToolDef[],
  streamTimeoutMs: number | undefined,
  retryPolicy: RetryPolicy = DEFAULT_RETRY_POLICY,
): AsyncGenerator<StreamConsumptionEvent> {
  for (let attempt = 1; attempt <= retryPolicy.maxAttempts; attempt++) {
    let yieldedAny = false;
    try {
      const rawStream = provider.streamWithTools({ model, messages, systemPrompt, tools: toolDefs });
      const stream = streamTimeoutMs ? withStreamTimeout(rawStream, streamTimeoutMs) : rawStream;
      let response: ModelResponse | null = null;
      let inputTokens = 0;
      let outputTokens = 0;

      for await (const event of stream) {
        if (event.type === "token") {
          yieldedAny = true;
          yield { type: "token", text: event.text };
          continue;
        }
        if (event.type === "thinking") {
          yieldedAny = true;
          yield { type: "thinking", text: event.text };
          continue;
        }
        if (event.type === "done") {
          response = event.response;
          if (response.usage) {
            inputTokens = response.usage.inputTokens;
            outputTokens = response.usage.outputTokens;
          }
          break;
        }
      }

      if (!response) {
        // Stream ended without a completion event — provider bug or truncation.
        if (attempt < retryPolicy.maxAttempts && !yieldedAny) {
          await new Promise((r) => setTimeout(r, retryPolicy.delayMs(attempt)));
          continue;
        }
        yield {
          type: "stream-error",
          message: `Provider stream ended without completion${yieldedAny ? " after emitting partial tokens" : ""}.`,
          fatal: true,
        };
        return;
      }

      yield { type: "stream-done", response, inputTokens, outputTokens };
      return;
    } catch (error) {
      const err = error instanceof Error ? error : new Error(String(error));
      if (attempt < retryPolicy.maxAttempts && retryPolicy.shouldRetry(err, attempt) && !yieldedAny) {
        await new Promise((r) => setTimeout(r, retryPolicy.delayMs(attempt)));
        continue;
      }
      yield { type: "stream-error", message: err.message, fatal: true };
      return;
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Call Argument Normalization
// ═══════════════════════════════════════════════════════════════════════════════

export function normalizeToolCallArgs(toolCall: ToolCall): Record<string, unknown> {
  const raw = toolCall.function.arguments.trim();
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    throw new Error("Tool arguments must be a JSON object.");
  } catch (error) {
    throw new Error(
      `Invalid JSON arguments for ${toolCall.function.name}: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
}

/** Safe variant that never throws — returns `{ _raw: ... }` on failure. */
export function safeArgs(toolCall: ToolCall): Record<string, unknown> {
  try {
    return normalizeToolCallArgs(toolCall);
  } catch {
    return { _raw: toolCall.function.arguments };
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Definition Conversion (Zod → JSON Schema)
// ═══════════════════════════════════════════════════════════════════════════════

export function toolDefsFromTools(tools: AgentTool[]): ToolDef[] {
  return tools
    .filter((tool): tool is AgentTool & { name: string } => typeof tool.name === "string" && tool.name.length > 0)
    .map((tool) => ({
      type: "function" as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: zodToJsonSchema(tool.schema, tool.name),
      },
    }));
}

export function zodToJsonSchema(schema: unknown, toolName?: string): Record<string, unknown> {
  if (schema instanceof z.ZodType) {
    return normalizeZodSchema(schema);
  }
  if (schema && typeof schema === "object" && !Array.isArray(schema)) {
    const maybeSchema = schema as Record<string, unknown>;
    if (
      typeof maybeSchema.type === "string" ||
      (maybeSchema.properties !== null && typeof maybeSchema.properties === "object" && !Array.isArray(maybeSchema.properties)) ||
      Array.isArray(maybeSchema.required)
    ) {
      return maybeSchema;
    }
  }
  // Schema was provided but did not match Zod or JSON-Schema shapes — warn
  // so developers know parameter validation has been stripped for this tool.
  if (schema !== undefined && schema !== null) {
    process.stderr.write(
      `[chorus] Warning: tool "${toolName ?? "unknown"}" has an unrecognised schema ` +
      `(expected Zod schema or JSON-Schema object). Falling back to open object schema. ` +
      `Parameter validation will be skipped.\n`,
    );
  }
  return { type: "object", properties: {}, additionalProperties: true };
}

function normalizeZodSchema(schema: z.ZodTypeAny): Record<string, unknown> {
  if (schema instanceof z.ZodOptional || schema instanceof z.ZodDefault) {
    return normalizeZodSchema(schema._def.innerType);
  }
  if (schema instanceof z.ZodString) {
    return { type: "string" };
  }
  if (schema instanceof z.ZodNumber) {
    return { type: "number" };
  }
  if (schema instanceof z.ZodBoolean) {
    return { type: "boolean" };
  }
  if (schema instanceof z.ZodEnum) {
    return { type: "string", enum: [...schema.options] };
  }
  if (schema instanceof z.ZodArray) {
    return { type: "array", items: normalizeZodSchema(schema.element) };
  }
  if (schema instanceof z.ZodObject) {
    const shape = schema.shape;
    const properties: Record<string, unknown> = {};
    const required: string[] = [];
    for (const [key, value] of Object.entries(shape)) {
      const field = value as z.ZodTypeAny;
      properties[key] = normalizeZodSchema(field);
      if (!(field instanceof z.ZodOptional) && !(field instanceof z.ZodDefault)) {
        required.push(key);
      }
    }
    return {
      type: "object",
      properties,
      ...(required.length > 0 ? { required } : {}),
      additionalProperties: false,
    };
  }
  return { type: "object", properties: {}, additionalProperties: true };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tool Execution
// ═══════════════════════════════════════════════════════════════════════════════

export async function executeToolCall(
  toolCall: ToolCall,
  toolsByName: Map<string, AgentTool>,
): Promise<{ result: string; attempts: number }> {
  const tool = toolsByName.get(toolCall.function.name);
  if (!tool) {
    throw new Error(`Unknown tool: ${toolCall.function.name}`);
  }

  const args = normalizeToolCallArgs(toolCall);
  const { value, attempts } = await withRetry(
    async () => tool.invoke(args),
    DEFAULT_RETRY_POLICY,
  );

  return {
    result: typeof value === "string" ? value : JSON.stringify(value, null, 2),
    attempts,
  };
}
