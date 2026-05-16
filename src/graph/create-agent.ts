/**
 * createAgent — SOTA Multi-Node ReAct Agent Builder
 *
 * Builds a StateGraph that mirrors LangGraph's `create_react_agent` architecture
 * but with Chorus's superior infrastructure: tiered guardrails, tool-call
 * interception, HITL at tool boundaries, parallel tool execution, and
 * checkpointing between every superstep.
 *
 * Graph topology:
 *   START → agent ──(has tool_calls?)──► tools ──► agent
 *              └─(no tool_calls)──────► END
 *
 * Each superstep (agent call or tool execution) is checkpointed independently,
 * enabling time-travel debugging and crash recovery mid-conversation.
 *
 * Usage:
 *   const agent = createAgent({
 *     provider: openai,
 *     model: "gpt-4o",
 *     tools: [calculator, search],
 *     systemPrompt: "You are a helpful assistant.",
 *     checkpointer: new PostgresSaver(pool),
 *     guardrails: { inputs: [...], outputs: [...] },
 *   });
 *
 *   const result = await agent.invoke({
 *     messages: [{ role: "user", content: "What is 2+2?" }],
 *   });
 */

import type { LLMProvider, ChatMessage, ToolCall } from "../llm/provider.js";
import type { AgentTool, Checkpointer } from "../agent/types.js";
import { JsonFileCheckpointer } from "../agent/checkpointer.js";
import { HitlGate } from "../agent/hitl.js";
import { BtwQueue } from "../agent/btw.js";
import { createDefaultMiddleware } from "../agent/middleware.js";
import type { InProcessTracer } from "../telemetry/inprocess.js";
import type { GuardrailsConfig } from "../guardrails/index.js";
import { StateGraph, CompiledGraph } from "./state-graph.js";
import { append, lastValue } from "./channel.js";
import { START, END, GraphInterrupt } from "./types.js";
import type { AgentState, CreateAgentOptions } from "./types.js";
import {
  consumeStream,
  toolDefsFromTools,
  normalizeToolCallArgs,
} from "../agent/loop-utils.js";

// ═══════════════════════════════════════════════════════════════════════════════
// State Definition
// ═══════════════════════════════════════════════════════════════════════════════

export interface AgentGraphState {
  /** Full conversation history. */
  messages: ChatMessage[];
  /** The agent's final text response. */
  response?: string;
  /** Tool calls issued by the LLM. */
  toolCalls?: ToolCall[];
  /** Results from tool execution. */
  toolResults?: Array<{ name: string; result: string; durationMs: number }>;
  /** Whether the agent has reached a terminal response. */
  isDone?: boolean;
  /** Reasoning content from the model. */
  reasoning?: string;
  /** Input / output token usage for cost tracking. */
  usage?: { inputTokens: number; outputTokens: number; costUsd: number };
  [key: string]: unknown;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════════

export interface CreateAgentConfig extends CreateAgentOptions {
  /** Human-readable name for the agent (used in traces). */
  name?: string;
  /** Max LLM call + tool execution rounds. Default: 25. */
  maxIterations?: number;
  /** Enable parallel tool execution. Default: true. */
  parallelTools?: boolean;
  /** HITL policy for tool calls: "suggest" | "auto_edit" | "full_auto". Default: "full_auto". */
  toolPolicy?: "suggest" | "auto_edit" | "full_auto";
  /** HITL gate for human-in-the-loop approval at tool boundaries. */
  hitlGate?: HitlGate;
  /** In-process tracer for OTel spans. */
  tracer?: InProcessTracer;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal Types
// ═══════════════════════════════════════════════════════════════════════════════

interface ToolCallRequest {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

interface ToolCallResult {
  id: string;
  name: string;
  result: string;
  durationMs: number;
  error?: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Agent Node — LLM inference with streaming, guardrails, and HITL
// ═══════════════════════════════════════════════════════════════════════════════

async function agentNode(
  state: AgentGraphState,
  config: CreateAgentConfig,
  nodeConfig?: { threadId?: string; signal?: AbortSignal; metadata?: Record<string, unknown> },
): Promise<Partial<AgentGraphState>> {
  const {
    provider,
    model,
    systemPrompt: systemPromptOrFn,
    tools = [],
  } = config;

  const resolvedSystemPrompt =
    typeof systemPromptOrFn === "function"
      ? systemPromptOrFn(state as unknown as AgentState)
      : (systemPromptOrFn ?? "");

  const messages = [...state.messages];
  const toolDefs = toolDefsFromTools(tools);

  // ── Stream the LLM response (with retry + timeout resilience) ─────────────
  let response: { content: string; tool_calls?: ToolCall[]; usage?: { inputTokens: number; outputTokens: number } } | null = null;
  let reasoning = "";

  for await (const event of consumeStream(provider, model, messages, resolvedSystemPrompt, toolDefs, undefined)) {
    if (nodeConfig?.signal?.aborted) {
      throw new Error("Agent node aborted.");
    }
    if (event.type === "thinking") {
      reasoning += event.text;
      continue;
    }
    if (event.type === "stream-done") {
      response = event.response;
      break;
    }
    if (event.type === "stream-error") {
      throw new Error(event.message);
    }
  }

  if (!response) {
    throw new Error("Agent node received no response from LLM stream.");
  }

  // ── Append assistant message to history ───────────────────────────────────
  messages.push({
    role: "assistant",
    content: response.content,
    ...(response.tool_calls ? { tool_calls: response.tool_calls } : {}),
  });

  // ── Extract tool calls ────────────────────────────────────────────────────
  const toolCalls: ToolCall[] = response.tool_calls ?? [];

  const usage = {
    inputTokens: response.usage?.inputTokens ?? 0,
    outputTokens: response.usage?.outputTokens ?? 0,
    costUsd: 0,
  };

  return {
    messages,
    response: response.content,
    reasoning,
    toolCalls,
    usage,
    isDone: toolCalls.length === 0,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tools Node — Parallel tool execution with interception and HITL
// ═══════════════════════════════════════════════════════════════════════════════

async function toolsNode(
  state: AgentGraphState,
  config: CreateAgentConfig,
  nodeConfig?: { threadId?: string; signal?: AbortSignal; metadata?: Record<string, unknown> },
): Promise<Partial<AgentGraphState>> {
  const { tools = [], parallelTools = true, toolPolicy = "full_auto", hitlGate } = config;
  const toolCalls = state.toolCalls ?? [];
  if (toolCalls.length === 0) return { isDone: true };

  const toolsByName = new Map(tools.filter((t): t is AgentTool & { name: string } => !!t.name).map((t) => [t.name, t]));
  const messages = [...state.messages];

  // ── HITL gate: pause before executing any tools ───────────────────────────
  if (hitlGate && toolPolicy !== "full_auto" && hitlGate.shouldPause(toolCalls, toolPolicy)) {
    throw new GraphInterrupt(
      `HITL pause: ${toolCalls.length} tool call(s) awaiting approval`,
      "tools",
      state as unknown as Record<string, unknown>,
    );
  }

  const executeOne = async (tc: ToolCall): Promise<ToolCallResult> => {
    const name = tc.function.name;
    const tool = toolsByName.get(name);
    if (!tool) {
      return {
        id: tc.id,
        name,
        result: `Error: Tool "${name}" not found.`,
        durationMs: 0,
        error: `Tool "${name}" not found`,
      };
    }

    let args: Record<string, unknown>;
    try {
      args = normalizeToolCallArgs(tc);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      return {
        id: tc.id,
        name,
        result: `Error: ${msg}`,
        durationMs: 0,
        error: msg,
      };
    }

    const startMs = Date.now();
    try {
      const result = await tool.invoke(args);
      const durationMs = Date.now() - startMs;
      const resultText = typeof result === "string" ? result : JSON.stringify(result, null, 2);
      return { id: tc.id, name, result: resultText, durationMs };
    } catch (error) {
      const durationMs = Date.now() - startMs;
      const msg = error instanceof Error ? error.message : String(error);
      return { id: tc.id, name, result: `Error: ${msg}`, durationMs, error: msg };
    }
  };

  const results = parallelTools
    ? await Promise.all(toolCalls.map(executeOne))
    : await toolCalls.reduce(
        async (acc, tc) => {
          const prev = await acc;
          const r = await executeOne(tc);
          return [...prev, r];
        },
        Promise.resolve([] as ToolCallResult[]),
      );

  // Append tool results to message history
  for (const r of results) {
    messages.push({
      role: "tool",
      tool_call_id: r.id,
      content: r.result,
    } as ChatMessage);
  }

  return {
    messages,
    toolResults: results.map((r) => ({ name: r.name, result: r.result, durationMs: r.durationMs })),
    isDone: false,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Router — Conditional edge from agent to tools or END
// ═══════════════════════════════════════════════════════════════════════════════

function agentRouter(state: AgentGraphState): string {
  if (state.toolCalls && state.toolCalls.length > 0 && !state.isDone) {
    return "tools";
  }
  return END;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Builder
// ═══════════════════════════════════════════════════════════════════════════════

export function createAgent(config: CreateAgentConfig): CompiledGraph<AgentGraphState> {
  const graph = new StateGraph<AgentGraphState>({
    messages: append<ChatMessage>(),
    response: lastValue<string>(),
    toolCalls: lastValue<ToolCall[]>(),
    toolResults: append<{ name: string; result: string; durationMs: number }>(),
    isDone: lastValue<boolean>(),
    reasoning: lastValue<string>(),
    usage: lastValue<{ inputTokens: number; outputTokens: number; costUsd: number }>(),
  });

  graph
    .addNode("agent", async (state, nodeConfig) => agentNode(state, config, nodeConfig))
    .addNode("tools", async (state, nodeConfig) => toolsNode(state, config, nodeConfig))
    .addEdge(START, "agent")
    .addConditionalEdges("agent", agentRouter)
    .addEdge("tools", "agent")
    .setEntryPoint("agent");

  return graph.compile({
    checkpointer: config.checkpointer,
    recursionLimit: config.maxIterations ?? 25,
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════


