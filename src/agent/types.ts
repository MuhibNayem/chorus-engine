import type { ApprovalPolicy } from "../harness/types.js";
import type { ChatMessage, LLMProvider, ModelResponse, ToolCall, ToolDef } from "../llm/provider.js";
import type { AgentMiddleware } from "./middleware.js";
import type { InProcessTracer } from "../telemetry/inprocess.js";
import type { GuardrailsConfig } from "../guardrails/index.js";
import type { ZodTypeAny } from "zod";

/**
 * Valid tool parameter schemas. Accepts a Zod schema or a raw JSON Schema object.
 * The `unknown` fallback is preserved for backwards-compat — new code should use
 * ZodTypeAny or Record<string, unknown>.
 */
export type ToolSchema = ZodTypeAny | Record<string, unknown>;

export type AgentTool = {
  name?: string;
  description?: string;
  /** Parameter schema — Zod or JSON Schema object. */
  schema?: ToolSchema;
  invoke(input: unknown): Promise<unknown>;
};

export type HitlDecision =
  | { type: "approve" }
  | { type: "approve_session"; toolNames?: string[] }
  | { type: "reject"; message?: string };

export type HitlRequest = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  description?: string;
};

/**
 * First-class control-flow exception for agent handoffs.
 * Thrown by handoff tools and caught by the agent loop to yield a native
 * `handoff` event, bypassing fragile JSON parsing.
 */
export class HandoffSignal extends Error {
  constructor(
    public readonly targetAgent: string,
    public readonly taskDescription: string,
    public readonly artifacts: string[],
    public readonly reasoning?: string,
  ) {
    super(`Handoff to ${targetAgent}: ${taskDescription}`);
    this.name = "HandoffSignal";
  }
}

export type AgentEvent =
  | { type: "token"; text: string }
  | { type: "thinking"; text: string }
  | { type: "tool-start"; id: string; name: string; args: Record<string, unknown> }
  | { type: "tool-done"; id: string; name: string; result: string; durationMs: number }
  | { type: "tool-error"; id: string; name: string; error: string; willRetry: boolean }
  | { type: "hitl"; requests: HitlRequest[]; resumeKey: string }
  | { type: "btw"; text: string }
  | { type: "checkpoint"; round: number; threadId: string }
  | { type: "compacted"; removedMessages: number; savedTokens: number }
  | { type: "done"; response: string; reasoning: string; toolCount: number; history: ChatMessage[]; inputTokens: number; outputTokens: number; costUsd: number; durationMs: number }
  | { type: "error"; message: string; fatal: boolean }
  | { type: "aborted"; message?: string }
  // Structured lifecycle events (OpenTelemetry-aligned)
  | { type: "round-start"; round: number; threadId: string; messageCount: number }
  | { type: "round-end"; round: number; threadId: string; toolCallsThisRound: number }
  | { type: "guardrail-triggered"; guardrail: string; severity: "info" | "warning" | "critical"; action: "halt" | "warn"; message: string }
  | { type: "memory-recall"; scope: string; query: string; resultsCount: number }
  | { type: "memory-compact"; scope: string; removedMessages: number; factsExtracted: number }
  | { type: "checkpoint-saved"; round: number; threadId: string; mode?: string }
  | { type: "checkpoint-loaded"; round: number; threadId: string; restored: boolean }
  | { type: "stream-start"; round: number; threadId: string; model: string }
  | { type: "stream-end"; round: number; threadId: string; tokensEmitted: number }
  | { type: "middleware-before"; round: number; hook: string }
  | { type: "middleware-after"; round: number; hook: string }
  | { type: "handoff"; targetAgent: string; taskDescription: string; artifacts: string[]; reasoning?: string };

export type CheckpointState = {
  messages: ChatMessage[];
  round: number;
  waitingForHitl?: {
    resumeKey: string;
    requests: HitlRequest[];
    toolCalls: ToolCall[];
    assistant: ModelResponse;
  };
};

export interface Checkpoint {
  threadId: string;
  round: number;
  messages: ChatMessage[];
  createdAt: number;
  waitingForHitl?: CheckpointState["waitingForHitl"];
}

export interface Checkpointer {
  save(threadId: string, state: CheckpointState): Promise<void>;
  load(threadId: string): Promise<Checkpoint | null>;
  loadAt(threadId: string, round: number): Promise<Checkpoint | null>;
  list(threadId: string): Promise<Checkpoint[]>;
  fork(threadId: string, round: number, newThreadId: string): Promise<void>;
  delete(threadId: string): Promise<void>;
}

export interface LoopOptions {
  provider: LLMProvider;
  model: string;
  tools: AgentTool[];
  messages: ChatMessage[];
  systemPrompt: string;
  threadId: string;
  hitlGate: {
    shouldPause(toolCalls: ToolCall[], policy: ApprovalPolicy): boolean;
    wait(resumeKey: string): Promise<HitlDecision>;
  };
  btwQueue: {
    drain(): string[];
  };
  policy: ApprovalPolicy;
  checkpointer: Checkpointer;
  maxRounds?: number;
  resumedDecision?: HitlDecision;
  middleware?: AgentMiddleware[];
  abortSignal?: AbortSignal;
  /**
   * Maximum milliseconds to wait for the next token/chunk from the provider stream.
   * If the stream stalls longer than this, the loop yields an `error` event and stops.
   * Default: 120_000 (2 minutes).
   */
  streamTimeoutMs?: number;
  /**
   * Zod schema to validate the final model response against.
   * When set, the loop will retry with a correction prompt if the response
   * does not parse as valid JSON matching this schema.
   * Works best when the model supports JSON mode / tool-call response format.
   */
  outputSchema?: ZodTypeAny;
  /**
   * In-process tracer for real-time OTel span creation. When provided,
   * the loop emits spans for agent turns and tool calls.
   */
  tracer?: InProcessTracer;
  /**
   * Guardrails configuration for input, output, and tool validation.
   * When provided, guardrails run at the appropriate lifecycle boundaries.
   */
  guardrails?: GuardrailsConfig;
}
