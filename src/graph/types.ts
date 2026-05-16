/**
 * StateGraph Core Types
 *
 * Typed-state graph execution inspired by LangGraph's Pregel runtime.
 * Key concepts:
 *   - Channels: typed state fields with reducer semantics
 *   - Nodes: functions that read state and return partial updates
 *   - Edges: static or conditional routing between nodes
 *   - CompiledGraph: executable runtime with invoke/stream/checkpoint support
 *   - Interrupt: pauses execution at node boundaries for human input
 *   - Command: resumes execution with new state values
 */

import type { ZodTypeAny } from "zod";
import type { Checkpointer } from "../agent/types.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Channels — Typed state fields with reducer semantics
// ═══════════════════════════════════════════════════════════════════════════════

/** A channel defines how a state field is initialized and updated. */
export interface Channel<T = unknown> {
  /** Initial value for this field. */
  default: () => T;
  /** Merge an update into the current value. */
  update: (current: T, update: T) => T;
}

/** Sentinel value for graph entry/exit. */
export const START = "__start__";
export const END = "__end__";

// ═══════════════════════════════════════════════════════════════════════════════
// Nodes
// ═══════════════════════════════════════════════════════════════════════════════

/** Configuration passed to each node invocation. */
export interface NodeConfig {
  /** Thread identifier for checkpointing. */
  threadId?: string;
  /** User-defined metadata. */
  metadata?: Record<string, unknown>;
  /** Signal for cancellation. */
  signal?: AbortSignal;
}

/** A node function reads state and returns partial updates. */
export type NodeFn<State extends Record<string, unknown>> = (
  state: State,
  config?: NodeConfig,
) => Promise<Partial<State>> | Partial<State>;

// ═══════════════════════════════════════════════════════════════════════════════
// Edges
// ═══════════════════════════════════════════════════════════════════════════════

/** A static edge from one node to another. */
export interface StaticEdge {
  source: string;
  target: string;
}

/** A conditional edge routes to one or more targets based on state. */
export interface ConditionalEdge<State extends Record<string, unknown>> {
  source: string;
  router: (state: State) => string | string[];
}

// ═══════════════════════════════════════════════════════════════════════════════
// Interrupts & Commands
// ═══════════════════════════════════════════════════════════════════════════════

/** Thrown by a node to pause graph execution for external input. */
export class GraphInterrupt extends Error {
  constructor(
    message: string,
    public readonly node: string,
    public readonly stateSnapshot: Record<string, unknown>,
  ) {
    super(message);
    this.name = "GraphInterrupt";
  }
}

/** Resume a graph after an interrupt with new state values. */
export interface Command<State extends Record<string, unknown>> {
  /** Values to merge into state before resuming. */
  update?: Partial<State>;
  /** Node to resume from (default: the interrupted node). */
  resumeNode?: string;
  /** Metadata for the resume operation. */
  metadata?: Record<string, unknown>;
}

/** Dynamic fan-out primitive — send a value to a specific node. */
export interface Send<T = unknown> {
  /** Target node name. */
  node: string;
  /** Argument to pass to the target node (merged into state). */
  arg: T;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Compilation
// ═══════════════════════════════════════════════════════════════════════════════

export interface CompileOptions {
  /** Checkpointer for persistence. */
  checkpointer?: Checkpointer;
  /** Max recursion depth (node executions). Default: 25. */
  recursionLimit?: number;
  /** Enable debug event streaming. */
  debug?: boolean;
  /** Static cycle detection at compile time. Default: true. */
  detectCycles?: boolean;
  /** Default per-node wall-clock timeout in ms. Default: 60_000. */
  nodeTimeoutMs?: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Runtime Events
// ═══════════════════════════════════════════════════════════════════════════════

export type GraphEvent =
  | { type: "start"; threadId: string; input: Record<string, unknown> }
  | { type: "node_start"; node: string; threadId: string; state: Record<string, unknown> }
  | { type: "node_end"; node: string; threadId: string; output: Record<string, unknown>; durationMs: number }
  | { type: "edge"; source: string; target: string; threadId: string }
  | { type: "interrupt"; node: string; threadId: string; message: string; state: Record<string, unknown> }
  | { type: "state"; threadId: string; values: Record<string, unknown> }
  | { type: "end"; threadId: string; state: Record<string, unknown> }
  | { type: "error"; threadId: string; node?: string; error: string }
  | { type: "cycle_detected"; threadId: string; message: string }
  | { type: "timeout"; threadId: string; node?: string; kind: "node" | "graph"; limitMs: number }
  | { type: "deadlock"; threadId: string; completedNodes: string[] };

// ═══════════════════════════════════════════════════════════════════════════════
// Execution Config
// ═══════════════════════════════════════════════════════════════════════════════

export interface RunConfig {
  threadId?: string;
  metadata?: Record<string, unknown>;
  signal?: AbortSignal;
  /** Stream mode: "values" (full state) | "updates" (partial updates) | "debug" (all events). */
  streamMode?: "values" | "updates" | "debug";
  /** Per-node wall-clock timeout in ms. Overrides compile-time default. */
  nodeTimeoutMs?: number;
  /** Total graph wall-clock timeout in ms. Default: 300_000 (5 min). */
  graphTimeoutMs?: number;
  /** Detect infinite loops by state fingerprint repetition. Default: true. */
  detectStateLoops?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Checkpoint extensions for graph semantics
// ═══════════════════════════════════════════════════════════════════════════════

export interface GraphCheckpoint {
  threadId: string;
  checkpointId: string;
  parentCheckpointId?: string;
  /** Full state at this checkpoint. */
  state: Record<string, unknown>;
  /** Which nodes have been executed. */
  completedNodes: string[];
  /** Which nodes are queued for next wave. */
  nextNodes: string[];
  /** Number of waves executed so far. */
  waveCount: number;
  createdAt: number;
  metadata?: Record<string, unknown>;
}

/** Extended checkpointer interface for graph execution. */
export interface GraphCheckpointer extends Checkpointer {
  saveGraphCheckpoint(cp: GraphCheckpoint): Promise<void>;
  loadGraphCheckpoint(threadId: string, checkpointId?: string): Promise<GraphCheckpoint | null>;
  listGraphCheckpoints(threadId: string): Promise<GraphCheckpoint[]>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Agent Builder Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface AgentState {
  messages: Array<{ role: string; content: string }>;
  /** The agent's response text. */
  response?: string;
  /** Tool calls from the LLM. */
  toolCalls?: Array<{ name: string; arguments: string }>;
  /** Results from tool execution. */
  toolResults?: Array<{ name: string; result: string }>;
  /** Whether the agent is done. */
  isDone?: boolean;
  /** Reasoning content from the model. */
  reasoning?: string;
}

export interface CreateAgentOptions {
  provider: import("../llm/provider.js").LLMProvider;
  model: string;
  tools?: import("../agent/types.js").AgentTool[];
  /** Static string or dynamic function from state. */
  systemPrompt?: string | ((state: AgentState) => string);
  checkpointer?: Checkpointer;
  /** Zod schema for structured output validation. */
  outputSchema?: ZodTypeAny;
  /** Max tool execution rounds. Default: 25. */
  maxIterations?: number;
  /** In-process tracer. */
  tracer?: import("../telemetry/inprocess.js").InProcessTracer;
  /** Guardrails configuration. */
  guardrails?: import("../guardrails/index.js").GuardrailsConfig;
}
