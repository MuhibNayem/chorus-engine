// ── Agent Loop ────────────────────────────────────────────────────────────────
export { runAgentLoop } from "./agent/loop.js";
export type {
  AgentEvent,
  AgentTool,
  ToolSchema,
  LoopOptions,
  Checkpointer,
  CheckpointState,
  HitlDecision,
  HitlRequest,
} from "./agent/types.js";

// ── Middleware ────────────────────────────────────────────────────────────────
export type {
  AgentMiddleware,
  RoundContext,
  ToolResultContext,
  CompactResult,
} from "./agent/middleware.js";
export {
  SummarizationMiddleware,
  ObservabilityMiddleware,
  LargeOutputOffloadMiddleware,
  TodoMiddleware,
  createDefaultMiddleware,
} from "./agent/middleware.js";
export { SkillMiddleware } from "./skills/middleware.js";

// ── HITL ──────────────────────────────────────────────────────────────────────
export { HitlGate, HitlGateTimeoutError, HitlGateDisposedError } from "./agent/hitl.js";

// ── Checkpointing ─────────────────────────────────────────────────────────────
export { JsonFileCheckpointer } from "./agent/checkpointer.js";

// ── Side-Channel ──────────────────────────────────────────────────────────────
export { BtwQueue } from "./agent/btw.js";

// ── Memory ────────────────────────────────────────────────────────────────────
export {
  createMemoryTools,
  createSharedMemoryTools,
} from "./agent/memory-tools.js";

// ── Retry ─────────────────────────────────────────────────────────────────────
export { withRetry, DEFAULT_RETRY_POLICY, RATE_LIMIT_RETRY_POLICY, isRetryable } from "./agent/retry.js";
export type { RetryPolicy, HttpError } from "./agent/retry.js";

// ── Swarm ─────────────────────────────────────────────────────────────────────
export { runSwarm } from "./swarm/orchestrator.js";
export { runSwarmGraph, computeWaves } from "./swarm/graph-executor.js";
export { buildSupervisorSwarm } from "./swarm/supervisor.js";
export { runGroupChat } from "./swarm/group-chat.js";
export type {
  SwarmConfig,
  SwarmAgent,
  SwarmEvent,
  SwarmSession,
  HandoffRequest,
  CostBudget,
  CostRoutingPolicy,
  AgentMetrics,
} from "./swarm/types.js";
export type {
  GroupChatConfig,
  GroupChatEvent,
  GroupChatAgent,
} from "./swarm/group-chat.js";

// ── Swarm Presets ─────────────────────────────────────────────────────────────
export { buildPresetSwarm, SWARM_PRESETS } from "./swarm/presets/index.js";
export { createPlanBuildReviewSwarm } from "./swarm/presets/plan-build-review.js";
export { createResearchSynthesizeSwarm } from "./swarm/presets/research-synthesize.js";
export { createParallelResearchSwarm } from "./swarm/presets/research-parallel.js";
export { createVaptReportSwarm } from "./swarm/presets/vapt-report.js";

// ── LLM Providers ─────────────────────────────────────────────────────────────
export { createProvider, getDefaultProvider } from "./llm/registry.js";
export type {
  LLMProvider,
  ChatMessage,
  ModelResponse,
  ToolDef,
  ToolCall,
  ProviderHealth,
} from "./llm/provider.js";
export type { ProviderName, KnownProviderName } from "./llm/config.js";

// ── Tools ─────────────────────────────────────────────────────────────────────
export { createFilesystemTools, filesystemTools } from "./tools/filesystem.js";
export { shellTools } from "./tools/shell.js";
export { gitTools } from "./tools/index.js";
export { assessCommandSafety, auditCommand } from "./tools/safety.js";

// ── Evals ─────────────────────────────────────────────────────────────────────
export { runEvalSuite, formatEvalRun } from "./evals/runner.js";
export type {
  EvalSuite,
  EvalRun,
  EvalCaseResult,
  EvalVerdict,
  ScorerConfig,
} from "./evals/types.js";
export type { EvalRunnerConfig } from "./evals/runner.js";

// ── Harness (pre-flight workers) ──────────────────────────────────────────────
export { executeWorkers, formatWorkerResults } from "./harness/workerEngine.js";
export { prepareTaskExecution, prepareTaskExecutionAsync } from "./harness/orchestrator.js";
export { SemanticTaskRouter, routeTaskSemantic } from "./harness/semanticRouter.js";
export type {
  WorkerEvent,
  WorkerEventCallback,
  WorkerExecutionOptions,
  WorkerExecutionResult,
  SharedWorkerContext,
} from "./harness/workerEngine.js";
export type {
  SemanticRouteResult,
  SemanticRouterOptions,
} from "./harness/semanticRouter.js";

// ── MCP ───────────────────────────────────────────────────────────────────────
export { getMcpTools } from "./mcp/client.js";

// ── A2A ───────────────────────────────────────────────────────────────────────
export { createSwarmA2AServer } from "./a2a/adapter.js";
export type { AgentCard } from "./a2a/types.js";

// ── Subagents ─────────────────────────────────────────────────────────────────
export { createDelegateTool } from "./subagents/delegateTool.js";
export type {
  SubagentEvent,
  SubagentEventCallback,
  SubagentExecutionOptions,
} from "./subagents/runtime.js";

// ── Settings / Engine config ───────────────────────────────────────────────────
export { configureEngine } from "./settings/storage.js";
export type { EngineConfig } from "./settings/storage.js";

// ── Skills ────────────────────────────────────────────────────────────────────
export { SkillRegistry } from "./skills/registry.js";
export { TrajectorySynthesizer, longestCommonSubsequence } from "./skills/synthesizer.js";
export type { SynthesizerOptions } from "./skills/synthesizer.js";
export type {
  SkillDef,
  PatternDef,
  PatternParameter,
  SkillWorkflowStep,
  SkillMatch,
  SkillMetrics,
  ToolTrajectory,
} from "./skills/types.js";

// ── Telemetry ─────────────────────────────────────────────────────────────────
export { OTelExporter, swarmEventsToSpans } from "./telemetry/index.js";
export type { TelemetryConfig } from "./telemetry/index.js";

// ── Cost Router ───────────────────────────────────────────────────────────────
export { checkCostBudget, resolveModel } from "./swarm/cost-router.js";
export type { RoutingContext } from "./swarm/cost-router.js";

// ── HITL config ───────────────────────────────────────────────────────────────
export type { HitlGateOptions } from "./agent/hitl.js";

// ── Middleware config ─────────────────────────────────────────────────────────
export type {
  BeforeToolContext,
  SummarizationOptions,
  LargeOutputOffloadOptions,
} from "./agent/middleware.js";
