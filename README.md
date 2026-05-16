# chorus-engine

**Headless multi-agent orchestration runtime for TypeScript**

`chorus-engine` is a framework-agnostic engine for building, executing, and observing LLM-powered agents — from single ReAct loops to multi-agent DAG swarms. It ships with production-grade infrastructure for HITL, checkpointing, telemetry, cost control, and adaptive skill synthesis.

```bash
npm install chorus-engine
```

---

## Package Exports

| Import | Contents |
|--------|----------|
| `chorus-engine` | Core agent loop, middleware, HITL, checkpointing, retry, memory tools |
| `chorus-engine/agent` | Agent loop, types, middleware classes, durable checkpointer |
| `chorus-engine/swarm` | Multi-agent orchestrator, graph executor, supervisor, group chat, 4 presets, cost router, circuit breaker |
| `chorus-engine/graph` | StateGraph builder, `createAgent`, GraphRestServer, channels, PostgresSaver |
| `chorus-engine/harness` | Semantic task router, worker engine, protocol builder, repo intelligence |
| `chorus-engine/tools` | Filesystem, shell, git, web search, safety auditing |
| `chorus-engine/mcp` | MCP client tools, server management, OAuth support |
| `chorus-engine/llm` | Provider registry, context windows, pricing, reasoning parser |
| `chorus-engine/evals` | Eval suite runner, scorer, storage |

---

## Feature Overview

### 1. Agent Loop — ReAct Execution Engine

The core `runAgentLoop` is a streaming async generator that drives a single agent through the ReAct pattern (Reason → Act → Observe → loop).

```typescript
import { runAgentLoop, HitlGate, BtwQueue, JsonFileCheckpointer } from "chorus-engine";

const stream = runAgentLoop({
  provider,
  model: "gpt-4o",
  tools: [calculatorTool, searchTool],
  messages: [{ role: "user", content: "What is 15% of 340?" }],
  systemPrompt: "You are a precise calculator. Use tools when needed.",
  threadId: "thread-1",
  hitlGate: new HitlGate(),
  btwQueue: new BtwQueue(),
  policy: "full_auto",
  checkpointer: new JsonFileCheckpointer(),
});

for await (const event of stream) {
  if (event.type === "token") process.stdout.write(event.text);
  if (event.type === "done") console.log(`Cost: $${event.costUsd.toFixed(4)}`);
}
```

**Capabilities:**
- Streaming token emission with per-chunk timeout (prevents hung providers)
- Automatic retry with exponential backoff (3 attempts, configurable policy)
- Fatal boundary after token emission (consistency > availability)
- Zod output schema validation with auto-retry correction prompts
- Tool execution with retry + parallel invocation
- BtwQueue for async side-channel message injection (`[/btw]`)
- AbortSignal support for cooperative cancellation

**30+ structured event types:** `token`, `thinking`, `tool-start`, `tool-done`, `tool-error`, `hitl`, `btw`, `checkpoint`, `compacted`, `done`, `error`, `aborted`, `round-start`, `round-end`, `guardrail-triggered`, `memory-recall`, `memory-compact`, `checkpoint-saved`, `checkpoint-loaded`, `stream-start`, `stream-end`, `middleware-before`, `middleware-after`, `handoff`

---

### 2. Middleware System — Plugin Pipeline

6 lifecycle hooks with priority-based parallel execution:

```typescript
import {
  SummarizationMiddleware,
  LargeOutputOffloadMiddleware,
  SkillMiddleware,
  createDefaultMiddleware,
  type AgentMiddleware,
} from "chorus-engine";
```

| Hook | Called When | Use For |
|------|-------------|---------|
| `beforeRound` | Before LLM call each round | Skill routing, context prep |
| `extraSystemPrompt` | System prompt assembly | Inject schemas, instructions |
| `extraTools` | Tool registry assembly | Dynamic tool injection |
| `maybeCompact` | Before each round (first wins) | Context window compaction |
| `beforeTool` | Before tool execution | Audit, block, or substitute tools |
| `afterTool` | After tool execution | Transform results, redact secrets |

**Built-in middleware:**
- **`SummarizationMiddleware`** — Compacts conversation history at configurable threshold (default 85% of context window)
- **`LargeOutputOffloadMiddleware`** — Offloads tool outputs exceeding a byte threshold to disk
- **`ObservabilityMiddleware`** — Emits structured events for every lifecycle stage
- **`TodoMiddleware`** — Enforces structured task planning
- **`SkillMiddleware`** — Integrates the Adaptive Skill Runtime for per-turn semantic skill routing

**Custom middleware example:**

```typescript
class AuditMiddleware implements AgentMiddleware {
  async beforeTool(ctx: BeforeToolContext) {
    console.log(`[audit] ${ctx.name}`, ctx.args);
    // Block writes outside /tmp
    if (ctx.name === "file_write" && !String(ctx.args.path).startsWith("/tmp")) {
      return { cancel: true, result: "[BLOCKED] path not allowed" };
    }
  }
  async afterTool(ctx: ToolResultContext) {
    return ctx.result.replace(/secret-\w+/g, "[REDACTED]");
  }
}
```

---

### 3. HITL — Human-in-the-Loop Gates

Pause agent execution at tool boundaries for human approval, then resume across process restarts.

```typescript
import { HitlGate, HitlGateTimeoutError, HitlGateDisposedError } from "chorus-engine";

const hitlGate = new HitlGate({
  additionalSensitiveTools: ["deploy_to_production"],
  timeoutMs: 300_000, // 5 minutes
});
```

When the agent calls a sensitive tool under an `auto_edit` policy:

```
→ hitl event yielded with resumeKey
→ Agent loop pauses and awaits approval
→ External process calls hitlGate.resolve(resumeKey, { type: "approve" })
→ Agent resumes with tool execution
```

**Decision types:**
- `approve` — Execute this tool call once
- `approve_session` — Auto-approve this tool for the rest of the session
- `reject` — Skip the tool, inject a rejection message into history

**Safety guarantees:**
- Configurable timeout prevents immortal resolvers
- `dispose()` rejects all pending gates (clean shutdown)
- Gates survive process restarts via checkpoint serialization

---

### 4. Checkpointing & Crash Recovery

Every agent round is persisted. Resume from any checkpoint, fork timelines, or reconstruct state from event logs after a crash.

```typescript
import { JsonFileCheckpointer, DurableCheckpointer } from "chorus-engine";

// File-based (good for dev/CLI)
const cp = new JsonFileCheckpointer();

// Production-grade with event sourcing
const cp = new DurableCheckpointer("sync"); // or "async" | "exit"

// PostgreSQL for high-concurrency deployments
import { PostgresSaver } from "chorus-engine/graph";
import { Pool } from "pg";
const cp = new PostgresSaver(new Pool({ connectionString: process.env.DATABASE_URL }));
await cp.setup();
```

**DurableCheckpointer modes:**
| Mode | Throughput | Durability |
|------|-----------|------------|
| `sync` | Low | Every step flushed before next starts |
| `async` | Medium | Checkpoints written asynchronously (tiny window of data loss) |
| `exit` | High | Only flushed on graceful exit / manual `flush()` |

**Crash recovery:**
```typescript
const checkpointer = new DurableCheckpointer();
const { checkpoint, recovered, events } = await recoverFromCrash(checkpointer, threadId);
if (recovered) {
  // Resume from checkpoint.round
}
```

---

### 5. Swarm — Multi-Agent Orchestration

Four execution paradigms in one API:

#### Handoff (default)
Agents transfer control via `handoff_to_agent` tool. Context is filtered to prevent token bloat.

```typescript
import { runSwarm, type SwarmConfig } from "chorus-engine/swarm";

const events = runSwarm({
  provider,
  modelName: "gpt-4o",
  task: "Audit the auth module for security issues",
  executionModel: "handoff",  // or omit for default
  agents: [
    { name: "auditor", systemPrompt: "You find vulnerabilities...", tools: [...], handoffDestinations: ["fixer"] },
    { name: "fixer", systemPrompt: "You apply security patches...", tools: [...], handoffDestinations: [] },
  ],
  initialAgent: "auditor",
});
```

#### Graph (DAG)
Agents run in topological waves with dependencies. Parallel agents execute in isolated Git worktrees.

```typescript
import { runSwarmGraph } from "chorus-engine/swarm";

const events = runSwarmGraph({
  ...config,
  executionModel: "graph",
  agents: [
    { name: "researcher", ... },
    { name: "analyst", ... },
    { name: "synthesizer", dependsOn: ["researcher", "analyst"], requiredArtifacts: ["research"] },
  ],
});
```

#### Supervisor
A central coordinator dynamically routes tasks to specialists, filtering context to prevent token bloat.

```typescript
import { buildSupervisorSwarm } from "chorus-engine/swarm";
```

#### Group Chat
Multi-agent debate with structural convergence strategies (`vote`, `concatenate`, `first-success`).

```typescript
import { runGroupChat, type GroupChatConfig } from "chorus-engine/swarm";
```

---

### 6. Swarm Presets — Battle-Tested Configurations

```typescript
import {
  createPlanBuildReviewSwarm,
  createResearchSynthesizeSwarm,
  createParallelResearchSwarm,
  createVaptReportSwarm,
} from "chorus-engine/swarm";
```

| Preset | Agents | Use Case |
|--------|--------|----------|
| `plan-build-review` | Planner → Coder → Reviewer | Feature implementation with review gate |
| `research-synthesize` | Researcher + Synthesizer | Gather info and produce structured report |
| `research-parallel` | 3 parallel researchers | Broad coverage with divergent exploration |
| `vapt-report` | Analyzer → Reporter → Validator | Security assessment pipeline |

---

### 7. Circuit Breaker & Cost Control

Financial guardrails prevent runaway LLM spend.

```typescript
const config: SwarmConfig = {
  costBudget: { totalUsd: 2.50 },  // Hard limit — never exceed $2.50

  circuitBreaker: {
    maxConsecutiveSameAgent: 3,
    maxConsecutiveRounds: 15,
    maxAgentRounds: 30,
    maxTokensPerAgent: 100_000,
    maxDurationMs: 600_000,
    maxCostUsdPerAgent: 1.00,
    maxCostUsd: 5.00,
    stallTimeoutMs: 120_000,
  },
};
```

When a breaker trips:
1. All in-flight LLM requests are aborted
2. State is checkpointed for recovery
3. A `circuit-break` event is yielded with the reason

**Cost routing:** Automatically selects cheaper models within budget constraints via `resolveModel()`.

---

### 8. StateGraph — Typed-State DAG Execution

LangGraph-compatible builder with reducer semantics, parallel waves, and full checkpointing.

```typescript
import { StateGraph, append, lastValue, START, END } from "chorus-engine/graph";

const graph = new StateGraph({
  messages: append<ChatMessage>(),
  answer: lastValue<string>(),
});

graph
  .addNode("agent", async (state) => { /* call LLM */ })
  .addNode("tools", async (state) => { /* execute tools */ })
  .addEdge(START, "agent")
  .addConditionalEdges("agent", (state) => state.toolCalls?.length ? "tools" : END)
  .addEdge("tools", "agent")
  .setEntryPoint("agent");

const app = graph.compile({ checkpointer, recursionLimit: 25 });
const result = await app.invoke({ messages: [{ role: "user", content: "Hello" }] });

// Stream events
for await (const event of app.stream(input, { streamMode: "debug" })) {
  // node_start, node_end, state, interrupt, end, error, timeout, deadlock
}
```

**`createAgent` — LangGraph-style ReAct agent builder:**

```typescript
import { createAgent } from "chorus-engine/graph";

const agent = createAgent({
  provider,
  model: "gpt-4o",
  tools: [calculator, search],
  systemPrompt: "You are a helpful assistant.",
  checkpointer: new PostgresSaver(pool),
  guardrails: { inputs: [...], outputs: [...] },
  hitlGate: new HitlGate(),
  toolPolicy: "auto_edit",
  outputSchema: z.object({ answer: z.string(), confidence: z.number() }),
});

const result = await agent.invoke({
  messages: [{ role: "user", content: "What is 2+2?" }],
});
```

**State channels (reducer semantics):**
| Channel | Behavior |
|---------|----------|
| `lastValue<T>()` | Last write wins (scalars) |
| `append<T>()` | Accumulate into array (messages, logs) |
| `prepend<T>()` | Newest first |
| `sum()` | Numeric counter |
| `setUnion<T>()` | Set merge |
| `mapMerge<T>()` | Shallow object merge |
| `binaryOperator(op, default)` | Custom merge function |

**Runtime safeguards:**
- Compile-time DFS cycle detection
- Runtime state fingerprinting for infinite loop detection
- Per-node and per-graph wall-clock timeouts
- Deadlock detection (unreachable finish points)
- `GraphInterrupt` for HITL pauses with `Command`-based resume

---

### 9. GraphRestServer — LangGraph Platform-Compatible REST API

```typescript
import { GraphRestServer } from "chorus-engine/graph";

const server = new GraphRestServer({
  graph: compiledGraph,
  port: 8123,
  apiKey: "optional-api-key",
});
await server.start();
```

Endpoints: `POST /threads`, `GET /threads/:id`, `PATCH /threads/:id`, `DELETE /threads/:id`, `POST /threads/:id/runs`, `GET /threads/:id/runs/:runId`, `POST /threads/:id/runs/:runId/stream` (SSE), `POST /threads/:id/runs/:runId/resume`, `GET /threads/:id/checkpoints`

---

### 10. Harness — Enterprise Task Routing & Worker Execution

Semantic intent classification with hybrid fallback + multi-worker parallel/pipeline execution.

```typescript
import {
  SemanticTaskRouter,
  routeTaskSemantic,
  executeWorkers,
  prepareTaskExecutionAsync,
  type WorkerExecutionResult,
} from "chorus-engine/harness";
```

**Semantic routing (embedding-based, ~94% accuracy):**
```typescript
const router = new SemanticTaskRouter({ confidenceThreshold: 0.75 });
const route = await router.route({ text: "Refactor the auth module to use JWT" });
// { kind: "multi_file_edit", confidence: 0.91, method: "semantic", ... }

// Multi-label scoring for ambiguity detection
const scores = await router.score({ text: "Check the login code" });
// [{ label: "debug", confidence: 0.72 }, { label: "single_file_edit", confidence: 0.68 }, ...]
```

**Worker execution (parallel or pipeline):**
```typescript
const results = await executeWorkers({
  assignments: [
    { workerId: "w1", role: "researcher", ... },
    { workerId: "w2", role: "planner", ... },
    { workerId: "w3", role: "coder", ... },
  ],
  executionMode: "pipeline",  // or "parallel"
  concurrency: 3,
  provider,
  model: "gpt-4o",
  taskText: "Build a REST API for user management",
  onEvent: (event) => console.log(event),
  parentTurnId: "turn-1",
});
```

**Task-to-worker routing:**
```typescript
const prepared = await prepareTaskExecutionAsync({
  text: "Fix the null pointer in auth.ts",
  expandedText: "...",
  basePrompt: "...",
  messages: [...],
});
// prepared.route.kind → "debug"
// prepared.route.lane → "foreground_sync"
// prepared.protocol.stages → ["classified", "inspected", "planned", "edited", "verified", "reviewed", "finalized"]
```

---

### 11. Adaptive Skill Runtime (ASR) — Skills That Learn

A four-layer stack that observes, synthesizes, and routes skills:

| Layer | Name | Description |
|-------|------|-------------|
| L0 | Primitives | Tool definitions (filesystem, shell, git, web search) |
| L1 | Skills | Human-authored SKILL.md files with workflows |
| L2 | Patterns | Auto-synthesized from successful tool trajectories via LCS alignment |
| L3 | Metaskills | Skills that orchestrate other skills (swarms) |

```typescript
import { SkillRegistry, TrajectorySynthesizer, SkillMiddleware, createSkillHarness } from "chorus-engine";

const harness = createSkillHarness(["./my-skills"]);
const selection = await harness.routeForTurn(history, contextWindow, systemPrompt);
// selection.skills → most relevant SkillDef[]
// selection.patterns → most relevant PatternDef[]
// selection.schemas → injectable prompt fragments
// selection.tokensUsed → budget consumed
```

**Skill synthesis pipeline:**
1. Observe successful tool trajectories
2. Cluster with k-medoids annealing (consensus selection)
3. Align via Longest Common Subsequence
4. Generalize into typed `PatternDef` with parameters
5. Register for future semantic routing

**Curation rules (automatic):**
- 5+ invocations, >80% success → promoted to **trusted**
- 10+ invocations, <40% success → **deprecated**
- 3-10 invocations, 40-60% success → **watch** status

---

### 12. Telemetry — OpenTelemetry Native

```typescript
import { OTelExporter, swarmEventsToSpans } from "chorus-engine";

const exporter = new OTelExporter({
  endpoint: "https://api.honeycomb.io/v1/traces",
  headers: { "X-Honeycomb-Team": process.env.HONEYCOMB_KEY! },
  batchSize: 100,
  flushIntervalMs: 5_000,
  redactSensitive: true,
});
```

**Spans produced:**
- `agent.loop` — Full run, thread ID, model
- `agent.round` — Per-round, token counts
- `agent.tool_call` — Per-tool, duration, error
- `harness.worker` — Per-worker execution

**Redaction:** API keys, bearer tokens, passwords, credit card numbers auto-scrubbed from span attributes.

---

### 13. Guardrails — Tiered Safety Validation

```typescript
import { runGuardrails, BuiltInGuardrails } from "chorus-engine";

const config: GuardrailsConfig = {
  runAll: true,        // Run all guardrails, not halt on first
  haltOn: "critical",  // or "warning" | "info"
  inputs: [BuiltInGuardrails.noPersonallyIdentifiableInfo],
  outputs: [BuiltInGuardrails.noCodeInjection],
  tools: [
    BuiltInGuardrails.noShellInjection,
    BuiltInGuardrails.pathTraversalPrevention,
  ],
};
```

Three tiers: **Input guardrails** (before LLM call), **Output guardrails** (before final response), **Tool guardrails** (before tool execution). Each can halt, warn, or pass.

---

### 14. MCP — Model Context Protocol

```typescript
import { getMcpTools } from "chorus-engine/mcp";

const tools = await getMcpTools();
// Returns AgentTool[] from all configured MCP servers (stdio, SSE, HTTP)
```

Supports server management (add, remove, list, update), stdio/SSE/HTTP transports, token limiting per server, and OAuth authentication.

---

### 15. A2A — Agent-to-Agent Protocol

```typescript
import { createSwarmA2AServer } from "chorus-engine/a2a";

const server = createSwarmA2AServer({
  agentCard: { name: "My Swarm", description: "...", capabilities: {...} },
  swarmConfig: mySwarmConfig,
  port: 9090,
});
await server.start();
```

JSON-RPC 2.0 over HTTP. Expose your swarm as a discoverable agent endpoint.

---

### 16. LLM Providers — Pluggable Abstraction

```typescript
import { createProvider, getDefaultProvider, type LLMProvider } from "chorus-engine/llm";
```

Built-in support: OpenAI, Anthropic, Ollama, vLLM, custom HTTP endpoints.

**Provider interface:**
```typescript
interface LLMProvider {
  readonly name: ProviderName;                           // any string
  generate(input: GenerationRequest): Promise<GenerationResult>;
  streamWithTools(input): AsyncIterable<ToolStreamEvent>;  // token + done events
  estimateCost?(inputTokens: number, outputTokens: number): number;
  health(): Promise<ProviderHealth>;
}
```

---

### 17. Tools — Built-in Tool Suite

```typescript
import {
  createFilesystemTools,  // read, write, edit, ls, glob, grep
  shellTools,             // run_command with safety audit
  gitTools,               // status, diff, log, branch, commit
  assessCommandSafety,    // scores shell commands for risk
} from "chorus-engine/tools";
```

| Tool | Description |
|------|-------------|
| `file_read` | Read file contents with offset/limit |
| `file_write` | Write content to file |
| `file_edit` | Exact string replacement in file |
| `list_dir` | List directory contents |
| `find_files` | Glob pattern matching |
| `search_files` | Grep / ripgrep file content search |
| `run_command` | Execute shell command (with safety audit) |
| `git_status`, `git_diff`, `git_log`, `git_branch`, `git_commit` | Git operations |
| `internet_search` | Web search (Serper/Google CSE) |
| `write_todos` | Structured task tracking |

---

### 18. Evals — Agent Evaluation Framework

```typescript
import { runEvalSuite, formatEvalRun, type EvalSuite } from "chorus-engine/evals";

const suite: EvalSuite = {
  name: "My Agent Eval",
  cases: [
    { id: "math-1", input: "What is 12 * 34?", expectedOutput: "408" },
    { id: "math-2", input: "Calculate 15% of 200", expectedOutput: "30" },
  ],
  scorers: [{ type: "exact_match" }, { type: "contains", value: "408" }],
};

const run = await runEvalSuite(suite, {
  provider: createProvider("openai"),
  model: "gpt-4o",
  tools: [calculatorTool],
  parallel: 5,
});
console.log(formatEvalRun(run));
// Passed: 2/2 | Avg score: 1.00 | Duration: 1.2s
```

---

### 19. Memory — Persistent Agent Memory

```typescript
import { createMemoryTools, createSharedMemoryTools } from "chorus-engine";

const memoryTools = createMemoryTools({ scope: "project-knowledge" });
// memory_recall, memory_store, memory_search, memory_compact
```

Structured memory with scoped namespaces, embedding-based recall, and automatic compaction.

---

### 20. Retry — Configurable Policies

```typescript
import { withRetry, DEFAULT_RETRY_POLICY, RATE_LIMIT_RETRY_POLICY, isRetryable } from "chorus-engine";

const { value, attempts } = await withRetry(
  async () => provider.generate(req),
  RATE_LIMIT_RETRY_POLICY,  // 5 attempts, exponential + jitter
);
```

| Policy | Attempts | Backoff | Use Case |
|--------|----------|---------|----------|
| `DEFAULT_RETRY_POLICY` | 3 | 500ms × 2^attempt, capped 8s | Tool calls |
| `RATE_LIMIT_RETRY_POLICY` | 5 | 1s × 2^attempt + 20% jitter, capped 30s | LLM calls |

---

## Quick Start

```bash
npm install chorus-engine
```

```typescript
import { configureEngine, runAgentLoop, HitlGate, BtwQueue, JsonFileCheckpointer } from "chorus-engine";
import { createProvider } from "chorus-engine/llm";

configureEngine({
  llm: {
    provider: "openai",
    providers: {
      openai: {
        apiKey: process.env.OPENAI_API_KEY!,
        model: "gpt-4o-mini",
      },
    },
  },
});

const provider = createProvider("openai");

const stream = runAgentLoop({
  provider,
  model: "gpt-4o-mini",
  tools: [],
  messages: [{ role: "user", content: "Explain async generators in 3 sentences." }],
  systemPrompt: "You are a concise TypeScript expert.",
  threadId: "demo-1",
  hitlGate: new HitlGate(),
  btwQueue: new BtwQueue(),
  policy: "full_auto",
  checkpointer: new JsonFileCheckpointer(),
});

for await (const event of stream) {
  if (event.type === "token") process.stdout.write(event.text);
  if (event.type === "done") console.log(`\n\nCost: $${event.costUsd.toFixed(6)}`);
  if (event.type === "error") console.error("Error:", event.message);
}
```

---

## Requirements

- Node.js >= 20.0.0
- TypeScript >= 5.7 (for consumers using type imports)

---

## License

MIT
