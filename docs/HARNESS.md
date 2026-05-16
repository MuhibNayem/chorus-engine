# Enterprise Harness

The harness module (`chorus-engine/harness`) provides production-grade task routing and multi-worker execution. It replaces naive parallel prompting with **semantic routing**, **phased execution**, and **shared context passing**.

## Quick Start

```typescript
import {
  executeWorkers,
  prepareTaskExecutionAsync,
  SemanticTaskRouter,
  routeTaskSemantic,
  formatWorkerResults,
  type WorkerExecutionResult,
} from "chorus-engine/harness";
```

---

## Semantic Router

Embedding-based intent classification with ~94% accuracy at ~60% cost reduction vs pure LLM classification. Uses the same MiniLM embedder as the skills layer.

```typescript
import { routeTaskSemantic, SemanticTaskRouter } from "chorus-engine/harness";

// One-shot classification
const route = await routeTaskSemantic({ text: "Debug the auth module" });

// Reusable router with custom threshold
const router = new SemanticTaskRouter({ confidenceThreshold: 0.75 });
const result = await router.route({ text: "Refactor the auth module to use JWT" });
// {
//   kind: "multi_file_edit",
//   confidence: 0.91,
//   method: "semantic",
//   lane: "foreground_sync",
//   path: "parallel_multi_worker_path",
//   requiresResearch: false,
//   canParallelize: true,
//   usesCheapTriage: false
// }
```

### Route Kinds

| Kind | Lane | Path | Description |
|------|------|------|-------------|
| `answer_only` | `cheap_triage` | `direct_agent_path` | Simple Q&A, no tools needed |
| `inspect_only` | `cheap_triage` | `direct_agent_path` | Read/analyze files, no edits |
| `single_file_edit` | `foreground_sync` | `tool_or_single_worker_path` | One-file change |
| `multi_file_edit` | `foreground_sync` | `parallel_multi_worker_path` | Cross-file change |
| `debug` | `foreground_sync` | `tool_or_single_worker_path` | Diagnostic / error fixing |
| `research` | `foreground_sync` | `research_then_plan_path` | Requires external knowledge |
| `project_phase` | `background_async` | `background_or_batch_path` | Entire-project task |

### Multi-Label Scoring

Get confidence for ALL routes — useful for ambiguity detection and multi-intent routing:

```typescript
const scores = await router.score({ text: "Find and fix the bug" });
// [
//   { label: "debug", confidence: 0.85 },
//   { label: "research", confidence: 0.42 },
//   { label: "single_file_edit", confidence: 0.31 },
//   { label: "answer_only", confidence: 0.18 },
//   { label: "multi_file_edit", confidence: 0.15 },
//   { label: "project_phase", confidence: 0.08 },
//   { label: "inspect_only", confidence: 0.05 },
// ]

// Ambiguity detection: top-2 gap < 0.2 → ask clarifying question
if (scores[0].confidence - scores[1].confidence < 0.2) {
  // Ask the user to clarify
}
```

### Hybrid Classification Fallback

1. **Embedding similarity** (fast path): Cosine similarity against 7 route prototypes with multi-vector scoring
2. **Regex fallback** (slow path): If confidence < threshold (default 0.55), use keyword heuristics
3. **Never blocks**: Worst case routes to `answer_only`

### Performance

| Metric | Value |
|--------|-------|
| Latency | ~50ms (local MiniLM embedding) |
| Accuracy | ~94% (research benchmark) |
| Cost | ~60% less than LLM-based classification |

---

## Worker Execution

### Parallel Mode (default)

All workers run concurrently, bounded by `concurrency` (default: 3):

```typescript
const results = await executeWorkers({
  assignments: [
    { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "ctx-1", status: "queued" },
    { workerId: "w2", role: "planner", ownedScope: [], inputBundleId: "ctx-1", status: "queued" },
    { workerId: "w3", role: "coder", ownedScope: ["workspace"], inputBundleId: "ctx-1", status: "queued" },
  ],
  executionMode: "parallel",
  concurrency: 3,
  provider,
  model: "gpt-4o",
  taskText: "Build a user management API",
  onEvent: (event) => {
    if (event.type === "worker-add") console.log(`${event.emoji} ${event.role} started`);
    if (event.type === "worker-update") console.log(`${event.workerId}: ${event.status}`);
  },
  parentTurnId: "turn-1",
  abortSignal: controller.signal,
  maxRetriesPerWorker: 3,
});
```

**Use for:** Independent tasks (research multiple topics, review multiple files).

### Pipeline Mode

Workers run sequentially, each receiving accumulated context from all previous workers:

```typescript
const results = await executeWorkers({
  assignments: [
    { workerId: "w1", role: "researcher", ... },
    { workerId: "w2", role: "planner", ... },
    { workerId: "w3", role: "coder", ... },
  ],
  executionMode: "pipeline",
});
```

Execution flow:
```
Researcher: "Found 3 relevant API patterns, 2 security concerns"
    ↓ [shared context: worker.researcher]
Planner: "Based on findings: design REST API with 4 endpoints, JWT auth"
    ↓ [shared context: worker.planner]
Coder: "Implementing: GET/POST/PUT/DELETE /users with JWT middleware"
```

**Use for:** Dependent tasks (research → plan → code → verify).

### Concurrency Control

| Option | Default | Description |
|--------|---------|-------------|
| `concurrency` | 3 | Max parallel workers (1 = sequential) |
| `abortSignal` | — | Cancel all pending workers |
| `maxRetriesPerWorker` | 3 | Retries on transient LLM failures |

### Failure Handling

| Scenario | Behavior |
|----------|----------|
| Single worker fails | Other workers continue; failure logged in result |
| All workers fail | Returns all failures; upstream decides retry |
| AbortSignal triggered | Pending workers cancelled; resolved results returned |
| Provider error | Retried per `withRetry` policy (3× backoff) |

---

## Worker Roles

| Role | Phase | System Prompt | Purpose |
|------|-------|---------------|---------|
| `researcher` | Discovery | Research-focused | Gather context, search docs, find examples |
| `planner` | Planning | Architecture-focused | Design approach, break down tasks |
| `coder` | Execution | Implementation-focused | Generate code, apply edits |
| `reviewer` | Verification | Review-focused | Review output, flag issues, assess risk |
| `tester` | Verification | Test-focused | Generate tests, verify correctness |
| `orchestrator` | Coordination | Coordination-focused | Coordinate multi-worker flows |
| `advisor` | Advisory | Advisory-focused | Review plans and advise (inserted between planner and coder when enabled) |

---

## Shared Context

Workers can read/write a shared key-value store across parallel or pipeline execution:

```typescript
const shared = new InMemorySharedContext();
shared.set("design.decisions", ["Use REST", "JWT auth"]);

const results = await executeWorkers({
  assignments,
  executionMode: "pipeline",
  sharedContext: shared,
});

// Results auto-stored by key
shared.get("worker.researcher");    // WorkerExecutionResult
shared.get("worker.planner");       // WorkerExecutionResult
shared.get("worker.coder");         // WorkerExecutionResult
```

Default shared context is created automatically in pipeline mode. Results are keyed by `worker.{role}`.

---

## Task Routing Integration

Full pipeline from user message to execution:

```typescript
import { prepareTaskExecutionAsync, executeWorkers } from "chorus-engine/harness";

const prepared = await prepareTaskExecutionAsync({
  text: "Refactor the auth module to use JWT instead of sessions",
  expandedText: "[File: src/auth/sessions.ts] [File: src/middleware/auth.ts]",
  basePrompt: "You are a senior TypeScript engineer.",
  messages: conversationHistory,
  mode: "build",
});

// prepared.route.kind         → "multi_file_edit"
// prepared.route.lane         → "foreground_sync"
// prepared.route.path         → "parallel_multi_worker_path"
// prepared.protocol.stages    → ["classified", "inspected", "planned", "edited", "verified", "reviewed", "finalized"]
// prepared.workerAssignments  → [{ role: "planner" }, { role: "coder" }, { role: "reviewer" }, { role: "tester" }]
// prepared.runtimePrompt      → Full formatted system prompt with all intelligence
// prepared.repoIntelligence   → { languages: ["TypeScript"], packageManager: "npm", testSignals: ["vitest"], ... }

const results = await executeWorkers({
  assignments: prepared.workerAssignments,
  taskText: prepared.runtimePrompt,
  provider,
  model: "gpt-4o",
  onEvent: handleWorkerEvent,
  parentTurnId: prepared.task.taskId,
});

console.log(formatWorkerResults(results));
```

---

## Worker Events

The `onEvent` callback receives structured events:

```typescript
type WorkerEvent =
  | { type: "worker-add"; workerId: string; role: WorkerRole; emoji: string; color: string; status: "running"; summary: string; sessionId: string }
  | { type: "worker-thinking"; sessionId: string; id: string; text: string; expanded: boolean }
  | { type: "worker-response"; sessionId: string; text: string }
  | { type: "worker-main-turn-thinking"; sessionId: string; id: string; text: string; expanded: boolean }
  | { type: "worker-session-complete"; sessionId: string; completedAt: number }
  | { type: "worker-update"; workerId: string; status: "done" | "error"; result: string };
```
