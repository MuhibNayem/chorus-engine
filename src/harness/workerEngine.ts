import type { LLMProvider } from "../llm/provider.js";
import type { WorkerAssignment, WorkerRole } from "./types.js";
import { getWorkerSystemPrompt } from "./workerPrompts.js";
import { withRetry, DEFAULT_RETRY_POLICY } from "../agent/retry.js";

// ── Worker event types (replaces Dispatch<FeedAction>) ────────────────────────

export type WorkerEvent =
  | {
      type: "worker-add";
      workerId: string;
      role: WorkerRole;
      emoji: string;
      color: string;
      status: "running";
      summary: string;
      sessionId: string;
    }
  | {
      type: "worker-thinking";
      sessionId: string;
      id: string;
      text: string;
      expanded: boolean;
    }
  | {
      type: "worker-response";
      sessionId: string;
      text: string;
    }
  | {
      type: "worker-main-turn-thinking";
      sessionId: string;
      id: string;
      text: string;
      expanded: boolean;
    }
  | {
      type: "worker-session-complete";
      sessionId: string;
      completedAt: number;
    }
  | {
      type: "worker-update";
      workerId: string;
      status: "done" | "error";
      result: string;
    };

export type WorkerEventCallback = (event: WorkerEvent) => void;

// ── Shared Context ────────────────────────────────────────────────────────────

/** Thread-safe (per-execution) key/value store for inter-worker communication. */
export interface SharedWorkerContext {
  get<T = unknown>(key: string): T | undefined;
  set<T>(key: string, value: T): void;
  has(key: string): boolean;
  entries(): Readonly<Record<string, unknown>>;
}

class InMemorySharedContext implements SharedWorkerContext {
  private store = new Map<string, unknown>();

  get<T = unknown>(key: string): T | undefined {
    return this.store.get(key) as T | undefined;
  }
  set<T>(key: string, value: T): void {
    this.store.set(key, value);
  }
  has(key: string): boolean {
    return this.store.has(key);
  }
  entries(): Readonly<Record<string, unknown>> {
    return Object.fromEntries(this.store);
  }
}

// ── Interfaces ────────────────────────────────────────────────────────────────

export interface WorkerExecutionOptions {
  assignments: WorkerAssignment[];
  taskText: string;
  provider: LLMProvider;
  model: string;
  onEvent: WorkerEventCallback;
  parentTurnId: string;
  /**
   * Maximum concurrent workers. Default: 3 (balanced parallelism without
   * overwhelming provider rate limits). Set to 1 for sequential execution,
   * or increase for higher throughput on high-capacity providers.
   */
  concurrency?: number;
  /**
   * Execution topology:
   *   - "parallel": All workers run concurrently (subject to `concurrency`).
   *   - "pipeline": Workers run sequentially, each receiving accumulated
   *     outputs from all previous workers. Enables researcher → planner →
   *     advisor chains.
   * Default: "parallel".
   */
  executionMode?: "parallel" | "pipeline";
  /**
   * Shared mutable state accessible to all workers. In pipeline mode,
   * workers may read/write this to coordinate. In parallel mode, workers
   * may write results here for aggregation.
   */
  sharedContext?: SharedWorkerContext;
  /** AbortSignal for cooperative cancellation. */
  abortSignal?: AbortSignal;
  /** Max retries per worker on transient LLM failures. Default: 3. */
  maxRetriesPerWorker?: number;
}

export interface WorkerExecutionResult {
  workerId: string;
  role: WorkerRole;
  summary: string;
  findings: string[];
  durationMs: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function roleEmoji(role: WorkerRole): string {
  switch (role) {
    case "researcher":   return "🔍";
    case "planner":      return "🏗️";
    case "coder":        return "💻";
    case "reviewer":     return "👁️";
    case "tester":       return "🧪";
    case "orchestrator": return "🎛️";
    case "advisor":      return "🧠";
    default:             return "🤖";
  }
}

function roleColor(role: WorkerRole): string {
  switch (role) {
    case "researcher":   return "cyan";
    case "planner":      return "blue";
    case "coder":        return "green";
    case "reviewer":     return "yellow";
    case "tester":       return "magenta";
    case "orchestrator": return "white";
    case "advisor":      return "cyanBright";
    default:             return "gray";
  }
}

function parseFindings(summary: string): string[] {
  const findings: string[] = [];
  const lines = summary.split("\n");
  let inList = false;
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || /^\d+\./.test(trimmed)) {
      findings.push(trimmed.replace(/^[-*\d.\s]+/, "").trim());
      inList = true;
    } else if (inList && trimmed && !trimmed.startsWith("#")) {
      if (findings.length > 0) {
        findings[findings.length - 1] += " " + trimmed;
      }
    } else if (trimmed.startsWith("##")) {
      inList = false;
    }
  }
  return findings.length > 0 ? findings : [summary.slice(0, 200)];
}

// ── Core execution ────────────────────────────────────────────────────────────

async function executeSingleWorker(
  assignment: WorkerAssignment,
  taskText: string,
  provider: LLMProvider,
  model: string,
  onEvent: WorkerEventCallback,
  parentTurnId: string,
  maxRetries: number,
): Promise<WorkerExecutionResult> {
  const startedAt = Date.now();
  const workerId = assignment.workerId;
  const role = assignment.role;
  const sessionId = `session-${workerId}`;

  onEvent({
    type: "worker-add",
    workerId,
    role,
    emoji: roleEmoji(role),
    color: roleColor(role),
    status: "running",
    summary: `${roleEmoji(role)} ${role} analyzing…`,
    sessionId,
  });

  onEvent({
    type: "worker-thinking",
    sessionId,
    id: `${sessionId}-think-0`,
    text: `Starting ${role} analysis…`,
    expanded: false,
  });

  try {
    const systemPrompt = getWorkerSystemPrompt(role);
    const { value: result, attempts } = await withRetry(
      async () =>
        provider.generate({
          model,
          systemPrompt,
          messages: [
            { role: "user", content: `Task: ${taskText}\n\nProvide your analysis.` },
          ],
        }),
      { ...DEFAULT_RETRY_POLICY, maxAttempts: maxRetries },
    );

    const summary = result.text.trim();
    const findings = parseFindings(summary);
    const durationMs = Date.now() - startedAt;

    onEvent({ type: "worker-response", sessionId, text: summary });
    onEvent({
      type: "worker-main-turn-thinking",
      sessionId,
      id: `${sessionId}-result`,
      text: `${roleEmoji(role)} ${role} completed in ${durationMs}ms${attempts > 1 ? ` (${attempts} attempts)` : ""}:\n\n${summary}`,
      expanded: false,
    });
    onEvent({ type: "worker-session-complete", sessionId, completedAt: Date.now() });
    onEvent({
      type: "worker-update",
      workerId,
      status: "done",
      result: `${roleEmoji(role)} ${role} — ${findings.length} finding${findings.length === 1 ? "" : "s"} (${durationMs}ms)`,
    });

    return { workerId, role, summary, findings, durationMs };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const durationMs = Date.now() - startedAt;

    onEvent({ type: "worker-response", sessionId, text: `Error: ${message}` });
    onEvent({ type: "worker-session-complete", sessionId, completedAt: Date.now() });
    onEvent({
      type: "worker-update",
      workerId,
      status: "error",
      result: `${roleEmoji(role)} ${role} failed: ${message}`,
    });

    return {
      workerId,
      role,
      summary: `Worker ${role} failed: ${message}`,
      findings: [message],
      durationMs,
    };
  }
}

/** Pipeline mode: sequential execution with accumulated context. */
async function executePipeline(
  assignments: WorkerAssignment[],
  options: Omit<WorkerExecutionOptions, "assignments" | "executionMode" | "concurrency">,
  sharedContext: SharedWorkerContext,
): Promise<WorkerExecutionResult[]> {
  const results: WorkerExecutionResult[] = [];
  const { taskText, provider, model, onEvent, parentTurnId, abortSignal, maxRetriesPerWorker = 3 } = options;

  for (const assignment of assignments) {
    if (abortSignal?.aborted) {
      break;
    }

    // Build accumulated context from all previous workers
    const accumulatedContext = [
      `Original task: ${taskText}`,
      results.length > 0
        ? `\nPrevious analyses:\n${results
            .map((r) => `[${r.role}]: ${r.summary.slice(0, 500)}`)
            .join("\n")}`
        : "",
    ].join("\n");

    const result = await executeSingleWorker(
      assignment,
      accumulatedContext,
      provider,
      model,
      onEvent,
      parentTurnId,
      maxRetriesPerWorker,
    );

    results.push(result);
    sharedContext.set(`worker.${assignment.role}`, result);
  }

  return results;
}

/** Parallel mode with bounded concurrency. */
async function executeParallel(
  assignments: WorkerAssignment[],
  options: Omit<WorkerExecutionOptions, "assignments" | "executionMode" | "concurrency">,
  sharedContext: SharedWorkerContext,
  concurrency: number,
): Promise<WorkerExecutionResult[]> {
  const results: WorkerExecutionResult[] = [];
  const { taskText, provider, model, onEvent, parentTurnId, abortSignal, maxRetriesPerWorker = 3 } = options;

  if (concurrency <= 1) {
    // Sequential but without pipeline context accumulation
    for (const assignment of assignments) {
      if (abortSignal?.aborted) break;
      const result = await executeSingleWorker(
        assignment, taskText, provider, model, onEvent, parentTurnId, maxRetriesPerWorker,
      );
      results.push(result);
      sharedContext.set(`worker.${assignment.role}`, result);
    }
    return results;
  }

  // Worker pool with bounded concurrency
  const queue = [...assignments];
  const inFlight = new Set<Promise<void>>();
  let aborted = false;

  abortSignal?.addEventListener("abort", () => {
    aborted = true;
  });

  while (queue.length > 0 || inFlight.size > 0) {
    if (aborted) {
      await Promise.all(inFlight);
      break;
    }

    while (inFlight.size < concurrency && queue.length > 0) {
      const assignment = queue.shift()!;
      const promise = executeSingleWorker(
        assignment, taskText, provider, model, onEvent, parentTurnId, maxRetriesPerWorker,
      )
        .then((result) => {
          results.push(result);
          sharedContext.set(`worker.${assignment.role}`, result);
        })
        .finally(() => {
          inFlight.delete(promise);
        });
      inFlight.add(promise);
    }

    if (inFlight.size > 0) {
      await Promise.race(inFlight);
    }
  }

  return results;
}

export async function executeWorkers(
  options: WorkerExecutionOptions,
): Promise<WorkerExecutionResult[]> {
  if (options.assignments.length === 0) return [];

  const sharedContext = options.sharedContext ?? new InMemorySharedContext();
  const mode = options.executionMode ?? "parallel";
  const concurrency = options.concurrency ?? 3;

  if (mode === "pipeline") {
    return executePipeline(options.assignments, options, sharedContext);
  }

  return executeParallel(options.assignments, options, sharedContext, concurrency);
}

export function formatWorkerResults(results: WorkerExecutionResult[]): string {
  if (results.length === 0) return "";
  const lines: string[] = ["\n--- Worker Analysis ---\n"];
  for (const result of results) {
    lines.push(`## ${result.role} (${result.durationMs}ms)`);
    lines.push(result.summary);
    lines.push("");
  }
  lines.push("--- End Worker Analysis ---\n");
  return lines.join("\n");
}
