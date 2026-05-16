/**
 * Graph Executor — DAG-based parallel wave execution for swarms.
 *
 * Agents declare `dependsOn` to form a dependency graph. The executor
 * computes topological waves and runs each wave's agents concurrently.
 * Agents share state through artifacts (set_artifact / get_artifact).
 *
 * Safety contract: agents with `permissionMode: "auto_edit"` cannot run in
 * parallel waves (they may pause awaiting HITL approval). They must be the
 * only agent in their wave, or use "full_auto" / "suggest" instead.
 */

import { randomUUID } from "crypto";
import { runAgentLoop } from "../agent/loop.js";
import { createDefaultMiddleware } from "../agent/middleware.js";
import { JsonFileCheckpointer } from "../agent/checkpointer.js";
import { HitlGate } from "../agent/hitl.js";
import { BtwQueue } from "../agent/btw.js";
import type { AgentMetrics, SwarmAgent, SwarmConfig, SwarmEvent } from "./types.js";
import { createArtifactTools } from "./session.js";
import { SwarmTracer } from "./trace.js";
import { createFilesystemTools } from "../tools/filesystem.js";
import { createWorktree, isGitRepo } from "./worktree.js";
import type { WorktreeHandle } from "./worktree.js";
import { SwarmCheckpointer } from "./swarm-checkpointer.js";
import { checkCostBudget, resolveModel } from "./cost-router.js";
import { globalBroadcaster } from "../channels/broadcaster.js";
import { InProcessTracer } from "../telemetry/inprocess.js";

// ─── Async Generator Merge ───────────────────────────────────────────────────

/**
 * Merge multiple AsyncGenerators into a single stream. Events arrive in
 * completion order, not declaration order. Exceptions in any generator are
 * surfaced as `error` events rather than propagating (so one failing agent
 * does not silently cancel its wave siblings).
 *
 * Cleanup contract:
 *   • Accepts an AbortSignal for external cancellation.
 *   • When the caller stops iterating (break/return/throw), the `finally`
 *     block calls `.return()` on every underlying iterator so background
 *     generators close promptly and release resources.
 *   • Background tasks are awaited before the generator returns, preventing
 *     unhandled promise rejections and generator leaks.
 */
async function* mergeGenerators<T extends SwarmEvent>(
  gens: AsyncGenerator<T>[],
  signal?: AbortSignal,
): AsyncGenerator<T> {
  if (gens.length === 0) return;

  const queue: T[] = [];
  let waiting: (() => void) | null = null;
  let done = 0;
  let cancelled = false;

  function enqueue(item: T): void {
    if (cancelled) return;
    queue.push(item);
    if (waiting) {
      const w = waiting;
      waiting = null;
      w();
    }
  }

  function finish(): void {
    done++;
    if (waiting) {
      const w = waiting;
      waiting = null;
      w();
    }
  }

  const abortHandler = (): void => {
    cancelled = true;
    if (waiting) {
      const w = waiting;
      waiting = null;
      w();
    }
  };

  signal?.addEventListener("abort", abortHandler, { once: true });

  // Obtain direct iterator references so we can call .return() on cleanup.
  const iterators = gens.map((gen) => gen[Symbol.asyncIterator]() as AsyncGenerator<T>);

  const tasks = iterators.map(async (iter) => {
    try {
      while (!cancelled) {
        const result = await iter.next();
        if (result.done) break;
        enqueue(result.value);
      }
    } catch (err) {
      if (!cancelled) {
        enqueue({
          type: "circuit-break",
          agent: "unknown",
          reason: `Generator error: ${String(err)}`,
        } as unknown as T);
      }
    } finally {
      finish();
    }
  });

  try {
    while (done < gens.length && !cancelled) {
      if (queue.length > 0) {
        yield queue.shift()!;
      } else {
        await new Promise<void>((resolve) => {
          waiting = resolve;
        });
      }
    }

    // Drain remaining queue after cancellation or normal completion.
    while (queue.length > 0 && !cancelled) {
      yield queue.shift()!;
    }
  } finally {
    cancelled = true;
    signal?.removeEventListener("abort", abortHandler);
    // Signal every underlying iterator to close so generators unwind.
    await Promise.all(
      iterators.map(async (iter) => {
        try {
          if (iter.return) {
            await iter.return(undefined);
          }
        } catch {
          // ignore cleanup errors
        }
      }),
    );
    // Wait for background consumption tasks to settle.
    await Promise.all(
      tasks.map(async (task) => {
        try {
          await task;
        } catch {
          // ignore
        }
      }),
    );
  }
}

// ─── Wave Computation ────────────────────────────────────────────────────────

/**
 * Topological sort of agents into execution waves.
 * Wave 0 contains agents with no dependencies; each subsequent wave contains
 * agents whose all dependencies appear in earlier waves.
 *
 * Throws on unknown dependency references or circular dependencies.
 */
export function computeWaves(agents: SwarmAgent[]): SwarmAgent[][] {
  const knownNames = new Set(agents.map((a) => a.name));

  for (const agent of agents) {
    for (const dep of agent.dependsOn ?? []) {
      if (!knownNames.has(dep)) {
        throw new Error(
          `Agent "${agent.name}" depends on unknown agent "${dep}". ` +
            `Known agents: ${[...knownNames].join(", ")}`,
        );
      }
    }
  }

  const completed = new Set<string>();
  const waves: SwarmAgent[][] = [];
  let remaining = [...agents];

  while (remaining.length > 0) {
    const wave = remaining.filter((agent) =>
      (agent.dependsOn ?? []).every((dep) => completed.has(dep)),
    );

    if (wave.length === 0) {
      throw new Error(
        `Circular dependency detected. Remaining agents: ${remaining.map((a) => a.name).join(", ")}`,
      );
    }

    waves.push(wave);
    for (const a of wave) completed.add(a.name);
    remaining = remaining.filter((a) => !completed.has(a.name));
  }

  return waves;
}

// ─── Filesystem Tool Names (for worktree replacement) ────────────────────────

const FS_TOOL_NAMES = new Set([
  "file_read",
  "file_write",
  "file_edit",
  "list_dir",
  "find_files",
  "search_files",
]);

// ─── Signal Utilities ────────────────────────────────────────────────────────

/** Combine two AbortSignals: aborts when either fires. */
function combineSignals(a: AbortSignal, b: AbortSignal): AbortSignal {
  const ctrl = new AbortController();
  if (a.aborted || b.aborted) { ctrl.abort(); return ctrl.signal; }
  a.addEventListener("abort", () => ctrl.abort(), { once: true });
  b.addEventListener("abort", () => ctrl.abort(), { once: true });
  return ctrl.signal;
}

// ─── Single-Agent Graph Runner ───────────────────────────────────────────────

async function* runAgentInGraph(
  agent: SwarmAgent,
  config: SwarmConfig,
  readArtifacts: Readonly<Record<string, string>>,
  writeArtifacts: Record<string, string>,
  swarmId: string,
  abortSignal?: AbortSignal,
  inProcessTracer?: InProcessTracer,
): AsyncGenerator<SwarmEvent> {
  const permissionMode = agent.permissionMode ?? config.policy ?? "full_auto";
  const threadId = `${swarmId}-${agent.name}`;
  const checkpointer = config.checkpointer ?? new JsonFileCheckpointer();
  const middleware = createDefaultMiddleware(threadId);

  // ─── Worktree isolation setup ───────────────────────────────────────────
  let worktreeHandle: WorktreeHandle | null = null;

  if (agent.isolation === "worktree") {
    if (!isGitRepo(process.cwd())) {
      yield {
        type: "worktree-error",
        agent: agent.name,
        reason: "Not in a git repository — running without worktree isolation.",
      };
    } else {
      try {
        worktreeHandle = createWorktree(agent.name, swarmId);
        yield {
          type: "worktree-created",
          agent: agent.name,
          path: worktreeHandle.path,
          branch: worktreeHandle.branch,
        };
      } catch (err) {
        yield {
          type: "worktree-error",
          agent: agent.name,
          reason: `Failed to create worktree: ${String(err)} — running without isolation.`,
        };
      }
    }
  }

  try {
    const artifactKeys = Object.keys(readArtifacts);
    const artifactContext =
      artifactKeys.length > 0
        ? `\n\nArtifacts available from prior stages: ${artifactKeys.join(", ")}. Use get_artifact to retrieve them.`
        : "";

    const worktreePath = worktreeHandle?.path;
    const workspaceNote = worktreePath
      ? `\n\nYou are working in an isolated git worktree at: ${worktreePath}`
      : "";

    const messages = [
      { role: "user" as const, content: `${config.task}${artifactContext}${workspaceNote}` },
    ];

    const systemPrompt = config.spec
      ? `${agent.systemPrompt}\n\n---\nInvariant goal: ${config.spec}`
      : agent.systemPrompt;

    // Replace filesystem tools with worktree-scoped versions when isolated
    let agentTools = agent.tools;
    if (worktreePath) {
      const wtFsTools = createFilesystemTools(worktreePath);
      agentTools = [
        ...agent.tools.filter((t) => !FS_TOOL_NAMES.has(t.name ?? "")),
        ...wtFsTools,
      ];
    }

    const artifactTools = createArtifactTools(readArtifacts, writeArtifacts);
    const allTools = [...agentTools, ...artifactTools];

    const hitlGate = new HitlGate();
    const btwQueue = new BtwQueue();

    yield {
      type: "agent-start",
      agent: agent.name,
      traceId: randomUUID(),
      contextMode: agent.contextMode,
    };

    let agentResponse = "";
    const agentStartMs = Date.now();
    let agentMetrics: AgentMetrics = {
      inputTokens: 0,
      outputTokens: 0,
      costUsd: 0,
      durationMs: 0,
      rounds: 0,
      toolCalls: 0,
    };

    const effectiveModel = resolveModel(
      {
        agentName: agent.name,
        defaultModel: agent.model ?? config.modelName,
        spentUsd: Object.values(readArtifacts)
          .filter((v: string) => v.startsWith("__cost__"))
          .reduce((s: number, v: string) => s + parseFloat(v.slice(8) || "0"), 0),
        budgetTotalUsd: config.costBudget?.totalUsd,
        maxRounds: agent.maxRounds,
      },
      config.costRouting,
    );

    for await (const event of runAgentLoop({
      provider: config.provider,
      model: effectiveModel,
      tools: allTools,
      messages,
      systemPrompt,
      threadId,
      hitlGate,
      btwQueue,
      policy: permissionMode,
      checkpointer,
      maxRounds: agent.maxRounds,
      middleware,
      abortSignal,
      tracer: inProcessTracer,
    })) {
      yield { ...event, agent: agent.name } as SwarmEvent;
      if (event.type === "done") {
        agentResponse = event.response;
        agentMetrics = {
          inputTokens: event.inputTokens,
          outputTokens: event.outputTokens,
          costUsd: event.costUsd,
          durationMs: event.durationMs,
          rounds: agentMetrics.rounds,
          toolCalls: event.toolCount,
        };
      }
      if (event.type === "checkpoint") {
        agentMetrics.rounds = event.round + 1;
      }
    }
    agentMetrics.durationMs = Math.max(agentMetrics.durationMs, Date.now() - agentStartMs);

    // Store the worktree path as an artifact so downstream agents can access it
    if (worktreePath) {
      writeArtifacts[`${agent.name}-worktree`] = worktreePath;
    }

    yield { type: "agent-done", agent: agent.name, responseText: agentResponse, metrics: agentMetrics };
  } finally {
    if (worktreeHandle) {
      const cleanedPath = worktreeHandle.path;
      await worktreeHandle.remove();
      yield { type: "worktree-removed", agent: agent.name, path: cleanedPath };
    }
  }
}

// ─── Graph Swarm Entry Point ─────────────────────────────────────────────────

export async function* runSwarmGraph(config: SwarmConfig): AsyncGenerator<SwarmEvent> {
  // Issue 12: Validate non-empty agents list before doing any I/O.
  if (config.agents.length === 0) {
    yield { type: "circuit-break", agent: "graph", reason: "SwarmConfig.agents must not be empty." };
    return;
  }

  const cp = new SwarmCheckpointer();

  // ─── Resume from checkpoint if requested ─────────────────────────────────
  let swarmId: string;
  const artifacts: Record<string, string> = {};
  let startWave = 0;

  if (config.resumeSwarmId) {
    const saved = cp.load(config.resumeSwarmId);
    if (saved) {
      swarmId = saved.swarmId;
      Object.assign(artifacts, saved.artifacts);
      startWave = saved.completedWaves;
    } else {
      // Checkpoint gone — start fresh with the requested ID
      swarmId = config.resumeSwarmId;
    }
  } else {
    swarmId = `swarm-graph-${Date.now()}`;
  }

  const tracer = new SwarmTracer(swarmId);
  const inProcessTracer = new InProcessTracer();
  const swarmSpan = inProcessTracer.startSpan("swarm.execution", {
    attributes: { "swarm.id": swarmId, "swarm.agents": config.agents.map((a) => a.name).join(",") },
  });
  const spansToExport: import("../telemetry/types.js").OTelSpan[] = [];

  const swarmStartMs = Date.now();
  let swarmInputTokens = 0;
  let swarmOutputTokens = 0;
  let swarmCostUsd = 0;

  // ─── Build waves ──────────────────────────────────────────────────────────
  let waves: SwarmAgent[][];
  try {
    waves = computeWaves(config.agents);
  } catch (err) {
    const cbEvent: SwarmEvent = {
      type: "circuit-break",
      agent: "graph",
      reason: String(err),
    };
    tracer.record(cbEvent);
    tracer.flush();
    spansToExport.push(inProcessTracer.endSpan(swarmSpan, { error: cbEvent.reason }));
    await inProcessTracer.export(spansToExport);
    cp.delete(swarmId); // Issue 5: clean up checkpoint on early exit
    yield cbEvent;
    return;
  }

  // Validate HITL safety: auto_edit agents cannot run in multi-agent waves
  for (const wave of waves) {
    if (wave.length > 1) {
      for (const agent of wave) {
        const mode = agent.permissionMode ?? config.policy ?? "full_auto";
        if (mode === "auto_edit") {
          const cbEvent: SwarmEvent = {
            type: "circuit-break",
            agent: agent.name,
            reason:
              `Agent "${agent.name}" has permissionMode "auto_edit" but is in a ` +
              `parallel wave (${wave.length} concurrent agents). "auto_edit" agents ` +
              `may pause for HITL approval and cannot run concurrently. ` +
              `Use "full_auto" or "suggest", or ensure this agent is in its own wave via dependsOn.`,
          };
          tracer.record(cbEvent);
          tracer.flush();
          spansToExport.push(inProcessTracer.endSpan(swarmSpan, { error: cbEvent.reason }));
          await inProcessTracer.export(spansToExport);
          cp.delete(swarmId); // Issue 5
          yield cbEvent;
          return;
        }
      }
    }
  }

  // ─── Emit start event ─────────────────────────────────────────────────────
  if (startWave > 0) {
    const resumeEvent: SwarmEvent = {
      type: "swarm-resumed",
      swarmId,
      fromWave: startWave,
      artifacts: Object.keys(artifacts),
    };
    tracer.record(resumeEvent);
    yield resumeEvent;
  } else {
    const startEvent: SwarmEvent = {
      type: "swarm-start",
      swarmId,
      agents: config.agents.map((a) => a.name),
    };
    tracer.record(startEvent);
    globalBroadcaster.broadcastSwarmEvent(swarmId, startEvent);
    yield startEvent;
  }

  // ─── Execute waves ────────────────────────────────────────────────────────
  for (let waveIdx = startWave; waveIdx < waves.length; waveIdx++) {
    if (config.abortSignal?.aborted) {
      const abortEvent: SwarmEvent = {
        type: "circuit-break",
        agent: "graph",
        reason: "Swarm aborted via AbortSignal.",
      };
      tracer.record(abortEvent);
      tracer.flush();
      spansToExport.push(inProcessTracer.endSpan(swarmSpan, { error: abortEvent.reason }));
      await inProcessTracer.export(spansToExport);
      cp.delete(swarmId); // Issue 5
      yield abortEvent;
      return;
    }

    const wave = waves[waveIdx];
    const waveAgentNames = wave.map((a) => a.name);

    const waveStartEvent: SwarmEvent = {
      type: "wave-start",
      wave: waveIdx,
      agents: waveAgentNames,
    };
    tracer.record(waveStartEvent);
    yield waveStartEvent;

    // Issue 1: Each agent gets an isolated write store (snapshot of shared artifacts as read).
    // After the wave, all write stores are merged into the shared artifacts.
    const readSnapshot: Readonly<Record<string, string>> = { ...artifacts };
    const waveWriteStores: Record<string, Record<string, string>> = {};

    // Issue 8: Per-wave AbortController so a circuit-breaking agent cancels its siblings.
    const waveCtrl = new AbortController();
    const waveSignal = config.abortSignal
      ? combineSignals(config.abortSignal, waveCtrl.signal)
      : waveCtrl.signal;

    const waveSpan = inProcessTracer.startSpan("swarm.wave", {
      parentSpanId: swarmSpan.spanId,
      attributes: { "swarm.wave": waveIdx, "swarm.agents": waveAgentNames.join(",") },
    });

    const gens = wave.map((agent) => {
      const localStore: Record<string, string> = {};
      waveWriteStores[agent.name] = localStore;
      return runAgentInGraph(agent, config, readSnapshot, localStore, swarmId, waveSignal, inProcessTracer);
    });

    let circuitTripped = false;
    for await (const event of mergeGenerators(gens, waveSignal)) {
      tracer.record(event);
      globalBroadcaster.broadcastSwarmEvent(swarmId, event);
      yield event;
      if (event.type === "agent-done") {
        swarmInputTokens += event.metrics.inputTokens;
        swarmOutputTokens += event.metrics.outputTokens;
        swarmCostUsd += event.metrics.costUsd;

        // Hard cost cap check after each agent completes
        const budgetViolation = checkCostBudget(
          event.agent,
          event.metrics.costUsd,
          swarmCostUsd,
          config.costBudget,
        );
        if (budgetViolation) {
          const budgetEvent: SwarmEvent = {
            type: "budget-exceeded",
            agent: event.agent,
            scope: budgetViolation.scope,
            limitUsd: budgetViolation.limitUsd,
            spentUsd: budgetViolation.spentUsd,
          };
          tracer.record(budgetEvent);
          yield budgetEvent;

          const cbEvent: SwarmEvent = {
            type: "circuit-break",
            agent: event.agent,
            reason:
              `Cost budget exceeded (${budgetViolation.scope}): ` +
              `spent $${budgetViolation.spentUsd.toFixed(4)} of ` +
              `$${budgetViolation.limitUsd.toFixed(4)} limit.`,
          };
          tracer.record(cbEvent);
          tracer.flush();
          waveCtrl.abort(); // Issue 8: cancel sibling agents
          cp.delete(swarmId); // Issue 5
          yield cbEvent;
          return;
        }
      }
      if (event.type === "circuit-break") {
        circuitTripped = true;
        waveCtrl.abort(); // Issue 8: cancel remaining agents in wave
        break;
      }
    }

    if (circuitTripped) {
      tracer.flush();
      spansToExport.push(inProcessTracer.endSpan(waveSpan, { error: "Circuit breaker tripped" }));
      spansToExport.push(inProcessTracer.endSpan(swarmSpan, { error: "Circuit breaker tripped" }));
      await inProcessTracer.export(spansToExport);
      cp.delete(swarmId); // Issue 5
      return;
    }

    // Merge per-agent write stores into shared artifacts (Issue 1).
    for (const localStore of Object.values(waveWriteStores)) {
      Object.assign(artifacts, localStore);
    }

    // Validate outcome contracts: check required artifacts were produced
    for (const agent of wave) {
      for (const key of agent.requiredArtifacts ?? []) {
        if (!artifacts[key]) {
          const missingEvent: SwarmEvent = {
            type: "artifact-missing",
            agent: agent.name,
            key,
          };
          tracer.record(missingEvent);
          yield missingEvent;

          const cbEvent: SwarmEvent = {
            type: "circuit-break",
            agent: agent.name,
            reason: `Required artifact "${key}" was not produced by agent "${agent.name}".`,
          };
          tracer.record(cbEvent);
          tracer.flush();
          cp.delete(swarmId); // Issue 5
          yield cbEvent;
          return;
        }
      }
    }

    // ─── Checkpoint after each successful wave ─────────────────────────────
    cp.save(swarmId, waveIdx + 1, artifacts);
    const cpEvent: SwarmEvent = {
      type: "wave-checkpoint",
      wave: waveIdx,
      artifacts: Object.keys(artifacts),
    };
    tracer.record(cpEvent);
    yield cpEvent;

    const waveDoneEvent: SwarmEvent = {
      type: "wave-done",
      wave: waveIdx,
      agents: waveAgentNames,
      artifacts: Object.keys(artifacts),
    };
    tracer.record(waveDoneEvent);
    spansToExport.push(inProcessTracer.endSpan(waveSpan));
    yield waveDoneEvent;
  }

  // ─── Clean up checkpoint on successful completion ─────────────────────────
  cp.delete(swarmId);

  const doneEvent: SwarmEvent = {
    type: "swarm-done",
    swarmId,
    handoffCount: 0,
    totalAgentRounds: 0,
    totalInputTokens: swarmInputTokens,
    totalOutputTokens: swarmOutputTokens,
    totalCostUsd: swarmCostUsd,
    durationMs: Date.now() - swarmStartMs,
  };
  tracer.record(doneEvent);
  tracer.flush();
  swarmSpan.setAttribute("swarm.input_tokens", swarmInputTokens);
  swarmSpan.setAttribute("swarm.output_tokens", swarmOutputTokens);
  swarmSpan.setAttribute("swarm.cost_usd", swarmCostUsd);
  swarmSpan.setAttribute("swarm.duration_ms", Date.now() - swarmStartMs);
  spansToExport.push(inProcessTracer.endSpan(swarmSpan));
  await inProcessTracer.export(spansToExport);
  yield doneEvent;
}

// ─── Graph Swarm Builder ─────────────────────────────────────────────────────

export interface GraphAgentDef {
  name: string;
  description: string;
  systemPrompt: string;
  tools?: AgentTool[];
  model?: string;
  maxRounds?: number;
  permissionMode?: SwarmAgent["permissionMode"];
  /** Names of agents that must complete before this agent runs. */
  dependsOn?: string[];
  /** Artifact keys this agent must produce (checked after execution). */
  requiredArtifacts?: string[];
  /** Isolation mode. "worktree" gives this agent its own git worktree. */
  isolation?: SwarmAgent["isolation"];
}

// Import AgentTool here to avoid circular issues
import type { AgentTool } from "../agent/types.js";
import type { LLMProvider } from "../llm/provider.js";

export interface GraphSwarmConfig {
  agents: GraphAgentDef[];
  task: string;
  spec?: string;
  provider: LLMProvider;
  modelName: string;
  checkpointer?: import("../agent/types.js").Checkpointer;
}

/**
 * Build a SwarmConfig for graph (parallel wave) execution from a simplified
 * agent definition list. Use this instead of buildSupervisorSwarm when you
 * want DAG-based parallel execution rather than coordinator-mediated handoffs.
 */
export function buildGraphSwarm(config: GraphSwarmConfig): SwarmConfig {
  const agents: SwarmAgent[] = config.agents.map((def) => ({
    name: def.name,
    description: def.description,
    systemPrompt: def.systemPrompt,
    tools: (def.tools ?? []) as AgentTool[],
    handoffDestinations: [],
    contextMode: "filtered",
    maxRounds: def.maxRounds ?? 30,
    model: def.model,
    permissionMode: def.permissionMode,
    dependsOn: def.dependsOn,
    requiredArtifacts: def.requiredArtifacts,
    isolation: def.isolation,
  }));

  return {
    agents,
    initialAgent: config.agents[0]?.name ?? "",
    task: config.task,
    spec: config.spec,
    provider: config.provider,
    modelName: config.modelName,
    checkpointer: config.checkpointer,
    executionModel: "graph",
    policy: "full_auto",
  };
}
