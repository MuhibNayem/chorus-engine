import { randomUUID } from "crypto";
import { runAgentLoop } from "../agent/loop.js";
import { createDefaultMiddleware } from "../agent/middleware.js";
import { JsonFileCheckpointer } from "../agent/checkpointer.js";
import { HitlGate } from "../agent/hitl.js";
import { BtwQueue } from "../agent/btw.js";
import type { SwarmAgent, SwarmConfig, SwarmEvent } from "./types.js";
import { createSession, broadcastToSharedState, applyHandoff, createArtifactTools } from "./session.js";
import { checkCircuitBreaker } from "./circuit-breaker.js";
import { validateOutput } from "./validator.js";
import { SwarmTracer } from "./trace.js";
import { buildAgentContext, buildSystemPrompt, createHandoffTools, isHandoffResult } from "./handoff.js";
import { runSwarmGraph } from "./graph-executor.js";
import { checkCostBudget, resolveModel } from "./cost-router.js";

function buildAgentsByName(agents: SwarmAgent[]): Map<string, SwarmAgent> {
  return new Map(agents.map((a) => [a.name, a]));
}

export async function* runSwarm(config: SwarmConfig): AsyncGenerator<SwarmEvent> {
  if (config.executionModel === "graph") {
    yield* runSwarmGraph(config);
    return;
  }

  // Issue 12: Validate non-empty agents list early.
  if (config.agents.length === 0) {
    yield { type: "circuit-break", agent: "swarm", reason: "SwarmConfig.agents must not be empty." };
    return;
  }

  const session = createSession(config);
  const tracer = new SwarmTracer(session.swarmId);
  const agentsByName = buildAgentsByName(config.agents);
  const checkpointer = config.checkpointer ?? new JsonFileCheckpointer();

  const startEvent: SwarmEvent = {
    type: "swarm-start",
    swarmId: session.swarmId,
    agents: config.agents.map((a) => a.name),
  };
  tracer.record(startEvent);
  yield startEvent;

  const initialAgent = agentsByName.get(config.initialAgent);
  if (!initialAgent) {
    throw new Error(`Initial agent "${config.initialAgent}" not found in swarm config.`);
  }

  session.activeAgent = config.initialAgent;
  session.agentHistory.push(config.initialAgent);
  session.lastHandoffDescription[config.initialAgent] = config.task;

  let totalAgentRounds = 0;
  const orchestratorStartMs = Date.now();
  let orchInputTokens = 0;
  let orchOutputTokens = 0;
  let orchCostUsd = 0;

  while (session.activeAgent !== null) {
    if (config.abortSignal?.aborted) {
      const abortEvent: SwarmEvent = {
        type: "circuit-break",
        agent: session.activeAgent ?? "swarm",
        reason: "Swarm aborted via AbortSignal.",
      };
      tracer.record(abortEvent);
      yield abortEvent;
      break;
    }

    const agentName = session.activeAgent;
    const agent = agentsByName.get(agentName);
    if (!agent) {
      throw new Error(`Active agent "${agentName}" not found.`);
    }

    const cbResult = checkCircuitBreaker(session, agent);
    if (cbResult.tripped) {
      const cbEvent: SwarmEvent = {
        type: "circuit-break",
        agent: agentName,
        reason: cbResult.reason!,
      };
      tracer.record(cbEvent);
      yield cbEvent;
      break;
    }

    const traceId = randomUUID();
    const agentStartEvent: SwarmEvent = {
      type: "agent-start",
      agent: agentName,
      traceId,
      contextMode: agent.contextMode,
    };
    tracer.record(agentStartEvent);
    yield agentStartEvent;

    // Sequential swarm: agents share one artifact store; read and write from the same object.
    const artifactTools = createArtifactTools(session.artifacts, session.artifacts);
    const handoffTools = createHandoffTools(session, agent, agentsByName);
    const allTools = [...agent.tools, ...artifactTools, ...handoffTools];

    const contextMessages = buildAgentContext(session, agent);
    const systemPrompt = buildSystemPrompt(session, agent);
    const threadId = `${session.swarmId}-${agentName}`;
    const middleware = createDefaultMiddleware(threadId);

    // Swarm agents always run full-auto — no HITL interrupts inside an orchestrated run.
    const hitlGate = new HitlGate();
    const btwQueue = new BtwQueue();

    const loopMessages = [...contextMessages];
    let agentResponse = "";
    let lastDoneEvent: (import("../agent/types.js").AgentEvent & { type: "done" }) | undefined;
    let handoffPayload: {
      targetAgent: string;
      taskDescription: string;
      artifacts: string[];
      reasoning?: string;
    } | null = null;

    const effectiveModel = resolveModel(
      {
        agentName: agentName,
        defaultModel: agent.model ?? config.modelName,
        spentUsd: orchCostUsd,
        budgetTotalUsd: config.costBudget?.totalUsd,
        maxRounds: agent.maxRounds,
      },
      config.costRouting,
    );

    const loopGen = runAgentLoop({
      provider: config.provider,
      model: effectiveModel,
      tools: allTools,
      messages: loopMessages,
      systemPrompt,
      threadId,
      hitlGate,
      btwQueue,
      policy: agent.permissionMode ?? config.policy ?? "full_auto",
      checkpointer,
      maxRounds: agent.maxRounds,
      middleware,
      abortSignal: config.abortSignal,
    });

    // Issue 9: Always dispose hitlGate — even if runAgentLoop throws mid-stream.
    try {
    for await (const event of loopGen) {
      const taggedEvent: SwarmEvent = { ...event, agent: agentName } as SwarmEvent;
      tracer.record(taggedEvent);
      yield taggedEvent;

      if (event.type === "handoff") {
        handoffPayload = {
          targetAgent: event.targetAgent,
          taskDescription: event.taskDescription,
          artifacts: event.artifacts,
          reasoning: event.reasoning,
        };
      }

      // Backward-compat: also detect JSON-based handoffs from tool-done events
      if (event.type === "tool-done") {
        try {
          const parsed = JSON.parse(event.result) as unknown;
          if (isHandoffResult(parsed)) {
            handoffPayload = {
              targetAgent: parsed.targetAgent,
              taskDescription: parsed.taskDescription,
              artifacts: parsed.artifacts,
              reasoning: parsed.reasoning,
            };
          }
        } catch {
          // not JSON — not a handoff
        }
      }

      if (event.type === "done") {
        agentResponse = event.response;
        totalAgentRounds += event.toolCount + 1;
        broadcastToSharedState(session, loopMessages, agentName);
        lastDoneEvent = event;
        orchInputTokens += event.inputTokens;
        orchOutputTokens += event.outputTokens;
        orchCostUsd += event.costUsd;

        // Hard cost cap check after each agent completes
        const budgetViolation = checkCostBudget(
          agentName,
          event.costUsd,
          orchCostUsd,
          config.costBudget,
        );
        if (budgetViolation) {
          const budgetEvent: SwarmEvent = {
            type: "budget-exceeded",
            agent: agentName,
            scope: budgetViolation.scope,
            limitUsd: budgetViolation.limitUsd,
            spentUsd: budgetViolation.spentUsd,
          };
          tracer.record(budgetEvent);
          yield budgetEvent;

          const cbEvent: SwarmEvent = {
            type: "circuit-break",
            agent: agentName,
            reason:
              `Cost budget exceeded (${budgetViolation.scope}): ` +
              `spent $${budgetViolation.spentUsd.toFixed(4)} of ` +
              `$${budgetViolation.limitUsd.toFixed(4)} limit.`,
          };
          tracer.record(cbEvent);
          yield cbEvent;
          session.activeAgent = null;
        }
      }
    }
    } finally {
      hitlGate.dispose();
    }

    const agentDoneEvent: SwarmEvent = {
      type: "agent-done",
      agent: agentName,
      responseText: agentResponse,
      metrics: {
        inputTokens: lastDoneEvent?.inputTokens ?? 0,
        outputTokens: lastDoneEvent?.outputTokens ?? 0,
        costUsd: lastDoneEvent?.costUsd ?? 0,
        durationMs: lastDoneEvent?.durationMs ?? 0,
        rounds: (lastDoneEvent?.toolCount ?? 0) + 1,
        toolCalls: lastDoneEvent?.toolCount ?? 0,
      },
    };
    tracer.record(agentDoneEvent);
    yield agentDoneEvent;

    const validation = validateOutput(agentResponse, agent);
    if (!validation.ok) {
      const valEvent: SwarmEvent = {
        type: "validation-fail",
        agent: agentName,
        reason: validation.reason ?? "Output validation failed.",
      };
      tracer.record(valEvent);
      yield valEvent;
    }

    if (handoffPayload) {
      const handoffEvent: SwarmEvent = {
        type: "handoff",
        from: agentName,
        to: handoffPayload.targetAgent,
        taskDescription: handoffPayload.taskDescription,
        reasoning: handoffPayload.reasoning,
      };
      tracer.record(handoffEvent);
      yield handoffEvent;

      if (!agentsByName.has(handoffPayload.targetAgent)) {
        const cbEvent: SwarmEvent = {
          type: "circuit-break",
          agent: agentName,
          reason: `Handoff target "${handoffPayload.targetAgent}" is not a registered agent.`,
        };
        tracer.record(cbEvent);
        yield cbEvent;
        break;
      }

      applyHandoff(session, handoffPayload);
      handoffPayload = null;
    } else {
      session.activeAgent = null;
    }
  }

  const doneEvent: SwarmEvent = {
    type: "swarm-done",
    swarmId: session.swarmId,
    handoffCount: session.handoffCount,
    totalAgentRounds,
    totalInputTokens: orchInputTokens,
    totalOutputTokens: orchOutputTokens,
    totalCostUsd: orchCostUsd,
    durationMs: Date.now() - orchestratorStartMs,
  };
  tracer.record(doneEvent);
  tracer.flush();
  yield doneEvent;
}
