import type { ChatMessage, ModelResponse, ToolCall } from "../llm/provider.js";
import type { AgentMiddleware, RoundContext } from "./middleware.js";
import { HandoffSignal, type AgentEvent, type AgentTool, type HitlDecision, type HitlRequest, type LoopOptions } from "./types.js";
import { estimateCost } from "../llm/pricing.js";
import type { InProcessTracer, MutableSpan } from "../telemetry/inprocess.js";
import { runGuardrails, shouldHalt, BuiltInGuardrails } from "../guardrails/index.js";
import type { GuardrailsConfig, GuardrailViolation } from "../guardrails/index.js";
import {
  consumeStream,
  type StreamConsumptionEvent,
  toolDefsFromTools,
  normalizeToolCallArgs,
  safeArgs,
  executeToolCall,
} from "./loop-utils.js";

type ToolByName = Map<string, AgentTool>;

function mergeAssistantMessage(
  history: readonly ChatMessage[],
  response: ModelResponse,
): ChatMessage[] {
  return [...history, {
    role: "assistant",
    content: response.content,
    ...(response.reasoning_content ? { reasoning_content: response.reasoning_content } : {}),
    ...(response.tool_calls ? { tool_calls: response.tool_calls } : {}),
  }];
}

function toHitlRequests(toolCalls: ToolCall[]): HitlRequest[] {
  return toolCalls.map((toolCall) => ({
    id: toolCall.id,
    name: toolCall.function.name,
    args: safeArgs(toolCall),
  }));
}

function applyHitlDecision(
  decision: HitlDecision,
  history: readonly ChatMessage[],
): ChatMessage[] {
  if (decision.type === "reject") {
    return [...history, {
      role: "user",
      content: decision.message?.trim() || "Tool execution denied by user.",
    }];
  }
  return [...history];
}

/** Run guardrails and yield events for any violations. */
async function* runGuardrailChecks<T>(
  guardrails: Array<(ctx: T) => Promise<GuardrailViolation | null>>,
  ctx: T,
  opts?: { runAll?: boolean; haltOn?: import("../guardrails/index.js").GuardrailSeverity; span?: MutableSpan },
): AsyncGenerator<AgentEvent, GuardrailViolation[]> {
  const violations = await runGuardrails(guardrails, ctx, opts);
  for (const v of violations) {
    yield {
      type: "guardrail-triggered",
      guardrail: v.guardrail,
      severity: v.severity,
      action: v.action,
      message: v.message,
    };
  }
  return violations;
}

/**
 * Execute middleware hooks grouped by priority. Hooks within the same priority
 * run in parallel via `Promise.all`; priority groups run sequentially from
 * lowest to highest. This eliminates linear I/O latency when multiple
 * middlewares perform independent work (e.g., logging + RAG fetch).
 *
 * Hooks with ordering semantics (`beforeTool` cancellation, `afterTool`
 * transformation chaining, `maybeCompact` first-wins) remain sequential and
 * are NOT routed through this helper.
 */
async function runMiddleware(
  middleware: AgentMiddleware[],
  hook: "beforeRound" | "afterRound",
  ...args: [RoundContext]
): Promise<void> {
  // Group by priority (default 0)
  const groups = new Map<number, AgentMiddleware[]>();
  for (const mw of middleware) {
    const p = mw.priority ?? 0;
    const list = groups.get(p) ?? [];
    list.push(mw);
    groups.set(p, list);
  }

  const sortedPriorities = [...groups.keys()].sort((a, b) => a - b);
  for (const p of sortedPriorities) {
    const hooks = groups.get(p) ?? [];
    await Promise.all(
      hooks.map(async (mw) => {
        const fn = mw[hook];
        if (fn) await fn.apply(mw, args);
      }),
    );
  }
}

export async function* runAgentLoop(options: LoopOptions): AsyncGenerator<AgentEvent> {
  const {
    provider,
    model,
    tools,
    messages,
    systemPrompt,
    threadId,
    hitlGate,
    btwQueue,
    policy,
    checkpointer,
    maxRounds = 500,
    resumedDecision,
    middleware = [],
    abortSignal,
    streamTimeoutMs,
    outputSchema,
    tracer,
    guardrails,
  } = options;

  const saved = await checkpointer.load(threadId);
  const restoreFromCheckpoint = saved?.waitingForHitl != null;
  yield { type: "checkpoint-loaded", round: saved?.round ?? 0, threadId, restored: restoreFromCheckpoint };

  // Only restore when a HITL-paused run exists for this thread. A completed turn
  // also writes a checkpoint, but the caller's messages array already contains the
  // new user turn and must not be overridden.
  let history: ChatMessage[] = [...(restoreFromCheckpoint ? saved!.messages : messages)];

  let round = restoreFromCheckpoint ? saved!.round : 0;
  let totalTools = 0;
  let pendingDecision = resumedDecision;
  const loopStartMs = Date.now();
  let totalInputTokens = 0;
  let totalOutputTokens = 0;

  // In-process telemetry: loop-level span
  const loopSpan = tracer?.startSpan("agent.loop", { attributes: { "agent.thread_id": threadId, "agent.model": model } });
  const spansToExport: import("../telemetry/types.js").OTelSpan[] = [];

  while (round < maxRounds) {
    if (abortSignal?.aborted) {
      yield { type: "aborted", message: "Interrupted by user." };
      return;
    }
    for (const text of btwQueue.drain()) {
      history = [...history, { role: "user", content: `[/btw] ${text}` }];
      yield { type: "btw", text };
    }

    yield { type: "round-start", round, threadId, messageCount: history.length };

    // Input guardrails (run against raw system prompt; effective prompt not yet built)
    if (guardrails?.inputs && guardrails.inputs.length > 0) {
      const inputViolations = yield* runGuardrailChecks(
        guardrails.inputs,
        { messages: history, systemPrompt, threadId, round },
        { runAll: guardrails.runAll, haltOn: guardrails.haltOn },
      );
      if (shouldHalt(inputViolations, guardrails.haltOn)) {
        if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan, { error: "Input guardrail halted" }));
        if (tracer) await tracer.export(spansToExport);
        yield { type: "error", message: `Input guardrail violation(s): ${inputViolations.map((v) => v.guardrail).join(", ")}`, fatal: true };
        return;
      }
    }

    // Middleware: beforeRound
    const roundCtx: RoundContext = { round, threadId, model, history, toolCallsThisRound: 0 };
    await runMiddleware(middleware, "beforeRound", roundCtx);
    yield { type: "middleware-before", round, hook: "beforeRound" };

    // Rebuild tools + system prompt each round (enables per-turn skill routing)
    const allTools = [...tools, ...middleware.flatMap((mw) => mw.extraTools?.() ?? [])];
    const toolsByName: ToolByName = new Map(
      allTools
        .filter((tool): tool is AgentTool & { name: string } => typeof tool.name === "string" && tool.name.length > 0)
        .map((tool) => [tool.name!, tool]),
    );

    // Pass tool registry to middlewares that need it for pattern execution
    for (const mw of middleware) {
      mw.setTools?.(toolsByName);
    }

    const toolDefs = toolDefsFromTools(allTools);

    const extraPrompts = middleware.flatMap((mw) => {
      const extra = mw.extraSystemPrompt?.();
      return extra ? [extra] : [];
    });
    const effectiveSystemPrompt = extraPrompts.length > 0
      ? `${systemPrompt}\n\n${extraPrompts.join("\n\n")}`
      : systemPrompt;

    // Middleware: maybeCompact — first matching middleware wins
    for (const mw of middleware) {
      if (!mw.maybeCompact) continue;
      const compactResult = await mw.maybeCompact(history, { model, systemPrompt: effectiveSystemPrompt });
      if (compactResult) {
        history = compactResult.replacement;
        yield { type: "compacted", removedMessages: compactResult.removedMessages, savedTokens: compactResult.savedTokens };
        break;
      }
    }

    // In-process telemetry: round-level span
    const roundSpan = tracer?.startSpan("agent.round", {
      parentSpanId: loopSpan?.spanId,
      attributes: { "agent.round": round, "agent.thread_id": threadId },
    });

    let response: ModelResponse | null = null;
    let tokensEmitted = 0;
    yield { type: "stream-start", round, threadId, model };
    for await (const event of consumeStream(provider, model, history, effectiveSystemPrompt, toolDefs, streamTimeoutMs)) {
      if (event.type === "token") {
        tokensEmitted++;
        yield { type: "token", text: event.text };
        continue;
      }
      if (event.type === "thinking") {
        yield { type: "thinking", text: event.text };
        continue;
      }
      if (event.type === "stream-done") {
        response = event.response;
        totalInputTokens += event.inputTokens;
        totalOutputTokens += event.outputTokens;
        break;
      }
      if (event.type === "stream-error") {
        yield { type: "stream-end", round, threadId, tokensEmitted };
        if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan, { error: event.message }));
        yield { type: "error", message: event.message, fatal: event.fatal };
        if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan, { error: event.message }));
        if (tracer) await tracer.export(spansToExport);
        return;
      }
    }

    yield { type: "stream-end", round, threadId, tokensEmitted };

    if (!response) {
      // Safety net: consumeStream should always produce either stream-done or stream-error,
      // but if it somehow returns early we emit a fatal error rather than silently substituting
      // an empty response (which would mask provider bugs).
      if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan, { error: "No response from stream" }));
      yield { type: "error", message: "Agent loop exited stream consumption without a response.", fatal: true };
      if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan, { error: "No response from stream" }));
      if (tracer) await tracer.export(spansToExport);
      return;
    }

    history = mergeAssistantMessage(history, response);

    if (!response.tool_calls || response.tool_calls.length === 0) {
      // Issue 15: Validate response against outputSchema before accepting it as final.
      if (outputSchema) {
        try {
          const parsed = JSON.parse(response.content) as unknown;
          outputSchema.parse(parsed);
        } catch (err) {
          const correctionMsg =
            `Your response must be a valid JSON object matching the required schema. ` +
            `Validation error: ${err instanceof Error ? err.message : String(err)}. ` +
            `Please respond with a valid JSON object only — no prose, no markdown fences.`;
          history = [...history, { role: "user", content: correctionMsg }];
          round += 1;
          await checkpointer.save(threadId, { messages: history, round });
          yield { type: "checkpoint", round, threadId };
          continue;
        }
      }

      // Output guardrails
      if (guardrails?.outputs && guardrails.outputs.length > 0) {
        const outputViolations = yield* runGuardrailChecks(
          guardrails.outputs,
          { response: response.content, toolCalls: response.tool_calls?.map((tc) => ({ name: tc.function.name, args: safeArgs(tc) })), threadId, round },
          { runAll: guardrails.runAll, haltOn: guardrails.haltOn, span: roundSpan ?? undefined },
        );
        if (shouldHalt(outputViolations, guardrails.haltOn)) {
          if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan, { error: "Output guardrail halted" }));
          if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan, { error: "Output guardrail halted" }));
          if (tracer) await tracer.export(spansToExport);
          yield { type: "error", message: `Output guardrail violation(s): ${outputViolations.map((v) => v.guardrail).join(", ")}`, fatal: true };
          return;
        }
      }

      await checkpointer.save(threadId, { messages: history, round });
      yield { type: "checkpoint", round, threadId };
      if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan));
      if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan));
      if (tracer) await tracer.export(spansToExport);
      yield {
        type: "done",
        response: response.content,
        reasoning: response.reasoning_content ?? "",
        toolCount: totalTools,
        history,
        inputTokens: totalInputTokens,
        outputTokens: totalOutputTokens,
        costUsd: provider.estimateCost?.(totalInputTokens, totalOutputTokens) ?? estimateCost(model, totalInputTokens, totalOutputTokens),
        durationMs: Date.now() - loopStartMs,
      };
      return;
    }

    totalTools += response.tool_calls.length;
    const requests = toHitlRequests(response.tool_calls);
    let decision = pendingDecision;
    pendingDecision = undefined;

    if (!decision && hitlGate.shouldPause(response.tool_calls, policy)) {
      const resumeKey = `hitl-${threadId}-${round}`;
      await checkpointer.save(threadId, {
        messages: history,
        round,
        waitingForHitl: {
          resumeKey,
          requests,
          toolCalls: response.tool_calls,
          assistant: response,
        },
      });
      yield { type: "checkpoint", round, threadId };
      yield { type: "hitl", requests, resumeKey };
      decision = await hitlGate.wait(resumeKey);
    }

    if (decision) {
      history = applyHitlDecision(decision, history);
      if (decision.type === "reject") {
        await checkpointer.save(threadId, { messages: history, round });
        yield { type: "checkpoint", round, threadId };
        if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan));
        if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan));
        if (tracer) await tracer.export(spansToExport);
        yield {
          type: "done",
          response: response.content,
          reasoning: response.reasoning_content ?? "",
          toolCount: totalTools,
          history,
          inputTokens: totalInputTokens,
          outputTokens: totalOutputTokens,
          costUsd: provider.estimateCost?.(totalInputTokens, totalOutputTokens) ?? estimateCost(model, totalInputTokens, totalOutputTokens),
          durationMs: Date.now() - loopStartMs,
        };
        return;
      }
    }

    let toolCallsThisRound = 0;
    for (const toolCall of response.tool_calls) {
      const name = toolCall.function.name;
      let args: Record<string, unknown>;
      try {
        args = normalizeToolCallArgs(toolCall);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        yield { type: "tool-error", id: toolCall.id, name, error: message, willRetry: false };
        history = [...history, {
          role: "tool",
          tool_call_id: toolCall.id,
          content: `Error: ${message}`,
        }];
        continue;
      }

      // Middleware: beforeTool — any middleware may cancel tool execution
      let cancelled: { result: string } | undefined;
      for (const mw of middleware) {
        if (!mw.beforeTool) continue;
        const directive = await mw.beforeTool({ id: toolCall.id, name, args });
        if (directive && directive.cancel) {
          cancelled = { result: directive.result };
          break;
        }
      }
      if (cancelled) {
        history = [...history, { role: "tool", tool_call_id: toolCall.id, content: cancelled.result }];
        yield { type: "tool-done", id: toolCall.id, name, result: cancelled.result, durationMs: 0 };
        toolCallsThisRound += 1;
        continue;
      }

      toolCallsThisRound += 1;
      yield { type: "tool-start", id: toolCall.id, name, args };
      const startedAt = Date.now();

      // In-process telemetry: tool-level span
      const toolSpan = tracer?.startSpan("agent.tool_call", {
        parentSpanId: roundSpan?.spanId,
        attributes: { "agent.tool_name": name, "agent.thread_id": threadId },
      });

      // Tool guardrails
      if (guardrails?.tools && guardrails.tools.length > 0) {
        const toolViolations = yield* runGuardrailChecks(
          guardrails.tools,
          { toolName: name, args, threadId, round },
          { runAll: guardrails.runAll, haltOn: guardrails.haltOn, span: toolSpan ?? undefined },
        );
        if (shouldHalt(toolViolations, guardrails.haltOn)) {
          if (toolSpan) spansToExport.push(tracer!.endSpan(toolSpan, { error: "Tool guardrail halted" }));
          history = [...history, { role: "tool", tool_call_id: toolCall.id, content: `Guardrail blocked: ${toolViolations.map((v) => v.message).join("; ")}` }];
          yield { type: "tool-error", id: toolCall.id, name, error: `Guardrail blocked: ${toolViolations.map((v) => v.guardrail).join(", ")}`, willRetry: false };
          continue;
        }
      }

      try {
        const { result: rawResult, attempts } = await executeToolCall(toolCall, toolsByName);
        const durationMs = Date.now() - startedAt;
        if (toolSpan) {
          toolSpan.setAttribute("agent.tool_duration_ms", durationMs);
          spansToExport.push(tracer!.endSpan(toolSpan));
        }

        // Middleware: afterTool — each mw may transform the result string
        let result = rawResult;
        for (const mw of middleware) {
          if (!mw.afterTool) continue;
          const transformed = await mw.afterTool({ id: toolCall.id, name, result, durationMs });
          if (transformed !== undefined) result = transformed;
        }

        history = [...history, {
          role: "tool",
          tool_call_id: toolCall.id,
          content: result,
        }];
        yield {
          type: "tool-done",
          id: toolCall.id,
          name,
          result: attempts > 1 ? `${result}\n\n[retried ${attempts - 1} time(s)]` : result,
          durationMs,
        };
      } catch (error) {
        if (error instanceof HandoffSignal) {
          // First-class handoff: push synthetic tool result, yield handoff + done, and exit.
          history = [...history, {
            role: "tool",
            tool_call_id: toolCall.id,
            content: `[Handoff to ${error.targetAgent}]`,
          }];
          yield {
            type: "handoff",
            targetAgent: error.targetAgent,
            taskDescription: error.taskDescription,
            artifacts: error.artifacts,
            reasoning: error.reasoning,
          };
          if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan));
          if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan));
          if (tracer) await tracer.export(spansToExport);
          await checkpointer.save(threadId, { messages: history, round });
          yield { type: "checkpoint", round, threadId };
          yield {
            type: "done",
            response: response?.content ?? "",
            reasoning: response?.reasoning_content ?? "",
            toolCount: totalTools,
            history,
            inputTokens: totalInputTokens,
            outputTokens: totalOutputTokens,
            costUsd: provider.estimateCost?.(totalInputTokens, totalOutputTokens) ?? estimateCost(model, totalInputTokens, totalOutputTokens),
            durationMs: Date.now() - loopStartMs,
          };
          return;
        }
        const message = error instanceof Error ? error.message : String(error);
        if (toolSpan) spansToExport.push(tracer!.endSpan(toolSpan, { error: message }));
        history = [...history, {
          role: "tool",
          tool_call_id: toolCall.id,
          content: `Error: ${message}`,
        }];
        yield {
          type: "tool-error",
          id: toolCall.id,
          name,
          error: message,
          willRetry: false,
        };
      }
    }

    round += 1;
    // Middleware: afterRound
    const afterCtx: RoundContext = { round, threadId, model, history, toolCallsThisRound };
    await runMiddleware(middleware, "afterRound", afterCtx);
    yield { type: "middleware-after", round, hook: "afterRound" };

    if (roundSpan) spansToExport.push(tracer!.endSpan(roundSpan));
    await checkpointer.save(threadId, { messages: history, round });
    yield { type: "checkpoint", round, threadId };
    yield { type: "checkpoint-saved", round, threadId };
    yield { type: "round-end", round, threadId, toolCallsThisRound };
  }

  if (loopSpan) spansToExport.push(tracer!.endSpan(loopSpan, { error: `Exceeded max rounds (${maxRounds})` }));
  if (tracer) await tracer.export(spansToExport);
  yield {
    type: "error",
    message: `Agent loop exceeded max rounds (${maxRounds}).`,
    fatal: true,
  };
}
