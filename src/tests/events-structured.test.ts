import { describe, it, expect } from "vitest";
import type { AgentEvent } from "../agent/types.js";

describe("Structured Event Taxonomy", () => {
  it("includes all lifecycle event types", () => {
    const events: AgentEvent[] = [
      { type: "token", text: "hello" },
      { type: "thinking", text: "..." },
      { type: "tool-start", id: "1", name: "search", args: {} },
      { type: "tool-done", id: "1", name: "search", result: "", durationMs: 0 },
      { type: "tool-error", id: "1", name: "search", error: "fail", willRetry: false },
      { type: "hitl", requests: [], resumeKey: "k" },
      { type: "btw", text: "note" },
      { type: "checkpoint", round: 1, threadId: "t" },
      { type: "compacted", removedMessages: 2, savedTokens: 100 },
      { type: "done", response: "ok", reasoning: "", toolCount: 0, history: [], inputTokens: 0, outputTokens: 0, costUsd: 0, durationMs: 0 },
      { type: "error", message: "err", fatal: true },
      { type: "aborted", message: "stop" },
      // New structured lifecycle events
      { type: "round-start", round: 1, threadId: "t", messageCount: 2 },
      { type: "round-end", round: 1, threadId: "t", toolCallsThisRound: 0 },
      { type: "guardrail-triggered", guardrail: "pii", severity: "critical", action: "halt", message: "found ssn" },
      { type: "memory-recall", scope: "user-1", query: "prefs", resultsCount: 3 },
      { type: "memory-compact", scope: "user-1", removedMessages: 10, factsExtracted: 5 },
      { type: "checkpoint-saved", round: 1, threadId: "t", mode: "sync" },
      { type: "checkpoint-loaded", round: 0, threadId: "t", restored: false },
      { type: "stream-start", round: 1, threadId: "t", model: "gpt-4" },
      { type: "stream-end", round: 1, threadId: "t", tokensEmitted: 42 },
      { type: "middleware-before", round: 1, hook: "beforeRound" },
      { type: "middleware-after", round: 1, hook: "afterRound" },
    ];

    expect(events.length).toBe(23);
    const types = events.map((e) => e.type);
    expect(new Set(types).size).toBe(23); // all unique
  });

  it("guardrail event carries severity and action", () => {
    const event: AgentEvent = {
      type: "guardrail-triggered",
      guardrail: "prompt_injection",
      severity: "critical",
      action: "halt",
      message: "Detected injection attempt",
    };
    expect(event.guardrail).toBe("prompt_injection");
    expect(event.severity).toBe("critical");
    expect(event.action).toBe("halt");
  });

  it("checkpoint events include round and threadId", () => {
    const saved: AgentEvent = { type: "checkpoint-saved", round: 5, threadId: "t-42", mode: "async" };
    expect(saved.round).toBe(5);
    expect(saved.threadId).toBe("t-42");
    expect(saved.mode).toBe("async");
  });

  it("stream events bookend token generation", () => {
    const start: AgentEvent = { type: "stream-start", round: 2, threadId: "t", model: "claude" };
    const end: AgentEvent = { type: "stream-end", round: 2, threadId: "t", tokensEmitted: 128 };
    expect(start.model).toBe("claude");
    expect(end.tokensEmitted).toBe(128);
  });
});
