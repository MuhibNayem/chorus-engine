import { describe, it, expect } from "vitest";
import { z } from "zod";
import { runAgentLoop, HitlGate, BtwQueue, JsonFileCheckpointer } from "../index.js";
import type { LLMProvider, ToolStreamEvent, ToolDef } from "../llm/provider.js";
import type { ProviderName } from "../llm/config.js";

function createMockProvider(
  streamFactory: (input: { model: string; messages: any[]; systemPrompt?: string; tools: ToolDef[] }) => AsyncGenerator<ToolStreamEvent>,
): LLMProvider {
  return {
    name: "mock" as ProviderName,
    async generate() {
      return { text: "mock", model: "mock" };
    },
    async *stream() {
      yield { type: "response.completed" };
    },
    async *streamWithTools(input) {
      yield* streamFactory(input);
    },
    async health() {
      return { ok: true, provider: "mock" };
    },
    estimateCost() {
      return 0;
    },
  };
}

describe("runAgentLoop — production runtime tests", () => {
  it("yields tokens and completes with a simple response", async () => {
    const provider = createMockProvider(async function* () {
      yield { type: "token", text: "Hello" };
      yield { type: "done", response: { content: "Hello world" } };
    });

    const events: Array<{ type: string }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [],
      messages: [{ role: "user", content: "Hi" }],
      systemPrompt: "",
      threadId: "test-1",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
    }

    expect(events.map((e) => e.type)).toEqual([
      "checkpoint-loaded", "round-start", "middleware-before", "stream-start",
      "token", "stream-end", "checkpoint", "done",
    ]);
    expect((events[4] as { type: "token"; text: string }).text).toBe("Hello");
    expect((events[7] as { type: "done"; response: string }).response).toBe("Hello world");
  });

  it("retries on retryable stream error before any tokens are yielded", async () => {
    let calls = 0;
    const provider = createMockProvider(async function* () {
      calls++;
      if (calls === 1) {
        throw Object.assign(new Error("429 rate limit exceeded"), { status: 429 });
      }
      yield { type: "token", text: "OK" };
      yield { type: "done", response: { content: "Success" } };
    });

    const events: Array<{ type: string }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [],
      messages: [{ role: "user", content: "Hi" }],
      systemPrompt: "",
      threadId: "test-retry",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
    }

    expect(calls).toBe(2);
    expect(events.some((e) => e.type === "done")).toBe(true);
  });

  it("fatals on stream error after tokens have been emitted", async () => {
    const provider = createMockProvider(async function* () {
      yield { type: "token", text: "Partial" };
      throw Object.assign(new Error("ECONNRESET"), { code: "ECONNRESET" });
    });

    const events: Array<{ type: string; fatal?: boolean }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [],
      messages: [{ role: "user", content: "Hi" }],
      systemPrompt: "",
      threadId: "test-fatal",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
    }

    expect(events.map((e) => e.type)).toEqual([
      "checkpoint-loaded", "round-start", "middleware-before", "stream-start",
      "token", "stream-end", "error",
    ]);
    expect(events[6].fatal).toBe(true);
  });

  it("fatals on truncated stream without done event", async () => {
    const provider = createMockProvider(async function* () {
      yield { type: "token", text: "Partial" };
      // Stream ends without done
    });

    const events: Array<{ type: string; fatal?: boolean }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [],
      messages: [{ role: "user", content: "Hi" }],
      systemPrompt: "",
      threadId: "test-trunc",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
    }

    expect(events.map((e) => e.type)).toEqual([
      "checkpoint-loaded", "round-start", "middleware-before", "stream-start",
      "token", "stream-end", "error",
    ]);
    expect(events[6].fatal).toBe(true);
  });

  it("executes tool calls and yields tool events", async () => {
    let callCount = 0;
    const provider = createMockProvider(async function* () {
      callCount++;
      if (callCount === 1) {
        yield {
          type: "done",
          response: {
            content: "",
            tool_calls: [
              { id: "tc1", type: "function", function: { name: "echo", arguments: JSON.stringify({ msg: "hi" }) } },
            ],
          },
        };
      } else {
        yield { type: "done", response: { content: "Done" } };
      }
    });

    const events: Array<{ type: string }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [
        {
          name: "echo",
          description: "echo",
          schema: undefined,
          invoke: async (input: unknown) => `echo: ${(input as { msg: string }).msg}`,
        },
      ],
      messages: [{ role: "user", content: "Call echo" }],
      systemPrompt: "",
      threadId: "test-tool",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
      if (event.type === "done") break;
    }

    const types = events.map((e) => e.type);
    expect(types).toContain("tool-start");
    expect(types).toContain("tool-done");
    const doneEvent = events.find((e): e is { type: "done"; toolCount: number } => e.type === "done");
    expect(doneEvent?.toolCount).toBe(1);
  });

  it("handles multiple parallel tool calls in one round", async () => {
    let callCount = 0;
    const provider = createMockProvider(async function* () {
      callCount++;
      if (callCount === 1) {
        yield {
          type: "done",
          response: {
            content: "",
            tool_calls: [
              { id: "tc1", type: "function", function: { name: "a", arguments: "{}" } },
              { id: "tc2", type: "function", function: { name: "b", arguments: "{}" } },
            ],
          },
        };
      } else {
        yield { type: "done", response: { content: "Done" } };
      }
    });

    const events: Array<{ type: string; name?: string }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [
        { name: "a", description: "a", schema: undefined, invoke: async () => "A" },
        { name: "b", description: "b", schema: undefined, invoke: async () => "B" },
      ],
      messages: [{ role: "user", content: "Call both" }],
      systemPrompt: "",
      threadId: "test-parallel",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
      if (event.type === "done") break;
    }

    const starts = events.filter((e) => e.type === "tool-start");
    expect(starts.length).toBe(2);
    expect(starts.map((s) => s.name).sort()).toEqual(["a", "b"]);
  });

  it("pauses on HITL and resumes with approval", async () => {
    const gate = new HitlGate();
    let callCount = 0;
    const provider = createMockProvider(async function* () {
      callCount++;
      if (callCount === 1) {
        yield {
          type: "done",
          response: {
            content: "",
            tool_calls: [
              { id: "tc1", type: "function", function: { name: "run_command", arguments: JSON.stringify({ cmd: "ls" }) } },
            ],
          },
        };
      } else {
        yield { type: "done", response: { content: "Done" } };
      }
    });

    const events: Array<{ type: string }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [{ name: "run_command", description: "run", schema: undefined, invoke: async () => "ok" }],
      messages: [{ role: "user", content: "Run ls" }],
      systemPrompt: "",
      threadId: "test-hitl",
      hitlGate: gate,
      btwQueue: new BtwQueue(),
      policy: "auto_edit",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
      if (event.type === "hitl") {
        const resumeKey = (event as { resumeKey: string }).resumeKey;
        gate.resolve(resumeKey, { type: "approve" });
      }
      if (event.type === "done") break;
    }

    expect(events.some((e) => e.type === "hitl")).toBe(true);
    expect(events.some((e) => e.type === "done")).toBe(true);
  });

  it("respects maxRounds and yields fatal error", async () => {
    let callCount = 0;
    const provider = createMockProvider(async function* () {
      callCount++;
      yield {
        type: "done",
        response: {
          content: "",
          tool_calls: [
            { id: `tc${callCount}`, type: "function", function: { name: "noop", arguments: "{}" } },
          ],
        },
      };
    });

    const events: Array<{ type: string; fatal?: boolean }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [{ name: "noop", description: "noop", schema: undefined, invoke: async () => "ok" }],
      messages: [{ role: "user", content: "Loop" }],
      systemPrompt: "",
      threadId: "test-max",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
      maxRounds: 2,
    })) {
      events.push(event);
      if (event.type === "error") break;
    }

    expect(events.some((e) => e.type === "error" && e.fatal)).toBe(true);
  });

  it("validates outputSchema and retries on schema failure", async () => {
    const schema = z.object({ answer: z.string() });
    let calls = 0;
    const provider = createMockProvider(async function* () {
      calls++;
      if (calls === 1) {
        yield { type: "done", response: { content: "not json" } };
      } else {
        yield { type: "done", response: { content: JSON.stringify({ answer: "42" }) } };
      }
    });

    const events: Array<{ type: string }> = [];
    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [],
      messages: [{ role: "user", content: "Answer" }],
      systemPrompt: "",
      threadId: "test-schema",
      hitlGate: new HitlGate(),
      btwQueue: new BtwQueue(),
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
      outputSchema: schema,
    })) {
      events.push(event);
    }

    expect(calls).toBe(2);
    expect(events.some((e) => e.type === "done")).toBe(true);
  });

  it("yields btw messages injected mid-loop", async () => {
    let callCount = 0;
    const provider = createMockProvider(async function* () {
      callCount++;
      if (callCount === 1) {
        yield { type: "done", response: { content: "First" } };
      } else {
        yield { type: "done", response: { content: "Second" } };
      }
    });

    const btwQueue = new BtwQueue();
    const events: Array<{ type: string }> = [];

    // Enqueue BEFORE starting so the first round picks it up
    btwQueue.enqueue("side note");

    for await (const event of runAgentLoop({
      provider,
      model: "mock",
      tools: [],
      messages: [{ role: "user", content: "Hi" }],
      systemPrompt: "",
      threadId: "test-btw",
      hitlGate: new HitlGate(),
      btwQueue,
      policy: "full_auto",
      checkpointer: new JsonFileCheckpointer(),
    })) {
      events.push(event);
      if (event.type === "done") break;
    }

    expect(events.some((e) => e.type === "btw")).toBe(true);
  });
});
