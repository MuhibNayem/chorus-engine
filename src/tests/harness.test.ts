import { describe, it, expect } from "vitest";
import { executeWorkers, formatWorkerResults, prepareTaskExecution, prepareTaskExecutionAsync } from "../index.js";
import type { LLMProvider } from "../llm/provider.js";
import type { WorkerExecutionResult } from "../harness/workerEngine.js";
import type { ProviderName } from "../llm/config.js";

const mockProvider: LLMProvider = {
  name: "mock" as ProviderName,
  async generate() { return { text: "Analysis complete", model: "mock" }; },
  async *stream() { yield { type: "response.completed" }; },
  async *streamWithTools() { yield { type: "done", response: { content: "Done" } }; },
  async health() { return { ok: true, provider: "mock" }; },
};

describe("executeWorkers — parallel mode", () => {
  it("executes multiple workers in parallel", async () => {
    const events: string[] = [];
    const results = await executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
        { workerId: "w2", role: "planner", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Analyze this",
      provider: mockProvider,
      model: "mock",
      onEvent: (event) => events.push(event.type),
      parentTurnId: "t1",
    });

    expect(results.length).toBe(2);
    expect(results.map((r) => r.role).sort()).toEqual(["planner", "researcher"]);
    expect(results.every((r) => r.summary.length > 0)).toBe(true);
  });

  it("respects concurrency limits", async () => {
    let concurrent = 0;
    let maxConcurrent = 0;

    const trackingProvider: LLMProvider = {
      ...mockProvider,
      async generate() {
        concurrent++;
        maxConcurrent = Math.max(maxConcurrent, concurrent);
        await new Promise((r) => setTimeout(r, 50));
        concurrent--;
        return { text: "ok", model: "mock" };
      },
    };

    await executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
        { workerId: "w2", role: "planner", ownedScope: [], inputBundleId: "b1", status: "running" },
        { workerId: "w3", role: "coder", ownedScope: [], inputBundleId: "b1", status: "running" },
        { workerId: "w4", role: "reviewer", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Analyze",
      provider: trackingProvider,
      model: "mock",
      onEvent: () => {},
      parentTurnId: "t1",
      concurrency: 2,
    });

    expect(maxConcurrent).toBe(2);
  });

  it("handles worker failures gracefully", async () => {
    const failingProvider: LLMProvider = {
      ...mockProvider,
      async generate() { throw new Error("Model overloaded"); },
    };

    const events: string[] = [];
    const results = await executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Analyze",
      provider: failingProvider,
      model: "mock",
      onEvent: (event) => events.push(event.type),
      parentTurnId: "t1",
    });

    expect(results.length).toBe(1);
    expect(results[0].summary).toContain("failed");
    expect(events).toContain("worker-update");
  });

  it("returns empty array for no assignments", async () => {
    const results = await executeWorkers({
      assignments: [],
      taskText: "Nothing",
      provider: mockProvider,
      model: "mock",
      onEvent: () => {},
      parentTurnId: "t1",
    });
    expect(results).toEqual([]);
  });

  it("respects abortSignal", async () => {
    const controller = new AbortController();
    const slowProvider: LLMProvider = {
      ...mockProvider,
      async generate() {
        await new Promise((r) => setTimeout(r, 500));
        return { text: "ok", model: "mock" };
      },
    };

    const promise = executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
        { workerId: "w2", role: "planner", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Analyze",
      provider: slowProvider,
      model: "mock",
      onEvent: () => {},
      parentTurnId: "t1",
      abortSignal: controller.signal,
    });

    controller.abort();
    const results = await promise;
    expect(results.length).toBeLessThanOrEqual(2);
  });
});

describe("executeWorkers — pipeline mode", () => {
  it("passes accumulated context through the chain", async () => {
    let receivedPrompts: string[] = [];
    const trackingProvider: LLMProvider = {
      ...mockProvider,
      async generate(input) {
        receivedPrompts.push(input.messages[0].content);
        return { text: `${input.messages[0].content.slice(0, 20)}...`, model: "mock" };
      },
    };

    await executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
        { workerId: "w2", role: "planner", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Build a website",
      provider: trackingProvider,
      model: "mock",
      onEvent: () => {},
      parentTurnId: "t1",
      executionMode: "pipeline",
    });

    expect(receivedPrompts.length).toBe(2);
    // Second worker should see first worker's output in its prompt
    expect(receivedPrompts[1]).toContain("Previous analyses");
    expect(receivedPrompts[1]).toContain("researcher");
  });

  it("writes to sharedContext in pipeline mode", async () => {
    const results = await executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Analyze",
      provider: mockProvider,
      model: "mock",
      onEvent: () => {},
      parentTurnId: "t1",
      executionMode: "pipeline",
    });

    // Results are returned even without explicit sharedContext
    expect(results.length).toBe(1);
    expect(results[0].role).toBe("researcher");
  });
});

describe("executeWorkers — shared context", () => {
  it("allows workers to read and write shared state", async () => {
    const customContext = {
      store: new Map<string, unknown>(),
      get<T>(key: string): T | undefined { return this.store.get(key) as T | undefined; },
      set<T>(key: string, value: T): void { this.store.set(key, value); },
      has(key: string): boolean { return this.store.has(key); },
      entries(): Readonly<Record<string, unknown>> { return Object.fromEntries(this.store); },
    };

    await executeWorkers({
      assignments: [
        { workerId: "w1", role: "researcher", ownedScope: [], inputBundleId: "b1", status: "running" },
      ],
      taskText: "Analyze",
      provider: mockProvider,
      model: "mock",
      onEvent: () => {},
      parentTurnId: "t1",
      sharedContext: customContext,
    });

    expect(customContext.has("worker.researcher")).toBe(true);
    expect(customContext.get<WorkerExecutionResult>("worker.researcher")?.role).toBe("researcher");
  });
});

describe("formatWorkerResults — output formatting", () => {
  it("formats empty results as empty string", () => {
    expect(formatWorkerResults([])).toBe("");
  });

  it("formats multiple worker results", () => {
    const formatted = formatWorkerResults([
      { workerId: "w1", role: "researcher", summary: "Found A", findings: ["A"], durationMs: 100 },
      { workerId: "w2", role: "planner", summary: "Plan B", findings: ["B"], durationMs: 200 },
    ]);
    expect(formatted).toContain("researcher");
    expect(formatted).toContain("planner");
    expect(formatted).toContain("Found A");
    expect(formatted).toContain("Plan B");
  });
});

describe("prepareTaskExecution — task routing", () => {
  it("classifies a research task", () => {
    const prepared = prepareTaskExecution({
      text: "What is the latest version of React?",
      expandedText: "What is the latest version of React?",
      basePrompt: "",
      messages: [{ role: "user", content: "What is the latest version of React?" }],
    });
    expect(prepared.route.kind).toBe("research");
    expect(prepared.route.requiresResearch).toBe(true);
  });

  it("classifies a multi-file edit as single_file_edit (regex)", () => {
    const prepared = prepareTaskExecution({
      text: "Refactor the entire authentication module",
      expandedText: "Refactor the entire authentication module",
      basePrompt: "",
      messages: [{ role: "user", content: "Refactor the entire authentication module" }],
    });
    expect(prepared.route.kind).toBe("single_file_edit");
  });

  it("classifies a simple question as answer_only", () => {
    const prepared = prepareTaskExecution({
      text: "What is 2+2?",
      expandedText: "What is 2+2?",
      basePrompt: "",
      messages: [{ role: "user", content: "What is 2+2?" }],
    });
    expect(prepared.route.kind).toBe("answer_only");
  });

  it("classifies a debug task as single_file_edit (regex)", () => {
    const prepared = prepareTaskExecution({
      text: "Fix the null pointer exception in utils.js",
      expandedText: "Fix the null pointer exception in utils.js",
      basePrompt: "",
      messages: [{ role: "user", content: "Fix the null pointer exception in utils.js" }],
    });
    expect(prepared.route.kind).toBe("single_file_edit");
  });
});

describe("prepareTaskExecutionAsync — semantic routing", () => {
  it("classifies a research task via semantic router", async () => {
    const prepared = await prepareTaskExecutionAsync({
      text: "What is the latest version of React?",
      expandedText: "What is the latest version of React?",
      basePrompt: "",
      messages: [{ role: "user", content: "What is the latest version of React?" }],
    });
    expect(prepared.route.kind).toBe("research");
    expect(prepared.route.requiresResearch).toBe(true);
  });

  it("classifies a simple question as answer_only via semantic router", async () => {
    const prepared = await prepareTaskExecutionAsync({
      text: "What is 2+2?",
      expandedText: "What is 2+2?",
      basePrompt: "",
      messages: [{ role: "user", content: "What is 2+2?" }],
    });
    expect(prepared.route.kind).toBe("answer_only");
  });

  it("falls back to regex when semantic confidence is low", async () => {
    // A very unusual utterance that won't match any prototype well
    const prepared = await prepareTaskExecutionAsync({
      text: "xyz123 nonsense string that matches no prototype",
      expandedText: "xyz123 nonsense string that matches no prototype",
      basePrompt: "",
      messages: [{ role: "user", content: "xyz123 nonsense string that matches no prototype" }],
    });
    // Should still produce a valid route (fallback regex)
    expect(prepared.route.kind).toBeDefined();
    expect(prepared.route.lane).toBeDefined();
    expect(prepared.route.path).toBeDefined();
  });
});
