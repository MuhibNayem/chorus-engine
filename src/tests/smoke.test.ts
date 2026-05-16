import { describe, it, expect } from "vitest";
import {
  HitlGate,
  BtwQueue,
  JsonFileCheckpointer,
  createFilesystemTools,
  assessCommandSafety,
  computeWaves,
} from "../index.js";
import type { SwarmAgent, WorkerEventCallback, SubagentEventCallback } from "../index.js";

describe("chorus-engine smoke tests", () => {
  it("HitlGate resolves approve correctly", async () => {
    const gate = new HitlGate();
    const key = "test-resume-key";
    gate.resolve(key, { type: "approve" });
    const decision = await gate.wait(key);
    expect(decision.type).toBe("approve");
  });

  it("BtwQueue drains correctly", () => {
    const q = new BtwQueue();
    q.enqueue("hello");
    q.enqueue("world");
    expect(q.drain()).toEqual(["hello", "world"]);
    expect(q.drain()).toEqual([]);
  });

  it("JsonFileCheckpointer round-trips a checkpoint", async () => {
    const cp = new JsonFileCheckpointer();
    const threadId = `smoke-test-${Date.now()}`;
    await cp.save(threadId, { messages: [{ role: "user", content: "hi" }], round: 0 });
    const loaded = await cp.load(threadId);
    expect(loaded?.round).toBe(0);
    expect(loaded?.messages[0].content).toBe("hi");
    await cp.delete(threadId);
  });

  it("createFilesystemTools sandboxes paths", () => {
    const tools = createFilesystemTools("/tmp/test-workspace");
    expect(tools.length).toBeGreaterThan(0);
    expect(tools.map((t) => t.name)).toContain("file_read");
  });

  it("assessCommandSafety blocks rm -rf", () => {
    const result = assessCommandSafety("rm", ["-rf"]);
    expect(result.ok).toBe(false);
  });

  it("assessCommandSafety allows safe commands", () => {
    const result = assessCommandSafety("ls", ["-la"]);
    expect(result.ok).toBe(true);
  });

  it("computeWaves resolves a simple DAG", () => {
    const agents: SwarmAgent[] = [
      { name: "A", description: "", systemPrompt: "", tools: [], handoffDestinations: [], contextMode: "isolated", maxRounds: 5 },
      { name: "B", description: "", systemPrompt: "", tools: [], handoffDestinations: [], contextMode: "isolated", maxRounds: 5, dependsOn: ["A"] },
      { name: "C", description: "", systemPrompt: "", tools: [], handoffDestinations: [], contextMode: "isolated", maxRounds: 5, dependsOn: ["A"] },
      { name: "D", description: "", systemPrompt: "", tools: [], handoffDestinations: [], contextMode: "isolated", maxRounds: 5, dependsOn: ["B", "C"] },
    ];
    const waves = computeWaves(agents);
    expect(waves.length).toBe(3);
    expect(waves[0].map((a) => a.name)).toEqual(["A"]);
    expect(waves[1].map((a) => a.name).sort()).toEqual(["B", "C"]);
    expect(waves[2].map((a) => a.name)).toEqual(["D"]);
  });

  it("WorkerEventCallback type is callable", () => {
    const events: string[] = [];
    const cb: WorkerEventCallback = (event) => events.push(event.type);
    cb({ type: "worker-add", workerId: "w1", role: "planner", emoji: "🏗️", color: "blue", status: "running", summary: "...", sessionId: "s1" });
    expect(events).toEqual(["worker-add"]);
  });

  it("SubagentEventCallback type is callable", () => {
    const events: string[] = [];
    const cb: SubagentEventCallback = (event) => events.push(event.type);
    cb({ type: "subagent-add", subagentId: "a1", name: "test", task: "do it", status: "running", sessionId: "s1" });
    expect(events).toEqual(["subagent-add"]);
  });
});
