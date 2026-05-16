import { describe, it, expect, beforeEach } from "vitest";
import {
  StateGraph,
  CompiledGraph,
  lastValue,
  append,
  prepend,
  sum,
  mapMerge,
  setUnion,
  binaryOperator,
  withDefault,
  START,
  END,
  GraphInterrupt,
} from "../graph/index.js";
import type { GraphCheckpointer, GraphCheckpoint } from "../graph/types.js";
import type { Checkpointer, CheckpointState } from "../agent/types.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Test Infrastructure
// ═══════════════════════════════════════════════════════════════════════════════

class MemoryGraphCheckpointer implements GraphCheckpointer {
  private checkpoints = new Map<string, GraphCheckpoint[]>();
  private agentCheckpoints = new Map<string, import("../agent/types.js").Checkpoint[]>();

  async saveGraphCheckpoint(cp: GraphCheckpoint): Promise<void> {
    const list = this.checkpoints.get(cp.threadId) ?? [];
    list.push(cp);
    this.checkpoints.set(cp.threadId, list);
  }

  async loadGraphCheckpoint(threadId: string, checkpointId?: string): Promise<GraphCheckpoint | null> {
    const list = this.checkpoints.get(threadId);
    if (!list || list.length === 0) return null;
    if (checkpointId) return list.find((c) => c.checkpointId === checkpointId) ?? null;
    return list[list.length - 1];
  }

  async listGraphCheckpoints(threadId: string): Promise<GraphCheckpoint[]> {
    return [...(this.checkpoints.get(threadId) ?? [])];
  }

  async save(threadId: string, state: CheckpointState): Promise<void> {
    const cp = { threadId, round: state.round, messages: state.messages, createdAt: Date.now() };
    const list = this.agentCheckpoints.get(threadId) ?? [];
    list.push(cp);
    this.agentCheckpoints.set(threadId, list);
  }

  async load(threadId: string) {
    const list = this.agentCheckpoints.get(threadId);
    return list && list.length > 0 ? list[list.length - 1] : null;
  }

  async loadAt(threadId: string, round: number) {
    const list = this.agentCheckpoints.get(threadId);
    return list?.find((c) => c.round === round) ?? null;
  }

  async list(threadId: string) {
    return [...(this.agentCheckpoints.get(threadId) ?? [])];
  }

  async fork(threadId: string, round: number, newThreadId: string): Promise<void> {
    const cp = await this.loadAt(threadId, round);
    if (cp) await this.save(newThreadId, { messages: cp.messages, round: cp.round });
  }

  async delete(threadId: string): Promise<void> {
    this.checkpoints.delete(threadId);
    this.agentCheckpoints.delete(threadId);
  }
}

interface CounterState {
  count: number;
  log: string[];
  meta: Record<string, number>;
  [key: string]: unknown;
}

function counterGraph(): StateGraph<CounterState> {
  return new StateGraph<CounterState>({
    count: withDefault(0),
    log: append<string>(),
    meta: mapMerge<number>(),
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Builder & Validation
// ═══════════════════════════════════════════════════════════════════════════════

describe("StateGraph builder validation", () => {
  it("constructs with typed channels", () => {
    const graph = counterGraph();
    expect(graph).toBeInstanceOf(StateGraph);
  });

  it("adds nodes, edges, hooks fluently", () => {
    const g = counterGraph()
      .addNode("inc", async (state) => ({ count: state.count + 1 }))
      .addEdge(START, "inc")
      .addEdge("inc", END)
      .setEntryPoint("inc")
      .beforeNode("inc", async () => {})
      .afterNode("inc", async () => {});
    expect(g._getNodes().has("inc")).toBe(true);
  });

  it("rejects START as node name", () => {
    expect(() => counterGraph().addNode(START, async () => ({}))).toThrow();
  });

  it("rejects END as node name", () => {
    expect(() => counterGraph().addNode(END, async () => ({}))).toThrow();
  });

  it("rejects missing entry point", () => {
    const g = counterGraph().addNode("a", async () => ({}));
    expect(() => g.compile()).toThrow("entry point not set");
  });

  it("rejects entry point that is not a node", () => {
    const g = counterGraph().setEntryPoint("ghost");
    expect(() => g.compile()).toThrow('entry point "ghost" is not a registered node');
  });

  it("rejects edge to unknown source", () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .addEdge("b", "a");
    expect(() => g.compile()).toThrow('edge source "b" is not a registered node');
  });

  it("rejects edge to unknown target", () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .addEdge("a", "b");
    expect(() => g.compile()).toThrow('edge target "b" is not a registered node');
  });

  it("rejects conditional edge from unknown source", () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .addConditionalEdges("ghost", () => END);
    expect(() => g.compile()).toThrow('conditional edge source "ghost" is not a registered node');
  });

  it("allows START as edge source", () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .addEdge(START, "a");
    expect(() => g.compile()).not.toThrow();
  });

  it("allows END as edge target", () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .addEdge("a", END);
    expect(() => g.compile()).not.toThrow();
  });

  it("supports multiple finish points", () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .addNode("b", async () => ({}))
      .addEdge(START, "a")
      .addEdge(START, "b")
      .setEntryPoint("a")
      .setFinishPoint("a")
      .setFinishPoint("b");
    expect(() => g.compile()).not.toThrow();
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Single-Node Execution
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph single-node execution", () => {
  it("invoke returns final state", async () => {
    const g = counterGraph()
      .addNode("double", async (state) => ({ count: state.count * 2 }))
      .setEntryPoint("double")
      .setFinishPoint("double");

    const result = await g.compile().invoke({ count: 5 });
    expect(result.count).toBe(10);
    expect(result.log).toEqual([]);
    expect(result.meta).toEqual({});
  });

  it("invoke uses channel defaults", async () => {
    const g = counterGraph()
      .addNode("noop", async () => ({}))
      .setEntryPoint("noop")
      .setFinishPoint("noop");

    const result = await g.compile().invoke({});
    expect(result.count).toBe(0);
    expect(result.log).toEqual([]);
    expect(result.meta).toEqual({});
  });

  it("invoke passes input through channels", async () => {
    const g = counterGraph()
      .addNode("pass", async () => ({}))
      .setEntryPoint("pass")
      .setFinishPoint("pass");

    const result = await g.compile().invoke({ count: 7, log: ["x"], meta: { k: 1 } });
    expect(result.count).toBe(7);
    expect(result.log).toEqual(["x"]);
    expect(result.meta).toEqual({ k: 1 });
  });

  it("stream emits complete lifecycle", async () => {
    const g = counterGraph()
      .addNode("inc", async () => ({ count: 1 }))
      .setEntryPoint("inc")
      .setFinishPoint("inc");

    const events = [];
    for await (const e of g.compile().stream({})) events.push(e.type);
    expect(events).toEqual(["start", "node_start", "node_end", "end"]);
  });

  it("stream includes threadId", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const events = [];
    for await (const e of g.compile().stream({}, { threadId: "tid-42" })) {
      events.push(e);
    }
    expect(events.every((e: any) => e.threadId === "tid-42")).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Linear Multi-Node Execution
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph linear chains", () => {
  it("executes a 2-node chain", async () => {
    const g = counterGraph()
      .addNode("a", async (state) => ({ count: state.count + 1, log: ["a"] }))
      .addNode("b", async (state) => ({ count: state.count * 2, log: ["b"] }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .addEdge("b", END)
      .setEntryPoint("a");

    const result = await g.compile().invoke({ count: 1 });
    expect(result.count).toBe(4); // (1 + 1) * 2
    expect(result.log).toEqual(["a", "b"]);
  });

  it("executes a 5-node chain", async () => {
    const g = counterGraph()
      .addNode("n1", async () => ({ count: 1 }))
      .addNode("n2", async (s) => ({ count: s.count + 10 }))
      .addNode("n3", async (s) => ({ count: s.count * 2 }))
      .addNode("n4", async (s) => ({ count: s.count - 3 }))
      .addNode("n5", async (s) => ({ count: s.count + 100 }))
      .addEdge(START, "n1")
      .addEdge("n1", "n2")
      .addEdge("n2", "n3")
      .addEdge("n3", "n4")
      .addEdge("n4", "n5")
      .addEdge("n5", END)
      .setEntryPoint("n1");

    const result = await g.compile().invoke({ count: 0 });
    expect(result.count).toBe(119); // ((1 + 10) * 2 - 3) + 100 = 119
  });

  it("emits state after each wave in stream", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .addNode("b", async (s) => ({ count: s.count + 10 }))
      .addNode("c", async (s) => ({ count: s.count * 2 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .addEdge("b", "c")
      .addEdge("c", END)
      .setEntryPoint("a");

    const states: number[] = [];
    for await (const e of g.compile().stream({})) {
      if (e.type === "state") states.push(e.values.count as number);
    }
    expect(states).toEqual([1, 11, 22]);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Parallel Wave Execution
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph parallel waves", () => {
  it("runs independent nodes in parallel", async () => {
    const g = new StateGraph<{ vals: number[] }>({
      vals: append<number>(),
    })
      .addNode("left", async () => ({ vals: [1] }))
      .addNode("right", async () => ({ vals: [2] }))
      .addNode("merge", async (s) => ({ vals: [s.vals.reduce((a, b) => a + b, 0)] }))
      .addEdge(START, "left")
      .addEdge(START, "right")
      .addEdge("left", "merge")
      .addEdge("right", "merge")
      .addEdge("merge", END)
      .setEntryPoint("left");

    const result = await g.compile().invoke({});
    expect(result.vals.sort((a, b) => a - b)).toEqual([1, 2, 3]);
  });

  it("parallel nodes see same pre-wave state", async () => {
    let leftState = 0;
    let rightState = 0;

    const g = counterGraph()
      .addNode("left", async (s) => { leftState = s.count; return { count: 100 }; })
      .addNode("right", async (s) => { rightState = s.count; return { count: 200 }; })
      .addEdge(START, "left")
      .addEdge(START, "right")
      .setEntryPoint("left");

    await g.compile().invoke({ count: 5 });
    expect(leftState).toBe(5);
    expect(rightState).toBe(5);
  });

  it("merges parallel outputs deterministically via channels", async () => {
    const g = new StateGraph<{ total: number; tags: string[] }>({
      total: sum(0),
      tags: append<string>(),
    })
      .addNode("a", async () => ({ total: 10, tags: ["a"] }))
      .addNode("b", async () => ({ total: 20, tags: ["b"] }))
      .addNode("c", async () => ({ total: 30, tags: ["c"] }))
      .addEdge(START, "a")
      .addEdge(START, "b")
      .addEdge(START, "c")
      .setEntryPoint("a");

    const result = await g.compile().invoke({});
    expect(result.total).toBe(60);
    expect(result.tags.sort()).toEqual(["a", "b", "c"]);
  });

  it("handles mixed parallel + sequential topology", async () => {
    // Wave 1: a, b (parallel)
    // Wave 2: c (depends on a)
    // Wave 3: d (depends on b and c)
    const g = new StateGraph<{ trace: string[] }>({
      trace: append<string>(),
    })
      .addNode("a", async () => ({ trace: ["a"] }))
      .addNode("b", async () => ({ trace: ["b"] }))
      .addNode("c", async (s) => ({ trace: ["c-after-" + s.trace.join("")] }))
      .addNode("d", async (s) => ({ trace: ["d-after-" + s.trace.join("")] }))
      .addEdge(START, "a")
      .addEdge(START, "b")
      .addEdge("a", "c")
      .addEdge("b", "d")
      .addEdge("c", "d")
      .setEntryPoint("a");

    const result = await g.compile().invoke({});
    expect(result.trace).toContain("a");
    expect(result.trace).toContain("b");
    expect(result.trace).toContain("c-after-ab");
    expect(result.trace.some((t: string) => t.startsWith("d-after-"))).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Conditional Edges
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph conditional edges", () => {
  it("routes to single target based on state", async () => {
    const g = counterGraph()
      .addNode("check", async (s) => ({ count: s.count }))
      .addNode("big", async () => ({ log: ["big"] }))
      .addNode("small", async () => ({ log: ["small"] }))
      .addEdge(START, "check")
      .addConditionalEdges("check", (s) => (s.count >= 10 ? "big" : "small"))
      .addEdge("big", END)
      .addEdge("small", END)
      .setEntryPoint("check");

    expect((await g.compile().invoke({ count: 15 })).log).toContain("big");
    expect((await g.compile().invoke({ count: 5 })).log).toContain("small");
  });

  it("routes to multiple targets (array return)", async () => {
    const g = new StateGraph<{ visited: string[] }>({
      visited: append<string>(),
    })
      .addNode("fork", async () => ({ visited: ["fork"] }))
      .addNode("a", async () => ({ visited: ["a"] }))
      .addNode("b", async () => ({ visited: ["b"] }))
      .addNode("join", async () => ({ visited: ["join"] }))
      .addEdge(START, "fork")
      .addConditionalEdges("fork", () => ["a", "b"])
      .addEdge("a", "join")
      .addEdge("b", "join")
      .addEdge("join", END)
      .setEntryPoint("fork");

    const result = await g.compile().invoke({});
    expect(result.visited).toContain("fork");
    expect(result.visited).toContain("a");
    expect(result.visited).toContain("b");
    expect(result.visited).toContain("join");
  });

  it("routes to END sentinel", async () => {
    const g = counterGraph()
      .addNode("gate", async (s) => ({ count: s.count }))
      .addNode("process", async () => ({ count: 999 }))
      .addEdge(START, "gate")
      .addConditionalEdges("gate", (s) => (s.count > 0 ? END : "process"))
      .addEdge("process", END)
      .setEntryPoint("gate");

    expect((await g.compile().invoke({ count: 5 })).count).toBe(5);
    expect((await g.compile().invoke({ count: 0 })).count).toBe(999);
  });

  it("handles dynamic routing in stream", async () => {
    const g = counterGraph()
      .addNode("decide", async () => ({}))
      .addNode("pathA", async () => ({ log: ["A"] }))
      .addNode("pathB", async () => ({ log: ["B"] }))
      .addEdge(START, "decide")
      .addConditionalEdges("decide", () => "pathA")
      .addEdge("pathA", END)
      .addEdge("pathB", END)
      .setEntryPoint("decide");

    const events = [];
    for await (const e of g.compile().stream({})) events.push(e.type);
    expect(events).toContain("node_start");
    expect(events.filter((t) => t === "node_start").length).toBe(2); // decide + pathA
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Channel Reducers — Exhaustive
// ═══════════════════════════════════════════════════════════════════════════════

describe("Channel reducers", () => {
  it("sum accumulates", async () => {
    const g = new StateGraph<{ total: number }>({ total: sum(0) })
      .addNode("a", async () => ({ total: 5 }))
      .addNode("b", async () => ({ total: 3 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).total).toBe(8);
  });

  it("sum with negative values", async () => {
    const g = new StateGraph<{ total: number }>({ total: sum(100) })
      .addNode("a", async () => ({ total: -30 }))
      .addNode("b", async () => ({ total: -20 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).total).toBe(50);
  });

  it("append accumulates arrays", async () => {
    const g = new StateGraph<{ items: string[] }>({ items: append<string>() })
      .addNode("a", async () => ({ items: ["x"] }))
      .addNode("b", async () => ({ items: ["y", "z"] }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).items).toEqual(["x", "y", "z"]);
  });

  it("append with single element (non-array)", async () => {
    const g = new StateGraph<{ items: number[] }>({ items: append<number>() })
      .addNode("a", async () => ({ items: 42 as unknown as number[] })) // type hack for test
      .setEntryPoint("a")
      .setFinishPoint("a");
    // Channel.update handles non-array by wrapping
  });

  it("prepend puts new items first", async () => {
    const g = new StateGraph<{ items: string[] }>({ items: prepend<string>() })
      .addNode("a", async () => ({ items: ["second"] }))
      .addNode("b", async () => ({ items: ["first"] }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).items).toEqual(["first", "second"]);
  });

  it("mapMerge shallow-merges objects", async () => {
    const g = new StateGraph<{ data: Record<string, number> }>({ data: mapMerge<number>() })
      .addNode("a", async () => ({ data: { x: 1 } }))
      .addNode("b", async () => ({ data: { y: 2 } }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).data).toEqual({ x: 1, y: 2 });
  });

  it("mapMerge overwrites same keys", async () => {
    const g = new StateGraph<{ data: Record<string, number> }>({ data: mapMerge<number>() })
      .addNode("a", async () => ({ data: { x: 1 } }))
      .addNode("b", async () => ({ data: { x: 99 } }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).data).toEqual({ x: 99 });
  });

  it("setUnion accumulates unique items", async () => {
    const g = new StateGraph<{ tags: Set<string> }>({ tags: setUnion<string>() })
      .addNode("a", async () => ({ tags: new Set(["x", "y"]) }))
      .addNode("b", async () => ({ tags: new Set(["y", "z"]) }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    const result = await g.compile().invoke({});
    expect(result.tags).toBeInstanceOf(Set);
    expect([...result.tags].sort()).toEqual(["x", "y", "z"]);
  });

  it("setUnion with single element", async () => {
    const g = new StateGraph<{ tags: Set<string> }>({ tags: setUnion<string>() })
      .addNode("a", async () => ({ tags: "solo" as unknown as Set<string> }))
      .setEntryPoint("a")
      .setFinishPoint("a");
    const result = await g.compile().invoke({});
    expect([...result.tags]).toEqual(["solo"]);
  });

  it("binaryOperator with custom logic", async () => {
    const g = new StateGraph<{ max: number }>({
      max: binaryOperator((a, b) => Math.max(a, b), 0),
    })
      .addNode("a", async () => ({ max: 5 }))
      .addNode("b", async () => ({ max: 10 }))
      .addNode("c", async () => ({ max: 3 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .addEdge("b", "c")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).max).toBe(10);
  });

  it("lastValue overwrites", async () => {
    const g = new StateGraph<{ name: string | undefined }>({ name: lastValue<string>() as import("../graph/types.js").Channel<string | undefined> })
      .addNode("a", async () => ({ name: "alice" }))
      .addNode("b", async () => ({ name: "bob" }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");
    expect((await g.compile().invoke({})).name).toBe("bob");
  });

  it("withDefault provides initial and is overwritten", async () => {
    const g = new StateGraph<{ flag: boolean }>({ flag: withDefault(false) })
      .addNode("set", async () => ({ flag: true }))
      .setEntryPoint("set")
      .setFinishPoint("set");
    expect((await g.compile().invoke({})).flag).toBe(true);
    expect((await g.compile().invoke({ flag: false })).flag).toBe(true);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Checkpointing
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph checkpointing", () => {
  let cp: MemoryGraphCheckpointer;

  beforeEach(() => {
    cp = new MemoryGraphCheckpointer();
  });

  it("saves after each wave", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .addNode("b", async (s) => ({ count: s.count + 10 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    await g.compile({ checkpointer: cp }).invoke({ count: 0 }, { threadId: "t1" });
    const cps = await cp.listGraphCheckpoints("t1");
    expect(cps.length).toBeGreaterThanOrEqual(2);
  });

  it("checkpoint contains correct state", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 42 }))
      .setEntryPoint("a")
      .setFinishPoint("a");

    await g.compile({ checkpointer: cp }).invoke({}, { threadId: "t2" });
    const last = await cp.loadGraphCheckpoint("t2");
    expect(last!.state.count).toBe(42);
    expect(last!.completedNodes).toContain("a");
    expect(last!.waveCount).toBeGreaterThanOrEqual(1);
  });

  it("checkpoint tracks parent reference", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .addNode("b", async (s) => ({ count: s.count + 1 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    await g.compile({ checkpointer: cp }).invoke({}, { threadId: "t3" });
    const cps = await cp.listGraphCheckpoints("t3");
    expect(cps.length).toBe(2);
    // Second checkpoint should reference first
    expect(cps[1].parentCheckpointId).toBe(cps[0].checkpointId);
  });

  it("resumes from latest checkpoint on re-invoke", async () => {
    let calls = 0;
    const g = counterGraph()
      .addNode("a", async () => { calls++; return { count: 1 }; })
      .addNode("b", async (s) => ({ count: s.count + 10 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    const app = g.compile({ checkpointer: cp });
    await app.invoke({ count: 0 }, { threadId: "t4" });
    expect(calls).toBe(1);

    // Re-invoke same thread — node "a" is already done
    await app.invoke({ count: 0 }, { threadId: "t4" });
    expect(calls).toBe(1);
  });

  it("getState returns latest checkpointed state", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 77 }))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const app = g.compile({ checkpointer: cp });
    await app.invoke({}, { threadId: "t5" });
    expect((await app.getState("t5"))!.count).toBe(77);
  });

  it("getState returns null for unknown thread", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const app = g.compile({ checkpointer: cp });
    expect(await app.getState("nonexistent")).toBeNull();
  });

  it("updateState patches checkpoint", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const app = g.compile({ checkpointer: cp });
    await app.invoke({}, { threadId: "t6" });
    await app.updateState("t6", { count: 999 });
    expect((await app.getState("t6"))!.count).toBe(999);
  });

  it("updateState throws for unknown thread", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const app = g.compile({ checkpointer: cp });
    await expect(app.updateState("ghost", { count: 1 })).rejects.toThrow("No checkpoint found");
  });

  it("multiple threads are isolated", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const app = g.compile({ checkpointer: cp });
    await app.invoke({ count: 10 }, { threadId: "alpha" });
    await app.invoke({ count: 20 }, { threadId: "beta" });

    expect((await app.getState("alpha"))!.count).toBe(10);
    expect((await app.getState("beta"))!.count).toBe(20);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Interrupt & Resume
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph interrupt & resume", () => {
  let cp: MemoryGraphCheckpointer;

  beforeEach(() => {
    cp = new MemoryGraphCheckpointer();
  });

  it("yields interrupt event", async () => {
    const g = counterGraph()
      .addNode("ask", async () => {
        throw new GraphInterrupt("Need approval", "ask", {});
      })
      .setEntryPoint("ask")
      .setFinishPoint("ask");

    const events = [];
    for await (const e of g.compile({ checkpointer: cp }).stream({}, { threadId: "i1" })) {
      events.push(e);
    }
    expect(events.some((e: any) => e.type === "interrupt")).toBe(true);
  });

  it("saves checkpoint on interrupt", async () => {
    const g = counterGraph()
      .addNode("ask", async () => {
        throw new GraphInterrupt("Need approval", "ask", {});
      })
      .setEntryPoint("ask")
      .setFinishPoint("ask");

    for await (const _e of g.compile({ checkpointer: cp }).stream({}, { threadId: "i2" })) {}
    const checkpoint = await cp.loadGraphCheckpoint("i2");
    expect(checkpoint).not.toBeNull();
    expect(checkpoint!.nextNodes).toContain("ask");
  });

  it("resumes and continues", async () => {
    let interrupted = false;
    const g = counterGraph()
      .addNode("ask", async (s) => {
        if (!interrupted) {
          interrupted = true;
          throw new GraphInterrupt("Need approval", "ask", { count: s.count });
        }
        return { count: s.count + 100 };
      })
      .setEntryPoint("ask")
      .setFinishPoint("ask");

    const app = g.compile({ checkpointer: cp });

    // First pass — interrupts
    for await (const _e of app.stream({ count: 5 }, { threadId: "i3" })) {}

    // Resume
    const events = [];
    for await (const e of app.resume("i3", { update: { count: 5 } })) {
      events.push(e.type);
    }
    expect(events).toContain("end");
    expect((await app.getState("i3"))!.count).toBe(105);
  });

  it("resume throws if no checkpoint", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({}))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const app = g.compile({ checkpointer: cp });
    await expect(
      (async () => {
        for await (const _e of app.resume("ghost", { update: {} })) {}
      })(),
    ).rejects.toThrow("No checkpoint found");
  });

  it("resume with explicit resumeNode", async () => {
    let step = 0;
    const g = counterGraph()
      .addNode("a", async () => {
        step++;
        if (step === 1) throw new GraphInterrupt("pause", "a", {});
        return { count: 1 };
      })
      .addNode("b", async (s) => ({ count: s.count + 10 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    const app = g.compile({ checkpointer: cp });
    for await (const _e of app.stream({}, { threadId: "i4" })) {}

    // Resume from "a" explicitly
    const events = [];
    for await (const e of app.resume("i4", { update: {}, resumeNode: "a" })) {
      events.push(e.type);
    }
    expect(events).toContain("end");
    expect((await app.getState("i4"))!.count).toBe(11);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Error Handling & Edge Cases
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph errors & edge cases", () => {
  it("throws on recursion limit", async () => {
    const g = counterGraph()
      .addNode("loop", async (s) => ({ count: s.count + 1 }))
      .addEdge(START, "loop")
      .addEdge("loop", "loop")
      .setEntryPoint("loop");

    await expect(g.compile({ recursionLimit: 3, detectCycles: false }).invoke({ count: 0 })).rejects.toThrow("recursion limit");
  });

  it("respects pre-aborted signal", async () => {
    const g = counterGraph()
      .addNode("slow", async () => ({ count: 1 }))
      .setEntryPoint("slow")
      .setFinishPoint("slow");

    const ctrl = new AbortController();
    ctrl.abort();
    await expect(g.compile().invoke({}, { signal: ctrl.signal })).rejects.toThrow("aborted");
  });

  it("respects signal aborted mid-stream", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .addNode("b", async () => {
        await new Promise((r) => setTimeout(r, 50));
        return { count: 2 };
      })
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    const ctrl = new AbortController();
    const app = g.compile();
    const stream = app.stream({}, { signal: ctrl.signal });

    // Read first event then abort
    const first = await stream.next();
    expect(first.value).toBeDefined();
    ctrl.abort();

    await expect(stream.next()).rejects.toThrow("aborted");
  });

  it("emits error event on node failure", async () => {
    const g = counterGraph()
      .addNode("bad", async () => {
        throw new Error("boom");
      })
      .setEntryPoint("bad")
      .setFinishPoint("bad");

    const events = [];
    try {
      for await (const e of g.compile().stream({})) events.push(e);
    } catch {
      /* expected */
    }
    const err = events.find((e: any) => e.type === "error");
    expect(err).toBeDefined();
    expect((err as any).error).toContain("boom");
  });

  it("node error propagates and terminates graph", async () => {
    const g = counterGraph()
      .addNode("bad", async () => {
        throw new Error("fatal");
      })
      .addNode("after", async () => ({ count: 999 }))
      .addEdge(START, "bad")
      .addEdge("bad", "after")
      .setEntryPoint("bad");

    await expect(g.compile().invoke({})).rejects.toThrow("fatal");
  });

  it("handles synchronous node functions", async () => {
    const g = counterGraph()
      .addNode("sync", (state) => ({ count: state.count + 5 }))
      .setEntryPoint("sync")
      .setFinishPoint("sync");

    const result = await g.compile().invoke({ count: 1 });
    expect(result.count).toBe(6);
  });

  it("handles empty graph (start → finish)", async () => {
    const g = counterGraph()
      .addNode("pass", async () => ({}))
      .setEntryPoint("pass")
      .setFinishPoint("pass");

    const result = await g.compile().invoke({ count: 5, log: ["x"] });
    expect(result.count).toBe(5);
    expect(result.log).toEqual(["x"]);
  });

  it("handles graph with no edges beyond entry", async () => {
    const g = counterGraph()
      .addNode("lonely", async () => ({ count: 1 }))
      .setEntryPoint("lonely");
    // No finish point, no outgoing edges
    const result = await g.compile().invoke({});
    expect(result.count).toBe(1);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Hooks
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph hooks", () => {
  it("beforeNode runs before node execution", async () => {
    const order: string[] = [];
    const g = counterGraph()
      .addNode("a", async () => { order.push("node"); return { count: 1 }; })
      .setEntryPoint("a")
      .setFinishPoint("a")
      .beforeNode("a", async () => { order.push("before"); });

    await g.compile().invoke({});
    expect(order).toEqual(["before", "node"]);
  });

  it("afterNode runs after node execution", async () => {
    const order: string[] = [];
    const g = counterGraph()
      .addNode("a", async () => { order.push("node"); return { count: 1 }; })
      .setEntryPoint("a")
      .setFinishPoint("a")
      .afterNode("a", async () => { order.push("after"); });

    await g.compile().invoke({});
    expect(order).toEqual(["node", "after"]);
  });

  it("multiple hooks run in registration order", async () => {
    const order: number[] = [];
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .setEntryPoint("a")
      .setFinishPoint("a")
      .beforeNode("a", async () => { order.push(1); })
      .beforeNode("a", async () => { order.push(2); })
      .afterNode("a", async () => { order.push(3); })
      .afterNode("a", async () => { order.push(4); });

    await g.compile().invoke({});
    expect(order).toEqual([1, 2, 3, 4]);
  });

  it("hook receives current state", async () => {
    let received = 0;
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .setEntryPoint("a")
      .setFinishPoint("a")
      .beforeNode("a", async (state) => { received = state.count; });

    await g.compile().invoke({ count: 42 });
    expect(received).toBe(42);
  });

  it("afterNode receives output", async () => {
    let received: Partial<CounterState> = {};
    const g = counterGraph()
      .addNode("a", async () => ({ count: 7 }))
      .setEntryPoint("a")
      .setFinishPoint("a")
      .afterNode("a", async (_state, output) => { received = output; });

    await g.compile().invoke({});
    expect(received.count).toBe(7);
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Complex Topologies
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph complex topologies", () => {
  it("diamond shape", async () => {
    const g = new StateGraph<{ value: number; tag: string | undefined }>({
      value: withDefault(0),
      tag: lastValue<string>(),
    })
      .addNode("start", async () => ({ value: 10 }))
      .addNode("left", async () => ({ value: 1, tag: "L" }))
      .addNode("right", async () => ({ value: 2, tag: "R" }))
      .addNode("end", async (s) => ({ value: s.value + 100 }))
      .addEdge(START, "start")
      .addConditionalEdges("start", () => ["left", "right"])
      .addEdge("left", "end")
      .addEdge("right", "end")
      .addEdge("end", END)
      .setEntryPoint("start");

    const result = await g.compile().invoke({});
    expect(result.value).toBe(102); // withDefault: start=10, left=1, right=2 (overwrites), end=102
    expect(result.tag).toMatch(/^[LR]$/);
  });

  it("deep graph (10 nodes)", async () => {
    const g = counterGraph();
    let prev = "start";
    g.addNode(prev, async () => ({ count: 1 }));
    g.setEntryPoint(prev);

    for (let i = 1; i <= 10; i++) {
      const name = `n${i}`;
      g.addNode(name, async (s) => ({ count: s.count + 1 }));
      g.addEdge(prev, name);
      prev = name;
    }
    g.addEdge(prev, END);

    const result = await g.compile().invoke({ count: 0 });
    expect(result.count).toBe(11);
  });

  it("fan-out then fan-in", async () => {
    const g = new StateGraph<{ parts: string[] }>({ parts: append<string>() })
      .addNode("source", async () => ({ parts: ["source"] }))
      .addNode("w1", async () => ({ parts: ["w1"] }))
      .addNode("w2", async () => ({ parts: ["w2"] }))
      .addNode("w3", async () => ({ parts: ["w3"] }))
      .addNode("sink", async () => ({ parts: ["sink"] }))
      .addEdge(START, "source")
      .addConditionalEdges("source", () => ["w1", "w2", "w3"])
      .addEdge("w1", "sink")
      .addEdge("w2", "sink")
      .addEdge("w3", "sink")
      .addEdge("sink", END)
      .setEntryPoint("source");

    const result = await g.compile().invoke({});
    expect(result.parts).toContain("source");
    expect(result.parts).toContain("w1");
    expect(result.parts).toContain("w2");
    expect(result.parts).toContain("w3");
    expect(result.parts).toContain("sink");
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Determinism & Stress
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph determinism", () => {
  it("produces identical results on repeated invocations", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1, log: ["a"] }))
      .addNode("b", async (s) => ({ count: s.count * 2, log: ["b"] }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    const app = g.compile();
    const r1 = await app.invoke({});
    const r2 = await app.invoke({});
    const r3 = await app.invoke({});

    expect(r1).toEqual(r2);
    expect(r2).toEqual(r3);
  });

  it("parallel wave results are deterministic with reducers", async () => {
    const g = new StateGraph<{ vals: number[] }>({ vals: append<number>() })
      .addNode("a", async () => ({ vals: [1] }))
      .addNode("b", async () => ({ vals: [2] }))
      .addNode("c", async () => ({ vals: [3] }))
      .addEdge(START, "a")
      .addEdge(START, "b")
      .addEdge(START, "c")
      .setEntryPoint("a");

    const app = g.compile();
    const results = await Promise.all([app.invoke({}), app.invoke({}), app.invoke({})]);
    for (const r of results) {
      expect(r.vals.sort((a, b) => a - b)).toEqual([1, 2, 3]);
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Stream Modes
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph stream modes", () => {
  it("values mode emits state after each wave", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .addNode("b", async (s) => ({ count: s.count + 10 }))
      .addEdge(START, "a")
      .addEdge("a", "b")
      .setEntryPoint("a");

    const events = [];
    for await (const e of g.compile().stream({}, { streamMode: "values" })) {
      events.push(e.type);
    }
    expect(events.filter((t) => t === "state").length).toBe(2);
  });

  it("updates mode suppresses state events", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const events = [];
    for await (const e of g.compile().stream({}, { streamMode: "updates" })) {
      events.push(e.type);
    }
    expect(events).not.toContain("state");
    expect(events).toContain("start");
    expect(events).toContain("end");
  });

  it("debug mode includes all events", async () => {
    const g = counterGraph()
      .addNode("a", async () => ({ count: 1 }))
      .setEntryPoint("a")
      .setFinishPoint("a");

    const events = [];
    for await (const e of g.compile().stream({}, { streamMode: "debug" })) {
      events.push(e.type);
    }
    expect(events).toContain("start");
    expect(events).toContain("node_start");
    expect(events).toContain("node_end");
    expect(events).toContain("end");
  });
});
