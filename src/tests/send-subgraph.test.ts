import { describe, it, expect } from "vitest";
import { StateGraph, CompiledGraph, withDefault, append, sum, START, END } from "../graph/index.js";
import type { Send } from "../graph/types.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Send (dynamic fan-out)
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph Send primitive", () => {
  it("sends to a single target node", async () => {
    const g = new StateGraph<{ count: number; log: string[] }>({
      count: sum(0),
      log: append<string>(),
    })
      .addNode("source", async (state) => {
        const send: Send = { node: "target", arg: { count: 10 } };
        return { count: 1, log: ["source"], send: send as unknown as string };
      })
      .addNode("target", async (state) => ({ count: 5, log: ["target"] }))
      .addEdge(START, "source")
      .addEdge("target", END)
      .setEntryPoint("source");

    // The send injects count: 10 into state before target runs
    // But target returns count: 5
    // Total: source(1) + send(10) + target(5) = 16
    const result = await g.compile().invoke({});
    expect(result.count).toBe(16);
    expect(result.log).toContain("source");
    expect(result.log).toContain("target");
  });

  it("sends to multiple target nodes (Map-Reduce)", async () => {
    const g = new StateGraph<{ values: number[]; results: string[] }>({
      values: append<number>(),
      results: append<string>(),
    })
      .addNode("mapper", async (state) => {
        const sends: Send[] = [
          { node: "worker", arg: { values: 1 } },
          { node: "worker", arg: { values: 2 } },
          { node: "worker", arg: { values: 3 } },
        ];
        return { results: ["mapped"], sends: sends as unknown as string[] };
      })
      .addNode("worker", async (state) => ({
        results: [`worker-${state.values[state.values.length - 1]}`],
      }))
      .addEdge(START, "mapper")
      .addEdge("worker", END)
      .setEntryPoint("mapper");

    const result = await g.compile().invoke({});
    expect(result.results).toContain("mapped");
    expect(result.results.filter((r: string) => r.startsWith("worker-")).length).toBe(3);
  });

  it("Send arg merges into state before target executes", async () => {
    const g = new StateGraph<{ tag: string; items: string[] }>({
      tag: withDefault<string>("default"),
      items: append<string>(),
    })
      .addNode("dispatch", async () => {
        const send: Send = { node: "handler", arg: { tag: "urgent", items: ["a"] } };
        return { send: send as unknown as string } as Partial<{ tag: string; items: string[] }>;
      })
      .addNode("handler", async (state) => ({
        items: [`handled-${state.tag}`],
      }))
      .addEdge(START, "dispatch")
      .addEdge("handler", END)
      .setEntryPoint("dispatch");

    const result = await g.compile().invoke({});
    expect(result.tag).toBe("urgent");
    expect(result.items).toContain("a");
    expect(result.items).toContain("handled-urgent");
  });
});

// ═══════════════════════════════════════════════════════════════════════════════
// Subgraphs
// ═══════════════════════════════════════════════════════════════════════════════

describe("CompiledGraph subgraphs", () => {
  it("embeds a subgraph as a node", async () => {
    // Inner graph: doubles a value
    const inner = new StateGraph<{ value: number }>({
      value: withDefault(0),
    })
      .addNode("double", async (state) => ({ value: (state.value as number) * 2 }))
      .setEntryPoint("double")
      .setFinishPoint("double")
      .compile();

    // Outer graph: increments, then runs subgraph, then increments again
    const outer = new StateGraph<{ value: number }>({
      value: withDefault(0),
    })
      .addNode("inc", async (state) => ({ value: (state.value as number) + 1 }))
      .addNode("sub", inner.asNode())
      .addNode("inc2", async (state) => ({ value: (state.value as number) + 1 }))
      .addEdge(START, "inc")
      .addEdge("inc", "sub")
      .addEdge("sub", "inc2")
      .addEdge("inc2", END)
      .setEntryPoint("inc");

    // 5 + 1 = 6, * 2 = 12, + 1 = 13
    const result = await outer.compile().invoke({ value: 5 });
    expect(result.value).toBe(13);
  });

  it("subgraph with state mapping", async () => {
    // Inner graph works on { innerValue }
    const inner = new StateGraph<{ innerValue: number }>({
      innerValue: withDefault(0),
    })
      .addNode("triple", async (state) => ({ innerValue: (state.innerValue as number) * 3 }))
      .setEntryPoint("triple")
      .setFinishPoint("triple")
      .compile();

    // Outer graph has { outerValue }
    const outer = new StateGraph<{ outerValue: number }>({
      outerValue: withDefault(0),
    })
      .addNode("sub", inner.asNode(
        (parent) => ({ innerValue: parent.outerValue }), // input map
        (sub) => ({ outerValue: sub.innerValue }),       // output map
      ))
      .setEntryPoint("sub")
      .setFinishPoint("sub");

    // 4 * 3 = 12
    const result = await outer.compile().invoke({ outerValue: 4 });
    expect(result.outerValue).toBe(12);
  });

  it("nested subgraphs", async () => {
    // Level 2: squares a value
    const level2 = new StateGraph<{ x: number }>({
      x: withDefault(0),
    })
      .addNode("square", async (state) => ({ x: (state.x as number) ** 2 }))
      .setEntryPoint("square")
      .setFinishPoint("square")
      .compile();

    // Level 1: adds 10, then runs level2
    const level1 = new StateGraph<{ x: number }>({
      x: withDefault(0),
    })
      .addNode("add10", async (state) => ({ x: (state.x as number) + 10 }))
      .addNode("sub", level2.asNode())
      .addEdge(START, "add10")
      .addEdge("add10", "sub")
      .setEntryPoint("add10")
      .compile();

    // Root: doubles, then runs level1
    const root = new StateGraph<{ x: number }>({
      x: withDefault(0),
    })
      .addNode("double", async (state) => ({ x: (state.x as number) * 2 }))
      .addNode("sub", level1.asNode())
      .addEdge(START, "double")
      .addEdge("double", "sub")
      .setEntryPoint("double")
      .compile();

    // 2 * 2 = 4, + 10 = 14, ^ 2 = 196
    const result = await root.invoke({ x: 2 });
    expect(result.x).toBe(196);
  });
});
