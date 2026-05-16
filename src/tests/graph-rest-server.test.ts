import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { GraphRestServer } from "../graph/rest-server.js";
import { StateGraph, withDefault, append } from "../graph/index.js";
import type { GraphCheckpointer, GraphCheckpoint } from "../graph/types.js";
import type { Checkpointer, CheckpointState } from "../agent/types.js";

// ── In-Memory Checkpointer for tests ─────────────────────────────────────────

class MemoryGraphCheckpointer implements GraphCheckpointer {
  private checkpoints = new Map<string, GraphCheckpoint[]>();
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
  async save(threadId: string, state: CheckpointState): Promise<void> {}
  async load(threadId: string) { return null; }
  async loadAt(threadId: string, round: number) { return null; }
  async list(threadId: string) { return []; }
  async fork(threadId: string, round: number, newThreadId: string): Promise<void> {}
  async delete(threadId: string): Promise<void> { this.checkpoints.delete(threadId); }
}

// ── Test Graph ───────────────────────────────────────────────────────────────

function buildTestGraph() {
  return new StateGraph<{ value: number; trace: string[] }>({
    value: withDefault(0),
    trace: append<string>(),
  })
    .addNode("inc", async (state) => ({ value: (state.value as number) + 1, trace: ["inc"] }))
    .addNode("double", async (state) => ({ value: (state.value as number) * 2, trace: ["double"] }))
    .addEdge("__start__", "inc")
    .addEdge("inc", "double")
    .addEdge("double", "__end__")
    .setEntryPoint("inc");
}

// ── HTTP Helper ──────────────────────────────────────────────────────────────

async function fetchJson(server: GraphRestServer<any>, path: string, opts?: { method?: string; body?: unknown }): Promise<{ status: number; data: unknown }> {
  const url = `${server.baseUrl}${path}`;
  const res = await fetch(url, {
    method: opts?.method ?? "GET",
    headers: { "Content-Type": "application/json" },
    body: opts?.body ? JSON.stringify(opts.body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  return { status: res.status, data };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("GraphRestServer", () => {
  let server: GraphRestServer<any>;

  beforeEach(async () => {
    const graph = buildTestGraph().compile({ checkpointer: new MemoryGraphCheckpointer() });
    server = new GraphRestServer({ graph, port: 18123 });
    await server.start();
  });

  afterEach(async () => {
    try { await server.stop(); } catch { /* ignore */ }
  });

  it("creates a thread", async () => {
    const { status, data } = await fetchJson(server, "/threads", { method: "POST", body: {} });
    expect(status).toBe(201);
    expect((data as any).threadId).toBeDefined();
  });

  it("creates a thread with custom id", async () => {
    const { status, data } = await fetchJson(server, "/threads", {
      method: "POST",
      body: { threadId: "my-thread" },
    });
    expect(status).toBe(201);
    expect((data as any).threadId).toBe("my-thread");
  });

  it("gets thread state", async () => {
    const createRes = await fetchJson(server, "/threads", { method: "POST", body: {} });
    const threadId = (createRes.data as any).threadId;

    const { status, data } = await fetchJson(server, `/threads/${threadId}`);
    expect(status).toBe(200);
    expect((data as any).threadId).toBe(threadId);
  });

  it("returns 404 for unknown thread", async () => {
    const { status } = await fetchJson(server, "/threads/nonexistent");
    expect(status).toBe(404);
  });

  it("deletes a thread", async () => {
    const createRes = await fetchJson(server, "/threads", { method: "POST", body: {} });
    const threadId = (createRes.data as any).threadId;

    const { status, data } = await fetchJson(server, `/threads/${threadId}`, { method: "DELETE" });
    expect(status).toBe(200);
    expect((data as any).deleted).toBe(true);
  });

  it("starts a run asynchronously", async () => {
    const createRes = await fetchJson(server, "/threads", { method: "POST", body: {} });
    const threadId = (createRes.data as any).threadId;

    const { status, data } = await fetchJson(server, `/threads/${threadId}/runs`, {
      method: "POST",
      body: { input: { value: 5 } },
    });
    expect(status).toBe(202);
    expect((data as any).runId).toBeDefined();
    expect((data as any).status).toBe("pending");
  });

  it("gets run status", async () => {
    const createRes = await fetchJson(server, "/threads", { method: "POST", body: {} });
    const threadId = (createRes.data as any).threadId;

    const runRes = await fetchJson(server, `/threads/${threadId}/runs`, {
      method: "POST",
      body: { input: { value: 5 } },
    });
    const runId = (runRes.data as any).runId;

    const { status, data } = await fetchJson(server, `/threads/${threadId}/runs/${runId}`);
    expect(status).toBe(200);
    expect((data as any).runId).toBe(runId);
  });

  it("streams run events via SSE", async () => {
    const createRes = await fetchJson(server, "/threads", { method: "POST", body: {} });
    const threadId = (createRes.data as any).threadId;
    const runId = "test-run-1";

    const url = `${server.baseUrl}/threads/${threadId}/runs/${runId}/stream`;
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: { value: 3 } }),
    });

    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain('"type":"start"');
    expect(text).toContain('"type":"node_start"');
    expect(text).toContain('"type":"node_end"');
    expect(text).toContain('"type":"end"');
  });

  it("patches thread state", async () => {
    const createRes = await fetchJson(server, "/threads", { method: "POST", body: {} });
    const threadId = (createRes.data as any).threadId;

    // Stream a run to create checkpoint synchronously
    await fetch(`${server.baseUrl}/threads/${threadId}/runs/test-run/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: { value: 1 } }),
    });

    const { status, data } = await fetchJson(server, `/threads/${threadId}`, {
      method: "PATCH",
      body: { state: { value: 99 } },
    });

    expect(status).toBe(200);
    expect((data as any).state.value).toBe(99);
  });

  it("returns 404 for unknown endpoint", async () => {
    const { status } = await fetchJson(server, "/unknown");
    expect(status).toBe(404);
  });

  it("handles CORS preflight", async () => {
    const res = await fetch(`${server.baseUrl}/threads`, { method: "OPTIONS" });
    expect(res.status).toBe(204);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
  });

  it("enforces api key when configured", async () => {
    await server.stop();

    const graph = buildTestGraph().compile();
    const secureServer = new GraphRestServer({ graph, port: 18124, apiKey: "secret123" });
    await secureServer.start();

    try {
      const { status: noKey } = await fetchJson(secureServer, "/threads", { method: "POST", body: {} });
      expect(noKey).toBe(401);

      const res = await fetch(`${secureServer.baseUrl}/threads`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": "secret123" },
        body: JSON.stringify({}),
      });
      expect(res.status).toBe(201);
    } finally {
      await secureServer.stop();
    }
  });
});
