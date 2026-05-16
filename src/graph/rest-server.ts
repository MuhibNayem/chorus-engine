/**
 * GraphRestServer — LangGraph Platform-compatible REST API for StateGraph.
 *
 * Exposes threads, runs, checkpoints, and streaming over HTTP.
 * All endpoints speak JSON. SSE is used for streaming run events.
 *
 * Endpoints:
 *   POST   /threads                    → Create thread
 *   GET    /threads/:id                → Get thread state
 *   PATCH  /threads/:id                → Update thread state
 *   DELETE /threads/:id                → Delete thread
 *   POST   /threads/:id/runs           → Start a run
 *   GET    /threads/:id/runs/:runId    → Get run status
 *   POST   /threads/:id/runs/:runId/stream → Stream run events (SSE)
 *   POST   /threads/:id/runs/:runId/resume → Resume interrupted run
 *   GET    /threads/:id/checkpoints    → List checkpoints
 */

import * as http from "http";
import { randomUUID } from "crypto";
import type { CompiledGraph } from "./state-graph.js";
import type { GraphEvent, RunConfig } from "./types.js";

export interface GraphRestServerConfig<State extends Record<string, unknown>> {
  /** The compiled graph to execute. */
  graph: CompiledGraph<State>;
  /** HTTP port. Default: 8123. */
  port?: number;
  /** HTTP host. Default: "127.0.0.1". */
  host?: string;
  /** CORS origin. Default: "*". */
  corsOrigin?: string;
  /** API key for simple auth (checked via X-API-Key header). */
  apiKey?: string;
  /** Maximum request body size in bytes. Default: 10 MB. */
  maxBodySize?: number;
}

interface ThreadRecord {
  threadId: string;
  createdAt: number;
  metadata?: Record<string, unknown>;
}

interface RunRecord {
  runId: string;
  threadId: string;
  status: "pending" | "running" | "completed" | "failed" | "interrupted";
  startedAt: number;
  finishedAt?: number;
  error?: string;
  events: GraphEvent[];
}

export class GraphRestServer<State extends Record<string, unknown>> {
  private server: http.Server;
  private config: Required<GraphRestServerConfig<State>>;
  private threads = new Map<string, ThreadRecord>();
  private runs = new Map<string, RunRecord>();

  constructor(config: GraphRestServerConfig<State>) {
    this.config = {
      graph: config.graph,
      port: config.port ?? 8123,
      host: config.host ?? "127.0.0.1",
      corsOrigin: config.corsOrigin ?? "*",
      apiKey: config.apiKey ?? "",
      maxBodySize: config.maxBodySize ?? 10_485_760,
    };
    this.server = http.createServer((req, res) => {
      void this.handleRequest(req, res);
    });
  }

  start(): Promise<void> {
    return new Promise((resolve) => {
      this.server.listen(this.config.port, this.config.host, () => resolve());
    });
  }

  stop(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server.close((err) => (err ? reject(err) : resolve()));
    });
  }

  get baseUrl(): string {
    return `http://${this.config.host}:${this.config.port}`;
  }

  // ── Request Router ──────────────────────────────────────────────────────────

  private async handleRequest(req: http.IncomingMessage, res: http.ServerResponse): Promise<void> {
    const cors = this.corsHeaders();

    if (req.method === "OPTIONS") {
      res.writeHead(204, cors);
      res.end();
      return;
    }

    if (this.config.apiKey && req.headers["x-api-key"] !== this.config.apiKey) {
      this.sendJson(res, 401, { error: "Unauthorized" }, cors);
      return;
    }

    const url = new URL(req.url ?? "/", `http://${req.headers.host}`);
    const segments = url.pathname.split("/").filter(Boolean);

    try {
      // /threads
      if (segments[0] === "threads" && segments.length === 1 && req.method === "POST") {
        await this.createThread(req, res, cors);
        return;
      }

      if (segments[0] === "threads" && segments.length === 2) {
        const threadId = segments[1];
        if (req.method === "GET") {
          await this.getThread(threadId, res, cors);
          return;
        }
        if (req.method === "PATCH") {
          await this.patchThread(threadId, req, res, cors);
          return;
        }
        if (req.method === "DELETE") {
          await this.deleteThread(threadId, res, cors);
          return;
        }
      }

      // /threads/:id/runs
      if (segments[0] === "threads" && segments[2] === "runs" && segments.length === 3 && req.method === "POST") {
        await this.createRun(segments[1], req, res, cors);
        return;
      }

      // /threads/:id/runs/:runId
      if (segments[0] === "threads" && segments[2] === "runs" && segments.length === 4) {
        const threadId = segments[1];
        const runId = segments[3];
        if (req.method === "GET") {
          await this.getRun(threadId, runId, res, cors);
          return;
        }
      }

      // /threads/:id/runs/:runId/stream
      if (segments[0] === "threads" && segments[2] === "runs" && segments[4] === "stream" && segments.length === 5 && req.method === "POST") {
        await this.streamRun(segments[1], segments[3], req, res, cors);
        return;
      }

      // /threads/:id/runs/:runId/resume
      if (segments[0] === "threads" && segments[2] === "runs" && segments[4] === "resume" && segments.length === 5 && req.method === "POST") {
        await this.resumeRun(segments[1], segments[3], req, res, cors);
        return;
      }

      // /threads/:id/checkpoints
      if (segments[0] === "threads" && segments[2] === "checkpoints" && segments.length === 3 && req.method === "GET") {
        await this.listCheckpoints(segments[1], res, cors);
        return;
      }

      this.sendJson(res, 404, { error: "Not found" }, cors);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.sendJson(res, 500, { error: msg }, cors);
    }
  }

  // ── Thread Endpoints ────────────────────────────────────────────────────────

  private async createThread(_req: http.IncomingMessage, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const body = await readBody(_req, this.config.maxBodySize);
    const payload = body ? (JSON.parse(body) as Record<string, unknown>) : {};
    const threadId = (payload.threadId as string) ?? randomUUID();

    const thread: ThreadRecord = {
      threadId,
      createdAt: Date.now(),
      metadata: (payload.metadata as Record<string, unknown>) ?? {},
    };

    this.threads.set(threadId, thread);
    this.sendJson(res, 201, { threadId, createdAt: thread.createdAt, metadata: thread.metadata }, cors);
  }

  private async getThread(threadId: string, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const thread = this.threads.get(threadId);
    if (!thread) {
      this.sendJson(res, 404, { error: `Thread "${threadId}" not found.` }, cors);
      return;
    }

    const state = await this.config.graph.getState(threadId);
    this.sendJson(res, 200, { threadId, createdAt: thread.createdAt, state, metadata: thread.metadata }, cors);
  }

  private async patchThread(threadId: string, req: http.IncomingMessage, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const thread = this.threads.get(threadId);
    if (!thread) {
      this.sendJson(res, 404, { error: `Thread "${threadId}" not found.` }, cors);
      return;
    }

    const body = await readBody(req, this.config.maxBodySize);
    const payload = body ? (JSON.parse(body) as Record<string, unknown>) : {};

    if (payload.state) {
      await this.config.graph.updateState(threadId, payload.state as Partial<State>);
    }
    if (payload.metadata) {
      thread.metadata = { ...thread.metadata, ...(payload.metadata as Record<string, unknown>) };
    }

    const state = await this.config.graph.getState(threadId);
    this.sendJson(res, 200, { threadId, state, metadata: thread.metadata }, cors);
  }

  private async deleteThread(threadId: string, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    this.threads.delete(threadId);
    this.sendJson(res, 200, { threadId, deleted: true }, cors);
  }

  // ── Run Endpoints ───────────────────────────────────────────────────────────

  private async createRun(threadId: string, req: http.IncomingMessage, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const body = await readBody(req, this.config.maxBodySize);
    const payload = body ? (JSON.parse(body) as Record<string, unknown>) : {};
    const runId = randomUUID();

    const run: RunRecord = {
      runId,
      threadId,
      status: "pending",
      startedAt: Date.now(),
      events: [],
    };
    this.runs.set(runId, run);

    // Execute asynchronously
    void this.executeRun(run, payload.input as Record<string, unknown> | undefined, payload.config as RunConfig);

    this.sendJson(res, 202, { runId, threadId, status: "pending" }, cors);
  }

  private async getRun(_threadId: string, runId: string, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const run = this.runs.get(runId);
    if (!run) {
      this.sendJson(res, 404, { error: `Run "${runId}" not found.` }, cors);
      return;
    }
    this.sendJson(res, 200, { runId, status: run.status, startedAt: run.startedAt, finishedAt: run.finishedAt, error: run.error }, cors);
  }

  private async streamRun(threadId: string, runId: string, req: http.IncomingMessage, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const body = await readBody(req, this.config.maxBodySize);
    const payload = body ? (JSON.parse(body) as Record<string, unknown>) : {};

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      ...cors,
    });

    const input = (payload.input ?? {}) as Record<string, unknown>;
    const config: RunConfig = { ...(payload.config as RunConfig), threadId };

    try {
      for await (const event of this.config.graph.stream(input as Partial<State>, config)) {
        res.write(`data: ${JSON.stringify(event)}\n\n`);
      }
      res.write(`data: ${JSON.stringify({ type: "stream-end" })}\n\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      res.write(`data: ${JSON.stringify({ type: "error", error: msg })}\n\n`);
    }
    res.end();
  }

  private async resumeRun(threadId: string, runId: string, req: http.IncomingMessage, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    const body = await readBody(req, this.config.maxBodySize);
    const payload = body ? (JSON.parse(body) as Record<string, unknown>) : {};

    const command = {
      update: (payload.update ?? {}) as Record<string, unknown>,
      resumeNode: payload.resumeNode as string | undefined,
      metadata: payload.metadata as Record<string, unknown> | undefined,
    };

    res.writeHead(200, {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      ...cors,
    });

    try {
      for await (const event of this.config.graph.resume(threadId, command as any)) {
        res.write(`data: ${JSON.stringify(event)}\n\n`);
      }
      res.write(`data: ${JSON.stringify({ type: "stream-end" })}\n\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      res.write(`data: ${JSON.stringify({ type: "error", error: msg })}\n\n`);
    }
    res.end();
  }

  private async listCheckpoints(threadId: string, res: http.ServerResponse, cors: Record<string, string>): Promise<void> {
    // Graph checkpointer must be available
    const cp = (this.config.graph as any).options?.checkpointer;
    if (!cp || typeof cp.listGraphCheckpoints !== "function") {
      this.sendJson(res, 501, { error: "Checkpointer does not support listing checkpoints." }, cors);
      return;
    }

    const list = await cp.listGraphCheckpoints(threadId);
    this.sendJson(res, 200, { threadId, checkpoints: list }, cors);
  }

  // ── Execution ───────────────────────────────────────────────────────────────

  private async executeRun(run: RunRecord, input?: Record<string, unknown>, config?: RunConfig): Promise<void> {
    run.status = "running";
    try {
      const finalState = await this.config.graph.invoke(input as Partial<State>, { ...config, threadId: run.threadId });
      run.status = "completed";
      run.finishedAt = Date.now();
      run.events.push({ type: "end", threadId: run.threadId, state: finalState as Record<string, unknown> });
    } catch (err) {
      run.status = "failed";
      run.finishedAt = Date.now();
      run.error = err instanceof Error ? err.message : String(err);
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  private corsHeaders(): Record<string, string> {
    return {
      "Access-Control-Allow-Origin": this.config.corsOrigin,
      "Access-Control-Allow-Methods": "GET, POST, PATCH, DELETE, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-Key",
    };
  }

  private sendJson(res: http.ServerResponse, status: number, data: unknown, cors: Record<string, string>): void {
    res.writeHead(status, { "Content-Type": "application/json", ...cors });
    res.end(JSON.stringify(data));
  }
}

function readBody(req: http.IncomingMessage, maxBodySize = 10_485_760): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalSize = 0;
    req.on("data", (c: Buffer) => {
      totalSize += c.length;
      if (totalSize > maxBodySize) {
        req.destroy();
        reject(new Error(`Request body exceeds ${maxBodySize} bytes`));
        return;
      }
      chunks.push(c);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf-8")));
    req.on("error", reject);
  });
}
