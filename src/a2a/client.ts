/**
 * A2A Client — calls external A2A-compatible agents from within a Chorus swarm.
 * Implements JSON-RPC 2.0 over HTTP with optional SSE streaming.
 */

import { z } from "zod";
import type {
  AgentCard,
  Task,
  TaskSendParams,
  TaskStreamEvent,
  JsonRpcRequest,
  JsonRpcResponse,
} from "./types.js";
import type { AgentTool } from "../agent/types.js";

export interface A2AClientConfig {
  /** Base URL of the remote A2A agent */
  baseUrl: string;
  /** Bearer token or API key for authentication */
  apiKey?: string;
  timeoutMs?: number;
}

/**
 * Thrown when a remote A2A agent transitions to `input-required` state.
 * Carries the task so the caller can send a follow-up message with the
 * required input and resume waiting via `continueTask()`.
 */
export class A2AInputRequiredError extends Error {
  readonly task: Task;
  constructor(task: Task) {
    const lastAgent = task.messages
      .filter((m) => m.role === "agent")
      .at(-1);
    const hint = lastAgent?.content.find((c) => c.type === "text")?.text ?? "No details provided.";
    super(`A2A task ${task.id} requires input: ${hint}`);
    this.task = task;
    this.name = "A2AInputRequiredError";
  }
}

export class A2AClient {
  private baseUrl: string;
  private headers: Record<string, string>;
  private timeoutMs: number;

  constructor(config: A2AClientConfig) {
    this.baseUrl = config.baseUrl.replace(/\/$/, "");
    this.timeoutMs = config.timeoutMs ?? 30_000;
    this.headers = {
      "Content-Type": "application/json",
      ...(config.apiKey ? { Authorization: `Bearer ${config.apiKey}` } : {}),
    };
  }

  async getAgentCard(): Promise<AgentCard> {
    const res = await this.fetch(`${this.baseUrl}/.well-known/agent.json`);
    return (await res.json()) as AgentCard;
  }

  async sendTask(params: TaskSendParams): Promise<Task> {
    const body: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: Date.now(),
      method: "tasks/send",
      params,
    };
    const res = await this.fetch(`${this.baseUrl}/tasks`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    const json = (await res.json()) as JsonRpcResponse;
    if (json.error) throw new Error(`A2A error ${json.error.code}: ${json.error.message}`);
    return json.result as Task;
  }

  /** Send a follow-up message to an existing task (e.g., after input-required). */
  async continueTask(taskId: string, messageText: string): Promise<Task> {
    return this.sendTask({
      id: taskId,
      message: { role: "user", content: [{ type: "text", text: messageText }] },
    });
  }

  async getTask(taskId: string): Promise<Task> {
    const body: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: Date.now(),
      method: "tasks/get",
      params: { id: taskId },
    };
    const res = await this.fetch(`${this.baseUrl}/tasks`, {
      method: "POST",
      body: JSON.stringify(body),
    });
    const json = (await res.json()) as JsonRpcResponse;
    if (json.error) throw new Error(`A2A error ${json.error.code}: ${json.error.message}`);
    return json.result as Task;
  }

  async cancelTask(taskId: string): Promise<void> {
    const body: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: Date.now(),
      method: "tasks/cancel",
      params: { id: taskId },
    };
    await this.fetch(`${this.baseUrl}/tasks`, {
      method: "POST",
      body: JSON.stringify(body),
    });
  }

  async *streamTask(params: TaskSendParams): AsyncGenerator<TaskStreamEvent> {
    const body: JsonRpcRequest = {
      jsonrpc: "2.0",
      id: Date.now(),
      method: "tasks/sendSubscribe",
      params,
    };

    const res = await this.fetch(`${this.baseUrl}/tasks/stream`, {
      method: "POST",
      body: JSON.stringify(body),
      headers: { ...this.headers, Accept: "text/event-stream" },
    });

    if (!res.body) throw new Error("No response body for streaming task");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6).trim();
            if (data === "[DONE]") return;
            try {
              yield JSON.parse(data) as TaskStreamEvent;
            } catch {
              // malformed SSE line — skip
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * Poll a task until it reaches a terminal state.
   *
   * Improvements over naive polling:
   *   • `input-required` is detected and throws {@link A2AInputRequiredError}
   *     so the caller can send follow-up input via {@link continueTask}.
   *   • `timeoutMs` caps total wait time (default: 5 min).
   *   • `onStateChange` callback receives every intermediate state.
   */
  async waitForTask(
    taskId: string,
    opts: { pollIntervalMs?: number; timeoutMs?: number; onStateChange?: (task: Task) => void } = {},
  ): Promise<Task> {
    const pollIntervalMs = opts.pollIntervalMs ?? 500;
    const timeoutMs = opts.timeoutMs ?? 300_000;
    const terminalStates = new Set(["completed", "failed", "canceled"]);
    const startedAt = Date.now();

    while (true) {
      if (Date.now() - startedAt > timeoutMs) {
        throw new Error(`A2A task ${taskId} timed out after ${timeoutMs}ms`);
      }

      const task = await this.getTask(taskId);
      opts.onStateChange?.(task);

      if (task.state === "input-required") {
        throw new A2AInputRequiredError(task);
      }

      if (terminalStates.has(task.state)) return task;
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    }
  }

  private async fetch(url: string, init?: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const res = await fetch(url, {
        ...init,
        headers: { ...this.headers, ...(init?.headers as Record<string, string> | undefined) },
        signal: controller.signal,
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      return res;
    } finally {
      clearTimeout(timer);
    }
  }
}

/**
 * Create an AgentTool that delegates to an external A2A agent.
 * Drop this into any agent's tools array to give it access to external A2A agents.
 *
 * Handles `input-required` gracefully: if the remote agent asks for more
 * information, the tool returns a structured JSON result with `state: "input-required"`
 * and the `taskId` so the agent can call the tool again with `taskId` and the
 * follow-up input.
 */
export function createA2ATool(name: string, client: A2AClient, description?: string): AgentTool {
  const schema = z.object({
    task: z.string().describe("The task to send to the external agent"),
    taskId: z.string().optional().describe("Existing task ID for follow-up messages (e.g., after input-required)"),
    await_completion: z.boolean().optional().describe("If true, wait for the task to complete and return the result. Default: true"),
  });

  return {
    name,
    description: description ?? `Delegate a task to the external A2A agent`,
    schema,
    async invoke(input: unknown) {
      const params = schema.parse(input);

      // Follow-up: send message to existing task
      const sent = params.taskId
        ? await client.continueTask(params.taskId, params.task)
        : await client.sendTask({
            message: { role: "user", content: [{ type: "text", text: params.task }] },
          });

      if (params.await_completion === false) {
        return JSON.stringify({ taskId: sent.id, state: sent.state });
      }

      try {
        const completed = await client.waitForTask(sent.id);
        const last = completed.messages.at(-1);
        const text = last?.content.find((c) => c.type === "text")?.text ?? "";
        return JSON.stringify({ taskId: completed.id, state: completed.state, result: text });
      } catch (error) {
        if (error instanceof A2AInputRequiredError) {
          return JSON.stringify({
            taskId: error.task.id,
            state: "input-required",
            message: error.message,
            hint: "Call this tool again with the same taskId and your follow-up input in the 'task' field.",
          });
        }
        throw error;
      }
    },
  };
}
