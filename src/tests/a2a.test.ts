import { describe, it, expect, vi } from "vitest";
import { A2AClient, A2AInputRequiredError, createA2ATool } from "../a2a/client.js";
import type { Task } from "../a2a/types.js";

function mockClient(overrides?: {
  sendTask?: (params: unknown) => Promise<Task>;
  getTask?: (taskId: string) => Promise<Task>;
  continueTask?: (taskId: string, message: string) => Promise<Task>;
}) {
  const client = new A2AClient({ baseUrl: "http://localhost:9999" });

  vi.spyOn(client, "sendTask").mockImplementation(
    overrides?.sendTask ??
      (async (params: unknown) => {
        const p = params as { id?: string; message: { role: "user"; content: Array<{ type: "text"; text?: string }> } };
        return {
          id: p.id ?? "task-1",
          agentId: "agent-1",
          state: "submitted" as const,
          messages: [p.message],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
      }),
  );

  vi.spyOn(client, "getTask").mockImplementation(
    overrides?.getTask ??
      (async () => ({
        id: "task-1",
        agentId: "agent-1",
        state: "completed" as const,
        messages: [
          { role: "user" as const, content: [{ type: "text" as const, text: "hello" }] },
          { role: "agent" as const, content: [{ type: "text" as const, text: "done" }] },
        ],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      })),
  );

  vi.spyOn(client, "continueTask").mockImplementation(
    overrides?.continueTask ??
      (async (taskId: string, messageText: string) => ({
        id: taskId,
        agentId: "agent-1",
        state: "working" as const,
        messages: [
          { role: "user" as const, content: [{ type: "text" as const, text: "hello" }] },
          { role: "user" as const, content: [{ type: "text" as const, text: messageText }] },
        ],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      })),
  );

  return client;
}

describe("A2AClient — waitForTask", () => {
  it("throws A2AInputRequiredError when task enters input-required", async () => {
    const client = mockClient({
      getTask: vi.fn().mockResolvedValue({
        id: "task-1",
        agentId: "agent-1",
        state: "input-required" as const,
        messages: [
          { role: "user" as const, content: [{ type: "text", text: "hello" }] },
          { role: "agent" as const, content: [{ type: "text", text: "What is your API key?" }] },
        ],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }),
    });

    await expect(client.waitForTask("task-1")).rejects.toThrow(A2AInputRequiredError);
    await expect(client.waitForTask("task-1")).rejects.toThrow("What is your API key?");
  });

  it("returns task when terminal state is reached", async () => {
    const client = mockClient();
    const task = await client.waitForTask("task-1");
    expect(task.state).toBe("completed");
  });

  it("throws on overall timeout", async () => {
    const client = mockClient({
      getTask: vi.fn().mockResolvedValue({
        id: "task-1",
        agentId: "agent-1",
        state: "working" as const,
        messages: [],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }),
    });

    await expect(client.waitForTask("task-1", { timeoutMs: 50 })).rejects.toThrow("timed out");
  });

  it("calls onStateChange for intermediate states", async () => {
    const states: string[] = [];
    const client = mockClient({
      getTask: vi
        .fn()
        .mockResolvedValueOnce({
          id: "task-1",
          agentId: "agent-1",
          state: "working" as const,
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        })
        .mockResolvedValueOnce({
          id: "task-1",
          agentId: "agent-1",
          state: "completed" as const,
          messages: [],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        }),
    });

    await client.waitForTask("task-1", { onStateChange: (t) => states.push(t.state) });
    expect(states).toEqual(["working", "completed"]);
  });
});

describe("A2AClient — continueTask", () => {
  it("sends follow-up message to existing task", async () => {
    const client = mockClient();
    const task = await client.continueTask("task-1", "My API key is xyz");
    expect(task.id).toBe("task-1");
    expect(task.messages).toHaveLength(2);
  });
});

describe("createA2ATool — input-required handling", () => {
  it("returns input-required hint instead of throwing", async () => {
    const client = mockClient({
      getTask: vi.fn().mockResolvedValue({
        id: "task-1",
        agentId: "agent-1",
        state: "input-required" as const,
        messages: [
          { role: "user" as const, content: [{ type: "text", text: "hello" }] },
          { role: "agent" as const, content: [{ type: "text", text: "Need more info" }] },
        ],
        createdAt: Date.now(),
        updatedAt: Date.now(),
      }),
    });

    const tool = createA2ATool("test_agent", client);
    const result = await tool.invoke({ task: "do something" });
    const parsed = JSON.parse(result as string);
    expect(parsed.state).toBe("input-required");
    expect(parsed.taskId).toBe("task-1");
    expect(parsed.message).toContain("Need more info");
    expect(parsed.hint).toContain("Call this tool again with the same taskId");
  });

  it("follows up with existing taskId", async () => {
    const client = mockClient();
    const tool = createA2ATool("test_agent", client);
    const result = await tool.invoke({ task: "my follow-up", taskId: "task-1" });
    const parsed = JSON.parse(result as string);
    expect(parsed.taskId).toBe("task-1");
    expect(client.continueTask).toHaveBeenCalledWith("task-1", "my follow-up");
  });
});
