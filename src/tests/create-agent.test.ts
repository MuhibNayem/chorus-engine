import { describe, it, expect, vi } from "vitest";
import { createAgent } from "../graph/create-agent.js";
import type { LLMProvider, ChatMessage, ToolDef, ModelResponse, ToolCall } from "../llm/provider.js";
import type { AgentTool } from "../agent/types.js";

// ── Mock LLM Provider ─────────────────────────────────────────────────────────

function mockProvider(responseText: string, toolCalls?: ToolCall[]): LLMProvider {
  return {
    name: "mock" as const,

    async generate() {
      return { text: responseText, model: "mock" };
    },

    async *stream() {
      yield { type: "response.delta" as const, text: responseText };
      yield { type: "response.completed" as const };
    },

    async *streamWithTools() {
      yield { type: "token" as const, text: responseText };
      yield {
        type: "done" as const,
        response: {
          content: responseText,
          usage: { inputTokens: 10, outputTokens: 5 },
          ...(toolCalls ? { tool_calls: toolCalls } : {}),
        },
      };
    },

    async health() {
      return { ok: true as const, provider: "mock" as const, detail: "healthy" };
    },
  };
}

// ── Mock Tools ───────────────────────────────────────────────────────────────

const mockCalculator: AgentTool = {
  name: "calculator",
  description: "Calculate math expressions",
  schema: {
    type: "object",
    properties: { expression: { type: "string" } },
    required: ["expression"],
  },
  async invoke(input: unknown) {
    const { expression } = input as { expression: string };
    // Safety: only allow simple arithmetic
    if (!/^[\d+\-*/.()\s]+$/.test(expression)) {
      return "Error: Invalid expression";
    }
    try {
      // eslint-disable-next-line no-eval
      return String(eval(expression));
    } catch {
      return "Error: Could not evaluate";
    }
  },
};

const mockEcho: AgentTool = {
  name: "echo",
  description: "Echo input back",
  async invoke(input: unknown) {
    return JSON.stringify(input);
  },
};

// ── Tests ────────────────────────────────────────────────────────────────────

describe("createAgent", () => {
  it("constructs a compiled agent graph", () => {
    const agent = createAgent({
      provider: mockProvider("hello"),
      model: "mock",
    });
    expect(agent).toBeDefined();
    expect(agent.invoke).toBeInstanceOf(Function);
    expect(agent.stream).toBeInstanceOf(Function);
  });

  it("returns response when no tools are called", async () => {
    const agent = createAgent({
      provider: mockProvider("The answer is 42."),
      model: "mock",
      systemPrompt: "You are a calculator.",
    });

    const result = await agent.invoke({
      messages: [{ role: "user", content: "What is 2+2?" }],
    });

    expect(result.response).toBe("The answer is 42.");
    expect(result.isDone).toBe(true);
    expect(result.messages.length).toBeGreaterThanOrEqual(2); // user + assistant
  });

  it("executes tools and returns final response", async () => {
    const toolCall: ToolCall = {
      id: "tc_1",
      type: "function",
      function: { name: "calculator", arguments: JSON.stringify({ expression: "2+2" }) },
    };

    // First call: LLM issues tool call
    // Second call: LLM sees tool result and responds
    let callCount = 0;
    const provider: LLMProvider = {
      name: "mock",
      async generate() {
        return { text: "done", model: "mock" };
      },
      async *stream() {
        yield { type: "response.completed" };
      },
      async *streamWithTools() {
        callCount++;
        if (callCount === 1) {
          yield { type: "token", text: "" };
          yield {
            type: "done",
            response: {
              content: "Let me calculate that.",
              usage: { inputTokens: 10, outputTokens: 5 },
              tool_calls: [toolCall],
            },
          };
        } else {
          yield { type: "token", text: "The answer is 4." };
          yield {
            type: "done",
            response: {
              content: "The answer is 4.",
              usage: { inputTokens: 15, outputTokens: 5 },
            },
          };
        }
      },
      async health() {
        return { ok: true, provider: "mock" as const, detail: "healthy" };
      },
    };

    const agent = createAgent({
      provider,
      model: "mock",
      tools: [mockCalculator],
      maxIterations: 5,
    });

    const result = await agent.invoke({
      messages: [{ role: "user", content: "Calculate 2+2" }],
    });

    expect(callCount).toBe(2);
    expect(result.response).toBe("The answer is 4.");
    expect(result.isDone).toBe(true);
    expect(result.toolResults).toBeDefined();
    expect(result.toolResults!.length).toBe(1);
    expect(result.toolResults![0].name).toBe("calculator");
    expect(result.toolResults![0].result).toBe("4");
    expect(result.messages.length).toBeGreaterThanOrEqual(4); // user, assistant, tool, assistant
  });

  it("executes multiple tools in parallel", async () => {
    const tc1: ToolCall = {
      id: "tc_1",
      type: "function",
      function: { name: "calculator", arguments: JSON.stringify({ expression: "1+1" }) },
    };
    const tc2: ToolCall = {
      id: "tc_2",
      type: "function",
      function: { name: "echo", arguments: JSON.stringify({ text: "hello" }) },
    };

    let callCount = 0;
    const provider: LLMProvider = {
      name: "mock",
      async generate() {
        return { text: "done", model: "mock" };
      },
      async *stream() {
        yield { type: "response.completed" };
      },
      async *streamWithTools() {
        callCount++;
        if (callCount === 1) {
          yield { type: "token", text: "" };
          yield {
            type: "done",
            response: {
              content: "Calling tools.",
              usage: { inputTokens: 10, outputTokens: 5 },
              tool_calls: [tc1, tc2],
            },
          };
        } else {
          yield { type: "token", text: "Done." };
          yield {
            type: "done",
            response: {
              content: "Done.",
              usage: { inputTokens: 20, outputTokens: 5 },
            },
          };
        }
      },
      async health() {
        return { ok: true, provider: "mock" as const, detail: "healthy" };
      },
    };

    const agent = createAgent({
      provider,
      model: "mock",
      tools: [mockCalculator, mockEcho],
      parallelTools: true,
      maxIterations: 5,
    });

    const result = await agent.invoke({
      messages: [{ role: "user", content: "Do two things" }],
    });

    expect(callCount).toBe(2);
    expect(result.toolResults).toBeDefined();
    expect(result.toolResults!.length).toBe(2);
    const names = result.toolResults!.map((r) => r.name).sort();
    expect(names).toEqual(["calculator", "echo"]);
  });

  it("streams agent superstep events", async () => {
    const agent = createAgent({
      provider: mockProvider("streaming response"),
      model: "mock",
    });

    const events = [];
    for await (const event of agent.stream({
      messages: [{ role: "user", content: "hi" }],
    })) {
      events.push(event.type);
    }

    expect(events).toContain("start");
    expect(events).toContain("node_start");
    expect(events).toContain("node_end");
    expect(events).toContain("end");
  });

  it("handles tool errors gracefully", async () => {
    const tc: ToolCall = {
      id: "tc_err",
      type: "function",
      function: { name: "calculator", arguments: JSON.stringify({ expression: "invalid!!!" }) },
    };

    let callCount = 0;
    const provider: LLMProvider = {
      name: "mock",
      async generate() {
        return { text: "done", model: "mock" };
      },
      async *stream() {
        yield { type: "response.completed" };
      },
      async *streamWithTools() {
        callCount++;
        if (callCount === 1) {
          yield { type: "token", text: "" };
          yield {
            type: "done",
            response: {
              content: "Let me try.",
              usage: { inputTokens: 10, outputTokens: 5 },
              tool_calls: [tc],
            },
          };
        } else {
          yield { type: "token", text: "Failed." };
          yield {
            type: "done",
            response: {
              content: "Failed.",
              usage: { inputTokens: 15, outputTokens: 5 },
            },
          };
        }
      },
      async health() {
        return { ok: true, provider: "mock" as const, detail: "healthy" };
      },
    };

    const agent = createAgent({
      provider,
      model: "mock",
      tools: [mockCalculator],
      maxIterations: 5,
    });

    const result = await agent.invoke({
      messages: [{ role: "user", content: "Calculate invalid!!!" }],
    });

    expect(result.toolResults).toBeDefined();
    expect(result.toolResults![0].result).toContain("Error");
    expect(result.response).toBe("Failed.");
  });

  it("uses dynamic system prompt from function", async () => {
    const agent = createAgent({
      provider: mockProvider("ok"),
      model: "mock",
      systemPrompt: (state) => `You are ${state.messages.length > 1 ? "chatty" : "brief"}.`,
    });

    const result = await agent.invoke({
      messages: [{ role: "user", content: "hi" }],
    });

    expect(result.response).toBe("ok");
    expect(result.isDone).toBe(true);
  });

  it("enforces maxIterations to prevent infinite loops", async () => {
    const tc: ToolCall = {
      id: "tc_loop",
      type: "function",
      function: { name: "echo", arguments: JSON.stringify({}) },
    };

    const provider: LLMProvider = {
      name: "mock",
      async generate() {
        return { text: "loop", model: "mock" };
      },
      async *stream() {
        yield { type: "response.completed" };
      },
      async *streamWithTools() {
        yield { type: "token", text: "" };
        yield {
          type: "done",
          response: {
            content: "Looping.",
            usage: { inputTokens: 10, outputTokens: 5 },
            tool_calls: [tc],
          },
        };
      },
      async health() {
        return { ok: true, provider: "mock" as const, detail: "healthy" };
      },
    };

    const agent = createAgent({
      provider,
      model: "mock",
      tools: [mockEcho],
      maxIterations: 3,
    });

    await expect(
      agent.invoke({ messages: [{ role: "user", content: "loop" }] }),
    ).rejects.toThrow("recursion limit");
  });
});
