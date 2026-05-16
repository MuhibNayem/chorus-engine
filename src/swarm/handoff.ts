import { z } from "zod";
import type { ChatMessage } from "../llm/provider.js";
import { HandoffSignal, type AgentTool } from "../agent/types.js";
import type { HandoffRequest, SwarmAgent, SwarmSession } from "./types.js";

/** @deprecated Kept for backward-compat with JSON-based handoff detection. */
const HANDOFF_SENTINEL = "__swarm_handoff__";

/** @deprecated Use HandoffSignal instead. */
export function isHandoffResult(result: unknown): result is HandoffRequest {
  return (
    result !== null &&
    typeof result === "object" &&
    "__type" in result &&
    (result as Record<string, unknown>).__type === HANDOFF_SENTINEL
  );
}

export function buildAgentContext(
  session: SwarmSession,
  agent: SwarmAgent,
): ChatMessage[] {
  switch (agent.contextMode) {
    case "shared":
      return [...session.sharedMessages];

    case "isolated":
      return [...(session.agentMessages[agent.name] ?? [])];

    case "filtered": {
      const messages: ChatMessage[] = [];
      const taskDesc = session.lastHandoffDescription[agent.name];
      if (taskDesc) {
        messages.push({ role: "user", content: taskDesc });
      }
      const artifactKeys = Object.keys(session.artifacts);
      if (artifactKeys.length > 0) {
        const summary = artifactKeys
          .map((k) => `- ${k}: ${session.artifacts[k].slice(0, 200)}${session.artifacts[k].length > 200 ? "…" : ""}`)
          .join("\n");
        messages.push({
          role: "user",
          content: `Available artifacts:\n${summary}`,
        });
      }
      return messages;
    }
  }
}

export function buildSystemPrompt(session: SwarmSession, agent: SwarmAgent): string {
  const parts: string[] = [agent.systemPrompt];

  if (session.spec) {
    parts.push(`\n## Swarm Invariant Spec\n${session.spec}`);
  }

  const taskDesc = session.lastHandoffDescription[agent.name];
  if (taskDesc && agent.contextMode === "filtered") {
    parts.push(`\n## Your Current Task\n${taskDesc}`);
  }

  parts.push(
    "\n## Artifact Tools\nUse `set_artifact` to share named outputs with other agents. Use `get_artifact` to retrieve artifacts set by previous agents.",
  );

  const dests = agent.handoffDestinations;
  if (dests.length > 0) {
    parts.push(
      `\n## Handoff\nWhen your task is complete or requires specialist expertise, hand off to one of: ${dests.join(", ")}. Use the appropriate handoff tool (handoff_to_<agentname>). Only hand off once per response.`,
    );
  }

  return parts.join("\n");
}

export function createHandoffTools(
  session: SwarmSession,
  agent: SwarmAgent,
  agentsByName: Map<string, SwarmAgent>,
): AgentTool[] {
  return agent.handoffDestinations.map((destName) => {
    const destAgent = agentsByName.get(destName);
    const desc = destAgent
      ? `Hand off to ${destName}: ${destAgent.description}`
      : `Hand off control to the ${destName} agent.`;

    return {
      name: `handoff_to_${destName}`,
      description: desc,
      schema: z.object({
        taskDescription: z.string().describe("Clear description of the task for the target agent"),
        artifacts: z
          .array(z.string())
          .optional()
          .describe("List of artifact keys to share with the target agent"),
        reasoning: z
          .string()
          .optional()
          .describe("Explain why you are handing off to this agent"),
      }),
      async invoke(input) {
        const { taskDescription, artifacts = [], reasoning } =
          input as { taskDescription: string; artifacts?: string[]; reasoning?: string };
        throw new HandoffSignal(destName, taskDescription, artifacts, reasoning);
      },
    };
  });
}
