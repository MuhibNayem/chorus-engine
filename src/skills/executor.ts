/**
 * Skill Executor — Executes skill workflows and swarm-orchestrated skills.
 *
 * Two execution modes:
 *   1. Single-mode: Run workflow steps sequentially, substituting parameters.
 *   2. Swarm-mode: Spawn a swarm with skill-declared agents, merge results.
 */

import type { SkillDef, PatternDef, SkillExecutionResult, SkillWorkflowStep } from "./types.js";
import type { AgentTool } from "../agent/types.js";
import type { LLMProvider } from "../llm/provider.js";
import { runSwarm } from "../swarm/orchestrator.js";
import type { SwarmConfig, SwarmAgent } from "../swarm/types.js";

/** Substitute {{parameter}} placeholders in workflow inputs. */
function substituteParams(
  input: Record<string, unknown>,
  params: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(input)) {
    if (typeof value === "string" && value.startsWith("{{") && value.endsWith("}}")) {
      const paramName = value.slice(2, -2);
      result[key] = params[paramName] ?? value;
    } else {
      result[key] = value;
    }
  }

  return result;
}

/** Execute a single workflow step against the tool registry. */
async function executeStep(
  step: SkillWorkflowStep,
  toolsByName: Map<string, AgentTool>,
  stepResults: Record<string, unknown>,
): Promise<unknown> {
  const tool = toolsByName.get(step.tool);
  if (!tool) {
    throw new Error(`Unknown tool in workflow: ${step.tool}`);
  }

  // Substitute step results from previous steps
  const input = substituteStepResults(step.input, stepResults);

  return tool.invoke(input);
}

/** Substitute {{step.field}} references to previous step outputs. */
function substituteStepResults(
  input: Record<string, unknown>,
  stepResults: Record<string, unknown>,
): Record<string, unknown> {
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(input)) {
    if (typeof value === "string" && value.startsWith("{{") && value.endsWith("}}")) {
      const path = value.slice(2, -2);
      result[key] = resolvePath(stepResults, path);
    } else {
      result[key] = value;
    }
  }

  return result;
}

/** Resolve a dotted path like "result.matches.0.path" from an object. */
function resolvePath(obj: Record<string, unknown>, path: string): unknown {
  const parts = path.split(".");
  let current: unknown = obj;

  for (const part of parts) {
    if (current === null || current === undefined) return undefined;

    if (Array.isArray(current)) {
      const idx = parseInt(part, 10);
      current = current[idx];
    } else if (typeof current === "object") {
      current = (current as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }

  return current;
}

/** Execute a pattern workflow sequentially. */
export async function executePatternWorkflow(
  pattern: PatternDef,
  params: Record<string, unknown>,
  toolsByName: Map<string, AgentTool>,
): Promise<SkillExecutionResult> {
  const start = Date.now();
  const stepResults: Record<string, unknown> = {};
  const outputs: string[] = [];
  let tokensUsed = 0;

  try {
    for (const step of pattern.workflow) {
      // Substitute user-provided params
      const stepInput = substituteParams(step.input, params);

      // Execute
      const result = await executeStep({ ...step, input: stepInput }, toolsByName, stepResults);
      const resultStr = typeof result === "string" ? result : JSON.stringify(result);

      // Store result for downstream steps
      stepResults[step.tool] = result;
      outputs.push(`[${step.tool}]: ${resultStr.slice(0, 500)}`);

      // Rough token estimation
      tokensUsed += resultStr.length / 4;
    }

    return {
      success: true,
      output: outputs.join("\n\n"),
      tokensUsed: Math.round(tokensUsed),
      durationMs: Date.now() - start,
    };
  } catch (error) {
    return {
      success: false,
      output: `Workflow failed at step: ${error instanceof Error ? error.message : String(error)}`,
      tokensUsed: Math.round(tokensUsed),
      durationMs: Date.now() - start,
    };
  }
}

/** Build a SwarmConfig from a skill's swarm declaration. */
function buildSwarmConfig(skill: SkillDef, params: Record<string, unknown>, provider: LLMProvider, modelName: string): SwarmConfig {
  const swarm = skill.swarm!;
  const agents: SwarmAgent[] = (swarm.agents ?? []).map((a) => ({
    name: a.role,
    description: a.description,
    systemPrompt: a.description,
    tools: [],
    handoffDestinations: [],
    contextMode: "filtered",
    maxRounds: 30,
    model: a.model,
  }));

  // Sequential handoff: each agent hands off to the next
  if (swarm.handoff?.strategy === "sequential" && agents.length > 1) {
    for (let i = 0; i < agents.length - 1; i++) {
      agents[i].handoffDestinations.push(agents[i + 1].name);
    }
  }

  return {
    agents,
    initialAgent: agents[0]?.name ?? "",
    task: (params.task as string) || skill.description || `Execute skill: ${skill.name}`,
    provider,
    modelName,
    executionModel: swarm.handoff?.strategy === "parallel" ? "graph" : "handoff",
    policy: "full_auto",
  };
}

/** Merge swarm events into a single execution result. */
function mergeSwarmResults(events: AsyncGenerator<import("../swarm/types.js").SwarmEvent>, mergeStrategy: string): Promise<{ output: string; tokensUsed: number; swarmResults: Array<{ agent: string; output: string }> }> {
  return new Promise(async (resolve, reject) => {
    const agentOutputs: Record<string, string> = {};
    let tokensUsed = 0;

    try {
      for await (const event of events) {
        if (event.type === "agent-done") {
          agentOutputs[event.agent] = event.responseText;
        }
        if (event.type === "done" && "inputTokens" in event) {
          tokensUsed += (event.inputTokens ?? 0) + (event.outputTokens ?? 0);
        }
      }

      const results = Object.entries(agentOutputs).map(([agent, output]) => ({ agent, output }));
      let output: string;

      switch (mergeStrategy) {
        case "vote":
          output = `Swarm results (${results.length} agents):\n\n${results.map((r) => `--- ${r.agent} ---\n${r.output}`).join("\n\n")}`;
          break;
        case "first_success":
          output = results.find((r) => r.output.length > 0)?.output ?? "No successful agent output.";
          break;
        case "concatenate_results":
        default:
          output = `Swarm results (${results.length} agents):\n\n${results.map((r) => `--- ${r.agent} ---\n${r.output}`).join("\n\n")}`;
          break;
      }

      resolve({ output, tokensUsed, swarmResults: results });
    } catch (error) {
      reject(error);
    }
  });
}

/** Execute a skill with swarm orchestration. */
export async function executeSkillWithSwarm(
  skill: SkillDef,
  params: Record<string, unknown>,
  provider: LLMProvider,
  modelName: string,
): Promise<SkillExecutionResult> {
  const start = Date.now();

  if (!skill.swarm?.enabled) {
    return {
      success: false,
      output: "Skill does not have swarm enabled",
      tokensUsed: 0,
      durationMs: Date.now() - start,
    };
  }

  const config = buildSwarmConfig(skill, params, provider, modelName);
  const events = runSwarm(config);
  const mergeStrategy = skill.swarm.handoff?.merge ?? "concatenate_results";
  const { output, tokensUsed, swarmResults } = await mergeSwarmResults(events, mergeStrategy);

  return {
    success: true,
    output,
    tokensUsed,
    durationMs: Date.now() - start,
    swarmResults,
  };
}

/** Main skill execution dispatcher. */
export async function executeSkill(
  skill: SkillDef | PatternDef,
  params: Record<string, unknown>,
  toolsByName: Map<string, AgentTool>,
  provider?: LLMProvider,
  modelName?: string,
): Promise<SkillExecutionResult> {
  // Check if it's a skill with swarm mode
  if ("swarm" in skill && skill.swarm?.enabled) {
    if (!provider || !modelName) {
      return {
        success: false,
        output: "Swarm execution requires provider and modelName parameters",
        tokensUsed: 0,
        durationMs: 0,
      };
    }
    return executeSkillWithSwarm(skill as SkillDef, params, provider, modelName);
  }

  // Check if it's a pattern with a workflow
  if ("workflow" in skill && skill.workflow?.length) {
    return executePatternWorkflow(skill as PatternDef, params, toolsByName);
  }

  // Fallback: skill with instructions but no workflow
  return {
    success: true,
    output: `Skill "${skill.name}" invoked. No executable workflow defined — follow instructions: ${"instructions" in skill ? skill.instructions.slice(0, 200) : ""}`,
    tokensUsed: 0,
    durationMs: 0,
  };
}
