/**
 * SkillMiddleware — Integrates Adaptive Skill Runtime into the agent loop.
 *
 * Implements AgentMiddleware to:
 *   • beforeRound(): route skills for the current turn
 *   • extraSystemPrompt(): inject selected skill schemas
 *   • extraTools(): expose synthesized patterns as callable tools
 *   • afterRound(): observe trajectories, update metrics
 */

import * as os from "os";
import * as path from "path";
import type { AgentMiddleware, RoundContext, ToolResultContext } from "../agent/middleware.js";
import type { AgentTool } from "../agent/types.js";
import type { ChatMessage } from "../llm/provider.js";
import { SkillHarness, createSkillHarness } from "./harness.js";
import { formatSkillSchema } from "./budget.js";
import { TrajectorySynthesizer } from "./synthesizer.js";
import { executePatternWorkflow } from "./executor.js";
import type { PatternDef, ToolTrajectory } from "./types.js";
import { getContextWindow } from "../llm/contextWindows.js";

export class SkillMiddleware implements AgentMiddleware {
  private harness: SkillHarness;
  private synthesizer: TrajectorySynthesizer;
  private contextWindow: number;
  private toolsThisRound: Array<{ name: string; input: Record<string, unknown>; output: string }> = [];
  private roundStartTime = 0;
  private toolsByName: Map<string, AgentTool> = new Map();

  constructor(
    contextWindow: number,
    extraSkillDirs: string[] = [],
  ) {
    this.contextWindow = contextWindow;
    this.harness = createSkillHarness(extraSkillDirs);
    this.synthesizer = new TrajectorySynthesizer(this.harness.getRegistry());
  }

  // ─── beforeRound ────────────────────────────────────────────────────────────

  async beforeRound(ctx: RoundContext): Promise<void> {
    this.roundStartTime = Date.now();
    this.toolsThisRound = [];

    // Route skills for this turn
    const selection = await this.harness.routeForTurn(
      ctx.history,
      this.contextWindow,
      "", // system prompt will be rebuilt by loop
    );

    // Store for extraSystemPrompt() / extraTools()
    // (harness already stores internally)
  }

  // ─── extraSystemPrompt ──────────────────────────────────────────────────────

  extraSystemPrompt(): string {
    const schemas = this.harness.getSchemasForTurn();
    if (schemas.length === 0) return "";

    return [
      "=== ACTIVE SKILLS ===",
      ...schemas,
      "=== END SKILLS ===",
    ].join("\n\n");
  }

  // ─── extraTools ─────────────────────────────────────────────────────────────

  extraTools(): AgentTool[] {
    try {
      const patterns = this.harness.getPatternsForTurn();
      return patterns
        .filter((p) => Array.isArray(p.parameters) && p.parameters.length > 0)
        .map((pattern) => patternToTool(pattern, this.toolsByName));
    } catch (error) {
      console.error(`[SkillMiddleware] extraTools error: ${error instanceof Error ? error.message : String(error)}`);
      return [];
    }
  }

  setTools(toolsByName: Map<string, AgentTool>): void {
    this.toolsByName = toolsByName;
  }

  // ─── afterTool ──────────────────────────────────────────────────────────────

  async afterTool(ctx: ToolResultContext): Promise<string | undefined> {
    // Record tool usage for trajectory observation
    this.toolsThisRound.push({
      name: ctx.name,
      input: {}, // args not directly available here; we'll infer from history
      output: ctx.result,
    });
    return undefined;
  }

  // ─── afterRound ─────────────────────────────────────────────────────────────

  async afterRound(ctx: RoundContext): Promise<void> {
    const duration = Date.now() - this.roundStartTime;

    // Extract trajectory from history for pattern synthesis
    const trajectory = TrajectorySynthesizer.extractTrajectory(ctx.history, {
      success: true, // assume success if round completed without error
      duration,
    });

    if (trajectory.tools.length > 0) {
      await this.synthesizer.observe(trajectory);
    }

    this.harness.updateMetrics();
  }

  // ─── Public API ─────────────────────────────────────────────────────────────

  getHarness(): SkillHarness {
    return this.harness;
  }

  dispose(): void {
    this.harness.dispose();
  }
}

/** Convert a PatternDef into an AgentTool the model can invoke. */
function patternToTool(pattern: PatternDef, toolsByName: Map<string, AgentTool>): AgentTool {
  return {
    name: pattern.name,
    description: `${pattern.description} (synthesized pattern, ~${pattern.estimatedTokens} tok)`,
    schema: buildPatternSchema(pattern),
    async invoke(input: unknown) {
      const params = (input as Record<string, unknown>) ?? {};
      const result = await executePatternWorkflow(pattern, params, toolsByName);
      return result.success ? result.output : `Error: ${result.output}`;
    },
  };
}

/** Build a Zod-like schema descriptor for a pattern's parameters. */
function buildPatternSchema(pattern: PatternDef): Record<string, unknown> {
  const properties: Record<string, unknown> = {};
  const required: string[] = [];
  const params = Array.isArray(pattern.parameters) ? pattern.parameters : [];

  for (const param of params) {
    properties[param.name] = {
      type: param.type,
      description: param.description,
    };
    if (param.default === undefined) {
      required.push(param.name);
    }
  }

  return {
    type: "object",
    properties,
    ...(required.length > 0 ? { required } : {}),
    additionalProperties: false,
  };
}
