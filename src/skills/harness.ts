/**
 * SkillHarness — Main orchestrator for the Adaptive Skill Runtime.
 *
 * Wires together:
 *   • SkillRegistry (loading, indexing, metrics)
 *   • SkillRouter (per-turn semantic routing)
 *   • Trajectory observation (for synthesis)
 *   • Token budget management
 */

import * as os from "os";
import * as path from "path";
import type {
  SkillDef,
  PatternDef,
  SkillSelection,
  TokenBudget,
  ToolTrajectory,
  SkillHarnessOptions,
  SkillHealthReport,
} from "./types.js";
import { SkillRegistry } from "./registry.js";
import { routeSkillsForTurn, type SkillRouterOptions } from "./router.js";
import { createTokenBudget } from "./budget.js";
import { countMessagesTokens } from "../context/tokenizer.js";
import type { ChatMessage } from "../llm/provider.js";

const DEFAULT_OPTIONS: Required<SkillHarnessOptions> = {
  skillDirs: [],
  similarityThreshold: 0.55,
  maxSkillsPerTurn: 6,
  maxPatternsPerTurn: 3,
  minTrajectoriesForSynthesis: 3,
  enableSynthesis: true,
  enableCuration: true,
  maxPatterns: 100,
};

export class SkillHarness {
  private registry: SkillRegistry;
  private options: Required<SkillHarnessOptions>;
  private currentSelection: SkillSelection | null = null;

  constructor(options: SkillHarnessOptions) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
    this.registry = new SkillRegistry(this.options.skillDirs);
  }

  // ─── Per-Turn Routing ───────────────────────────────────────────────────────

  /**
   * Route skills for the current turn.
   * Call this from middleware.beforeRound() BEFORE the model call.
   */
  async routeForTurn(
    history: readonly ChatMessage[],
    contextWindow: number,
    systemPrompt: string,
    cwd?: string,
  ): Promise<SkillSelection> {
    // Calculate reserved tokens (history + system prompt)
    const reservedTokens = countMessagesTokens(history as ChatMessage[], systemPrompt);

    // Create token budget for skills
    const budget = createTokenBudget(contextWindow, reservedTokens);

    // Route skills
    const selection = await routeSkillsForTurn(this.registry, history, budget, {
      maxSkills: this.options.maxSkillsPerTurn,
      maxPatterns: this.options.maxPatternsPerTurn,
      minScore: this.options.similarityThreshold,
      cwd: cwd ?? process.cwd(),
    });

    this.currentSelection = selection;
    return selection;
  }

  /** Get the current turn's skill schemas for system prompt injection. */
  getSchemasForTurn(): string[] {
    return this.currentSelection?.schemas ?? [];
  }

  /** Get the current turn's patterns as callable tools. */
  getPatternsForTurn(): PatternDef[] {
    return this.currentSelection?.patterns ?? [];
  }

  /** Get the current turn's selected skills. */
  getSkillsForTurn(): SkillDef[] {
    return this.currentSelection?.skills ?? [];
  }

  // ─── Trajectory Observation ─────────────────────────────────────────────────

  /**
   * Observe a tool trajectory after a round completes.
   * Call this from middleware.afterRound().
   */
  observe(trajectory: ToolTrajectory): void {
    if (!this.options.enableSynthesis) return;

    // Store trajectory for pattern synthesis
    // Actual synthesis happens in the Synthesizer (Phase C)
    // For now, just record metrics if a skill was used
    if (trajectory.skillUsed) {
      this.registry.recordInvocation(trajectory.skillUsed, {
        success: trajectory.success,
        tokens: trajectory.tokens,
        latency: trajectory.duration,
      });
    }
  }

  // ─── Metrics & Curation ─────────────────────────────────────────────────────

  /** Update metrics and trigger curation rules. */
  updateMetrics(): void {
    if (!this.options.enableCuration) return;

    // Apply curation rules
    for (const metric of this.registry.getAllMetrics()) {
      // Promotion: 5+ invocations, >80% success
      if (metric.invocations >= 5 && metric.successes / metric.invocations > 0.8 && metric.status === "active") {
        this.registry.setStatus(metric.name, "trusted");
      }

      // Deprecation: 10+ invocations, <40% success
      if (metric.invocations >= 10 && metric.successes / metric.invocations < 0.4 && metric.status !== "deprecated") {
        this.registry.setStatus(metric.name, "deprecated");
      }

      // Watch: 3-10 invocations, 40-60% success
      if (
        metric.invocations >= 3 &&
        metric.invocations < 10 &&
        metric.successes / metric.invocations >= 0.4 &&
        metric.successes / metric.invocations < 0.6 &&
        metric.status === "active"
      ) {
        this.registry.setStatus(metric.name, "watch");
      }
    }

    // Persist metrics
    this.registry.saveMetrics();
  }

  /** Generate a health report for all skills. */
  generateHealthReport(): SkillHealthReport {
    const metrics = this.registry.getAllMetrics();
    return {
      generatedAt: Date.now(),
      totalSkills: this.registry.getAllSkills().length,
      totalPatterns: this.registry.getAllPatterns().length,
      active: metrics.filter((m) => m.status === "active"),
      trusted: metrics.filter((m) => m.status === "trusted"),
      watch: metrics.filter((m) => m.status === "watch"),
      deprecated: metrics.filter((m) => m.status === "deprecated"),
      annealing: metrics.filter((m) => m.status === "annealing"),
    };
  }

  // ─── Registry Access ────────────────────────────────────────────────────────

  getRegistry(): SkillRegistry {
    return this.registry;
  }

  /** Register a skill programmatically. */
  async registerSkill(skill: SkillDef): Promise<void> {
    await this.registry.registerSkill(skill);
  }

  /** Register a pattern programmatically. */
  async registerPattern(pattern: PatternDef): Promise<void> {
    await this.registry.registerPattern(pattern);
  }

  // ─── Cleanup ────────────────────────────────────────────────────────────────

  dispose(): void {
    this.registry.dispose();
  }
}

/** Factory with default skill directories. */
export function createSkillHarness(extraDirs: string[] = []): SkillHarness {
  const defaultDirs = [
    path.join(os.homedir(), ".chorus", "skills"),
    path.join(process.cwd(), ".chorus", "skills"),
  ];

  return new SkillHarness({
    skillDirs: [...defaultDirs, ...extraDirs],
  });
}
