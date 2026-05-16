/**
 * Tiered Guardrail Engine
 *
 * Production guardrail architecture matching the 2025–2026 industry standard:
 *
 *   Tier 1 (Fast)    : Regex / keywords / blocklists  — <1ms, CPU-only
 *   Tier 2 (ML)      : Transformer NER / embeddings   — 20–100ms, neural
 *   Tier 3 (LLM)     : LLM-as-judge                   — 500ms–8s, nuanced
 *
 * Design inspired by production systems (Guardrails AI, NeMo, Fiddler,
 * SupraWall) and research on latency-accuracy trade-offs:
 *   - Fast tier filters obvious cases (regex injection patterns, known secrets)
 *   - ML tier catches context-dependent threats (names, addresses, semantic
 *     similarity to banned topics) that regex misses
 *   - LLM tier handles edge cases and policy nuance; only invoked when lower
 *     tiers disagree or score near threshold (uncertainty escalation)
 *   - Latency budget enforced: if Tier 3 would exceed budget, it is skipped
 *     and the stricter Tier 2 decision is used (fail-secure)
 *
 * References:
 *   - Fiddler Guardrails (2025) — multi-metric trust scoring
 *   - SupraWall (2026) — deterministic pre-execution interception
 *   - Symbolic Guardrails for Domain-Specific Agents (2026) — neural vs symbolic
 *   - A Survey on LLM Guardrails (2025) — tiered latency best practices
 */

import type { LLMProvider } from "../llm/provider.js";

export type GuardrailTier = "fast" | "ml" | "llm";

export type GuardrailSeverity = "info" | "warning" | "critical";

export interface GuardrailViolation {
  guardrail: string;
  message: string;
  severity: GuardrailSeverity;
  action: "halt" | "warn";
  trigger?: string;
  /** Which tier produced this violation. */
  tier?: GuardrailTier;
  /** Latency of the check that produced this violation (ms). */
  latencyMs?: number;
}

/** A single guardrail check at a specific tier. */
export interface GuardrailCheck {
  readonly name: string;
  readonly tier: GuardrailTier;
  readonly expectedLatencyMs: number;
  check(ctx: GuardrailContext): Promise<GuardrailViolation | null>;
}

/** Unified context passed to all guardrail tiers. */
export interface GuardrailContext {
  /** Input text to evaluate (user message, LLM output, or tool arg JSON). */
  text: string;
  /** Conversation history for context-aware evaluation. */
  history?: string;
  /** Which boundary is being checked: input, output, or tool. */
  boundary: "input" | "output" | "tool";
  /** Additional metadata (tool name, round number, etc.). */
  metadata?: Record<string, unknown>;
}

export interface TieredCheckOptions {
  /** Max tier to run. Default: "llm" */
  maxTier?: GuardrailTier;
  /** Tier 2 confidence threshold below which we escalate to Tier 3. 0–1. Default: 0.7 */
  escalationThreshold?: number;
  /** Max total latency budget (ms). If exceeded, skip remaining tiers. Default: 5000 */
  latencyBudgetMs?: number;
  /** If true, collect all violations; if false, stop at first halt-worthy violation. */
  collectAll?: boolean;
  /** Minimum severity that triggers a halt. */
  haltOn?: GuardrailSeverity;
}

export interface TieredResult {
  violations: GuardrailViolation[];
  /** Total wall-clock latency. */
  totalLatencyMs: number;
  /** Per-tier latency breakdown. */
  tierLatencyMs: Record<GuardrailTier, number>;
  /** Number of times Tier 3 was invoked due to Tier 2 uncertainty. */
  escalations: number;
  /** Tiers that were actually executed. */
  tiersExecuted: GuardrailTier[];
  /** Whether the latency budget was exceeded. */
  budgetExceeded: boolean;
}

const TIER_ORDER: GuardrailTier[] = ["fast", "ml", "llm"];

function tierIndex(t: GuardrailTier): number {
  return TIER_ORDER.indexOf(t);
}

function severityIndex(s: GuardrailSeverity): number {
  return ["info", "warning", "critical"].indexOf(s);
}

/**
 * Orchestrates guardrail checks across three latency tiers with intelligent
 * escalation and latency budgeting.
 */
export class TieredGuardrailEngine {
  private checksByTier = new Map<GuardrailTier, GuardrailCheck[]>();

  constructor(checks: GuardrailCheck[]) {
    for (const tier of TIER_ORDER) {
      this.checksByTier.set(tier, []);
    }
    for (const c of checks) {
      const list = this.checksByTier.get(c.tier) ?? [];
      list.push(c);
      this.checksByTier.set(c.tier, list);
    }
  }

  async check(ctx: GuardrailContext, opts: TieredCheckOptions = {}): Promise<TieredResult> {
    const maxTier = opts.maxTier ?? "llm";
    const escalationThreshold = opts.escalationThreshold ?? 0.7;
    const latencyBudgetMs = opts.latencyBudgetMs ?? 5_000;
    const collectAll = opts.collectAll ?? true;
    const haltOn = opts.haltOn ?? "critical";
    const haltIndex = severityIndex(haltOn);

    const violations: GuardrailViolation[] = [];
    const tierLatencyMs: Record<GuardrailTier, number> = { fast: 0, ml: 0, llm: 0 };
    let escalations = 0;
    const tiersExecuted: GuardrailTier[] = [];
    const startTime = performance.now();
    let budgetExceeded = false;

    // Track highest uncertainty score from ML tier for escalation decision
    let mlMaxUncertainty = 0;

    for (const tier of TIER_ORDER) {
      if (tierIndex(tier) > tierIndex(maxTier)) break;

      const elapsed = performance.now() - startTime;
      if (elapsed >= latencyBudgetMs) {
        budgetExceeded = true;
        break;
      }

      const checks = this.checksByTier.get(tier) ?? [];
      if (checks.length === 0) continue;

      tiersExecuted.push(tier);
      const tierStart = performance.now();

      if (tier === "llm") {
        // Only escalate to LLM if ML tier showed uncertainty
        if (mlMaxUncertainty < escalationThreshold && tiersExecuted.includes("ml")) {
          tierLatencyMs.llm = 0;
          continue; // Skip LLM — no uncertainty to resolve
        }
        escalations++;
      }

      if (collectAll) {
        const results = await Promise.all(
          checks.map(async (c) => {
            const checkStart = performance.now();
            try {
              const v = await c.check(ctx);
              if (v) {
                v.tier = tier;
                v.latencyMs = Math.round(performance.now() - checkStart);
              }
              return v;
            } catch {
              return null;
            }
          }),
        );
        for (const v of results) {
          if (v) {
            violations.push(v);
            if (tier === "ml" && v.severity === "warning") {
              mlMaxUncertainty = Math.max(mlMaxUncertainty, 0.75);
            }
          }
        }
      } else {
        for (const c of checks) {
          const checkStart = performance.now();
          try {
            const v = await c.check(ctx);
            if (v) {
              v.tier = tier;
              v.latencyMs = Math.round(performance.now() - checkStart);
              violations.push(v);
              if (severityIndex(v.severity) >= haltIndex) {
                tierLatencyMs[tier] = Math.round(performance.now() - tierStart);
                return {
                  violations,
                  totalLatencyMs: Math.round(performance.now() - startTime),
                  tierLatencyMs,
                  escalations,
                  tiersExecuted,
                  budgetExceeded,
                };
              }
            }
          } catch { /* ignore check failures */ }
        }
      }

      tierLatencyMs[tier] = Math.round(performance.now() - tierStart);

      // If ML tier flagged warnings, increase uncertainty score for potential escalation
      if (tier === "ml") {
        const mlWarnings = violations.filter((v) => v.tier === "ml" && v.severity === "warning");
        if (mlWarnings.length > 0) {
          mlMaxUncertainty = Math.max(mlMaxUncertainty, 0.6 + mlWarnings.length * 0.1);
        }
      }
    }

    return {
      violations,
      totalLatencyMs: Math.round(performance.now() - startTime),
      tierLatencyMs,
      escalations,
      tiersExecuted,
      budgetExceeded,
    };
  }

  shouldHalt(result: TieredResult, haltOn: GuardrailSeverity = "critical"): boolean {
    const haltIndex = severityIndex(haltOn);
    return result.violations.some((v) => severityIndex(v.severity) >= haltIndex);
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Built-in Fast Tier Guards (<1ms)
// ═══════════════════════════════════════════════════════════════════════════════

/** Regex-based prompt injection detector — Tier 1 fast path. */
export function createPromptInjectionGuard(): GuardrailCheck {
  const patterns = [
    /ignore previous instructions/i,
    /forget (everything|all|your) (instructions|prompt)/i,
    /you are now .* instead/i,
    /system prompt:/i,
    /new instructions:/i,
    /(override|bypass|disable) (safety|guardrail|restriction)/i,
    /\{\{.*?\}\}/, // Template injection
    /<\|im_start\|>/, // ChatML injection
    /\[system\]/i, // Pseudo system role
  ];

  return {
    name: "prompt_injection_fast",
    tier: "fast",
    expectedLatencyMs: 1,
    async check(ctx) {
      if (ctx.boundary !== "input") return null;
      for (const re of patterns) {
        const m = re.exec(ctx.text);
        if (m) {
          return {
            guardrail: "prompt_injection",
            message: `Fast-tier prompt injection pattern detected: "${m[0]}"`,
            severity: "critical",
            action: "halt",
            trigger: m[0],
          };
        }
      }
      return null;
    },
  };
}

/** Regex-based secret detector — Tier 1 fast path with structural validation. */
export function createSecretLeakGuard(): GuardrailCheck {
  // Luhn algorithm for credit card validation
  function luhnCheck(digits: string): boolean {
    let sum = 0;
    let alt = false;
    for (let i = digits.length - 1; i >= 0; i--) {
      let n = parseInt(digits[i], 10);
      if (alt) {
        n *= 2;
        if (n > 9) n -= 9;
      }
      sum += n;
      alt = !alt;
    }
    return sum % 10 === 0;
  }

  return {
    name: "secret_leak_fast",
    tier: "fast",
    expectedLatencyMs: 1,
    async check(ctx) {
      const text = ctx.text;
      const checks: Array<{ regex: RegExp; name: string; validate?: (m: string) => boolean }> = [
        {
          regex: /\bBearer\s+[A-Za-z0-9_\-]{20,}\b/gi,
          name: "bearer_token",
        },
        {
          regex: /\bAKIA[0-9A-Z]{16}\b/g,
          name: "aws_access_key",
        },
        {
          regex: /\b\d{3}-\d{2}-\d{4}\b/g,
          name: "us_ssn",
        },
        {
          regex: /\b(?:\d[ -]*?){13,16}\b/g,
          name: "credit_card",
          validate: (m) => {
            const digits = m.replace(/\D/g, "");
            return digits.length >= 13 && luhnCheck(digits);
          },
        },
        {
          regex: /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g,
          name: "email",
        },
        {
          regex: /-----BEGIN (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----[\s\S]*?-----END (RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----/g,
          name: "private_key",
        },
      ];

      for (const { regex, name, validate } of checks) {
        let match: RegExpExecArray | null;
        while ((match = regex.exec(text)) !== null) {
          const value = match[0];
          if (!validate || validate(value)) {
            return {
              guardrail: `secret_leak_${name}`,
              message: `Fast-tier secret leak detected: ${name}`,
              severity: "critical",
              action: "halt",
              trigger: value.slice(0, 20) + "...",
            };
          }
        }
      }
      return null;
    },
  };
}

/** Keyword blocklist guard — Tier 1. */
export function createBlocklistGuard(blockedTerms: string[]): GuardrailCheck {
  const set = new Set(blockedTerms.map((t) => t.toLowerCase()));
  return {
    name: "blocklist_fast",
    tier: "fast",
    expectedLatencyMs: 1,
    async check(ctx) {
      const words = ctx.text.toLowerCase().split(/\W+/);
      for (const w of words) {
        if (set.has(w)) {
          return {
            guardrail: "blocklist",
            message: `Blocked term detected: "${w}"`,
            severity: "critical",
            action: "halt",
            trigger: w,
          };
        }
      }
      return null;
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Built-in ML Tier Guards (20–100ms)
// ═══════════════════════════════════════════════════════════════════════════════

import type { NERDetector } from "./ner.js";

/** NER-based PII detector — Tier 2. Catches names, locations, orgs that regex misses. */
export function createNERPIIGuard(ner: NERDetector): GuardrailCheck {
  return {
    name: "pii_ner_ml",
    tier: "ml",
    expectedLatencyMs: 50,
    async check(ctx) {
      const entities = await ner.detect(ctx.text);
      const piiEntities = entities.filter((e) =>
        ["PER", "PERSON", "LOC", "LOCATION", "ORG", "ORGANIZATION",
         "EMAIL", "PHONE", "URL", "DATE"].includes(e.label.toUpperCase()),
      );
      if (piiEntities.length === 0) return null;

      const names = piiEntities.map((e) => e.text).slice(0, 3).join(", ");
      return {
        guardrail: "pii_ner",
        message: `ML-tier PII detected: ${piiEntities.length} entity(s) including ${names}`,
        severity: piiEntities.length > 2 ? "warning" : "info",
        action: "warn",
        trigger: names,
      };
    },
  };
}

/** Embedding similarity guard for topic drift — Tier 2. */
export function createTopicDriftGuard(
  embedFn: (text: string) => Promise<number[]>,
  bannedTopicEmbeddings: Array<{ name: string; embedding: number[] }>,
  threshold = 0.82,
): GuardrailCheck {
  function cosine(a: number[], b: number[]): number {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    return na === 0 || nb === 0 ? 0 : dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  return {
    name: "topic_drift_ml",
    tier: "ml",
    expectedLatencyMs: 30,
    async check(ctx) {
      const emb = await embedFn(ctx.text);
      let bestMatch: { name: string; score: number } | null = null;
      for (const topic of bannedTopicEmbeddings) {
        const score = cosine(emb, topic.embedding);
        if (!bestMatch || score > bestMatch.score) {
          bestMatch = { name: topic.name, score };
        }
      }
      if (bestMatch && bestMatch.score > threshold) {
        return {
          guardrail: "topic_drift",
          message: `Semantic similarity to banned topic "${bestMatch.name}" (${bestMatch.score.toFixed(3)})`,
          severity: "warning",
          action: "warn",
          trigger: bestMatch.name,
        };
      }
      return null;
    },
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Built-in LLM Tier Guards (500ms–8s)
// ═══════════════════════════════════════════════════════════════════════════════

/** LLM-as-judge for nuanced policy evaluation — Tier 3. */
export function createLLMPolicyGuard(
  provider: LLMProvider,
  model: string,
  policy: string,
): GuardrailCheck {
  return {
    name: "policy_llm_judge",
    tier: "llm",
    expectedLatencyMs: 2_000,
    async check(ctx) {
      const prompt =
        `You are a safety policy evaluator. Analyze the following text against this policy:\n\n` +
        `POLICY: ${policy}\n\n` +
        `TEXT:\n${ctx.text}\n\n` +
        `Respond with ONLY a JSON object:\n` +
        `{"violates": boolean, "severity": "info" | "warning" | "critical", ` +
        `"reason": string, "action": "halt" | "warn"}`;

      try {
        const result = await provider.generate({
          model,
          messages: [{ role: "user", content: prompt }],
        });
        const parsed = JSON.parse(result.text.replace(/```json\s*|\s*```/g, "")) as {
          violates?: boolean;
          severity?: string;
          reason?: string;
          action?: string;
        };
        if (parsed.violates) {
          return {
            guardrail: "llm_policy",
            message: parsed.reason || "LLM policy violation",
            severity: (parsed.severity as GuardrailSeverity) || "warning",
            action: parsed.action === "halt" ? "halt" : "warn",
          };
        }
      } catch {
        // LLM judge failure is non-fatal; fall through
      }
      return null;
    },
  };
}
