/**
 * Enterprise Guardrails System — Tiered Defense Architecture
 *
 * Three-tier validation matching 2025–2026 production standards:
 *   Tier 1 (Fast)    : Regex / keywords / deterministic interception  (<1ms)
 *   Tier 2 (ML)      : Transformer NER / embedding similarity         (20–100ms)
 *   Tier 3 (LLM)     : LLM-as-judge policy evaluation                 (500ms–8s)
 *
 * Components:
 *   - TieredGuardrailEngine   — orchestrates tiers with escalation + latency budgets
 *   - ToolCallInterceptor     — SupraWall-style deterministic pre-execution validation
 *   - HybridRedactionEngine   — three-tier PII redaction (regex + NER + LLM)
 *   - AdaptiveThreshold       — self-improving thresholds from operator feedback
 *   - NERDetector             — transformer-based named entity recognition
 *
 * Legacy API (BuiltInGuardrails, runGuardrails, shouldHalt) is preserved for
 * backward compatibility but wraps the new tiered engine internally.
 */

import type { ChatMessage } from "../llm/provider.js";
import type { MutableSpan } from "../telemetry/inprocess.js";
import {
  TieredGuardrailEngine,
  createPromptInjectionGuard,
  createSecretLeakGuard,
  createBlocklistGuard,
  createNERPIIGuard,
  createTopicDriftGuard,
  createLLMPolicyGuard,
  type GuardrailCheck,
  type GuardrailContext,
  type GuardrailTier,
  type GuardrailSeverity,
  type GuardrailViolation,
  type TieredResult,
  type TieredCheckOptions,
} from "./tiered.js";

export { redactString, redactObject, redactSpanAttributes } from "./redaction.js";
export type { RedactionConfig } from "./redaction.js";

export type {
  GuardrailCheck,
  GuardrailContext,
  GuardrailTier,
  GuardrailSeverity,
  GuardrailViolation,
  TieredResult,
  TieredCheckOptions,
};

export {
  TieredGuardrailEngine,
  createPromptInjectionGuard,
  createSecretLeakGuard,
  createBlocklistGuard,
  createNERPIIGuard,
  createTopicDriftGuard,
  createLLMPolicyGuard,
};

export {
  ToolCallInterceptor,
  DangerousToolPolicies,
  type PolicyDecision,
  type ToolPolicy,
  type ParamConstraint,
  type InterceptResult,
} from "./interceptor.js";

export {
  AdaptiveThreshold,
  AdaptiveThresholdManager,
  type GuardrailFeedback,
  type AdaptiveThresholdState,
} from "./adaptive.js";

export {
  TransformersNERDetector,
  KeywordNERFallback,
  HybridNERDetector,
  createNERDetector,
  type NERDetector,
  type EntitySpan,
} from "./ner.js";

export { HybridRedactionEngine, type RedactionSpan, type RedactionResult } from "./redaction.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Legacy Context Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface InputGuardrailContext {
  messages: readonly ChatMessage[];
  systemPrompt: string;
  threadId: string;
  round: number;
}

export interface OutputGuardrailContext {
  response: string;
  toolCalls?: Array<{ name: string; args: Record<string, unknown> }>;
  threadId: string;
  round: number;
}

export interface ToolGuardrailContext {
  toolName: string;
  args: Record<string, unknown>;
  threadId: string;
  round: number;
}

export type InputGuardrail = (ctx: InputGuardrailContext) => Promise<GuardrailViolation | null>;
export type OutputGuardrail = (ctx: OutputGuardrailContext) => Promise<GuardrailViolation | null>;
export type ToolGuardrail = (ctx: ToolGuardrailContext) => Promise<GuardrailViolation | null>;

export interface GuardrailsConfig {
  inputs?: InputGuardrail[];
  outputs?: OutputGuardrail[];
  tools?: ToolGuardrail[];
  haltOn?: GuardrailSeverity;
  runAll?: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Legacy Built-in Guardrails (wrap new tiered checks)
// ═══════════════════════════════════════════════════════════════════════════════

export namespace BuiltInGuardrails {
  export const promptInjection: InputGuardrail = async (ctx) => {
    const guard = createPromptInjectionGuard();
    const result = await guard.check({
      text: ctx.messages.map((m) => m.content).join(" "),
      boundary: "input",
      metadata: { threadId: ctx.threadId, round: ctx.round },
    });
    return result;
  };

  export const piiRedaction: ToolGuardrail = async (ctx) => {
    const json = JSON.stringify(ctx.args);
    const guard = createSecretLeakGuard();
    const result = await guard.check({
      text: json,
      boundary: "tool",
      metadata: { toolName: ctx.toolName, threadId: ctx.threadId, round: ctx.round },
    });
    return result;
  };

  export const dangerousCommand: ToolGuardrail = async (ctx) => {
    if (!["run_command", "shell", "execute", "exec"].includes(ctx.toolName)) return null;
    const command = String(ctx.args.command ?? "").toLowerCase();
    const dangerous = [
      "rm -rf /", "rm -rf ~", "rm -rf *", ":(){ :|:& };:", "mkfs",
      "dd if=/dev/zero", "> /dev/sda", "chmod 777 /",
    ];
    for (const d of dangerous) {
      if (command.includes(d)) {
        return {
          guardrail: "dangerous_command",
          message: `Dangerous command blocked: ${d}`,
          severity: "critical",
          action: "halt",
          trigger: d,
        };
      }
    }
    return null;
  };

  export const outputQuality: OutputGuardrail = async (ctx) => {
    if (!ctx.response || ctx.response.trim().length < 3) {
      return {
        guardrail: "output_quality",
        message: "LLM produced empty or near-empty output",
        severity: "warning",
        action: "warn",
      };
    }
    return null;
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Legacy runGuardrails (preserved for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

export async function runGuardrails<T>(
  guardrails: Array<(ctx: T) => Promise<GuardrailViolation | null>>,
  ctx: T,
  opts?: { runAll?: boolean; haltOn?: GuardrailSeverity; span?: MutableSpan },
): Promise<GuardrailViolation[]> {
  const runAll = opts?.runAll ?? true;
  const haltOn = opts?.haltOn ?? "critical";
  const severityOrder: GuardrailSeverity[] = ["info", "warning", "critical"];
  const haltIndex = severityOrder.indexOf(haltOn);
  const violations: GuardrailViolation[] = [];

  if (runAll) {
    const results = await Promise.all(guardrails.map((g) => g(ctx)));
    for (const v of results) {
      if (v) {
        violations.push(v);
        opts?.span?.setAttribute(`guardrail.${v.guardrail}.violated`, true);
        opts?.span?.setAttribute(`guardrail.${v.guardrail}.severity`, v.severity);
      }
    }
  } else {
    for (const g of guardrails) {
      const v = await g(ctx);
      if (v) {
        violations.push(v);
        opts?.span?.setAttribute(`guardrail.${v.guardrail}.violated`, true);
        if (severityOrder.indexOf(v.severity) >= haltIndex) break;
      }
    }
  }

  return violations;
}

export function shouldHalt(violations: GuardrailViolation[], haltOn: GuardrailSeverity = "critical"): boolean {
  const severityOrder: GuardrailSeverity[] = ["info", "warning", "critical"];
  const haltIndex = severityOrder.indexOf(haltOn);
  return violations.some((v) => severityOrder.indexOf(v.severity) >= haltIndex);
}
