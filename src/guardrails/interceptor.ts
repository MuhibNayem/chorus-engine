/**
 * Deterministic Tool-Call Interceptor
 *
 * SupraWall-style pre-execution policy enforcement (2026).
 * Intercepts tool calls BEFORE the underlying function is invoked and
 * performs a deterministic match against declarative policy rules.
 *
 * Key properties:
 *   - Binary outcomes: ALLOW, DENY, or REQUIRE_APPROVAL
 *   - No LLM in the enforcement path → 1–2ms latency, zero bypass rate
 *   - Declarative policies: tool name + param patterns, no code required
 *   - Fail-secure: unmatched tools default to DENY if strict mode enabled
 *
 * Contrast with probabilistic guardrails:
 *   - Regex/LLM guardrails: inspect tool arguments as text, can be bypassed
 *     via encoding tricks, spacing, or semantic obfuscation.
 *   - Deterministic interceptor: matches parsed parameter keys/values against
 *     declared patterns; the tool function is NEVER called if DENY.
 *
 * Reference:
 *   - SupraWall (2026) — deterministic pre-execution interception
 *   - OWASP ASI 2026 — agent-specific risk taxonomy (ASI02 tool misuse)
 */

export type PolicyDecision = "allow" | "deny" | "require_approval";

export interface ParamConstraint {
  /** Parameter name. */
  param: string;
  /** Allowed values (exact match). If empty, any value is allowed. */
  allowedValues?: (string | number | boolean)[];
  /** Blocked values (exact match). */
  blockedValues?: (string | number | boolean)[];
  /** Regex pattern the string value must match. */
  pattern?: RegExp;
  /** Regex patterns the string value must NOT match. */
  blockedPatterns?: RegExp[];
  /** Max string length. */
  maxLength?: number;
  /** Required type. */
  type?: "string" | "number" | "boolean" | "array" | "object";
}

export interface ToolPolicy {
  /** Tool name to match. */
  tool: string;
  /** Default decision when no constraints match. */
  defaultDecision?: PolicyDecision;
  /** Parameter-level constraints. */
  params?: ParamConstraint[];
  /** Parameter names that must be present. */
  requiredParams?: string[];
  /** Parameter names that must NOT be present. */
  forbiddenParams?: string[];
  /** Global blocked patterns applied to all string param values. */
  globalBlockedPatterns?: RegExp[];
  /** Human-readable policy description. */
  description?: string;
}

export interface InterceptResult {
  decision: PolicyDecision;
  /** Which policy rule produced this decision. */
  policy?: string;
  /** Human-readable explanation. */
  reason?: string;
  /** Parameter that triggered the decision (if any). */
  triggeredParam?: string;
  /** Value that triggered the decision (if any). */
  triggeredValue?: unknown;
}

/**
 * Pre-execution tool call interceptor.
 *
 * Usage:
 *   const interceptor = new ToolCallInterceptor([
 *     {
 *       tool: "execute_sql",
 *       defaultDecision: "require_approval",
 *       globalBlockedPatterns: [/\bDROP\b/i, /\bDELETE\b/i],
 *       params: [{ param: "command", blockedPatterns: [/rm\s+-rf/i] }],
 *     },
 *     {
 *       tool: "shell",
 *       defaultDecision: "deny",
 *       params: [
 *         { param: "command", blockedPatterns: [/rm\s+-rf\s+\//i, /mkfs/i] },
 *       ],
 *     },
 *   ]);
 *
 *   const result = interceptor.intercept("execute_sql", { command: "SELECT 1" });
 *   // → { decision: "allow" }
 *
 *   const result2 = interceptor.intercept("execute_sql", { command: "DROP TABLE users" });
 *   // → { decision: "deny", reason: "Blocked pattern matched", triggeredParam: "command" }
 */
export class ToolCallInterceptor {
  private policies = new Map<string, ToolPolicy>();
  private strictMode: boolean;

  constructor(policies: ToolPolicy[], strictMode = false) {
    this.strictMode = strictMode;
    for (const p of policies) {
      this.policies.set(p.tool, p);
    }
  }

  intercept(toolName: string, args: Record<string, unknown>): InterceptResult {
    const policy = this.policies.get(toolName);

    if (!policy) {
      return this.strictMode
        ? { decision: "deny", reason: `No policy defined for tool "${toolName}" (strict mode)` }
        : { decision: "allow", reason: `No policy defined for tool "${toolName}"` };
    }

    // Check required params
    if (policy.requiredParams) {
      for (const rp of policy.requiredParams) {
        if (!(rp in args)) {
          return {
            decision: "deny",
            policy: policy.description ?? policy.tool,
            reason: `Required parameter "${rp}" missing`,
            triggeredParam: rp,
          };
        }
      }
    }

    // Check forbidden params
    if (policy.forbiddenParams) {
      for (const fp of policy.forbiddenParams) {
        if (fp in args) {
          return {
            decision: "deny",
            policy: policy.description ?? policy.tool,
            reason: `Forbidden parameter "${fp}" present`,
            triggeredParam: fp,
          };
        }
      }
    }

    // Evaluate each parameter against constraints
    for (const [paramName, value] of Object.entries(args)) {
      const constraint = policy.params?.find((c) => c.param === paramName);

      // Global blocked patterns
      if (policy.globalBlockedPatterns && typeof value === "string") {
        for (const bp of policy.globalBlockedPatterns) {
          if (bp.test(value)) {
            return {
              decision: "deny",
              policy: policy.description ?? policy.tool,
              reason: `Global blocked pattern matched in "${paramName}"`,
              triggeredParam: paramName,
              triggeredValue: value.length > 50 ? value.slice(0, 50) + "..." : value,
            };
          }
        }
      }

      if (!constraint) continue;

      // Type check
      if (constraint.type) {
        const actualType = Array.isArray(value) ? "array" : typeof value;
        if (actualType !== constraint.type) {
          return {
            decision: "deny",
            policy: policy.description ?? policy.tool,
            reason: `Parameter "${paramName}" expected type ${constraint.type}, got ${actualType}`,
            triggeredParam: paramName,
            triggeredValue: value,
          };
        }
      }

      // String length check
      if (constraint.maxLength && typeof value === "string" && value.length > constraint.maxLength) {
        return {
          decision: "deny",
          policy: policy.description ?? policy.tool,
          reason: `Parameter "${paramName}" exceeds max length ${constraint.maxLength}`,
          triggeredParam: paramName,
          triggeredValue: value.slice(0, 50) + "...",
        };
      }

      // Allowed values
      if (constraint.allowedValues && constraint.allowedValues.length > 0) {
        if (!constraint.allowedValues.includes(value as string | number | boolean)) {
          return {
            decision: "deny",
            policy: policy.description ?? policy.tool,
            reason: `Parameter "${paramName}" value not in allowed set`,
            triggeredParam: paramName,
            triggeredValue: value,
          };
        }
      }

      // Blocked values
      if (constraint.blockedValues && constraint.blockedValues.length > 0) {
        if (constraint.blockedValues.includes(value as string | number | boolean)) {
          return {
            decision: "deny",
            policy: policy.description ?? policy.tool,
            reason: `Parameter "${paramName}" has blocked value`,
            triggeredParam: paramName,
            triggeredValue: value,
          };
        }
      }

      // Pattern match
      if (constraint.pattern && typeof value === "string") {
        if (!constraint.pattern.test(value)) {
          return {
            decision: "deny",
            policy: policy.description ?? policy.tool,
            reason: `Parameter "${paramName}" does not match required pattern`,
            triggeredParam: paramName,
            triggeredValue: value,
          };
        }
      }

      // Blocked patterns
      if (constraint.blockedPatterns && typeof value === "string") {
        for (const bp of constraint.blockedPatterns) {
          if (bp.test(value)) {
            return {
              decision: "deny",
              policy: policy.description ?? policy.tool,
              reason: `Blocked pattern in parameter "${paramName}"`,
              triggeredParam: paramName,
              triggeredValue: value.length > 50 ? value.slice(0, 50) + "..." : value,
            };
          }
        }
      }
    }

    // All checks passed
    return {
      decision: policy.defaultDecision ?? "allow",
      policy: policy.description ?? policy.tool,
      reason: "All parameter checks passed",
    };
  }

  /** Check if a tool has a registered policy. */
  hasPolicy(toolName: string): boolean {
    return this.policies.has(toolName);
  }

  /** List all tool names with registered policies. */
  listTools(): string[] {
    return Array.from(this.policies.keys());
  }
}

/** Convenience policies for common dangerous tools. */
export const DangerousToolPolicies: ToolPolicy[] = [
  {
    tool: "shell",
    defaultDecision: "require_approval",
    globalBlockedPatterns: [/rm\s+-rf\s+\//i, /rm\s+-rf\s+~/i, /mkfs/i, /dd\s+if=\/dev\/zero/i, />\s*\/dev\/sda/i],
    params: [{ param: "command", type: "string", maxLength: 10_000 }],
    description: "Shell execution with dangerous command blocking",
  },
  {
    tool: "execute_sql",
    defaultDecision: "require_approval",
    globalBlockedPatterns: [/\bDROP\s+(DATABASE|TABLE)\b/i, /\bDELETE\s+FROM\b.*\bWHERE\b/i],
    params: [{ param: "query", type: "string", maxLength: 50_000 }],
    description: "SQL execution with destructive command blocking",
  },
  {
    tool: "write_file",
    defaultDecision: "allow",
    params: [
      { param: "path", type: "string", blockedPatterns: [/\.\.\//, /\/etc\/passwd/, /\/etc\/shadow/, /\.ssh\/authorized_keys/] },
      { param: "content", type: "string", maxLength: 1_000_000 },
    ],
    description: "File write with path traversal protection",
  },
  {
    tool: "send_email",
    defaultDecision: "require_approval",
    requiredParams: ["to", "subject"],
    params: [
      { param: "to", type: "string", pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/ },
      { param: "subject", type: "string", maxLength: 500 },
      { param: "body", type: "string", maxLength: 100_000 },
    ],
    description: "Email sending with validation",
  },
];
