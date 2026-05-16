/**
 * PII Redaction Engine — Backward-Compatible Re-exports
 *
 * Re-exports the hybrid three-tier redaction system with legacy API signatures.
 * New code should use HybridRedactionEngine directly from telemetry/redaction.
 */

import {
  HybridRedactionEngine,
  redactString as hybridRedactString,
  redactObject as hybridRedactObject,
  redactSpanAttributes as hybridRedactSpanAttributes,
  type RedactionConfig as TelemetryConfig,
  type RedactionSpan,
  type RedactionResult,
} from "../telemetry/redaction.js";

export interface RedactionConfig {
  email?: boolean;
  ssn?: boolean;
  creditCard?: boolean;
  phone?: boolean;
  apiKey?: boolean;
  ipAddress?: boolean;
}

const DEFAULT_CONFIG: Required<RedactionConfig> = {
  email: true,
  ssn: true,
  creditCard: true,
  phone: true,
  apiKey: true,
  ipAddress: true,
};

/** Backward-compatible string redaction returning a string. */
export function redactString(text: string, config: RedactionConfig = {}): string {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  // Run synchronously using the fast-tier only (regex patterns)
  const result = hybridRedactString(text, { patterns: [], strip: false });
  return result.text;
}

/** Backward-compatible object redaction. */
export function redactObject<T extends Record<string, unknown>>(obj: T, config?: RedactionConfig): T {
  const result = hybridRedactObject(obj, { patterns: [], strip: false });
  return result.obj as T;
}

/** Backward-compatible span attribute redaction. */
export function redactSpanAttributes(attrs: Array<{ key: string; value: unknown }>): Array<{ key: string; value: unknown }> {
  return attrs.map((attr) => ({
    key: attr.key,
    value: typeof attr.value === "object" && attr.value !== null
      ? redactObject(attr.value as Record<string, unknown>)
      : attr.value,
  }));
}

export { HybridRedactionEngine, type RedactionSpan, type RedactionResult };
