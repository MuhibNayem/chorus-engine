/**
 * OTLP/HTTP exporter — sends OTel spans to a collector endpoint or stdout.
 *
 * Hardened for production:
 *   - Failed exports are buffered (up to MAX_BUFFER_SIZE spans).
 *   - Retried with exponential backoff on next flush call.
 *   - Buffer overflow drops oldest spans (newest telemetry is most valuable).
 *   - Non-fatal: export failures never throw or block the agent run.
 */

import type { OTelExportRequest, OTelSpan, OTelAttribute, TelemetryConfig } from "./types.js";
import { redactString } from "../guardrails/redaction.js";

export function attr(key: string, value: string | number | boolean): OTelAttribute {
  if (typeof value === "string") return { key, value: { stringValue: value } };
  if (typeof value === "boolean") return { key, value: { boolValue: value } };
  if (Number.isInteger(value)) return { key, value: { intValue: String(value) } };
  return { key, value: { doubleValue: value as number } };
}

export function toNanos(ms: number): string {
  return String(ms * 1_000_000);
}

const MAX_BUFFER_SIZE = 1_000;
const MAX_RETRY_ATTEMPTS = 3;

export class OTelExporter {
  private config: TelemetryConfig;
  /** Spans that failed export and are queued for retry on next flush. */
  private retryBuffer: OTelSpan[] = [];
  private retryAttempt = 0;
  private nextRetryAt = 0;

  constructor(config: TelemetryConfig = {}) {
    this.config = config;
  }

  async exportSpans(spans: OTelSpan[]): Promise<void> {
    if (spans.length === 0 && this.retryBuffer.length === 0) return;

    const now = Date.now();

    // Redact PII from all span attributes before export
    const redactedSpans = spans.map((span) => ({
      ...span,
      attributes: span.attributes.map((attr) => ({
        key: attr.key,
        value: this.redactAttributeValue(attr.value),
      })),
    }));
    const redactedBuffer = this.retryBuffer.map((span) => ({
      ...span,
      attributes: span.attributes.map((attr) => ({
        key: attr.key,
        value: this.redactAttributeValue(attr.value),
      })),
    }));

    // Merge retry buffer with new spans (retry buffer goes first — it's older)
    const toExport = [...redactedBuffer, ...redactedSpans];
    this.retryBuffer = [];

    if (this.config.stdoutExport) {
      this.emitToStdout(toExport);
    }

    if (!this.config.endpoint) return;

    // Honour backoff: if we're still cooling down, re-buffer and return.
    if (now < this.nextRetryAt) {
      this.bufferForRetry(toExport);
      return;
    }

    try {
      await this.send(toExport);
      this.retryAttempt = 0; // reset on success
    } catch (err) {
      this.retryAttempt += 1;
      if (this.retryAttempt <= MAX_RETRY_ATTEMPTS) {
        // Exponential backoff: 2s, 4s, 8s
        const delayMs = Math.min(1_000 * 2 ** this.retryAttempt, 30_000);
        this.nextRetryAt = now + delayMs;
        this.bufferForRetry(toExport);
        process.stderr.write(
          `[chorus] OTel export failed (attempt ${this.retryAttempt}/${MAX_RETRY_ATTEMPTS}), ` +
          `retrying in ${delayMs}ms: ${String(err)}\n`,
        );
      } else {
        // Give up after MAX_RETRY_ATTEMPTS — drop spans to avoid unbounded growth.
        process.stderr.write(
          `[chorus] OTel export permanently failed after ${MAX_RETRY_ATTEMPTS} attempts. ` +
          `${toExport.length} spans dropped: ${String(err)}\n`,
        );
        this.retryAttempt = 0;
      }
    }
  }

  private bufferForRetry(spans: OTelSpan[]): void {
    this.retryBuffer = [...this.retryBuffer, ...spans];
    // Drop oldest if buffer overflows to prevent unbounded memory growth.
    if (this.retryBuffer.length > MAX_BUFFER_SIZE) {
      const dropped = this.retryBuffer.length - MAX_BUFFER_SIZE;
      this.retryBuffer = this.retryBuffer.slice(dropped);
      process.stderr.write(`[chorus] OTel retry buffer overflow — dropped ${dropped} oldest spans.\n`);
    }
  }

  private async send(spans: OTelSpan[]): Promise<void> {
    const payload: OTelExportRequest = {
      resourceSpans: [
        {
          resource: {
            attributes: [
              attr("service.name", this.config.serviceName ?? "chorus-engine"),
              attr("service.version", "1.0.0"),
              ...Object.entries(this.config.resourceAttributes ?? {}).map(([k, v]) => attr(k, v)),
            ],
          },
          scopeSpans: [
            {
              scope: { name: "chorus.swarm", version: "1.0.0" },
              spans,
            },
          ],
        },
      ],
    };

    const res = await fetch(this.config.endpoint!, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(`OTLP endpoint returned HTTP ${res.status}: ${res.statusText}`);
    }
  }

  private redactAttributeValue(value: OTelAttribute["value"]): OTelAttribute["value"] {
    if ("stringValue" in value && typeof value.stringValue === "string") {
      return { stringValue: redactString(value.stringValue) };
    }
    return value;
  }

  private emitToStdout(spans: OTelSpan[]): void {
    const payload: OTelExportRequest = {
      resourceSpans: [
        {
          resource: { attributes: [attr("service.name", this.config.serviceName ?? "chorus-engine")] },
          scopeSpans: [{ scope: { name: "chorus.swarm", version: "1.0.0" }, spans }],
        },
      ],
    };
    process.stdout.write(`[otel] ${JSON.stringify(payload)}\n`);
  }
}
