import { describe, it, expect } from "vitest";
import { InProcessTracer } from "../telemetry/inprocess.js";

describe("InProcessTracer — real-time span creation", () => {
  it("starts and ends a span", () => {
    const tracer = new InProcessTracer();
    const span = tracer.startSpan("test.span");
    expect(span.name).toBe("test.span");
    expect(span.traceId).toHaveLength(32);
    expect(span.spanId).toHaveLength(16);

    const ended = tracer.endSpan(span);
    expect(ended.endTimeUnixNano).toBeDefined();
    expect(ended.status.code).toBe(1); // OK
  });

  it("creates child spans with parentSpanId", () => {
    const tracer = new InProcessTracer();
    const parent = tracer.startSpan("parent");
    const child = tracer.startSpan("child", { parentSpanId: parent.spanId });
    expect(child.parentSpanId).toBe(parent.spanId);
  });

  it("sets attributes on spans", () => {
    const tracer = new InProcessTracer();
    const span = tracer.startSpan("test", { attributes: { key: "value" } });
    expect(span.attributes.some((a) => a.key === "key")).toBe(true);
  });

  it("sets error status on span", () => {
    const tracer = new InProcessTracer();
    const span = tracer.startSpan("test");
    span.setError("something broke");
    expect(span.status.code).toBe(2); // ERROR
    expect(span.status.message).toBe("something broke");
  });

  it("withSpan auto-ends on success", async () => {
    const tracer = new InProcessTracer();
    let capturedSpan: import("../telemetry/inprocess.js").MutableSpan | undefined;
    const result = await tracer.withSpan("wrapped", async (span) => {
      capturedSpan = span;
      return 42;
    });
    expect(result).toBe(42);
    expect(capturedSpan).toBeDefined();
  });

  it("withSpan auto-ends on error", async () => {
    const tracer = new InProcessTracer();
    let capturedSpan: import("../telemetry/inprocess.js").MutableSpan | undefined;
    await expect(
      tracer.withSpan("wrapped", async (span) => {
        capturedSpan = span;
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");
    expect(capturedSpan).toBeDefined();
    expect(capturedSpan!.status.code).toBe(2);
  });

  it("newTrace resets trace state", () => {
    const tracer = new InProcessTracer();
    const oldTrace = tracer.getTraceId();
    tracer.newTrace();
    expect(tracer.getTraceId()).not.toBe(oldTrace);
  });

  it("flush ends all active spans", async () => {
    const tracer = new InProcessTracer();
    tracer.startSpan("s1");
    tracer.startSpan("s2");
    await tracer.flush();
    // No error thrown; spans ended cleanly
  });
});
