import { describe, it, expect } from "vitest";
import {
  EvalRunner,
  ExactMatchScorer,
  ContainsScorer,
  ToolCallScorer,
  ExperimentTracker,
  hashDataset,
  serializeEvalRun,
} from "../evals/index.js";

describe("Enhanced Evaluation Framework", () => {
  it("hashes datasets for versioning", () => {
    const cases = [
      { id: "c1", input: "hello", expected: "hi" },
      { id: "c2", input: "world", expected: "earth" },
    ];
    const hashA = hashDataset(cases);
    const hashB = hashDataset([...cases]);
    expect(hashA).toBe(hashB);
    expect(hashA.length).toBe(16);

    const hashC = hashDataset([...cases, { id: "c3", input: "foo", expected: "bar" }]);
    expect(hashC).not.toBe(hashA);
  });

  it("evaluates with multiple scorers and computes average", async () => {
    const runner = new EvalRunner({ passThreshold: 0.5 });
    runner.loadDataset([
      { id: "1", input: "hello", expected: "hello" },
      { id: "2", input: "world", expected: "world" },
    ]);

    const invoke = async (input: string) => input;
    const run = await runner.evaluate([new ExactMatchScorer(), new ContainsScorer()], invoke);

    expect(run.summary.total).toBe(2);
    expect(run.summary.passed).toBe(2);
    expect(run.summary.failed).toBe(0);
    expect(run.datasetVersion).toBeDefined();
    expect(run.datasetVersion!.length).toBe(16);
  });

  it("detects regressions against baseline", async () => {
    const baseline = {
      runId: "base",
      timestamp: 0,
      results: [
        { caseId: "1", input: "a", actual: "a", expected: "a", scores: { exact_match: 1.0 }, passed: true, latencyMs: 0, tokensUsed: 0 },
        { caseId: "2", input: "b", actual: "x", expected: "b", scores: { exact_match: 0.0 }, passed: false, latencyMs: 0, tokensUsed: 0 },
      ],
      summary: { total: 2, passed: 1, failed: 1, averageScore: 0.5, regressions: [] },
      regressed: false,
    };

    const runner = new EvalRunner({ baseline, regressionScorers: ["exact_match"] });
    runner.loadDataset([
      { id: "1", input: "a", expected: "a" },
      { id: "2", input: "b", expected: "b" },
    ]);

    // Regression: case 1 drops from 1.0 to 0.0
    const invoke = async (_input: string) => "wrong";
    const run = await runner.evaluate(new ExactMatchScorer(), invoke);

    expect(run.regressed).toBe(true);
    expect(run.summary.regressions.length).toBeGreaterThan(0);
    expect(run.summary.regressions[0]).toContain("exact_match");
  });

  it("scores tool calls correctly", () => {
    const scorer = new ToolCallScorer();
    scorer.actualToolCalls = [
      { name: "search", args: { query: "foo" } },
      { name: "calculate", args: { a: 1, b: 2 } },
    ];

    const score = scorer.scoreCalls([
      { name: "search", args: { query: "foo" } },
      { name: "calculate", args: { a: 1 } },
    ]);
    expect(score).toBe(1.0); // both match (partial args ok)

    const scoreMissing = scorer.scoreCalls([
      { name: "search", args: { query: "foo" } },
      { name: "missing", args: {} },
    ]);
    expect(scoreMissing).toBe(0.5);
  });

  it("tracks experiments and compares runs", () => {
    const tracker = new ExperimentTracker({ experimentName: "prompt-v2" });

    const runA = {
      runId: "a",
      timestamp: 0,
      results: [
        { caseId: "1", input: "a", actual: "a", expected: "a", scores: { s: 0.8 }, passed: true, latencyMs: 0, tokensUsed: 0 },
      ],
      summary: { total: 1, passed: 1, failed: 0, averageScore: 0.8, regressions: [] },
      regressed: false,
    };

    const runB = {
      runId: "b",
      timestamp: 1,
      results: [
        { caseId: "1", input: "a", actual: "a", expected: "a", scores: { s: 0.6 }, passed: true, latencyMs: 0, tokensUsed: 0 },
      ],
      summary: { total: 1, passed: 1, failed: 0, averageScore: 0.6, regressions: [] },
      regressed: false,
    };

    tracker.track(runA);
    tracker.track(runB);

    expect(tracker.latest()?.runId).toBe("b");

    const deltas = tracker.compare(runA, runB);
    expect(deltas.length).toBe(1);
    expect(deltas[0].delta).toBeCloseTo(-0.2, 10);

    const regressions = tracker.detectRegressions(runB, runA, 0.1);
    expect(regressions.length).toBe(1);
    expect(regressions[0].current).toBe(0.6);
    expect(regressions[0].previous).toBe(0.8);
  });

  it("serializes eval runs to JSON", () => {
    const run = {
      runId: "test",
      timestamp: 0,
      results: [],
      summary: { total: 0, passed: 0, failed: 0, averageScore: 0, regressions: [] },
      regressed: false,
    };
    const json = serializeEvalRun(run);
    expect(JSON.parse(json).runId).toBe("test");
  });
});
