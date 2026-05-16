/**
 * Adaptive Guardrail Thresholds
 *
 * Self-improving guardrail system that tracks false-positive/negative feedback
 * and adjusts thresholds over time. Inspired by:
 *   - Fiddler Trust Models (2025) — continuous monitoring and threshold tuning
 *   - Galileo Autotune (2026) — auto-improves metric accuracy from 2–5 examples
 *   - Symbolic Guardrails (2026) — deterministic policy with feedback-driven refinement
 *
 * Problem: Static thresholds produce either too many false positives (blocking
 * legitimate traffic) or too many false negatives (letting attacks through).
 *
 * Solution: Each guardrail maintains a running accuracy score. When human
 * operators provide feedback ("this was a false positive"), the system nudges
 * the threshold in the direction that would have prevented the error.
 *
 * Design:
 *   - Per-guardrail accuracy tracking (TP, FP, TN, FN counts)
 *   - Threshold nudging via exponential moving average
 *   - Confidence scoring for ML-tier detections
 *   - Exportable metrics for dashboarding
 */

export interface GuardrailFeedback {
  guardrail: string;
  /** What the guardrail predicted. */
  predictedViolation: boolean;
  /** What actually happened (human-verified). */
  actualViolation: boolean;
  /** Confidence score of the prediction (0–1). */
  confidence?: number;
  /** Timestamp. */
  ts?: number;
}

export interface AdaptiveThresholdState {
  threshold: number;
  totalChecks: number;
  truePositives: number;
  falsePositives: number;
  trueNegatives: number;
  falseNegatives: number;
  /** Recent accuracy (0–1). */
  rollingAccuracy: number;
  /** Recent false positive rate (0–1). */
  rollingFPR: number;
  /** Target FPR. Default: 0.05 (5%). */
  targetFPR: number;
}

export class AdaptiveThreshold {
  private state: AdaptiveThresholdState;
  private alpha: number; // EMA smoothing factor
  private minThreshold: number;
  private maxThreshold: number;
  private nudgeSize: number;

  constructor(
    initialThreshold = 0.7,
    opts: {
      alpha?: number;
      minThreshold?: number;
      maxThreshold?: number;
      nudgeSize?: number;
      targetFPR?: number;
    } = {},
  ) {
    this.alpha = opts.alpha ?? 0.1;
    this.minThreshold = opts.minThreshold ?? 0.3;
    this.maxThreshold = opts.maxThreshold ?? 0.95;
    this.nudgeSize = opts.nudgeSize ?? 0.02;
    this.state = {
      threshold: initialThreshold,
      totalChecks: 0,
      truePositives: 0,
      falsePositives: 0,
      trueNegatives: 0,
      falseNegatives: 0,
      rollingAccuracy: 0.5,
      rollingFPR: 0.05,
      targetFPR: opts.targetFPR ?? 0.05,
    };
  }

  get threshold(): number {
    return this.state.threshold;
  }

  get stats(): Readonly<AdaptiveThresholdState> {
    return { ...this.state };
  }

  /** Evaluate a prediction against a confidence score. */
  shouldTrigger(confidence: number): boolean {
    this.state.totalChecks++;
    return confidence >= this.state.threshold;
  }

  /** Record feedback to adjust the threshold. */
  recordFeedback(feedback: GuardrailFeedback): void {
    const { predictedViolation, actualViolation, confidence = 0.5 } = feedback;

    this.state.totalChecks++;

    // Update confusion matrix
    if (predictedViolation && actualViolation) this.state.truePositives++;
    if (predictedViolation && !actualViolation) this.state.falsePositives++;
    if (!predictedViolation && !actualViolation) this.state.trueNegatives++;
    if (!predictedViolation && actualViolation) this.state.falseNegatives++;

    // Update rolling accuracy via EMA
    const correct = predictedViolation === actualViolation ? 1 : 0;
    this.state.rollingAccuracy =
      this.alpha * correct + (1 - this.alpha) * this.state.rollingAccuracy;

    // Update rolling FPR
    const fpRate = this.state.falsePositives + this.state.trueNegatives > 0
      ? this.state.falsePositives / (this.state.falsePositives + this.state.trueNegatives)
      : 0;
    this.state.rollingFPR =
      this.alpha * fpRate + (1 - this.alpha) * this.state.rollingFPR;

    // Nudge threshold based on error type
    if (!predictedViolation && actualViolation) {
      // False negative: threshold was too high → lower it
      this.state.threshold = Math.max(
        this.minThreshold,
        this.state.threshold - this.nudgeSize,
      );
    } else if (predictedViolation && !actualViolation) {
      // False positive: threshold was too low → raise it
      this.state.threshold = Math.min(
        this.maxThreshold,
        this.state.threshold + this.nudgeSize,
      );
    }

    // If FPR is above target, aggressively raise threshold
    if (this.state.rollingFPR > this.state.targetFPR * 1.5) {
      this.state.threshold = Math.min(
        this.maxThreshold,
        this.state.threshold + this.nudgeSize * 2,
      );
    }
  }

  /** Serialize state for persistence. */
  serialize(): string {
    return JSON.stringify(this.state);
  }

  /** Deserialize state from persistence. */
  static deserialize(data: string): AdaptiveThreshold {
    const parsed = JSON.parse(data) as AdaptiveThresholdState;
    const at = new AdaptiveThreshold(parsed.threshold);
    at.state = { ...parsed };
    return at;
  }
}

/**
 * Multi-guardrail adaptive threshold manager.
 *
 * Maintains a separate adaptive threshold per guardrail, identified by name.
 */
export class AdaptiveThresholdManager {
  private thresholds = new Map<string, AdaptiveThreshold>();
  private defaults: ConstructorParameters<typeof AdaptiveThreshold>[1];

  constructor(defaults: ConstructorParameters<typeof AdaptiveThreshold>[1] = {}) {
    this.defaults = defaults;
  }

  getThreshold(guardrailName: string): AdaptiveThreshold {
    let t = this.thresholds.get(guardrailName);
    if (!t) {
      t = new AdaptiveThreshold(0.7, this.defaults);
      this.thresholds.set(guardrailName, t);
    }
    return t;
  }

  record(feedback: GuardrailFeedback): void {
    this.getThreshold(feedback.guardrail).recordFeedback(feedback);
  }

  shouldTrigger(guardrailName: string, confidence: number): boolean {
    return this.getThreshold(guardrailName).shouldTrigger(confidence);
  }

  getAllStats(): Record<string, Readonly<AdaptiveThresholdState>> {
    const out: Record<string, Readonly<AdaptiveThresholdState>> = {};
    for (const [name, t] of this.thresholds) {
      out[name] = t.stats;
    }
    return out;
  }

  serialize(): string {
    const obj: Record<string, string> = {};
    for (const [name, t] of this.thresholds) {
      obj[name] = t.serialize();
    }
    return JSON.stringify(obj);
  }

  static deserialize(data: string, defaults?: ConstructorParameters<typeof AdaptiveThreshold>[1]): AdaptiveThresholdManager {
    const parsed = JSON.parse(data) as Record<string, string>;
    const mgr = new AdaptiveThresholdManager(defaults);
    for (const [name, stateData] of Object.entries(parsed)) {
      mgr.thresholds.set(name, AdaptiveThreshold.deserialize(stateData));
    }
    return mgr;
  }
}
