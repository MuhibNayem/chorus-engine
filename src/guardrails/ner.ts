/**
 * Neural Named Entity Recognition for PII Detection
 *
 * Replaces regex-only PII detection with transformer-based NER that understands
 * context — catching names, addresses, organizations, and implicit identifiers
 * that regex patterns miss.
 *
 * Architecture:
 *   • TransformersNERDetector — uses @huggingface/transformers pipeline
 *     for token-classification with aggregation. Supports any NER model.
 *   • KeywordNERFallback — deterministic keyword + pattern matching for
 *     environments where model loading is undesirable (CI, edge, tests).
 *   • HybridNERDetector — tries transformer first, falls back to keyword
 *     on model failure. Matches the Presidio/GLiNER hybrid approach.
 *
 * Production deployments should use a lightweight NER model
 * (e.g., dslim/bert-base-NER or urchade/gliner_multi_pii-v1).
 * The fallback is acceptable for tests and offline mode but should NOT be
 * used in production where accuracy matters.
 *
 * References:
 *   - Presidio (Microsoft) — spaCy NER + regex recognizers hybrid
 *   - GLiNER (2025) — zero-shot NER with label prompting
 *   - GLiNER Guard (2026) — unified encoder for safety + PII
 */

import { pipeline, type TokenClassificationPipeline, env } from "@huggingface/transformers";
import * as os from "os";
import * as path from "path";

function chorusHome(): string {
  return process.env.CHORUS_HOME_DIR ?? path.join(os.homedir(), ".chorus");
}

function getModelCacheDir(): string {
  return process.env.CHORUS_MODELS_DIR ?? path.join(chorusHome(), "models");
}

function configureTransformersCache(): void {
  env.cacheDir = getModelCacheDir();
  env.allowLocalModels = true;
  env.allowRemoteModels = true;
  env.useFSCache = true;
}

/** A detected entity span with position and confidence. */
export interface EntitySpan {
  text: string;
  label: string;
  start: number;
  end: number;
  confidence: number;
}

/** Abstract NER detector interface. */
export interface NERDetector {
  detect(text: string): Promise<EntitySpan[]>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Transformers-based NER (production path)
// ═══════════════════════════════════════════════════════════════════════════════

const NER_MODEL_DEFAULT = process.env.CHORUS_NER_MODEL ?? "dslim/bert-base-NER";
const extractorPromises = new Map<string, Promise<TokenClassificationPipeline>>();

/**
 * Production NER detector using Hugging Face transformers.
 *
 * Runs token classification with simple aggregation to merge sub-word tokens
 * into coherent entity spans. Returns PER, LOC, ORG, MISC by default;
 * compatible with any token-classification model.
 */
export class TransformersNERDetector implements NERDetector {
  readonly modelId: string;
  private extractorPromise: Promise<TokenClassificationPipeline> | null = null;
  private warned = false;

  constructor(modelId = NER_MODEL_DEFAULT) {
    this.modelId = modelId;
  }

  private getExtractor(): Promise<TokenClassificationPipeline> {
    const shared = extractorPromises.get(this.modelId);
    if (shared) {
      this.extractorPromise = shared;
      return shared;
    }
    if (!this.extractorPromise) {
      configureTransformersCache();
      this.extractorPromise = pipeline("token-classification", this.modelId);
      extractorPromises.set(this.modelId, this.extractorPromise);
    }
    return this.extractorPromise;
  }

  async detect(text: string): Promise<EntitySpan[]> {
    try {
      const extractor = await this.getExtractor();
      const results = await extractor(text, { aggregation_strategy: "simple" });
      const items = Array.isArray(results) ? results : [results];

      return items
        .filter((r) => r && typeof r.word === "string")
        .map((r) => ({
          text: r.word as string,
          label: (r.entity_group ?? r.entity) as string,
          start: (r.start as number) ?? 0,
          end: (r.end as number) ?? 0,
          confidence: (r.score as number) ?? 0.5,
        }));
    } catch (error) {
      if (!this.warned) {
        this.warned = true;
        process.stderr.write(
          `Chorus NER model "${this.modelId}" unavailable: ${error instanceof Error ? error.message : String(error)}\n`,
        );
      }
      return [];
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Keyword-based NER Fallback (deterministic, no model download)
// ═══════════════════════════════════════════════════════════════════════════════

/** Lightweight deterministic NER using keyword lists and heuristic patterns. */
export class KeywordNERFallback implements NERDetector {
  private personIndicators = new Set([
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "lord", "lady",
  ]);

  private locationIndicators = new Set([
    "street", "avenue", "road", "boulevard", "lane", "drive", "way",
    "city", "town", "village", "county", "state", "country", "zip", "postal",
  ]);

  private orgIndicators = new Set([
    "inc", "corp", "ltd", "llc", "limited", "company", "corporation",
    "foundation", "institute", "university", "college", "bank", "group",
  ]);

  private emailRegex = /[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g;
  private phoneRegex = /\+?\d{1,3}[\s\-]?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}/g;
  private urlRegex = /https?:\/\/[^\s]+/g;

  async detect(text: string): Promise<EntitySpan[]> {
    const spans: EntitySpan[] = [];

    // Email detection
    let match: RegExpExecArray | null;
    while ((match = this.emailRegex.exec(text)) !== null) {
      spans.push({
        text: match[0],
        label: "EMAIL",
        start: match.index,
        end: match.index + match[0].length,
        confidence: 0.95,
      });
    }

    // Phone detection
    this.phoneRegex.lastIndex = 0;
    while ((match = this.phoneRegex.exec(text)) !== null) {
      const digits = match[0].replace(/\D/g, "");
      if (digits.length >= 7) {
        spans.push({
          text: match[0],
          label: "PHONE",
          start: match.index,
          end: match.index + match[0].length,
          confidence: 0.85,
        });
      }
    }

    // URL detection
    this.urlRegex.lastIndex = 0;
    while ((match = this.urlRegex.exec(text)) !== null) {
      spans.push({
        text: match[0],
        label: "URL",
        start: match.index,
        end: match.index + match[0].length,
        confidence: 0.95,
      });
    }

    // Heuristic word-level detection
    const tokens = text.split(/(\s+)/);
    let pos = 0;
    const capitalizedWords: Array<{ word: string; start: number; end: number }> = [];

    for (const token of tokens) {
      if (/^[A-Z][a-z]+$/.test(token) && token.length > 2) {
        capitalizedWords.push({ word: token, start: pos, end: pos + token.length });
      }
      pos += token.length;
    }

    // Merge adjacent capitalized words as potential names/org/locations
    let i = 0;
    while (i < capitalizedWords.length) {
      let j = i + 1;
      while (j < capitalizedWords.length && capitalizedWords[j].start - capitalizedWords[j - 1].end <= 1) {
        j++;
      }
      const chunk = capitalizedWords.slice(i, j);
      const chunkText = chunk.map((c) => c.word).join(" ");
      const chunkStart = chunk[0].start;
      const chunkEnd = chunk[chunk.length - 1].end;

      const lowerText = chunkText.toLowerCase();
      let label = "MISC";
      let confidence = 0.55;

      if (this.personIndicators.has(lowerText.split(" ")[0].toLowerCase())) {
        label = "PER";
        confidence = 0.7;
      } else if (this.locationIndicators.has(lowerText.split(" ").pop()!.toLowerCase())) {
        label = "LOC";
        confidence = 0.65;
      } else if (this.orgIndicators.has(lowerText.split(" ").pop()!.toLowerCase())) {
        label = "ORG";
        confidence = 0.65;
      } else if (chunk.length >= 2) {
        // Multi-word capitalized phrase — likely a named entity
        label = "MISC";
        confidence = 0.55;
      }

      if (chunk.length >= 2 || label !== "MISC") {
        spans.push({ text: chunkText, label, start: chunkStart, end: chunkEnd, confidence });
      }

      i = j;
    }

    // Deduplicate overlapping spans (prefer higher confidence)
    spans.sort((a, b) => a.start - b.start);
    const deduped: EntitySpan[] = [];
    for (const span of spans) {
      const overlap = deduped.find((d) => !(span.end <= d.start || span.start >= d.end));
      if (!overlap) {
        deduped.push(span);
      } else if (span.confidence > overlap.confidence) {
        const idx = deduped.indexOf(overlap);
        deduped[idx] = span;
      }
    }

    return deduped;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Hybrid NER — tries transformer, falls back to keyword
// ═══════════════════════════════════════════════════════════════════════════════

export class HybridNERDetector implements NERDetector {
  private primary: TransformersNERDetector;
  private fallback: KeywordNERFallback;
  private useFallback: boolean;

  constructor(modelId?: string, forceFallback = false) {
    this.primary = new TransformersNERDetector(modelId);
    this.fallback = new KeywordNERFallback();
    this.useFallback = forceFallback || process.env.CHORUS_NER_FALLBACK === "1";
  }

  async detect(text: string): Promise<EntitySpan[]> {
    if (this.useFallback) {
      return this.fallback.detect(text);
    }
    const primaryResult = await this.primary.detect(text);
    if (primaryResult.length > 0) {
      return primaryResult;
    }
    // If primary returns empty (possible model failure), try fallback
    return this.fallback.detect(text);
  }
}

/** Create the best available NER detector. */
export function createNERDetector(modelId?: string): NERDetector {
  if (process.env.CHORUS_NER_FALLBACK === "1") {
    return new KeywordNERFallback();
  }
  return new HybridNERDetector(modelId);
}
