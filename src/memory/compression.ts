/**
 * Semantic Memory Compression Engine
 *
 * Modern memory compression replacing naive RAKE (Rose et al., 2010) with
 * research-backed approaches from the 2025–2026 agent-memory literature:
 *
 *   • SimpleMem (Liu et al., ICLR 2026) — semantic structured compression,
 *     online semantic synthesis, memory atomization into fine-grained units.
 *   • Mem0 (Chhikara et al., 2025) — LLM-driven dynamic fact extraction
 *     from conversational inputs, storing structured facts with graph links.
 *   • ReadAgent (Lee et al., 2024) / LightMem (Fang et al., 2025) —
 *     embedding-based semantic clustering before summarization.
 *   • A-MEM (Xu et al., 2025) — networked notes with explicit related-atom
 *     links following the Zettelkasten method.
 *   • MemoryBank (Zhong et al., 2024) — forgetting-curve temporal dynamics.
 *
 * Design choices:
 *   - Pluggable compressor: LLM-driven (best quality) or embedding-only (local).
 *   - Atomic memory units (MemoryAtom) with dense embeddings for semantic ops.
 *   - Semantic synthesis: merge/update related atoms instead of duplicating.
 *   - Temporal decay: access-count tracking for future forgetting-curve pruning.
 *   - Verbatim grounding: every atom references source message indices.
 */

import type { ChatMessage } from "../llm/provider.js";
import type { LLMProvider } from "../llm/provider.js";
import type { SkillEmbedder } from "../skills/types.js";
import { cosineSimilarity } from "../skills/embedder.js";

// ═══════════════════════════════════════════════════════════════════════════════
// Data Models
// ═══════════════════════════════════════════════════════════════════════════════

/** An atomic unit of memory — fine-grained, embeddable, traceable. */
export interface MemoryAtom {
  /** Stable UUID for graph linking and deduplication. */
  id: string;
  /** The distilled fact, summary, or observation. */
  content: string;
  /** Memory category. */
  type: "fact" | "summary" | "observation" | "preference" | "procedure";
  /** Dense embedding vector for semantic comparison (optional until indexed). */
  embedding?: number[];
  /** Source message indices in the original conversation batch. */
  sources: number[];
  /** Confidence score (0–1). */
  confidence: number;
  /** Creation timestamp. */
  createdAt: number;
  /** Last update timestamp. */
  updatedAt: number;
  /** How many times this atom was retrieved (for forgetting-curve decay). */
  accessCount: number;
  /** Last retrieval timestamp. */
  lastAccessedAt: number;
  /** IDs of semantically related atoms (A-MEM graph links). */
  relatedAtomIds: string[];
  /** Free-form metadata (extractor-specific). */
  metadata: Record<string, unknown>;
}

/** A compressed archival chunk covering a range of messages. */
export interface ArchivalChunk {
  /** High-level summary of the covered message range. */
  summary: string;
  /** Top semantic keywords (derived from cluster centroids, not regex). */
  keywords: string[];
  /** Atomic memory units extracted from this range. */
  atoms: MemoryAtom[];
  /** Which message indices this chunk covers. */
  sourceRange: [number, number];
  /** Compression timestamp. */
  compressedAt: number;
}

/** Legacy alias for backward compatibility. */
export interface MemoryFact {
  key: string;
  value: string;
  sources: number[];
  confidence: number;
  extractedAt: number;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════════

function generateId(): string {
  return `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

function now(): number {
  return Date.now();
}

/** Simple local embedder wrapper that caches results per batch. */
async function embedBatch(texts: string[], embedder: SkillEmbedder): Promise<number[][]> {
  return Promise.all(texts.map((t) => embedder.embed(t)));
}

/** Agglomerative clustering by cosine-similarity threshold. */
function clusterByEmbedding(
  vectors: number[][],
  threshold = 0.72,
): number[][] {
  const n = vectors.length;
  if (n === 0) return [];

  // Build similarity matrix (upper triangular)
  const sim: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    sim[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const s = cosineSimilarity(vectors[i], vectors[j]);
      sim[i][j] = s;
      sim[j][i] = s;
    }
  }

  // Greedy agglomerative: start with each point as its own cluster,
  // repeatedly merge the pair with highest average intra-cluster sim
  // until no pair exceeds threshold.
  let clusters: number[][] = Array.from({ length: n }, (_, i) => [i]);

  while (true) {
    let bestI = -1;
    let bestJ = -1;
    let bestScore = -1;

    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        let sum = 0;
        let count = 0;
        for (const a of clusters[i]) {
          for (const b of clusters[j]) {
            sum += sim[a][b];
            count++;
          }
        }
        const avg = count > 0 ? sum / count : 0;
        if (avg > bestScore) {
          bestScore = avg;
          bestI = i;
          bestJ = j;
        }
      }
    }

    if (bestScore < threshold || bestI === -1) break;
    clusters[bestI] = clusters[bestI].concat(clusters[bestJ]);
    clusters.splice(bestJ, 1);
  }

  return clusters;
}

/** Extract topic keywords from a cluster using TF-IDF-like scoring. */
function extractClusterKeywords(
  texts: string[],
  topN = 5,
): string[] {
  if (texts.length === 0) return [];

  const stopWords = new Set([
    "a","an","the","and","or","but","in","on","at","to","for","of","with","by","from",
    "is","are","was","were","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","must","shall","can","need",
    "this","that","these","those","i","you","he","she","it","we","they",
    "me","him","her","us","them","my","your","his","its","our","their",
    "am","so","if","no","not","yes","ok","okay","um","uh","well",
  ]);

  // Document frequency
  const docFreq = new Map<string, number>();
  const docTokens: string[][] = [];

  for (const text of texts) {
    const tokens = text
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, " ")
      .split(/\s+/)
      .filter((t) => t.length > 2 && !stopWords.has(t));
    const unique = new Set(tokens);
    docTokens.push(tokens);
    for (const t of unique) {
      docFreq.set(t, (docFreq.get(t) ?? 0) + 1);
    }
  }

  const N = texts.length;
  const scores = new Map<string, number>();

  for (let i = 0; i < docTokens.length; i++) {
    const tf: Map<string, number> = new Map();
    for (const t of docTokens[i]) {
      tf.set(t, (tf.get(t) ?? 0) + 1);
    }
    for (const [term, count] of tf.entries()) {
      const idf = Math.log(N / (docFreq.get(term) ?? 1));
      scores.set(term, (scores.get(term) ?? 0) + count * idf);
    }
  }

  return Array.from(scores.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, topN)
    .map(([t]) => t);
}

/** Parse a JSON array of facts from LLM output, with graceful fallback. */
function parseExtractedFacts(jsonText: string): Array<{
  content: string;
  type: MemoryAtom["type"];
  confidence: number;
}> {
  try {
    // Try to extract JSON array from markdown code fences or raw text
    const match = jsonText.match(/\[[\s\S]*\]/);
    const raw = match ? match[0] : jsonText;
    const parsed = JSON.parse(raw) as Array<{
      content?: string;
      fact?: string;
      type?: string;
      confidence?: number;
    }>;

    return parsed
      .filter((p) => (p.content || p.fact))
      .map((p) => ({
        content: (p.content || p.fact || "").trim(),
        type: normalizeType(p.type),
        confidence: typeof p.confidence === "number" ? Math.max(0, Math.min(1, p.confidence)) : 0.85,
      }));
  } catch {
    // Fallback: treat each non-empty line as a fact
    return jsonText
      .split("\n")
      .map((l) => l.trim())
      .filter((l) => l.length > 10 && !l.startsWith("```"))
      .map((l) => ({
        content: l.replace(/^[-*\d.]+\s*/, ""),
        type: "fact" as const,
        confidence: 0.7,
      }));
  }
}

function normalizeType(t: string | undefined): MemoryAtom["type"] {
  switch (t?.toLowerCase()) {
    case "summary": return "summary";
    case "observation": return "observation";
    case "preference": return "preference";
    case "procedure": return "procedure";
    default: return "fact";
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Semantic Synthesizer — merges related atoms (SimpleMem-style)
// ═══════════════════════════════════════════════════════════════════════════════

export interface SynthesisOptions {
  /** Cosine-similarity threshold for considering two atoms duplicates. */
  mergeThreshold?: number;
  /** Threshold for linking as related (but not merging). */
  relateThreshold?: number;
  /** Max related links per atom. */
  maxRelated?: number;
}

/**
 * Merges newly extracted atoms into an existing knowledge base.
 *
 * Implements online semantic synthesis (SimpleMem):
 *   - If embedding similarity > mergeThreshold: update existing atom (combine
 *     sources, boost confidence, refresh timestamp).
 *   - If similarity > relateThreshold: add graph link between atoms (A-MEM).
 *   - Otherwise: append as new atom.
 */
export class SemanticSynthesizer {
  private opts: Required<SynthesisOptions>;

  constructor(opts: SynthesisOptions = {}) {
    this.opts = {
      mergeThreshold: opts.mergeThreshold ?? 0.88,
      relateThreshold: opts.relateThreshold ?? 0.72,
      maxRelated: opts.maxRelated ?? 5,
    };
  }

  /**
   * Integrate `incoming` atoms into `existing` knowledge base.
   * Returns the updated knowledge base (mutates in-place for efficiency).
   */
  synthesize(existing: MemoryAtom[], incoming: MemoryAtom[]): MemoryAtom[] {
    const result = existing;

    for (const atom of incoming) {
      if (!atom.embedding) {
        // Cannot synthesize without embeddings — append blindly
        result.push(atom);
        continue;
      }

      let merged = false;
      let bestMatch: MemoryAtom | null = null;
      let bestScore = -1;

      for (const existingAtom of result) {
        if (!existingAtom.embedding) continue;
        const sim = cosineSimilarity(atom.embedding, existingAtom.embedding);

        if (sim > this.opts.mergeThreshold && sim > bestScore) {
          bestScore = sim;
          bestMatch = existingAtom;
        }

        // Add related link if in relate range but below merge range
        if (sim >= this.opts.relateThreshold && sim < this.opts.mergeThreshold) {
          if (!existingAtom.relatedAtomIds.includes(atom.id)) {
            existingAtom.relatedAtomIds.push(atom.id);
            if (existingAtom.relatedAtomIds.length > this.opts.maxRelated) {
              existingAtom.relatedAtomIds.shift();
            }
          }
          if (!atom.relatedAtomIds.includes(existingAtom.id)) {
            atom.relatedAtomIds.push(existingAtom.id);
            if (atom.relatedAtomIds.length > this.opts.maxRelated) {
              atom.relatedAtomIds.shift();
            }
          }
        }
      }

      if (bestMatch) {
        // Merge: combine content heuristically, union sources, boost confidence
        bestMatch.content = this.mergeContent(bestMatch.content, atom.content);
        bestMatch.sources = [...new Set([...bestMatch.sources, ...atom.sources])];
        bestMatch.confidence = Math.min(0.99, bestMatch.confidence + 0.05);
        bestMatch.updatedAt = now();
        bestMatch.accessCount += 1;
        merged = true;
      }

      if (!merged) {
        result.push(atom);
      }
    }

    return result;
  }

  private mergeContent(a: string, b: string): string {
    // Simple merge: if one contains the other, keep the longer one
    if (a.includes(b)) return a;
    if (b.includes(a)) return b;
    // Otherwise concatenate with delimiter
    return `${a}; ${b}`;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Semantic Compressor Interface
// ═══════════════════════════════════════════════════════════════════════════════

export interface SemanticCompressor {
  /** Compress a batch of messages into atomic memory units. */
  compress(messages: ChatMessage[]): Promise<MemoryAtom[]>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// LLM-Driven Semantic Compressor (Mem0-style)
// ═══════════════════════════════════════════════════════════════════════════════

export interface LLMCompressorOptions {
  provider: LLMProvider;
  embedder: SkillEmbedder;
  model?: string;
  /** Max messages per extraction call (sliding window). */
  windowSize?: number;
  /** Overlap between windows. */
  windowOverlap?: number;
}

/**
 * Uses the LLM itself to extract structured facts from conversation batches.
 *
 * Inspired by Mem0 (Chhikara et al., 2025) and SimpleMem's semantic structured
 * compression. The LLM performs implicit semantic density gating — identifying
 * high-entropy spans and distilling them into atomic facts.
 */
export class LLMSemanticCompressor implements SemanticCompressor {
  private provider: LLMProvider;
  private embedder: SkillEmbedder;
  private model: string;
  private windowSize: number;
  private windowOverlap: number;

  constructor(opts: LLMCompressorOptions) {
    this.provider = opts.provider;
    this.embedder = opts.embedder;
    this.model = opts.model ?? "gpt-4o-mini";
    this.windowSize = opts.windowSize ?? 8;
    this.windowOverlap = opts.windowOverlap ?? 2;
  }

  async compress(messages: ChatMessage[]): Promise<MemoryAtom[]> {
    if (messages.length === 0) return [];

    // Sliding-window extraction to handle long batches
    const windows = this.slidingWindows(messages);
    const atoms: MemoryAtom[] = [];

    for (const { start, msgs } of windows) {
      const extracted = await this.extractFromWindow(msgs, start);
      atoms.push(...extracted);
    }

    // Embed all atoms for semantic operations
    const embeddings = await embedBatch(
      atoms.map((a) => a.content),
      this.embedder,
    );
    for (let i = 0; i < atoms.length; i++) {
      atoms[i].embedding = embeddings[i];
    }

    // Deduplicate via synthesis
    const synthesizer = new SemanticSynthesizer();
    return synthesizer.synthesize([], atoms);
  }

  private slidingWindows(messages: ChatMessage[]): Array<{ start: number; msgs: ChatMessage[] }> {
    const windows: Array<{ start: number; msgs: ChatMessage[] }> = [];
    let i = 0;
    while (i < messages.length) {
      const end = Math.min(i + this.windowSize, messages.length);
      windows.push({ start: i, msgs: messages.slice(i, end) });
      i += this.windowSize - this.windowOverlap;
      if (end === messages.length) break;
    }
    return windows;
  }

  private async extractFromWindow(msgs: ChatMessage[], startIndex: number): Promise<MemoryAtom[]> {
    const transcript = msgs
      .map((m, i) => `[${startIndex + i}] ${m.role}: ${m.content}`)
      .join("\n");

    const prompt =
      `You are a semantic memory extraction engine. Given a conversation transcript, extract atomic facts, preferences, observations, and procedures as a JSON array.\n\n` +
      `Rules:\n` +
      `- Each item must have: "content" (string), "type" (one of: fact, preference, observation, procedure, summary), "confidence" (0.0–1.0).\n` +
      `- Extract only salient, non-redundant information. Skip greetings, filler, and chit-chat.\n` +
      `- Use precise language. Replace pronouns with proper nouns where context permits.\n` +
      `- "confidence" reflects certainty: direct statements = 0.9+, inferred = 0.6–0.8, speculative = 0.5.\n` +
      `- Return ONLY a JSON array. No markdown fences, no explanations.\n\n` +
      `Transcript:\n${transcript}\n\n` +
      `Extracted memories:`;

    try {
      const result = await this.provider.generate({
        model: this.model,
        messages: [{ role: "user", content: prompt }],
      });

      const parsed = parseExtractedFacts(result.text);
      return parsed.map((p, i) => ({
        id: generateId(),
        content: p.content,
        type: p.type,
        sources: [startIndex + (i % msgs.length)], // Approximate source mapping
        confidence: p.confidence,
        createdAt: now(),
        updatedAt: now(),
        accessCount: 0,
        lastAccessedAt: 0,
        relatedAtomIds: [],
        metadata: { extractor: "llm", windowStart: startIndex },
      }));
    } catch {
      // Fallback: return a single summary atom
      return [{
        id: generateId(),
        content: `Conversation segment ${startIndex}–${startIndex + msgs.length - 1}`,
        type: "summary",
        sources: Array.from({ length: msgs.length }, (_, i) => startIndex + i),
        confidence: 0.5,
        createdAt: now(),
        updatedAt: now(),
        accessCount: 0,
        lastAccessedAt: 0,
        relatedAtomIds: [],
        metadata: { extractor: "llm-fallback", windowStart: startIndex },
      }];
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Embedding-Only Semantic Compressor (SimpleMem-style local pipeline)
// ═══════════════════════════════════════════════════════════════════════════════

export interface EmbeddingCompressorOptions {
  embedder: SkillEmbedder;
  /** Cosine similarity threshold for clustering messages. */
  clusterThreshold?: number;
  /** Max atoms to extract per cluster. */
  maxAtomsPerCluster?: number;
}

/**
 * Local, deterministic compression using sentence embeddings.
 *
 * Inspired by ReadAgent / LightMem semantic clustering + SimpleMem atomization.
 * No LLM API calls required — operates entirely via dense vector similarity.
 *
 * Pipeline:
 *   1. Embed each message.
 *   2. Agglomerative clustering by cosine similarity.
 *   3. Per cluster: extract TF-IDF keywords, generate summary atom,
 *      extract entity-predicate patterns via lightweight heuristics.
 *   4. Embed all atoms and deduplicate via synthesis.
 */
export class EmbeddingSemanticCompressor implements SemanticCompressor {
  private embedder: SkillEmbedder;
  private clusterThreshold: number;
  private maxAtomsPerCluster: number;

  constructor(opts: EmbeddingCompressorOptions) {
    this.embedder = opts.embedder;
    this.clusterThreshold = opts.clusterThreshold ?? 0.72;
    this.maxAtomsPerCluster = opts.maxAtomsPerCluster ?? 4;
  }

  async compress(messages: ChatMessage[]): Promise<MemoryAtom[]> {
    if (messages.length === 0) return [];

    // 1. Embed all messages
    const embeddings = await embedBatch(
      messages.map((m) => m.content),
      this.embedder,
    );

    // 2. Cluster by embedding similarity
    const clusters = clusterByEmbedding(embeddings, this.clusterThreshold);

    // 3. Extract atoms per cluster
    const atoms: MemoryAtom[] = [];
    for (const indices of clusters) {
      const clusterAtoms = await this.extractFromCluster(messages, indices);
      atoms.push(...clusterAtoms.slice(0, this.maxAtomsPerCluster));
    }

    // Handle outliers (messages in no cluster above threshold)
    const clusteredIndices = new Set(clusters.flat());
    for (let i = 0; i < messages.length; i++) {
      if (!clusteredIndices.has(i)) {
        atoms.push(this.messageToAtom(messages[i], i));
      }
    }

    // 4. Embed all atoms
    const atomEmbeddings = await embedBatch(
      atoms.map((a) => a.content),
      this.embedder,
    );
    for (let i = 0; i < atoms.length; i++) {
      atoms[i].embedding = atomEmbeddings[i];
    }

    // 5. Semantic synthesis for deduplication
    const synthesizer = new SemanticSynthesizer();
    return synthesizer.synthesize([], atoms);
  }

  private async extractFromCluster(
    messages: ChatMessage[],
    indices: number[],
  ): Promise<MemoryAtom[]> {
    const texts = indices.map((i) => messages[i].content);
    const keywords = extractClusterKeywords(texts, 5);

    // Summary atom for the cluster
    const summaryAtom: MemoryAtom = {
      id: generateId(),
      content: `Discussion about ${keywords.slice(0, 3).join(", ") || "general topics"}`,
      type: "summary",
      sources: indices,
      confidence: 0.6 + Math.min(0.3, indices.length * 0.02),
      createdAt: now(),
      updatedAt: now(),
      accessCount: 0,
      lastAccessedAt: 0,
      relatedAtomIds: [],
      metadata: { keywords, clusterSize: indices.length },
    };

    // Extract predicate patterns from combined text
    const combined = texts.join("\n");
    const facts = extractPredicatePatterns(combined, indices);

    return [summaryAtom, ...facts];
  }

  private messageToAtom(msg: ChatMessage, index: number): MemoryAtom {
    // Single-message atom: use first sentence or truncated content
    const content = msg.content.split(/\.(\s|$)/)[0]?.trim() || msg.content.slice(0, 120);
    return {
      id: generateId(),
      content,
      type: "observation",
      sources: [index],
      confidence: 0.55,
      createdAt: now(),
      updatedAt: now(),
      accessCount: 0,
      lastAccessedAt: 0,
      relatedAtomIds: [],
      metadata: { role: msg.role, singleton: true },
    };
  }
}

/** Lightweight predicate extraction using modern NLP patterns (not 2010 RAKE). */
function extractPredicatePatterns(
  text: string,
  sources: number[],
): MemoryAtom[] {
  const atoms: MemoryAtom[] = [];

  // Entity–attribute patterns (far more robust than RAKE triples)
  // Pattern: "X prefers/needs/requires/uses Y"
  const patterns: Array<{ regex: RegExp; type: MemoryAtom["type"]; relation: string }> = [
    { regex: /\b([A-Z][A-Za-z0-9\s]{1,30}?)\s+prefer(?:s|red|ring)\s+([A-Za-z0-9\s,]{3,60}?)(?:\.|,|;|$)/gi, type: "preference", relation: "prefers" },
    { regex: /\b([A-Z][A-Za-z0-9\s]{1,30}?)\s+need(?:s|ed|ing)\s+([A-Za-z0-9\s,]{3,60}?)(?:\.|,|;|$)/gi, type: "fact", relation: "needs" },
    { regex: /\b([A-Z][A-Za-z0-9\s]{1,30}?)\s+require(?:s|d|ing)\s+([A-Za-z0-9\s,]{3,60}?)(?:\.|,|;|$)/gi, type: "fact", relation: "requires" },
    { regex: /\b([A-Z][A-Za-z0-9\s]{1,30}?)\s+use(?:s|d|ing)\s+([A-Za-z0-9\s,]{3,60}?)(?:\.|,|;|$)/gi, type: "fact", relation: "uses" },
    { regex: /\b([A-Z][A-Za-z0-9\s]{1,30}?)\s+is\s+(?:a |an )?([A-Za-z0-9\s,]{3,60}?)(?:\.|,|;|$)/gi, type: "fact", relation: "is" },
    { regex: /\b([A-Z][A-Za-z0-9\s]{1,30}?)\s+has\s+([A-Za-z0-9\s,]{3,60}?)(?:\.|,|;|$)/gi, type: "fact", relation: "has" },
  ];

  for (const { regex, type, relation } of patterns) {
    let match: RegExpExecArray | null;
    const seen = new Set<string>();
    while ((match = regex.exec(text)) !== null) {
      const subject = match[1].trim();
      const object = match[2].trim();
      const key = `${subject}|${relation}|${object}`.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);

      atoms.push({
        id: generateId(),
        content: `${subject} ${relation} ${object}`,
        type,
        sources,
        confidence: 0.72,
        createdAt: now(),
        updatedAt: now(),
        accessCount: 0,
        lastAccessedAt: 0,
        relatedAtomIds: [],
        metadata: { relation, subject, object },
      });
    }
  }

  return atoms;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Archival Chunk Builder
// ═══════════════════════════════════════════════════════════════════════════════

export interface ChunkBuilderOptions {
  compressor: SemanticCompressor;
}

/** Build an ArchivalChunk from a message batch using any compressor. */
export async function buildArchivalChunk(
  messages: ChatMessage[],
  opts: ChunkBuilderOptions,
  startIndex = 0,
): Promise<ArchivalChunk> {
  const atoms = await opts.compressor.compress(messages);

  // Derive keywords from atom content via simple frequency
  const allWords = atoms
    .flatMap((a) => a.content.split(/\W+/))
    .map((w) => w.toLowerCase())
    .filter((w) => w.length > 3);
  const freq = new Map<string, number>();
  for (const w of allWords) freq.set(w, (freq.get(w) ?? 0) + 1);
  const keywords = Array.from(freq.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([w]) => w);

  const summary = atoms.find((a) => a.type === "summary")?.content
    ?? (keywords.length > 0
      ? `Conversation covering ${keywords.slice(0, 4).join(", ")}`
      : "General conversation");

  return {
    summary,
    keywords,
    atoms,
    sourceRange: [startIndex, startIndex + messages.length - 1],
    compressedAt: now(),
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Legacy API compatibility (deprecated but preserved)
// ═══════════════════════════════════════════════════════════════════════════════

let defaultCompressor: SemanticCompressor | null = null;

/** Set the default compressor for legacy sync-style calls. */
export function setDefaultCompressor(compressor: SemanticCompressor): void {
  defaultCompressor = compressor;
}

/**
 * Legacy sync-style fact extraction.
 *
 * @deprecated Use {@link SemanticCompressor.compress} with an explicit compressor.
 */
export async function compressMessagesToFacts(messages: ChatMessage[]): Promise<MemoryFact[]> {
  if (!defaultCompressor) {
    throw new Error(
      "No default compressor set. Call setDefaultCompressor() or use SemanticCompressor directly.",
    );
  }
  const atoms = await defaultCompressor.compress(messages);
  return atoms.map((a) => ({
    key: `${a.type}_${a.id}`,
    value: a.content,
    sources: a.sources,
    confidence: a.confidence,
    extractedAt: a.createdAt,
  }));
}

/**
 * Legacy sync-style chunk extraction.
 *
 * @deprecated Use {@link buildArchivalChunk} with an explicit compressor.
 */
export async function compressMessagesToChunk(
  messages: ChatMessage[],
  startIndex = 0,
): Promise<ArchivalChunk> {
  if (!defaultCompressor) {
    throw new Error(
      "No default compressor set. Call setDefaultCompressor() or use buildArchivalChunk directly.",
    );
  }
  return buildArchivalChunk(messages, { compressor: defaultCompressor }, startIndex);
}
