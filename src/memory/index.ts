/**
 * Tiered Memory System
 *
 * Four-layer memory architecture inspired by Mastra, CrewAI, and SimpleMem:
 *   L1: Message History     — raw conversation turns (short-term)
 *   L2: Working Memory      — compressed atomic facts (mid-term)
 *   L3: Semantic Recall     — vector-based RAG for cross-session retrieval
 *   L4: Observational       — extracted facts from tool outputs (long-term)
 *
 * Design:
 *   - Pluggable backends: in-memory (default), disk, vector DB
 *   - Pluggable compressor: LLM-driven or embedding-only (SimpleMem-style)
 *   - Scoped per user/thread for multi-tenant isolation
 *   - Automatic compression when context window pressure exceeds threshold
 *   - Semantic synthesis prevents redundant facts (online deduplication)
 *   - Integrates with agent loop via middleware
 */

import type { ChatMessage } from "../llm/provider.js";
import type { SkillEmbedder } from "../skills/types.js";
import type { MemoryAtom, ArchivalChunk, SemanticCompressor, MemoryFact } from "./compression.js";
import {
  EmbeddingSemanticCompressor,
  SemanticSynthesizer,
  buildArchivalChunk,
  compressMessagesToFacts,
  compressMessagesToChunk,
  setDefaultCompressor,
} from "./compression.js";

export interface MemoryBackend {
  get<T>(scope: string, key: string): Promise<T | undefined>;
  set<T>(scope: string, key: string, value: T, ttlMs?: number): Promise<void>;
  delete(scope: string, key: string): Promise<void>;
  list(scope: string, prefix?: string): Promise<string[]>;
  search(scope: string, query: string, limit?: number): Promise<Array<{ key: string; score: number }>>;
}

/** In-memory backend with TTL support */
export class InMemoryBackend implements MemoryBackend {
  private store = new Map<string, { value: unknown; expiresAt?: number }>();

  private key(scope: string, k: string): string {
    return `${scope}::${k}`;
  }

  async get<T>(scope: string, key: string): Promise<T | undefined> {
    const entry = this.store.get(this.key(scope, key));
    if (!entry) return undefined;
    if (entry.expiresAt && entry.expiresAt < Date.now()) {
      this.store.delete(this.key(scope, key));
      return undefined;
    }
    return entry.value as T;
  }

  async set<T>(scope: string, key: string, value: T, ttlMs?: number): Promise<void> {
    this.store.set(this.key(scope, key), {
      value,
      expiresAt: ttlMs ? Date.now() + ttlMs : undefined,
    });
  }

  async delete(scope: string, key: string): Promise<void> {
    this.store.delete(this.key(scope, key));
  }

  async list(scope: string, prefix?: string): Promise<string[]> {
    const prefixStr = `${scope}::${prefix ?? ""}`;
    const keys: string[] = [];
    for (const k of this.store.keys()) {
      if (k.startsWith(prefixStr)) {
        keys.push(k.slice(prefixStr.length));
      }
    }
    return keys;
  }

  async search(_scope: string, _query: string, _limit?: number): Promise<Array<{ key: string; score: number }>> {
    // In-memory backend does not support semantic search; use VectorBackend instead
    return [];
  }
}

/** Disk-backed backend using JSON files */
export class DiskBackend implements MemoryBackend {
  constructor(private dir: string) {}

  private path(scope: string, key: string): string {
    return `${this.dir}/${scope}/${key}.json`;
  }

  async get<T>(scope: string, key: string): Promise<T | undefined> {
    try {
      const fs = await import("fs");
      const data = fs.readFileSync(this.path(scope, key), "utf-8");
      const parsed = JSON.parse(data) as { value: T; expiresAt?: number };
      if (parsed.expiresAt && parsed.expiresAt < Date.now()) {
        fs.unlinkSync(this.path(scope, key));
        return undefined;
      }
      return parsed.value;
    } catch {
      return undefined;
    }
  }

  async set<T>(scope: string, key: string, value: T, ttlMs?: number): Promise<void> {
    const fs = await import("fs");
    const path = this.path(scope, key);
    fs.mkdirSync(path.substring(0, path.lastIndexOf("/")), { recursive: true });
    fs.writeFileSync(path, JSON.stringify({ value, expiresAt: ttlMs ? Date.now() + ttlMs : undefined }));
  }

  async delete(scope: string, key: string): Promise<void> {
    try {
      const fs = await import("fs");
      fs.unlinkSync(this.path(scope, key));
    } catch { /* ignore */ }
  }

  async list(scope: string, prefix?: string): Promise<string[]> {
    try {
      const fs = await import("fs");
      const dir = `${this.dir}/${scope}`;
      if (!fs.existsSync(dir)) return [];
      const files = fs.readdirSync(dir);
      return files
        .filter((f) => f.endsWith(".json") && (!prefix || f.startsWith(prefix)))
        .map((f) => f.slice(0, -5));
    } catch {
      return [];
    }
  }

  async search(): Promise<Array<{ key: string; score: number }>> {
    return [];
  }
}

/** Vector backend for semantic search */
export class VectorBackend implements MemoryBackend {
  private vectors = new Map<string, number[]>();

  constructor(
    private embedder: SkillEmbedder,
    private base?: MemoryBackend,
  ) {}

  async get<T>(scope: string, key: string): Promise<T | undefined> {
    return this.base?.get(scope, key);
  }

  async set<T>(scope: string, key: string, value: T, ttlMs?: number): Promise<void> {
    await this.base?.set(scope, key, value, ttlMs);
    const text = typeof value === "string" ? value : JSON.stringify(value);
    const vec = await this.embedder.embed(text);
    this.vectors.set(`${scope}::${key}`, vec);
  }

  async delete(scope: string, key: string): Promise<void> {
    await this.base?.delete(scope, key);
    this.vectors.delete(`${scope}::${key}`);
  }

  async list(scope: string, prefix?: string): Promise<string[]> {
    return this.base?.list(scope, prefix) ?? [];
  }

  async search(scope: string, query: string, limit = 5): Promise<Array<{ key: string; score: number }>> {
    const qVec = await this.embedder.embed(query);
    const results: Array<{ key: string; score: number }> = [];
    for (const [key, vec] of this.vectors) {
      if (!key.startsWith(`${scope}::`)) continue;
      const score = cosineSimilarity(qVec, vec);
      results.push({ key: key.slice(scope.length + 2), score });
    }
    return results.sort((a, b) => b.score - a.score).slice(0, limit);
  }
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export interface TieredMemoryOptions {
  backend?: MemoryBackend;
  embedder?: SkillEmbedder;
  compressor?: SemanticCompressor;
  /** Compress working memory when token count exceeds this fraction of context window. Default: 0.85 */
  compressionThreshold?: number;
  /** Max messages to keep in L1 before compression. Default: 50 */
  maxMessages?: number;
  /** Max working memory entries. Default: 100 */
  maxWorkingMemory?: number;
}

/**
 * Tiered memory manager.
 *
 * L1 (Message History): Raw conversation turns, stored as-is.
 * L2 (Working Memory): Atomic facts extracted via semantic compression.
 * L3 (Semantic Recall): Vector embeddings for cross-session retrieval.
 * L4 (Observational): Facts extracted from tool outputs.
 */
export class TieredMemory {
  private backend: MemoryBackend;
  private vectorBackend?: VectorBackend;
  private compressor: SemanticCompressor;
  private synthesizer: SemanticSynthesizer;
  private opts: Required<Pick<TieredMemoryOptions, "compressionThreshold" | "maxMessages" | "maxWorkingMemory">>;

  constructor(opts: TieredMemoryOptions = {}) {
    this.backend = opts.backend ?? new InMemoryBackend();
    this.opts = {
      compressionThreshold: opts.compressionThreshold ?? 0.85,
      maxMessages: opts.maxMessages ?? 50,
      maxWorkingMemory: opts.maxWorkingMemory ?? 100,
    };

    // Default compressor: embedding-only if embedder provided, otherwise
    // we'll fail at compression time with a clear error.
    if (opts.compressor) {
      this.compressor = opts.compressor;
    } else if (opts.embedder) {
      this.compressor = new EmbeddingSemanticCompressor({ embedder: opts.embedder });
    } else {
      throw new Error(
        "TieredMemory requires either a `compressor` or an `embedder`. " +
          "Provide an embedder for local EmbeddingSemanticCompressor, or a custom SemanticCompressor.",
      );
    }

    // Set as default for legacy API compatibility
    setDefaultCompressor(this.compressor);

    this.synthesizer = new SemanticSynthesizer();

    if (opts.embedder) {
      this.vectorBackend = new VectorBackend(opts.embedder, this.backend);
    }
  }

  /** L1: Append a message to the conversation history */
  async appendMessage(scope: string, message: ChatMessage): Promise<void> {
    const key = "l1.messages";
    const history = (await this.backend.get<ChatMessage[]>(scope, key)) ?? [];
    history.push(message);
    if (history.length > this.opts.maxMessages) {
      // Compress oldest messages into working memory
      const toCompress = history.splice(0, history.length - this.opts.maxMessages);
      await this.compressMessages(scope, toCompress);
    }
    await this.backend.set(scope, key, history);
  }

  /** L1: Get full conversation history */
  async getMessages(scope: string): Promise<ChatMessage[]> {
    return (await this.backend.get<ChatMessage[]>(scope, "l1.messages")) ?? [];
  }

  /** L2: Store a working memory atom */
  async setAtom(scope: string, atom: MemoryAtom): Promise<void> {
    const atoms = (await this.backend.get<MemoryAtom[]>(scope, "l2.atoms")) ?? [];
    atoms.push(atom);

    // Enforce max working memory via synthesis + trim
    const synthesized = this.synthesizer.synthesize([], atoms);
    const trimmed = synthesized.slice(-this.opts.maxWorkingMemory);
    await this.backend.set(scope, "l2.atoms", trimmed);

    // Also index in L3 if vector backend available
    await this.vectorBackend?.set(scope, `l3.atom.${atom.id}`, atom.content);
  }

  /** L2: Get all working memory atoms */
  async getAtoms(scope: string): Promise<MemoryAtom[]> {
    return (await this.backend.get<MemoryAtom[]>(scope, "l2.atoms")) ?? [];
  }

  /** L2 (legacy): Store a working memory fact as a simple atom */
  async setFact(scope: string, key: string, value: string): Promise<void> {
    const atom: MemoryAtom = {
      id: key,
      content: value,
      type: "fact",
      sources: [],
      confidence: 0.8,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      accessCount: 0,
      lastAccessedAt: 0,
      relatedAtomIds: [],
      metadata: { legacy: true },
    };
    await this.setAtom(scope, atom);
  }

  /** L2 (legacy): Get all working memory facts */
  async getFacts(scope: string): Promise<Record<string, string>> {
    const atoms = await this.getAtoms(scope);
    const facts: Record<string, string> = {};
    for (const atom of atoms) {
      facts[atom.id] = atom.content;
    }
    return facts;
  }

  /** L3: Semantic search across memory */
  async recall(scope: string, query: string, limit = 5): Promise<Array<{ key: string; value: string; score: number }>> {
    if (!this.vectorBackend) return [];
    const results = await this.vectorBackend.search(scope, query, limit);
    const out: Array<{ key: string; value: string; score: number }> = [];
    for (const r of results) {
      const value = await this.vectorBackend.get<string>(scope, r.key);
      if (value) out.push({ key: r.key, value, score: r.score });
    }
    return out;
  }

  /** L4: Extract and store observations from tool outputs */
  async observe(scope: string, toolName: string, output: string): Promise<void> {
    const observations = (await this.backend.get<Array<{ tool: string; output: string; ts: number }>>(scope, "l4.observations")) ?? [];
    observations.push({ tool: toolName, output, ts: Date.now() });
    if (observations.length > 100) observations.shift();
    await this.backend.set(scope, "l4.observations", observations);
    await this.vectorBackend?.set(scope, `l4.obs.${toolName}.${Date.now()}`, output);
  }

  /** L4: Get recent observations */
  async getObservations(scope: string): Promise<Array<{ tool: string; output: string; ts: number }>> {
    return (await this.backend.get<Array<{ tool: string; output: string; ts: number }>>(scope, "l4.observations")) ?? [];
  }

  /** Clear all memory layers for a scope */
  async clear(scope: string): Promise<void> {
    for (const key of await this.backend.list(scope)) {
      await this.backend.delete(scope, key);
    }
  }

  /** Format working memory as a system prompt injection */
  async formatWorkingMemory(scope: string): Promise<string> {
    const atoms = await this.getAtoms(scope);
    if (atoms.length === 0) return "";
    const lines = atoms.map((a) => `- ${a.content}`);
    return `\n\n## Working Memory\n${lines.join("\n")}`;
  }

  private async compressMessages(scope: string, messages: ChatMessage[]): Promise<void> {
    // Modern semantic compression: embed → cluster → extract atoms → synthesize
    const atoms = await this.compressor.compress(messages);

    // Merge into existing working memory
    const existing = (await this.backend.get<MemoryAtom[]>(scope, "l2.atoms")) ?? [];
    const merged = this.synthesizer.synthesize(existing, atoms);
    const trimmed = merged.slice(-this.opts.maxWorkingMemory);
    await this.backend.set(scope, "l2.atoms", trimmed);

    // Index each atom in L3
    if (this.vectorBackend) {
      for (const atom of atoms) {
        await this.vectorBackend.set(scope, `l3.atom.${atom.id}`, atom.content);
      }
    }

    // Also store an archival chunk with full provenance
    const chunk = await buildArchivalChunk(messages, { compressor: this.compressor });
    const chunks = (await this.backend.get<ArchivalChunk[]>(scope, "l4.chunks")) ?? [];
    chunks.push(chunk);
    if (chunks.length > 50) chunks.shift();
    await this.backend.set(scope, "l4.chunks", chunks);
  }
}

export { cosineSimilarity };
export { compressMessagesToFacts, compressMessagesToChunk, buildArchivalChunk, setDefaultCompressor } from "./compression.js";
export type { MemoryAtom, MemoryFact, ArchivalChunk, SemanticCompressor, SynthesisOptions } from "./compression.js";
export { EmbeddingSemanticCompressor, LLMSemanticCompressor, SemanticSynthesizer } from "./compression.js";
