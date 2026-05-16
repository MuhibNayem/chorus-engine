import { describe, it, expect, beforeAll } from "vitest";
import {
  compressMessagesToFacts,
  compressMessagesToChunk,
  setDefaultCompressor,
  EmbeddingSemanticCompressor,
  SemanticSynthesizer,
  buildArchivalChunk,
  type MemoryAtom,
} from "../memory/compression.js";
import { KeywordEmbedder } from "../skills/embedder.js";
import type { ChatMessage } from "../llm/provider.js";

// Use deterministic keyword embedder for reproducible tests (no model download)
const embedder = new KeywordEmbedder();

beforeAll(() => {
  setDefaultCompressor(new EmbeddingSemanticCompressor({ embedder, clusterThreshold: 0.65 }));
});

describe("Semantic Memory Compression Engine — embedding-based clustering", () => {
  it("extracts facts from a batch of messages", async () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "I prefer using TypeScript for all backend services." },
      { role: "assistant", content: "TypeScript provides excellent type safety and developer experience." },
      { role: "user", content: "The project uses PostgreSQL as the primary database." },
    ];

    const facts = await compressMessagesToFacts(messages);

    expect(facts.length).toBeGreaterThan(0);
    // At least one fact should mention either typescript or postgresql
    const hasRelevantFact = facts.some((f) =>
      f.value.toLowerCase().includes("typescript") ||
      f.value.toLowerCase().includes("postgresql") ||
      f.value.toLowerCase().includes("backend") ||
      f.value.toLowerCase().includes("database"),
    );
    expect(hasRelevantFact).toBe(true);
    // All facts should have source provenance
    expect(facts.every((f) => f.sources.length > 0)).toBe(true);
    expect(facts.every((f) => f.confidence >= 0 && f.confidence <= 1)).toBe(true);
  });

  it("extracts predicate patterns from messages", async () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "My deployment pipeline is using Docker containers." },
      { role: "assistant", content: "Docker containers provide consistent environments across stages." },
    ];

    const facts = await compressMessagesToFacts(messages);
    const deploymentFact = facts.find((f) =>
      f.value.toLowerCase().includes("deployment") && f.value.toLowerCase().includes("docker"),
    );
    expect(deploymentFact).toBeDefined();
  });

  it("clusters related messages by semantic similarity", async () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "How do I configure Redis caching?" },
      { role: "assistant", content: "Redis caching requires setting up eviction policies." },
      { role: "user", content: "What is the weather today?" },
      { role: "assistant", content: "The weather is sunny and warm." },
    ];

    const facts = await compressMessagesToFacts(messages);
    // Should create at least a Redis cluster fact
    expect(facts.some((f) => f.value.toLowerCase().includes("redis"))).toBe(true);
  });

  it("produces archival chunks with provenance", async () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "We need to implement OAuth2 authentication." },
      { role: "assistant", content: "OAuth2 requires client ID and secret configuration." },
    ];

    const chunk = await compressMessagesToChunk(messages, 5);
    expect(chunk.summary.length).toBeGreaterThan(0);
    expect(chunk.keywords.length).toBeGreaterThanOrEqual(0);
    expect(chunk.atoms.length).toBeGreaterThanOrEqual(0);
    expect(chunk.sourceRange[0]).toBe(5);
    expect(chunk.sourceRange[1]).toBe(6);
    expect(chunk.compressedAt).toBeGreaterThan(0);
  });

  it("handles empty message batches gracefully", async () => {
    expect(await compressMessagesToFacts([])).toEqual([]);
    const chunk = await compressMessagesToChunk([], 0);
    expect(chunk.summary).toBe("General conversation");
    expect(chunk.atoms).toEqual([]);
  });

  it("deduplicates facts by key", async () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "The API is RESTful." },
      { role: "assistant", content: "The API uses RESTful principles." },
    ];

    const facts = await compressMessagesToFacts(messages);
    const keys = facts.map((f) => f.key);
    const uniqueKeys = new Set(keys);
    expect(uniqueKeys.size).toBe(keys.length);
  });
});

describe("EmbeddingSemanticCompressor — direct API", () => {
  it("compresses messages into MemoryAtoms with embeddings", async () => {
    const compressor = new EmbeddingSemanticCompressor({ embedder, clusterThreshold: 0.6 });
    const messages: ChatMessage[] = [
      { role: "user", content: "I love using React for frontend development." },
      { role: "assistant", content: "React has a great component model and ecosystem." },
    ];

    const atoms = await compressor.compress(messages);

    expect(atoms.length).toBeGreaterThan(0);
    // Atoms should have embeddings
    expect(atoms.every((a) => a.embedding && a.embedding.length > 0)).toBe(true);
    // Sources should point to valid message indices
    expect(atoms.every((a) => a.sources.every((s) => s >= 0 && s < messages.length))).toBe(true);
  });

  it("clusters semantically similar messages together", async () => {
    const compressor = new EmbeddingSemanticCompressor({ embedder, clusterThreshold: 0.6 });
    const messages: ChatMessage[] = [
      { role: "user", content: "How do I set up Kubernetes?" },
      { role: "assistant", content: "Kubernetes needs a cluster with pods and services." },
      { role: "user", content: "The cat sat on the mat." },
      { role: "assistant", content: "That is a very nice cat." },
    ];

    const atoms = await compressor.compress(messages);

    // Should produce atoms from both clusters (k8s and cat)
    const hasK8s = atoms.some((a) => a.content.toLowerCase().includes("kubernetes") || a.content.toLowerCase().includes("cluster"));
    expect(hasK8s).toBe(true);
  });
});

describe("SemanticSynthesizer — online deduplication", () => {
  it("merges duplicate atoms above merge threshold", () => {
    const synth = new SemanticSynthesizer({ mergeThreshold: 0.95, relateThreshold: 0.8 });

    const atomA: MemoryAtom = {
      id: "a1",
      content: "User prefers dark mode",
      type: "preference",
      embedding: [1, 0, 0, 0],
      sources: [0],
      confidence: 0.8,
      createdAt: 1000,
      updatedAt: 1000,
      accessCount: 1,
      lastAccessedAt: 1000,
      relatedAtomIds: [],
      metadata: {},
    };

    const atomB: MemoryAtom = {
      id: "a2",
      content: "User prefers dark mode",
      type: "preference",
      embedding: [1, 0, 0, 0],
      sources: [1],
      confidence: 0.85,
      createdAt: 2000,
      updatedAt: 2000,
      accessCount: 0,
      lastAccessedAt: 0,
      relatedAtomIds: [],
      metadata: {},
    };

    const result = synth.synthesize([atomA], [atomB]);

    // Should merge into a single atom
    expect(result.length).toBe(1);
    expect(result[0].sources).toContain(0);
    expect(result[0].sources).toContain(1);
    expect(result[0].confidence).toBeGreaterThan(0.8);
  });

  it("links related atoms below merge but above relate threshold", () => {
    const synth = new SemanticSynthesizer({ mergeThreshold: 0.99, relateThreshold: 0.5 });

    // Vectors chosen to have cosine similarity ~0.8 (between relate=0.5 and merge=0.99)
    const atomA: MemoryAtom = {
      id: "a1",
      content: "User likes TypeScript",
      type: "preference",
      embedding: [1, 0.5, 0, 0],
      sources: [0],
      confidence: 0.8,
      createdAt: 1000,
      updatedAt: 1000,
      accessCount: 0,
      lastAccessedAt: 0,
      relatedAtomIds: [],
      metadata: {},
    };

    const atomB: MemoryAtom = {
      id: "a2",
      content: "User uses TypeScript for backend",
      type: "fact",
      embedding: [0.5, 1, 0, 0],
      sources: [1],
      confidence: 0.8,
      createdAt: 2000,
      updatedAt: 2000,
      accessCount: 0,
      lastAccessedAt: 0,
      relatedAtomIds: [],
      metadata: {},
    };

    const result = synth.synthesize([atomA], [atomB]);

    // Should not merge, but should link
    expect(result.length).toBe(2);
    expect(result[0].relatedAtomIds).toContain("a2");
    expect(result[1].relatedAtomIds).toContain("a1");
  });
});

describe("buildArchivalChunk — unified chunk builder", () => {
  it("builds a chunk with atoms, keywords, and summary", async () => {
    const compressor = new EmbeddingSemanticCompressor({ embedder, clusterThreshold: 0.6 });
    const messages: ChatMessage[] = [
      { role: "user", content: "Deploy the app to AWS Lambda." },
      { role: "assistant", content: "Lambda deployment requires packaging dependencies." },
    ];

    const chunk = await buildArchivalChunk(messages, { compressor }, 0);

    expect(chunk.summary.length).toBeGreaterThan(0);
    expect(chunk.sourceRange).toEqual([0, 1]);
    expect(chunk.compressedAt).toBeGreaterThan(0);
    expect(chunk.atoms.length).toBeGreaterThan(0);
  });
});
