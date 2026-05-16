import { describe, it, expect, vi, beforeEach } from "vitest";
import { PostgresSaver } from "../graph/postgres-saver.js";
import type { GraphCheckpoint } from "../graph/types.js";
import type { CheckpointState } from "../agent/types.js";

// ── Mock pg Pool ─────────────────────────────────────────────────────────────

function createMockPool() {
  const queries: Array<{ sql: string; params: unknown[] }> = [];
  let rows: unknown[] = [];

  const pool = {
    query: vi.fn(async (sql: string, params?: unknown[]) => {
      queries.push({ sql, params: params ?? [] });
      return { rows: [...rows], rowCount: rows.length };
    }),
    connect: vi.fn(async () => ({
      query: vi.fn(async (sql: string, params?: unknown[]) => {
        queries.push({ sql, params: params ?? [] });
        return { rows: [...rows], rowCount: rows.length };
      }),
      release: vi.fn(),
    })),
  };

  return { pool, queries, setRows: (r: unknown[]) => { rows = r; }, getQueries: () => queries };
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe("PostgresSaver", () => {
  let mock: ReturnType<typeof createMockPool>;
  let saver: PostgresSaver;

  beforeEach(() => {
    mock = createMockPool();
    saver = new PostgresSaver(mock.pool as any, { schema: "public", prefix: "chorus" });
  });

  it("constructs with default options", () => {
    const defaultSaver = new PostgresSaver(mock.pool as any);
    expect(defaultSaver).toBeInstanceOf(PostgresSaver);
  });

  describe("setup", () => {
    it("creates table and index", async () => {
      await saver.setup();
      const queries = mock.getQueries();
      expect(queries.length).toBe(2);
      expect(queries[0].sql).toContain("CREATE TABLE IF NOT EXISTS");
      expect(queries[0].sql).toContain("chorus_checkpoints");
      expect(queries[1].sql).toContain("CREATE INDEX");
    });
  });

  describe("saveGraphCheckpoint / loadGraphCheckpoint", () => {
    it("saves and loads the latest graph checkpoint", async () => {
      const cp: GraphCheckpoint = {
        threadId: "t1",
        checkpointId: "t1_cp_1_12345",
        parentCheckpointId: "t1_cp_0_12344",
        state: { count: 42 },
        completedNodes: ["a", "b"],
        nextNodes: ["c"],
        waveCount: 2,
        createdAt: 12345,
        metadata: { source: "test" },
      };

      await saver.saveGraphCheckpoint(cp);

      const insertQuery = mock.getQueries().find((q) => q.sql.includes("INSERT INTO"));
      expect(insertQuery).toBeDefined();
      expect(insertQuery!.params[0]).toBe("t1");
      expect(insertQuery!.params[3]).toBe("t1_cp_0_12344");
      expect(JSON.parse(insertQuery!.params[5] as string)).toMatchObject({
        state: { count: 42 },
        completedNodes: ["a", "b"],
        nextNodes: ["c"],
        waveCount: 2,
      });

      // Simulate load returning the saved row
      mock.setRows([
        {
          thread_id: "t1",
          checkpoint_id: "t1_cp_1_12345",
          parent_id: "t1_cp_0_12344",
          checkpoint: {
            state: { count: 42 },
            completedNodes: ["a", "b"],
            nextNodes: ["c"],
            waveCount: 2,
          },
          metadata: { source: "test" },
          created_at: new Date(12345).toISOString(),
        },
      ]);

      const loaded = await saver.loadGraphCheckpoint("t1");
      expect(loaded).not.toBeNull();
      expect(loaded!.state.count).toBe(42);
      expect(loaded!.completedNodes).toEqual(["a", "b"]);
      expect(loaded!.nextNodes).toEqual(["c"]);
      expect(loaded!.waveCount).toBe(2);
      expect(loaded!.parentCheckpointId).toBe("t1_cp_0_12344");
    });

    it("loads a specific checkpoint by id", async () => {
      mock.setRows([
        {
          thread_id: "t1",
          checkpoint_id: "specific-id",
          parent_id: null,
          checkpoint: { state: {}, completedNodes: [], nextNodes: [], waveCount: 0 },
          metadata: {},
          created_at: new Date().toISOString(),
        },
      ]);

      const loaded = await saver.loadGraphCheckpoint("t1", "specific-id");
      expect(loaded).not.toBeNull();
      expect(loaded!.checkpointId).toBe("specific-id");

      const query = mock.getQueries()[mock.getQueries().length - 1];
      expect(query.sql).toContain("checkpoint_id = $2");
      expect(query.params[1]).toBe("specific-id");
    });

    it("returns null when no checkpoint exists", async () => {
      mock.setRows([]);
      const loaded = await saver.loadGraphCheckpoint("nonexistent");
      expect(loaded).toBeNull();
    });
  });

  describe("listGraphCheckpoints", () => {
    it("returns all checkpoints ordered by created_at", async () => {
      mock.setRows([
        {
          thread_id: "t1",
          checkpoint_id: "cp1",
          parent_id: null,
          checkpoint: { state: { v: 1 }, completedNodes: ["a"], nextNodes: ["b"], waveCount: 1 },
          metadata: {},
          created_at: new Date(1000).toISOString(),
        },
        {
          thread_id: "t1",
          checkpoint_id: "cp2",
          parent_id: "cp1",
          checkpoint: { state: { v: 2 }, completedNodes: ["a", "b"], nextNodes: [], waveCount: 2 },
          metadata: {},
          created_at: new Date(2000).toISOString(),
        },
      ]);

      const list = await saver.listGraphCheckpoints("t1");
      expect(list).toHaveLength(2);
      expect(list[0].checkpointId).toBe("cp1");
      expect(list[1].checkpointId).toBe("cp2");
      expect(list[1].parentCheckpointId).toBe("cp1");
    });
  });

  describe("base Checkpointer interface", () => {
    it("saves agent checkpoints", async () => {
      const state: CheckpointState = {
        messages: [{ role: "user", content: "hi" }],
        round: 3,
      };

      await saver.save("thread-a", state);

      const query = mock.getQueries().find((q) => q.sql.includes("INSERT INTO"));
      expect(query).toBeDefined();
      expect(query!.params[0]).toBe("thread-a");
      expect(query!.params[3]).toBe("agent");
      const checkpointJson = JSON.parse(query!.params[4] as string);
      expect(checkpointJson.round).toBe(3);
      expect(checkpointJson.messages).toEqual([{ role: "user", content: "hi" }]);
    });

    it("loads the latest agent checkpoint", async () => {
      mock.setRows([
        {
          checkpoint: {
            threadId: "thread-a",
            round: 5,
            messages: [{ role: "assistant", content: "hello" }],
            createdAt: 12345,
          },
        },
      ]);

      const loaded = await saver.load("thread-a");
      expect(loaded).not.toBeNull();
      expect(loaded!.round).toBe(5);
      expect(loaded!.messages).toHaveLength(1);
    });

    it("loads agent checkpoint at a specific round", async () => {
      mock.setRows([
        {
          checkpoint: {
            threadId: "thread-a",
            round: 2,
            messages: [],
            createdAt: 1000,
          },
        },
      ]);

      const loaded = await saver.loadAt("thread-a", 2);
      expect(loaded).not.toBeNull();
      expect(loaded!.round).toBe(2);

      const query = mock.getQueries()[mock.getQueries().length - 1];
      expect(query.sql).toContain("checkpoint->>'round' = $2");
      expect(query.params[1]).toBe("2");
    });

    it("lists all agent checkpoints", async () => {
      mock.setRows([
        { checkpoint: { threadId: "t1", round: 1, messages: [], createdAt: 1000 } },
        { checkpoint: { threadId: "t1", round: 2, messages: [], createdAt: 2000 } },
      ]);

      const list = await saver.list("t1");
      expect(list).toHaveLength(2);
      expect(list[0].round).toBe(1);
      expect(list[1].round).toBe(2);
    });

    it("forks a checkpoint to a new thread", async () => {
      mock.setRows([
        {
          checkpoint: {
            threadId: "src",
            round: 3,
            messages: [{ role: "user", content: "fork me" }],
            createdAt: 5000,
          },
        },
      ]);

      await saver.fork("src", 3, "dst");

      const queries = mock.getQueries();
      const selectQuery = queries.find((q) => q.sql.includes("checkpoint->>'round' = $2"));
      expect(selectQuery).toBeDefined();

      const insertQuery = queries.find((q) => q.sql.includes("INSERT INTO") && q.params[3] === "agent");
      expect(insertQuery).toBeDefined();
      expect(insertQuery!.params[0]).toBe("dst");
      const cp = JSON.parse(insertQuery!.params[4] as string);
      expect(cp.round).toBe(3);
    });

    it("deletes all checkpoints for a thread", async () => {
      await saver.delete("thread-x");
      const query = mock.getQueries()[0];
      expect(query.sql).toContain("DELETE FROM");
      expect(query.params[0]).toBe("thread-x");
    });
  });
});
