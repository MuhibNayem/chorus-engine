/**
 * PostgresSaver — Production-Grade Database Checkpointer
 *
 * Async PostgreSQL checkpointer with connection pooling, JSONB state storage,
 * and automatic schema migrations. Designed for high-concurrency production
 * deployments where file-based checkpointing is insufficient.
 *
 * Schema:
 *   - checkpoints table: thread_id, checkpoint_id, parent_id, state (JSONB),
 *     completed_nodes (TEXT[]), next_nodes (TEXT[]), wave_count, metadata (JSONB)
 *   - event_log table: thread_id, seq, event_type, payload (JSONB), ts
 *
 * Features:
 *   - Connection pooling via pg.Pool
 *   - Automatic table creation / migration on first use
 *   - JSONB for efficient state queries and indexing
 *   - Batch operations for checkpoint + event log writes
 *   - Compatible with both Checkpointer and GraphCheckpointer interfaces
 *
 * Usage:
 *   import { Pool } from "pg";
 *   const pool = new Pool({ connectionString: process.env.DATABASE_URL });
 *   const saver = new PostgresSaver(pool);
 *   await saver.setup(); // create tables
 *
 *   const graph = new StateGraph(...).compile({ checkpointer: saver });
 */

import type { Pool, PoolClient } from "pg";
import type {
  Checkpointer,
  Checkpoint,
  CheckpointState,
} from "./types.js";
import type { GraphCheckpointer, GraphCheckpoint } from "../graph/types.js";

export interface PostgresSaverOptions {
  /** Database schema name. Default: "public" */
  schema?: string;
  /** Table name for checkpoints. Default: "chorus_checkpoints" */
  checkpointsTable?: string;
  /** Table name for event logs. Default: "chorus_event_log" */
  eventLogTable?: string;
}

export class PostgresSaver implements Checkpointer, GraphCheckpointer {
  private pool: Pool;
  private schema: string;
  private checkpointsTable: string;
  private eventLogTable: string;
  private initialized = false;

  constructor(pool: Pool, opts: PostgresSaverOptions = {}) {
    this.pool = pool;
    this.schema = opts.schema ?? "public";
    this.checkpointsTable = opts.checkpointsTable ?? "chorus_checkpoints";
    this.eventLogTable = opts.eventLogTable ?? "chorus_event_log";
  }

  /** Create tables if they don't exist. Idempotent. */
  async setup(): Promise<void> {
    if (this.initialized) return;
    const client = await this.pool.connect();
    try {
      await client.query(`
        CREATE TABLE IF NOT EXISTS ${this.schema}.${this.checkpointsTable} (
          id SERIAL PRIMARY KEY,
          thread_id TEXT NOT NULL,
          checkpoint_id TEXT NOT NULL,
          parent_checkpoint_id TEXT,
          state JSONB NOT NULL DEFAULT '{}',
          completed_nodes TEXT[] DEFAULT '{}',
          next_nodes TEXT[] DEFAULT '{}',
          wave_count INT NOT NULL DEFAULT 0,
          metadata JSONB DEFAULT '{}',
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          UNIQUE(thread_id, checkpoint_id)
        )
      `);

      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_${this.checkpointsTable}_thread
        ON ${this.schema}.${this.checkpointsTable}(thread_id)
      `);

      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_${this.checkpointsTable}_checkpoint
        ON ${this.schema}.${this.checkpointsTable}(thread_id, checkpoint_id)
      `);

      await client.query(`
        CREATE TABLE IF NOT EXISTS ${this.schema}.${this.eventLogTable} (
          id SERIAL PRIMARY KEY,
          thread_id TEXT NOT NULL,
          seq INT NOT NULL,
          event_type TEXT NOT NULL,
          payload JSONB NOT NULL DEFAULT '{}',
          ts TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          UNIQUE(thread_id, seq)
        )
      `);

      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_${this.eventLogTable}_thread
        ON ${this.schema}.${this.eventLogTable}(thread_id)
      `);

      this.initialized = true;
    } finally {
      client.release();
    }
  }

  // ── Checkpointer interface ────────────────────────────────────────────────

  async save(threadId: string, state: CheckpointState): Promise<void> {
    await this.ensureSetup();
    await this.pool.query(
      `INSERT INTO ${this.schema}.${this.checkpointsTable}
        (thread_id, checkpoint_id, state, wave_count)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (thread_id, checkpoint_id)
       DO UPDATE SET state = EXCLUDED.state, wave_count = EXCLUDED.wave_count`,
      [threadId, `${threadId}_${state.round}`, JSON.stringify(state), state.round],
    );
  }

  async load(threadId: string): Promise<Checkpoint | null> {
    await this.ensureSetup();
    const result = await this.pool.query(
      `SELECT state, wave_count as round, created_at
       FROM ${this.schema}.${this.checkpointsTable}
       WHERE thread_id = $1
       ORDER BY wave_count DESC
       LIMIT 1`,
      [threadId],
    );
    if (result.rows.length === 0) return null;
    const row = result.rows[0];
    return {
      threadId,
      round: row.round,
      messages: row.state.messages ?? [],
      createdAt: new Date(row.created_at).getTime(),
      waitingForHitl: row.state.waitingForHitl,
    };
  }

  async loadAt(threadId: string, round: number): Promise<Checkpoint | null> {
    await this.ensureSetup();
    const result = await this.pool.query(
      `SELECT state, wave_count as round, created_at
       FROM ${this.schema}.${this.checkpointsTable}
       WHERE thread_id = $1 AND wave_count = $2
       LIMIT 1`,
      [threadId, round],
    );
    if (result.rows.length === 0) return null;
    const row = result.rows[0];
    return {
      threadId,
      round: row.round,
      messages: row.state.messages ?? [],
      createdAt: new Date(row.created_at).getTime(),
      waitingForHitl: row.state.waitingForHitl,
    };
  }

  async list(threadId: string): Promise<Checkpoint[]> {
    await this.ensureSetup();
    const result = await this.pool.query(
      `SELECT state, wave_count as round, created_at
       FROM ${this.schema}.${this.checkpointsTable}
       WHERE thread_id = $1
       ORDER BY wave_count ASC`,
      [threadId],
    );
    return result.rows.map((row) => ({
      threadId,
      round: row.round,
      messages: row.state.messages ?? [],
      createdAt: new Date(row.created_at).getTime(),
      waitingForHitl: row.state.waitingForHitl,
    }));
  }

  async fork(threadId: string, round: number, newThreadId: string): Promise<void> {
    await this.ensureSetup();
    await this.pool.query(
      `INSERT INTO ${this.schema}.${this.checkpointsTable}
        (thread_id, checkpoint_id, parent_checkpoint_id, state, completed_nodes, next_nodes, wave_count, metadata)
       SELECT $1, $2, checkpoint_id, state, completed_nodes, next_nodes, wave_count, metadata
       FROM ${this.schema}.${this.checkpointsTable}
       WHERE thread_id = $3 AND wave_count = $4
       LIMIT 1`,
      [newThreadId, `${newThreadId}_${round}`, threadId, round],
    );
  }

  async delete(threadId: string): Promise<void> {
    await this.ensureSetup();
    await this.pool.query(
      `DELETE FROM ${this.schema}.${this.checkpointsTable} WHERE thread_id = $1`,
      [threadId],
    );
    await this.pool.query(
      `DELETE FROM ${this.schema}.${this.eventLogTable} WHERE thread_id = $1`,
      [threadId],
    );
  }

  // ── GraphCheckpointer interface ───────────────────────────────────────────

  async saveGraphCheckpoint(cp: GraphCheckpoint): Promise<void> {
    await this.ensureSetup();
    await this.pool.query(
      `INSERT INTO ${this.schema}.${this.checkpointsTable}
        (thread_id, checkpoint_id, parent_checkpoint_id, state, completed_nodes, next_nodes, wave_count, metadata)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
       ON CONFLICT (thread_id, checkpoint_id)
       DO UPDATE SET state = EXCLUDED.state, completed_nodes = EXCLUDED.completed_nodes,
                     next_nodes = EXCLUDED.next_nodes, wave_count = EXCLUDED.wave_count,
                     metadata = EXCLUDED.metadata`,
      [
        cp.threadId,
        cp.checkpointId,
        cp.parentCheckpointId ?? null,
        JSON.stringify(cp.state),
        cp.completedNodes,
        cp.nextNodes,
        cp.waveCount,
        JSON.stringify(cp.metadata ?? {}),
      ],
    );
  }

  async loadGraphCheckpoint(threadId: string, checkpointId?: string): Promise<GraphCheckpoint | null> {
    await this.ensureSetup();
    let result;
    if (checkpointId) {
      result = await this.pool.query(
        `SELECT thread_id, checkpoint_id, parent_checkpoint_id, state, completed_nodes,
                next_nodes, wave_count, metadata, created_at
         FROM ${this.schema}.${this.checkpointsTable}
         WHERE thread_id = $1 AND checkpoint_id = $2
         LIMIT 1`,
        [threadId, checkpointId],
      );
    } else {
      result = await this.pool.query(
        `SELECT thread_id, checkpoint_id, parent_checkpoint_id, state, completed_nodes,
                next_nodes, wave_count, metadata, created_at
         FROM ${this.schema}.${this.checkpointsTable}
         WHERE thread_id = $1
         ORDER BY wave_count DESC
         LIMIT 1`,
        [threadId],
      );
    }
    if (result.rows.length === 0) return null;
    return this.rowToGraphCheckpoint(result.rows[0]);
  }

  async listGraphCheckpoints(threadId: string): Promise<GraphCheckpoint[]> {
    await this.ensureSetup();
    const result = await this.pool.query(
      `SELECT thread_id, checkpoint_id, parent_checkpoint_id, state, completed_nodes,
              next_nodes, wave_count, metadata, created_at
       FROM ${this.schema}.${this.checkpointsTable}
       WHERE thread_id = $1
       ORDER BY wave_count ASC`,
      [threadId],
    );
    return result.rows.map((r) => this.rowToGraphCheckpoint(r));
  }

  // ── Event log (append-only) ───────────────────────────────────────────────

  async appendEvent(
    threadId: string,
    seq: number,
    eventType: string,
    payload: Record<string, unknown>,
  ): Promise<void> {
    await this.ensureSetup();
    await this.pool.query(
      `INSERT INTO ${this.schema}.${this.eventLogTable}
        (thread_id, seq, event_type, payload)
       VALUES ($1, $2, $3, $4)
       ON CONFLICT (thread_id, seq) DO NOTHING`,
      [threadId, seq, eventType, JSON.stringify(payload)],
    );
  }

  async readEvents(threadId: string): Promise<Array<{ seq: number; type: string; payload: Record<string, unknown>; ts: number }>> {
    await this.ensureSetup();
    const result = await this.pool.query(
      `SELECT seq, event_type, payload, ts
       FROM ${this.schema}.${this.eventLogTable}
       WHERE thread_id = $1
       ORDER BY seq ASC`,
      [threadId],
    );
    return result.rows.map((r) => ({
      seq: r.seq,
      type: r.event_type,
      payload: r.payload,
      ts: new Date(r.ts).getTime(),
    }));
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private async ensureSetup(): Promise<void> {
    if (!this.initialized) await this.setup();
  }

  private rowToGraphCheckpoint(row: Record<string, unknown>): GraphCheckpoint {
    return {
      threadId: row.thread_id as string,
      checkpointId: row.checkpoint_id as string,
      parentCheckpointId: (row.parent_checkpoint_id as string) ?? undefined,
      state: (row.state as Record<string, unknown>) ?? {},
      completedNodes: (row.completed_nodes as string[]) ?? [],
      nextNodes: (row.next_nodes as string[]) ?? [],
      waveCount: (row.wave_count as number) ?? 0,
      createdAt: new Date(row.created_at as string).getTime(),
      metadata: (row.metadata as Record<string, unknown>) ?? {},
    };
  }
}
