/**
 * PostgresSaver — PostgreSQL-backed GraphCheckpointer.
 *
 * Stores checkpoints in a relational database with JSONB state columns
 * for efficient querying and versioning.
 *
 * Schema:
 *   chorus_checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_id,
 *                       type, checkpoint JSONB, metadata JSONB, created_at)
 *
 * Usage:
 *   import { Pool } from "pg";
 *   const pool = new Pool({ connectionString: "..." });
 *   const saver = new PostgresSaver(pool);
 *   await saver.setup();
 */

import type { Pool, PoolClient } from "pg";
import type { Checkpointer, Checkpoint, CheckpointState } from "../agent/types.js";
import type { GraphCheckpointer, GraphCheckpoint } from "./types.js";

export interface PostgresSaverOptions {
  /** Database schema name. Default: "public". */
  schema?: string;
  /** Table name prefix. Default: "chorus". */
  prefix?: string;
}

export class PostgresSaver implements GraphCheckpointer {
  private pool: Pool;
  private schema: string;
  private table: string;

  constructor(pool: Pool, options: PostgresSaverOptions = {}) {
    this.pool = pool;
    this.schema = options.schema ?? "public";
    this.table = `${options.prefix ?? "chorus"}_checkpoints`;
  }

  /** Create the checkpoints table if it does not exist. */
  async setup(): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query(`
        CREATE TABLE IF NOT EXISTS ${this.schema}.${this.table} (
          id            SERIAL PRIMARY KEY,
          thread_id     TEXT NOT NULL,
          checkpoint_ns TEXT NOT NULL DEFAULT '',
          checkpoint_id TEXT NOT NULL,
          parent_id     TEXT,
          type          TEXT NOT NULL DEFAULT 'graph',
          checkpoint    JSONB NOT NULL DEFAULT '{}',
          metadata      JSONB NOT NULL DEFAULT '{}',
          created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
          UNIQUE (thread_id, checkpoint_ns, checkpoint_id)
        )
      `);
      await client.query(`
        CREATE INDEX IF NOT EXISTS idx_${this.table}_thread
        ON ${this.schema}.${this.table} (thread_id, checkpoint_ns, created_at)
      `);
    } finally {
      client.release();
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // GraphCheckpointer interface
  // ═══════════════════════════════════════════════════════════════════════════

  async saveGraphCheckpoint(cp: GraphCheckpoint): Promise<void> {
    await this.pool.query(
      `INSERT INTO ${this.schema}.${this.table}
       (thread_id, checkpoint_ns, checkpoint_id, parent_id, type, checkpoint, metadata, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, to_timestamp($8 / 1000.0))`,
      [
        cp.threadId,
        "",
        cp.checkpointId,
        cp.parentCheckpointId ?? null,
        "graph",
        JSON.stringify({
          state: cp.state,
          completedNodes: cp.completedNodes,
          nextNodes: cp.nextNodes,
          waveCount: cp.waveCount,
        }),
        JSON.stringify(cp.metadata ?? {}),
        cp.createdAt,
      ],
    );
  }

  async loadGraphCheckpoint(
    threadId: string,
    checkpointId?: string,
  ): Promise<GraphCheckpoint | null> {
    const result = checkpointId
      ? await this.pool.query(
          `SELECT * FROM ${this.schema}.${this.table}
           WHERE thread_id = $1 AND checkpoint_id = $2 AND type = 'graph'
           ORDER BY created_at DESC LIMIT 1`,
          [threadId, checkpointId],
        )
      : await this.pool.query(
          `SELECT * FROM ${this.schema}.${this.table}
           WHERE thread_id = $1 AND type = 'graph'
           ORDER BY created_at DESC LIMIT 1`,
          [threadId],
        );

    if (result.rows.length === 0) return null;
    return this.rowToGraphCheckpoint(result.rows[0]);
  }

  async listGraphCheckpoints(threadId: string): Promise<GraphCheckpoint[]> {
    const result = await this.pool.query(
      `SELECT * FROM ${this.schema}.${this.table}
       WHERE thread_id = $1 AND type = 'graph'
       ORDER BY created_at ASC`,
      [threadId],
    );
    return result.rows.map((r) => this.rowToGraphCheckpoint(r));
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Base Checkpointer interface
  // ═══════════════════════════════════════════════════════════════════════════

  async save(threadId: string, state: CheckpointState): Promise<void> {
    const checkpoint: Checkpoint = {
      threadId,
      round: state.round,
      messages: state.messages,
      createdAt: Date.now(),
      waitingForHitl: state.waitingForHitl,
    };
    await this.pool.query(
      `INSERT INTO ${this.schema}.${this.table}
       (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, created_at)
       VALUES ($1, $2, $3, $4, $5, to_timestamp($6 / 1000.0))`,
      [
        threadId,
        "",
        `${threadId}_cp_${state.round}_${Date.now()}`,
        "agent",
        JSON.stringify(checkpoint),
        Date.now(),
      ],
    );
  }

  async load(threadId: string): Promise<Checkpoint | null> {
    const result = await this.pool.query(
      `SELECT checkpoint FROM ${this.schema}.${this.table}
       WHERE thread_id = $1 AND type = 'agent'
       ORDER BY created_at DESC LIMIT 1`,
      [threadId],
    );
    if (result.rows.length === 0) return null;
    return result.rows[0].checkpoint as Checkpoint;
  }

  async loadAt(threadId: string, round: number): Promise<Checkpoint | null> {
    const result = await this.pool.query(
      `SELECT checkpoint FROM ${this.schema}.${this.table}
       WHERE thread_id = $1 AND type = 'agent'
         AND checkpoint->>'round' = $2
       ORDER BY created_at DESC LIMIT 1`,
      [threadId, String(round)],
    );
    if (result.rows.length === 0) return null;
    return result.rows[0].checkpoint as Checkpoint;
  }

  async list(threadId: string): Promise<Checkpoint[]> {
    const result = await this.pool.query(
      `SELECT checkpoint FROM ${this.schema}.${this.table}
       WHERE thread_id = $1 AND type = 'agent'
       ORDER BY created_at ASC`,
      [threadId],
    );
    return result.rows.map((r) => r.checkpoint as Checkpoint);
  }

  async fork(threadId: string, round: number, newThreadId: string): Promise<void> {
    const cp = await this.loadAt(threadId, round);
    if (!cp) return;
    await this.save(newThreadId, {
      messages: cp.messages,
      round: cp.round,
      waitingForHitl: cp.waitingForHitl,
    });
  }

  async delete(threadId: string): Promise<void> {
    await this.pool.query(
      `DELETE FROM ${this.schema}.${this.table} WHERE thread_id = $1`,
      [threadId],
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Helpers
  // ═══════════════════════════════════════════════════════════════════════════

  private rowToGraphCheckpoint(row: Record<string, unknown>): GraphCheckpoint {
    const cp = row.checkpoint as Record<string, unknown>;
    return {
      threadId: row.thread_id as string,
      checkpointId: row.checkpoint_id as string,
      parentCheckpointId: (row.parent_id as string) ?? undefined,
      state: (cp.state ?? {}) as Record<string, unknown>,
      completedNodes: (cp.completedNodes ?? []) as string[],
      nextNodes: (cp.nextNodes ?? []) as string[],
      waveCount: (cp.waveCount ?? 0) as number,
      createdAt: new Date(row.created_at as string).getTime(),
      metadata: (row.metadata ?? {}) as Record<string, unknown>,
    };
  }
}
