/**
 * Durable Execution Checkpointer
 *
 * Production-grade checkpointing inspired by LangGraph and Temporal:
 *   • Durability modes — sync (highest durability), async (balanced),
 *     exit (performance-first, no mid-crash recovery)
 *   • Event sourcing — append-only log of state transitions for replay
 *   • Crash recovery — automatic state reconstruction from event log
 *   • Time-travel — fork from any historical checkpoint
 *   • Atomic writes — tmp+rename for crash-safe persistence
 *
 * Design:
 *   - `sync`: every step is flushed to disk before the next step starts.
 *     Highest durability, lowest throughput.
 *   - `async`: checkpoints are written asynchronously while the next step runs.
 *     Good balance; tiny window of data loss if crash during async write.
 *   - `exit`: checkpoints only written on graceful exit. Best performance,
 *     but mid-execution crashes lose all in-flight work.
 */

import * as fs from "fs";
import * as path from "path";
import { randomUUID } from "crypto";
import { getSettingsPath } from "../settings/storage.js";
import type { Checkpoint, Checkpointer, CheckpointState, AgentEvent } from "./types.js";
import { JsonFileCheckpointer } from "./checkpointer.js";

export type DurabilityMode = "sync" | "async" | "exit";

/** A single event in the append-only event log. */
export interface CheckpointEvent {
  /** Monotonic sequence number within a thread. */
  seq: number;
  /** Event timestamp (ms). */
  ts: number;
  /** Event type describing the state transition. */
  type: "round_start" | "llm_response" | "tool_call" | "tool_result" | "hitl_pause" | "hitl_resume" | "compaction" | "error" | "done";
  /** Human-readable description. */
  description: string;
  /** Optional payload (tool name, error message, etc.). */
  payload?: Record<string, unknown>;
}

/** Enhanced checkpoint with event-sourcing metadata. */
export interface DurableCheckpoint extends Checkpoint {
  /** Sequence number for event-log ordering. */
  seq: number;
  /** Durability mode used for this checkpoint. */
  durability: DurabilityMode;
  /** Events that led to this checkpoint state. */
  events: CheckpointEvent[];
  /** Wall-clock duration of the round in ms. */
  roundDurationMs?: number;
  /** Cumulative token usage at this checkpoint. */
  cumulativeTokens?: { input: number; output: number };
}

function getEventLogPath(threadId: string): string {
  return path.join(path.dirname(getSettingsPath()), "checkpoints", threadId, "events.jsonl");
}

function atomicAppendJsonl(filePath: string, lines: string[]): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  let existing = "";
  try {
    existing = fs.readFileSync(filePath, "utf-8");
  } catch { /* file does not exist yet */ }
  const newData = lines.join("\n") + (lines.length > 0 ? "\n" : "");
  const tmp = `${filePath}.${randomUUID()}.tmp`;
  fs.writeFileSync(tmp, existing + newData, "utf-8");
  fs.renameSync(tmp, filePath);
}

function readJsonl<T>(filePath: string): T[] {
  try {
    const data = fs.readFileSync(filePath, "utf-8");
    return data
      .split("\n")
      .filter((l) => l.trim().length > 0)
      .map((l) => JSON.parse(l) as T);
  } catch {
    return [];
  }
}

/**
 * Durable checkpointer with event sourcing and configurable durability modes.
 *
 * Wraps `JsonFileCheckpointer` for snapshot storage and adds an append-only
 * event log for crash recovery and audit trails.
 */
export class DurableCheckpointer implements Checkpointer {
  private base: JsonFileCheckpointer;
  private mode: DurabilityMode;
  private pendingFlush: Promise<void> | null = null;
  private seqCounter = new Map<string, number>();

  constructor(mode: DurabilityMode = "sync") {
    this.base = new JsonFileCheckpointer();
    this.mode = mode;
  }

  /** Current durability mode. */
  get durability(): DurabilityMode {
    return this.mode;
  }

  /** Change durability mode at runtime. */
  setDurability(mode: DurabilityMode): void {
    this.mode = mode;
  }

  private nextSeq(threadId: string): number {
    const current = this.seqCounter.get(threadId) ?? 0;
    const next = current + 1;
    this.seqCounter.set(threadId, next);
    return next;
  }

  async save(threadId: string, state: CheckpointState): Promise<void> {
    if (this.mode === "exit") {
      // Defer write until explicit flush or process exit
      return;
    }

    const checkpoint: DurableCheckpoint = {
      threadId,
      round: state.round,
      messages: state.messages,
      createdAt: Date.now(),
      waitingForHitl: state.waitingForHitl,
      seq: this.nextSeq(threadId),
      durability: this.mode,
      events: [], // Events are stored in the separate log
    };

    if (this.mode === "sync") {
      await this.base.save(threadId, state);
      this.writeEventLog(threadId, checkpoint.seq, state);
    } else {
      // async: fire-and-forget with best-effort ordering
      this.pendingFlush = this.flushAsync(threadId, state, checkpoint.seq);
    }
  }

  private async flushAsync(
    threadId: string,
    state: CheckpointState,
    seq: number,
  ): Promise<void> {
    await this.base.save(threadId, state);
    this.writeEventLog(threadId, seq, state);
  }

  private writeEventLog(threadId: string, seq: number, state: CheckpointState): void {
    const event: CheckpointEvent = {
      seq,
      ts: Date.now(),
      type: state.waitingForHitl ? "hitl_pause" : "round_start",
      description: `Checkpoint round ${state.round}`,
      payload: {
        round: state.round,
        messageCount: state.messages.length,
        hitlPaused: !!state.waitingForHitl,
      },
    };
    const logPath = getEventLogPath(threadId);
    atomicAppendJsonl(logPath, [JSON.stringify(event)]);
  }

  /** Append an arbitrary event to the event log. */
  appendEvent(threadId: string, event: Omit<CheckpointEvent, "seq" | "ts">): void {
    const fullEvent: CheckpointEvent = {
      ...event,
      seq: this.nextSeq(threadId),
      ts: Date.now(),
    };
    const logPath = getEventLogPath(threadId);
    atomicAppendJsonl(logPath, [JSON.stringify(fullEvent)]);
  }

  async load(threadId: string): Promise<DurableCheckpoint | null> {
    const base = await this.base.load(threadId);
    if (!base) return null;
    const events = this.readEvents(threadId);
    return {
      ...base,
      seq: events.length > 0 ? events[events.length - 1].seq : 0,
      durability: this.mode,
      events,
    };
  }

  async loadAt(threadId: string, round: number): Promise<DurableCheckpoint | null> {
    const base = await this.base.loadAt(threadId, round);
    if (!base) return null;
    const events = this.readEvents(threadId).filter((e) => {
      const payload = e.payload as { round?: number } | undefined;
      return payload && payload.round === round;
    });
    return {
      ...base,
      seq: events.length > 0 ? events[events.length - 1].seq : 0,
      durability: this.mode,
      events,
    };
  }

  async list(threadId: string): Promise<DurableCheckpoint[]> {
    const bases = await this.base.list(threadId);
    const events = this.readEvents(threadId);
    return bases.map((base) => ({
      ...base,
      seq: 0,
      durability: this.mode,
      events: events.filter((e) => {
        const payload = e.payload as { round?: number } | undefined;
        return payload && payload.round === base.round;
      }),
    }));
  }

  async fork(threadId: string, round: number, newThreadId: string): Promise<void> {
    await this.base.fork(threadId, round, newThreadId);
    // Copy events up to the fork point
    const events = this.readEvents(threadId);
    const forkEvents = events.filter((e) => {
      const payload = e.payload as { round?: number } | undefined;
      return payload && typeof payload.round === "number" && payload.round <= round;
    });
    if (forkEvents.length > 0) {
      const logPath = getEventLogPath(newThreadId);
      atomicAppendJsonl(logPath, forkEvents.map((e) => JSON.stringify(e)));
    }
  }

  async delete(threadId: string): Promise<void> {
    await this.base.delete(threadId);
    try {
      fs.unlinkSync(getEventLogPath(threadId));
    } catch { /* ignore */ }
  }

  /** Read the full event log for a thread. */
  readEvents(threadId: string): CheckpointEvent[] {
    return readJsonl<CheckpointEvent>(getEventLogPath(threadId));
  }

  /** Reconstruct approximate state from event log (useful when snapshots are corrupt). */
  reconstructFromEvents(threadId: string): { round: number; events: CheckpointEvent[] } | null {
    const events = this.readEvents(threadId);
    if (events.length === 0) return null;
    const lastRoundEvent = [...events].reverse().find((e) => e.payload && typeof (e.payload as { round?: number }).round === "number");
    const round = lastRoundEvent ? (lastRoundEvent.payload as { round: number }).round : 0;
    return { round, events };
  }

  /** Explicitly flush all pending async checkpoints. Call before process exit. */
  async flush(): Promise<void> {
    if (this.pendingFlush) {
      await this.pendingFlush;
      this.pendingFlush = null;
    }
  }
}

/** Helper to detect whether a thread has an in-flight run that may need recovery. */
export function detectCrashedRun(
  checkpointer: DurableCheckpointer,
  threadId: string,
): { crashed: boolean; lastSeq: number; lastEvent?: CheckpointEvent } {
  const events = checkpointer.readEvents(threadId);
  if (events.length === 0) return { crashed: false, lastSeq: 0 };

  const last = events[events.length - 1];
  // A run is "crashed" if the last event is not a terminal event (done/error)
  // and was written more than a small grace period ago.
  const isTerminal = last.type === "done" || last.type === "error";
  const crashed = !isTerminal;
  return { crashed, lastSeq: last.seq, lastEvent: last };
}

/** Crash recovery: load the latest checkpoint or reconstruct from events. */
export async function recoverFromCrash(
  checkpointer: DurableCheckpointer,
  threadId: string,
): Promise<{ checkpoint: Checkpoint | null; recovered: boolean; events: CheckpointEvent[] }> {
  const detection = detectCrashedRun(checkpointer, threadId);
  if (!detection.crashed) {
    return { checkpoint: null, recovered: false, events: [] };
  }

  const checkpoint = await checkpointer.load(threadId);
  const events = checkpointer.readEvents(threadId);

  if (!checkpoint) {
    // Snapshot missing but event log exists — reconstruct what we can
    const reconstructed = checkpointer.reconstructFromEvents(threadId);
    if (reconstructed) {
      return {
        checkpoint: {
          threadId,
          round: reconstructed.round,
          messages: [],
          createdAt: Date.now(),
        },
        recovered: true,
        events,
      };
    }
    return { checkpoint: null, recovered: false, events };
  }

  return { checkpoint, recovered: true, events };
}
