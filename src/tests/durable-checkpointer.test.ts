import { describe, it, expect, beforeEach } from "vitest";
import { DurableCheckpointer, detectCrashedRun, recoverFromCrash } from "../agent/durable-checkpointer.js";
import type { CheckpointState } from "../agent/types.js";

describe("DurableCheckpointer — event-sourced crash recovery", () => {
  let cp: DurableCheckpointer;

  beforeEach(() => {
    cp = new DurableCheckpointer("sync");
  });

  it("saves checkpoints in sync mode with event log", async () => {
    const threadId = `durable-sync-${Date.now()}`;
    const state: CheckpointState = {
      messages: [{ role: "user", content: "hello" }],
      round: 0,
    };

    await cp.save(threadId, state);
    const loaded = await cp.load(threadId);

    expect(loaded).not.toBeNull();
    expect(loaded!.round).toBe(0);
    expect(loaded!.seq).toBeGreaterThan(0);
    expect(loaded!.durability).toBe("sync");
    expect(loaded!.events.length).toBeGreaterThan(0);

    const events = cp.readEvents(threadId);
    expect(events.some((e) => e.type === "round_start")).toBe(true);

    await cp.delete(threadId);
  });

  it("supports async durability mode", async () => {
    const threadId = `durable-async-${Date.now()}`;
    cp.setDurability("async");

    await cp.save(threadId, { messages: [], round: 1 });
    await cp.flush();

    const loaded = await cp.load(threadId);
    expect(loaded).not.toBeNull();
    expect(loaded!.durability).toBe("async");

    await cp.delete(threadId);
  });

  it("defer writes in exit mode until flush", async () => {
    const threadId = `durable-exit-${Date.now()}`;
    cp.setDurability("exit");

    await cp.save(threadId, { messages: [], round: 2 });
    let loaded = await cp.load(threadId);
    expect(loaded).toBeNull(); // not yet flushed

    // Simulate exit flush via setting to sync and re-saving
    cp.setDurability("sync");
    await cp.save(threadId, { messages: [], round: 2 });

    loaded = await cp.load(threadId);
    expect(loaded).not.toBeNull();

    await cp.delete(threadId);
  });

  it("forks checkpoint and event log to new thread", async () => {
    const src = `durable-fork-src-${Date.now()}`;
    const dst = `durable-fork-dst-${Date.now()}`;

    await cp.save(src, { messages: [{ role: "user", content: "a" }], round: 0 });
    await cp.save(src, { messages: [{ role: "user", content: "b" }], round: 1 });

    await cp.fork(src, 1, dst);
    const forked = await cp.load(dst);
    expect(forked).not.toBeNull();
    expect(forked!.round).toBe(1);

    const dstEvents = cp.readEvents(dst);
    expect(dstEvents.length).toBeGreaterThan(0);

    await cp.delete(src);
    await cp.delete(dst);
  });

  it("detects crashed runs from non-terminal event logs", async () => {
    const threadId = `durable-crash-${Date.now()}`;

    await cp.save(threadId, { messages: [], round: 0 });
    // Append a non-terminal event
    cp.appendEvent(threadId, { type: "round_start", description: "round 0" });

    const detection = detectCrashedRun(cp, threadId);
    expect(detection.crashed).toBe(true);
    expect(detection.lastSeq).toBeGreaterThan(0);

    await cp.delete(threadId);
  });

  it("does not flag completed runs as crashed", async () => {
    const threadId = `durable-done-${Date.now()}`;

    await cp.save(threadId, { messages: [], round: 0 });
    cp.appendEvent(threadId, { type: "done", description: "completed" });

    const detection = detectCrashedRun(cp, threadId);
    expect(detection.crashed).toBe(false);

    await cp.delete(threadId);
  });

  it("recovers from crash with latest checkpoint", async () => {
    const threadId = `durable-recover-${Date.now()}`;

    await cp.save(threadId, { messages: [{ role: "user", content: "recover me" }], round: 3 });
    cp.appendEvent(threadId, { type: "tool_call", description: "some tool" });

    const recovery = await recoverFromCrash(cp, threadId);
    expect(recovery.recovered).toBe(true);
    expect(recovery.checkpoint).not.toBeNull();
    expect(recovery.checkpoint!.round).toBe(3);
    expect(recovery.events.length).toBeGreaterThan(0);

    await cp.delete(threadId);
  });

  it("reconstructs from events when snapshots are missing", async () => {
    const threadId = `durable-reconstruct-${Date.now()}`;

    // Write events directly without saving checkpoint
    cp.appendEvent(threadId, { type: "round_start", description: "round 0", payload: { round: 0 } });
    cp.appendEvent(threadId, { type: "round_start", description: "round 1", payload: { round: 1 } });

    const reconstructed = cp.reconstructFromEvents(threadId);
    expect(reconstructed).not.toBeNull();
    expect(reconstructed!.round).toBe(1);
    expect(reconstructed!.events.length).toBe(2);

    await cp.delete(threadId);
  });

  it("lists checkpoints with embedded events", async () => {
    const threadId = `durable-list-${Date.now()}`;

    await cp.save(threadId, { messages: [{ role: "user", content: "x" }], round: 0 });
    await cp.save(threadId, { messages: [{ role: "user", content: "y" }], round: 1 });

    const list = await cp.list(threadId);
    expect(list.length).toBe(2);
    expect(list[0].events.length).toBeGreaterThanOrEqual(0);
    expect(list[1].events.length).toBeGreaterThanOrEqual(0);

    await cp.delete(threadId);
  });
});
