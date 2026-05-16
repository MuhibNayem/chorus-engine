import { describe, it, expect } from "vitest";
import { clusterTrajectories, selectConsensusTrajectory } from "../skills/annealer.js";
import type { ToolTrajectory } from "../skills/types.js";

function makeTrajectory(id: string, tools: string[], success = true): ToolTrajectory {
  return {
    id,
    task: "test",
    tools: tools.map((name) => ({ name, input: {}, output: "" })),
    success,
    tokens: 100,
    duration: 1000,
    timestamp: Date.now(),
  };
}

describe("TrajectoryClustering — k-medoids", () => {
  it("returns single cluster for 1 trajectory", () => {
    const trajectories = [makeTrajectory("t1", ["a", "b", "c"])];
    const clusters = clusterTrajectories(trajectories);
    expect(clusters).toHaveLength(1);
    expect(clusters[0].members).toHaveLength(1);
    expect(clusters[0].medoidIndex).toBe(0);
    expect(clusters[0].cohesion).toBe(1.0);
  });

  it("clusters similar trajectories together", () => {
    const trajectories = [
      makeTrajectory("t1", ["search", "read", "write"]),
      makeTrajectory("t2", ["search", "read", "write"]),
      makeTrajectory("t3", ["search", "read", "write"]),
      makeTrajectory("t4", ["git_status", "git_commit"]),
      makeTrajectory("t5", ["git_status", "git_commit"]),
    ];

    const clusters = clusterTrajectories(trajectories, { k: 2 });
    expect(clusters.length).toBeGreaterThanOrEqual(1);

    // The two git trajectories should end up in the same cluster
    const gitCluster = clusters.find((c) =>
      c.members.some((m) => m.id === "t4") && c.members.some((m) => m.id === "t5"),
    );
    expect(gitCluster).toBeDefined();
    expect(gitCluster!.members.length).toBe(2);

    // The search/read/write trajectories should end up together
    const searchCluster = clusters.find((c) =>
      c.members.some((m) => m.id === "t1") && c.members.some((m) => m.id === "t2"),
    );
    expect(searchCluster).toBeDefined();
    expect(searchCluster!.members.length).toBeGreaterThanOrEqual(2);
  });

  it("selects medoid as representative", () => {
    const trajectories = [
      makeTrajectory("t1", ["a", "b", "c", "d"]),
      makeTrajectory("t2", ["a", "b", "c"]),
      makeTrajectory("t3", ["a", "b", "c", "e"]),
    ];

    const consensus = selectConsensusTrajectory(trajectories);
    expect(consensus).not.toBeNull();
    expect(consensus!.cluster.members.length).toBeGreaterThanOrEqual(1);
    expect(consensus!.trajectory).toBeDefined();
  });

  it("auto-computes k when not provided", () => {
    const trajectories = Array.from({ length: 8 }, (_, i) =>
      makeTrajectory(`t${i}`, i < 4 ? ["a", "b"] : ["c", "d"]),
    );
    const clusters = clusterTrajectories(trajectories);
    expect(clusters.length).toBeGreaterThanOrEqual(1);
  });

  it("returns empty array for no trajectories", () => {
    expect(clusterTrajectories([])).toEqual([]);
  });
});
