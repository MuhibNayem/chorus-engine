/**
 * Trajectory Clustering & Consensus Annealer
 *
 * When multiple agents solve the same task using slightly different tool
 * sequences, the synthesizer needs a merge strategy to anneal them into a
 * single optimal pattern. This module implements k-medoids clustering over
 * trajectories, then selects the "medoid" — the trajectory with the minimum
 * average distance to all others in its cluster — as the representative.
 *
 * Algorithm:
 *   1. Compute an all-pairs similarity matrix using LCS.
 *   2. Convert similarity → distance (distance = 1 - similarity).
 *   3. Run k-medoids clustering (auto-k via sqrt heuristic).
 *   4. Select the medoid per cluster (min average distance).
 *   5. Return clustered trajectories + medoid indices.
 */

import type { ToolTrajectory } from "./types.js";
import { longestCommonSubsequence } from "./synthesizer.js";

/** A cluster of trajectories with its medoid (representative). */
export interface TrajectoryCluster {
  /** Trajectories in this cluster. */
  members: ToolTrajectory[];
  /** Index of the medoid within `members`. */
  medoidIndex: number;
  /** Average intra-cluster similarity. */
  cohesion: number;
}

export interface ClusteringOptions {
  /** Number of clusters. If omitted, auto-computed as ceil(sqrt(n/2)). */
  k?: number;
  /** Minimum similarity for two trajectories to be considered related. */
  similarityThreshold?: number;
  /** Maximum iterations for k-medoids convergence. */
  maxIterations?: number;
}

/** Compute pairwise LCS similarity for all trajectory pairs. */
function computeSimilarityMatrix(trajectories: ToolTrajectory[]): number[][] {
  const n = trajectories.length;
  const matrix: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    matrix[i][i] = 1.0;
    const namesI = trajectories[i].tools.map((t) => t.name);

    for (let j = i + 1; j < n; j++) {
      const namesJ = trajectories[j].tools.map((t) => t.name);
      const lcs = longestCommonSubsequence(namesI, namesJ);
      const maxLen = Math.max(namesI.length, namesJ.length);
      const sim = maxLen === 0 ? 1.0 : lcs.length / maxLen;
      matrix[i][j] = sim;
      matrix[j][i] = sim;
    }
  }

  return matrix;
}

/** Convert similarity matrix to distance matrix. */
function toDistanceMatrix(similarity: number[][]): number[][] {
  return similarity.map((row) => row.map((s) => 1 - s));
}

/** Find the medoid of a cluster: the member with minimum average distance to others. */
function findMedoid(clusterIndices: number[], distMatrix: number[][]): number {
  let bestIdx = clusterIndices[0];
  let bestAvg = Infinity;

  for (const i of clusterIndices) {
    const avg =
      clusterIndices.reduce((sum, j) => sum + distMatrix[i][j], 0) /
      clusterIndices.length;
    if (avg < bestAvg) {
      bestAvg = avg;
      bestIdx = i;
    }
  }

  return bestIdx;
}

/** Compute average intra-cluster distance (cohesion; higher = tighter). */
function clusterCohesion(clusterIndices: number[], simMatrix: number[][]): number {
  if (clusterIndices.length <= 1) return 1.0;
  let sum = 0;
  let count = 0;
  for (const i of clusterIndices) {
    for (const j of clusterIndices) {
      if (i !== j) {
        sum += simMatrix[i][j];
        count++;
      }
    }
  }
  return sum / count;
}

/**
 * K-medoids clustering. Simpler and more robust than k-means for sequence data
 * because it uses actual trajectories as centroids rather than averaging.
 */
function kMedoids(
  distMatrix: number[][],
  k: number,
  maxIterations: number,
): number[][] {
  const n = distMatrix.length;
  if (k >= n) return distMatrix.map((_, i) => [i]);

  // Deterministic farthest-first initialization for diversity
  const medoids = new Set<number>();
  medoids.add(0); // start with first point
  while (medoids.size < k) {
    let bestIdx = 0;
    let bestDist = -1;
    for (let i = 0; i < n; i++) {
      if (medoids.has(i)) continue;
      const minDistToMedoids = Math.min(...[...medoids].map((m) => distMatrix[i][m]));
      if (minDistToMedoids > bestDist) {
        bestDist = minDistToMedoids;
        bestIdx = i;
      }
    }
    medoids.add(bestIdx);
  }
  let medoidArray = [...medoids];

  for (let iter = 0; iter < maxIterations; iter++) {
    // Assignment step: each point goes to nearest medoid
    const clusters: number[][] = medoidArray.map(() => []);
    for (let i = 0; i < n; i++) {
      let bestMedoid = 0;
      let bestDist = Infinity;
      for (let m = 0; m < medoidArray.length; m++) {
        const d = distMatrix[i][medoidArray[m]];
        if (d < bestDist) {
          bestDist = d;
          bestMedoid = m;
        }
      }
      clusters[bestMedoid].push(i);
    }

    // Update step: find new medoid per cluster
    const newMedoids = clusters.map((cluster) =>
      cluster.length > 0 ? findMedoid(cluster, distMatrix) : medoidArray[clusters.indexOf(cluster)],
    );

    // Convergence check
    if (newMedoids.every((m, i) => m === medoidArray[i])) {
      return clusters;
    }
    medoidArray = newMedoids;
  }

  // Final assignment
  const clusters: number[][] = medoidArray.map(() => []);
  for (let i = 0; i < n; i++) {
    let bestMedoid = 0;
    let bestDist = Infinity;
    for (let m = 0; m < medoidArray.length; m++) {
      const d = distMatrix[i][medoidArray[m]];
      if (d < bestDist) {
        bestDist = d;
        bestMedoid = m;
      }
    }
    clusters[bestMedoid].push(i);
  }
  return clusters;
}

/**
 * Cluster trajectories and return each cluster with its medoid.
 *
 * If `k` is not provided, it defaults to `ceil(sqrt(n / 2))` which is a
 * common heuristic for moderate dataset sizes.
 */
export function clusterTrajectories(
  trajectories: ToolTrajectory[],
  opts: ClusteringOptions = {},
): TrajectoryCluster[] {
  if (trajectories.length === 0) return [];
  if (trajectories.length === 1) {
    return [{ members: trajectories, medoidIndex: 0, cohesion: 1.0 }];
  }

  const simMatrix = computeSimilarityMatrix(trajectories);
  const distMatrix = toDistanceMatrix(simMatrix);

  const defaultK = Math.ceil(Math.sqrt(trajectories.length / 2));
  const k = Math.max(1, Math.min(opts.k ?? defaultK, trajectories.length - 1));
  const maxIterations = opts.maxIterations ?? 100;

  const clusters = kMedoids(distMatrix, k, maxIterations);

  return clusters
    .filter((indices) => indices.length > 0)
    .map((indices) => {
      const members = indices.map((i) => trajectories[i]);
      const medoidGlobalIdx = findMedoid(indices, distMatrix);
      const medoidIndex = indices.indexOf(medoidGlobalIdx);
      const cohesion = clusterCohesion(indices, simMatrix);
      return { members, medoidIndex, cohesion };
    });
}

/**
 * Pick the best cluster (highest cohesion) and return its medoid trajectory.
 * This is the entry-point for the synthesizer: given a set of candidate
 * trajectories, find the consensus representative.
 */
export function selectConsensusTrajectory(
  trajectories: ToolTrajectory[],
  opts?: ClusteringOptions,
): { trajectory: ToolTrajectory; cluster: TrajectoryCluster } | null {
  const clusters = clusterTrajectories(trajectories, opts);
  if (clusters.length === 0) return null;

  // Pick the cluster with the highest cohesion (tightest grouping)
  const best = clusters.reduce((a, b) => (a.cohesion >= b.cohesion ? a : b));
  return { trajectory: best.members[best.medoidIndex], cluster: best };
}
