/**
 * Batched Operations for Unified Node Type
 *
 * BLAS-optimized message passing and scoring operations.
 * Uses single matmul calls instead of per-node loops.
 *
 * Key insight: Graph structure is fixed, only embeddings change.
 * Pre-build incidence matrices once, reuse for all forward passes.
 *
 * @module shgat/core/batched-ops
 */

import * as math from "../utils/math.ts";
import type { BatchedEmbeddings, Node } from "./types.ts";
import {
  batchGetEmbeddingsByLevel,
  buildIncidenceMatrix,
  groupNodesByLevel,
} from "./types.ts";

// ============================================================================
// Types
// ============================================================================

/**
 * Pre-computed graph structure for batched operations
 *
 * Build once, reuse for all forward passes.
 */
export interface BatchedGraphStructure {
  /** Nodes grouped by level */
  nodesByLevel: Map<number, Node[]>;
  /** Embeddings matrix per level: level → [numNodes × dim] */
  embeddingsByLevel: Map<number, BatchedEmbeddings>;
  /** Incidence matrices: parentLevel → {matrix, childIndex, parentIndex} */
  incidenceMatrices: Map<number, {
    matrix: number[][];
    childIndex: Map<string, number>;
    parentIndex: Map<string, number>;
  }>;
  /** Max hierarchy level */
  maxLevel: number;
  /** Embedding dimension */
  embDim: number;
}

/**
 * Result of batched forward pass
 */
export interface BatchedForwardResult {
  /** Final embeddings per level after message passing */
  E: Map<number, number[][]>;
  /** Attention weights upward: level → [numChildren × numParents] */
  attentionUp: Map<number, number[][]>;
  /** Attention weights downward: level → [numParents × numChildren] */
  attentionDown: Map<number, number[][]>;
}

/**
 * Cache for batched K-head scoring
 */
export interface BatchedScoringCache {
  /** K vectors per level: level → [numNodes × scoringDim] */
  K_byLevel: Map<number, number[][]>;
  /** Node ID → (level, index) mapping */
  nodeIndex: Map<string, { level: number; idx: number }>;
}

// ============================================================================
// Graph Structure Pre-computation
// ============================================================================

/**
 * Pre-compute graph structure for batched operations
 *
 * Call once when graph changes, reuse for all forward passes.
 *
 * @param nodes Map of all nodes
 * @returns BatchedGraphStructure for efficient operations
 */
export function precomputeGraphStructure(
  nodes: Map<string, Node>,
): BatchedGraphStructure {
  const nodesByLevel = groupNodesByLevel(nodes);

  // Find max level and embedding dim
  let maxLevel = 0;
  let embDim = 0;
  for (const node of nodes.values()) {
    if (node.level > maxLevel) maxLevel = node.level;
    if (node.embedding.length > embDim) embDim = node.embedding.length;
  }

  // Pre-compute embeddings matrices per level
  const embeddingsByLevel = new Map<number, BatchedEmbeddings>();
  for (let level = 0; level <= maxLevel; level++) {
    embeddingsByLevel.set(level, batchGetEmbeddingsByLevel(nodes, level));
  }

  // Pre-compute incidence matrices
  const incidenceMatrices = new Map<number, {
    matrix: number[][];
    childIndex: Map<string, number>;
    parentIndex: Map<string, number>;
  }>();
  for (let level = 1; level <= maxLevel; level++) {
    incidenceMatrices.set(level, buildIncidenceMatrix(nodes, level - 1, level));
  }

  return {
    nodesByLevel,
    embeddingsByLevel,
    incidenceMatrices,
    maxLevel,
    embDim,
  };
}

// ============================================================================
// Batched Message Passing
// ============================================================================

/**
 * Batched upward pass: aggregate children → parents
 *
 * E_parent[p] = Σ_c∈children(p) softmax(A[c,p]) * E_child[c]
 *
 * Optimized: E_parent = softmax(A)^T @ E_child  (1 matmul)
 *
 * @param E_child Child embeddings [numChildren × dim]
 * @param incidence Incidence matrix [numChildren × numParents]
 * @param temperature Softmax temperature (lower = sharper attention)
 * @returns Parent embeddings [numParents × dim] and attention weights
 */
export function batchedUpwardPass(
  E_child: number[][],
  incidence: number[][],
  temperature: number = 1.0,
): { E_parent: number[][]; attention: number[][] } {
  const numChildren = E_child.length;
  const numParents = incidence[0]?.length ?? 0;

  if (numChildren === 0 || numParents === 0) {
    return { E_parent: [], attention: [] };
  }

  // Compute attention: softmax over children for each parent
  // attention[c][p] = exp(A[c][p]/τ) / Σ_c' exp(A[c'][p]/τ)
  const attention: number[][] = new Array(numChildren);

  for (let c = 0; c < numChildren; c++) {
    attention[c] = new Array(numParents);
  }

  // Column-wise softmax (per parent)
  for (let p = 0; p < numParents; p++) {
    // Gather column values
    const col: number[] = new Array(numChildren);
    for (let c = 0; c < numChildren; c++) {
      col[c] = incidence[c][p] / temperature;
    }

    // Softmax
    const softmaxCol = math.softmax(col);

    // Store back
    for (let c = 0; c < numChildren; c++) {
      attention[c][p] = softmaxCol[c];
    }
  }

  // E_parent = attention^T @ E_child
  // [numParents × numChildren] @ [numChildren × dim] = [numParents × dim]
  const attention_T = math.transpose(attention);
  const E_parent = math.matmul(attention_T, E_child);

  return { E_parent, attention };
}

/**
 * Batched downward pass: propagate parents → children
 *
 * E_child'[c] = E_child[c] + α * Σ_p∈parents(c) softmax(A[c,p]) * E_parent[p]
 *
 * Optimized: E_child' = E_child + α * softmax(A) @ E_parent  (1 matmul)
 *
 * @param E_child Current child embeddings [numChildren × dim]
 * @param E_parent Parent embeddings [numParents × dim]
 * @param incidence Incidence matrix [numChildren × numParents]
 * @param residual Residual weight (0-1, how much original to keep)
 * @param temperature Softmax temperature
 * @returns Updated child embeddings and attention weights
 */
export function batchedDownwardPass(
  E_child: number[][],
  E_parent: number[][],
  incidence: number[][],
  residual: number = 0.3,
  temperature: number = 1.0,
): { E_child_updated: number[][]; attention: number[][] } {
  const numChildren = E_child.length;
  const numParents = E_parent.length;

  if (numChildren === 0 || numParents === 0) {
    return { E_child_updated: E_child, attention: [] };
  }

  // Compute attention: softmax over parents for each child
  // attention[c][p] = exp(A[c][p]/τ) / Σ_p' exp(A[c][p']/τ)
  const attention: number[][] = new Array(numChildren);

  for (let c = 0; c < numChildren; c++) {
    // Row-wise softmax (per child)
    const row = incidence[c].map((v) => v / temperature);
    attention[c] = math.softmax(row);
  }

  // propagated = attention @ E_parent
  // [numChildren × numParents] @ [numParents × dim] = [numChildren × dim]
  const propagated = math.matmul(attention, E_parent);

  // E_child' = residual * E_child + (1-residual) * propagated
  const dim = E_child[0]?.length ?? 0;
  const E_child_updated: number[][] = new Array(numChildren);

  for (let c = 0; c < numChildren; c++) {
    E_child_updated[c] = new Array(dim);
    for (let d = 0; d < dim; d++) {
      E_child_updated[c][d] =
        residual * E_child[c][d] + (1 - residual) * propagated[c][d];
    }
  }

  return { E_child_updated, attention };
}

/**
 * Full batched forward pass through all hierarchy levels
 *
 * 1. Upward: level 0 → 1 → ... → maxLevel
 * 2. Downward: maxLevel → ... → 1 → 0
 *
 * @param structure Pre-computed graph structure
 * @param temperature Attention temperature
 * @param residual Residual weight for downward pass
 * @returns Final embeddings per level and attention weights
 */
export function batchedForward(
  structure: BatchedGraphStructure,
  temperature: number = 1.0,
  residual: number = 0.3,
): BatchedForwardResult {
  const { embeddingsByLevel, incidenceMatrices, maxLevel } = structure;

  // Clone initial embeddings
  const E = new Map<number, number[][]>();
  for (const [level, batch] of embeddingsByLevel) {
    E.set(level, batch.matrix.map((row) => [...row]));
  }

  const attentionUp = new Map<number, number[][]>();
  const attentionDown = new Map<number, number[][]>();

  // === UPWARD PASS ===
  for (let level = 1; level <= maxLevel; level++) {
    const incidence = incidenceMatrices.get(level);
    if (!incidence) continue;

    const E_child = E.get(level - 1) ?? [];
    const { E_parent, attention } = batchedUpwardPass(
      E_child,
      incidence.matrix,
      temperature,
    );

    // Merge with existing embeddings at parent level (if any)
    const E_existing = E.get(level) ?? [];
    if (E_existing.length > 0 && E_parent.length > 0) {
      // Average aggregated with original
      const dim = E_parent[0]?.length ?? 0;
      for (let i = 0; i < E_existing.length; i++) {
        for (let d = 0; d < dim; d++) {
          E_existing[i][d] = 0.5 * E_existing[i][d] + 0.5 * (E_parent[i]?.[d] ?? 0);
        }
      }
    } else if (E_parent.length > 0) {
      E.set(level, E_parent);
    }

    attentionUp.set(level, attention);
  }

  // === DOWNWARD PASS ===
  for (let level = maxLevel; level >= 1; level--) {
    const incidence = incidenceMatrices.get(level);
    if (!incidence) continue;

    const E_child = E.get(level - 1) ?? [];
    const E_parent = E.get(level) ?? [];

    const { E_child_updated, attention } = batchedDownwardPass(
      E_child,
      E_parent,
      incidence.matrix,
      residual,
      temperature,
    );

    E.set(level - 1, E_child_updated);
    attentionDown.set(level, attention);
  }

  return { E, attentionUp, attentionDown };
}

// ============================================================================
// Batched K-Head Scoring
// ============================================================================

/**
 * Pre-compute K vectors for all nodes (all levels)
 *
 * K = E @ W_k^T for each head
 *
 * @param structure Graph structure with embeddings
 * @param W_k Key projection matrices [numHeads][scoringDim × embDim]
 * @returns K vectors per head per level, and node index
 */
export function precomputeAllK(
  structure: BatchedGraphStructure,
  W_k: number[][][],
): {
  K_byHead: Map<number, Map<number, number[][]>>; // head → level → [N × scoringDim]
  nodeIndex: Map<string, { level: number; idx: number }>;
} {
  const numHeads = W_k.length;
  const K_byHead = new Map<number, Map<number, number[][]>>();
  const nodeIndex = new Map<string, { level: number; idx: number }>();

  // Build node index
  for (const [level, batch] of structure.embeddingsByLevel) {
    for (let i = 0; i < batch.ids.length; i++) {
      nodeIndex.set(batch.ids[i], { level, idx: i });
    }
  }

  // Compute K for each head
  for (let h = 0; h < numHeads; h++) {
    const K_byLevel = new Map<number, number[][]>();

    for (const [level, batch] of structure.embeddingsByLevel) {
      if (batch.matrix.length === 0) continue;

      // K = E @ W_k^T: [N × embDim] @ [embDim × scoringDim] = [N × scoringDim]
      const K = math.matmulTranspose(batch.matrix, W_k[h]);
      K_byLevel.set(level, K);
    }

    K_byHead.set(h, K_byLevel);
  }

  return { K_byHead, nodeIndex };
}

/**
 * Batch score all nodes against a batch of intents
 *
 * scores[intent][node] = sigmoid(Q[intent] · K[node] / √dim)
 *
 * @param Q_batch Query vectors [batchSize × scoringDim]
 * @param K_all K vectors for all nodes [numNodes × scoringDim]
 * @param nodeIds Node IDs in order of K_all rows
 * @returns Scores map: nodeId → [batchSize scores]
 */
export function batchScoreAllNodes(
  Q_batch: number[][],
  K_all: number[][],
  nodeIds: string[],
): Map<string, number[]> {
  const batchSize = Q_batch.length;
  const numNodes = K_all.length;
  const scoringDim = K_all[0]?.length ?? 1;
  const scale = Math.sqrt(scoringDim);

  // scores = Q @ K^T / √dim: [batch × scoringDim] @ [scoringDim × N] = [batch × N]
  const logitsMatrix = math.matmulTranspose(Q_batch, K_all);

  // Build result map with sigmoid
  const scores = new Map<string, number[]>();

  for (let n = 0; n < numNodes; n++) {
    const nodeScores: number[] = new Array(batchSize);
    for (let b = 0; b < batchSize; b++) {
      const logit = logitsMatrix[b][n] / scale;
      nodeScores[b] = math.sigmoid(logit);
    }
    scores.set(nodeIds[n], nodeScores);
  }

  return scores;
}

/**
 * Full batched K-head scoring for all nodes
 *
 * @param intents Batch of intent embeddings [batchSize × embDim]
 * @param W_intent Intent projection [hiddenDim × embDim]
 * @param W_q Query projections [numHeads][scoringDim × hiddenDim]
 * @param K_byHead Pre-computed K vectors from precomputeAllK
 * @param nodeIndex Node ID → (level, idx) mapping
 * @param structure Graph structure
 * @returns Average scores across heads: nodeId → [batchSize scores]
 */
export function batchedKHeadScoring(
  intents: number[][],
  W_intent: number[][],
  W_q: number[][][],
  K_byHead: Map<number, Map<number, number[][]>>,
  nodeIndex: Map<string, { level: number; idx: number }>,
  structure: BatchedGraphStructure,
): Map<string, number[]> {
  const batchSize = intents.length;
  const numHeads = W_q.length;

  // Project intents: [batch × embDim] @ [embDim × hiddenDim] = [batch × hiddenDim]
  const intentsProjected = math.matmulTranspose(intents, W_intent);

  // Accumulate scores across heads
  const scoresAccum = new Map<string, number[]>();

  // Initialize with zeros
  for (const nodeId of nodeIndex.keys()) {
    scoresAccum.set(nodeId, new Array(batchSize).fill(0));
  }

  // Score for each head
  for (let h = 0; h < numHeads; h++) {
    // Q = intentsProjected @ W_q^T: [batch × hidden] @ [hidden × scoringDim]
    const Q_batch = math.matmulTranspose(intentsProjected, W_q[h]);

    const K_byLevel = K_byHead.get(h);
    if (!K_byLevel) continue;

    // Score all nodes at each level
    for (const [level, batch] of structure.embeddingsByLevel) {
      const K_level = K_byLevel.get(level);
      if (!K_level || K_level.length === 0) continue;

      const levelScores = batchScoreAllNodes(Q_batch, K_level, batch.ids);

      // Accumulate
      for (const [nodeId, scores] of levelScores) {
        const accum = scoresAccum.get(nodeId)!;
        for (let b = 0; b < batchSize; b++) {
          accum[b] += scores[b] / numHeads;
        }
      }
    }
  }

  return scoresAccum;
}

// ============================================================================
// Batched Gradient Computation (for training)
// ============================================================================

/**
 * Batched backward through K-head scoring
 *
 * Computes gradients for W_q, W_k, and embeddings.
 *
 * @param dScores Gradient of loss w.r.t. scores: nodeId → [batchSize]
 * @param Q_batch Query vectors [batchSize × scoringDim]
 * @param K_byLevel K vectors: level → [numNodes × scoringDim]
 * @param intentsProjected Projected intents [batchSize × hiddenDim]
 * @param structure Graph structure
 * @param W_q Query projection [scoringDim × hiddenDim]
 * @param W_k Key projection [scoringDim × embDim]
 * @returns Gradients for W_q, W_k, intents, and embeddings
 */
export function batchedBackwardKHead(
  dScores: Map<string, number[]>,
  Q_batch: number[][],
  K_byLevel: Map<number, number[][]>,
  intentsProjected: number[][],
  structure: BatchedGraphStructure,
  W_q: number[][],
  W_k: number[][],
): {
  dW_q: number[][];
  dW_k: number[][];
  dIntents: number[][];
  dE: Map<number, number[][]>;
} {
  const batchSize = Q_batch.length;
  const scoringDim = W_q.length;
  const hiddenDim = W_q[0]?.length ?? 0;
  const embDim = W_k[0]?.length ?? 0;
  const scale = Math.sqrt(scoringDim);

  // Initialize gradients
  const dW_q: number[][] = Array.from({ length: scoringDim }, () =>
    new Array(hiddenDim).fill(0)
  );
  const dW_k: number[][] = Array.from({ length: scoringDim }, () =>
    new Array(embDim).fill(0)
  );
  const dIntents: number[][] = Array.from({ length: batchSize }, () =>
    new Array(hiddenDim).fill(0)
  );
  const dE = new Map<number, number[][]>();

  // Initialize dE per level
  for (const [level, batch] of structure.embeddingsByLevel) {
    dE.set(
      level,
      Array.from({ length: batch.ids.length }, () => new Array(embDim).fill(0)),
    );
  }

  // Backprop through each node
  for (const [level, batch] of structure.embeddingsByLevel) {
    const K_level = K_byLevel.get(level);
    const dE_level = dE.get(level);
    if (!K_level || !dE_level) continue;

    for (let n = 0; n < batch.ids.length; n++) {
      const nodeId = batch.ids[n];
      const dScore = dScores.get(nodeId);
      if (!dScore) continue;

      const K = K_level[n];
      const E = batch.matrix[n];

      for (let b = 0; b < batchSize; b++) {
        const Q = Q_batch[b];
        const intent = intentsProjected[b];

        // dLoss/dLogit = dScore * sigmoid'(logit) / scale
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        const dotQK = math.dot(Q, K);
        const logit = dotQK / scale;
        const sig = math.sigmoid(logit);
        const dLogit = dScore[b] * sig * (1 - sig) / scale;

        // dQ = dLogit * K
        // dK = dLogit * Q
        for (let d = 0; d < scoringDim; d++) {
          const dQ_d = dLogit * K[d];
          const dK_d = dLogit * Q[d];

          // dW_q[d] += dQ_d * intent
          // dW_k[d] += dK_d * E
          for (let h = 0; h < hiddenDim; h++) {
            dW_q[d][h] += dQ_d * intent[h];
          }
          for (let e = 0; e < embDim; e++) {
            dW_k[d][e] += dK_d * E[e];
          }

          // dIntent += W_q^T @ dQ
          for (let h = 0; h < hiddenDim; h++) {
            dIntents[b][h] += W_q[d][h] * dQ_d;
          }

          // dE += W_k^T @ dK
          for (let e = 0; e < embDim; e++) {
            dE_level[n][e] += W_k[d][e] * dK_d;
          }
        }
      }
    }
  }

  return { dW_q, dW_k, dIntents, dE };
}
