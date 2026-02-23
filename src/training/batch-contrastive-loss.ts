/**
 * Batch Contrastive Loss — Forward & Backward (Plain JS + OpenBLAS)
 *
 * In-batch negative sampling with symmetric cross-entropy.
 * Replaces the per-example InfoNCE loop for ~2-4x speedup.
 *
 * Forward:
 *   sim[i][j] = mean_h( Q_h[i] · K_h[j] ) / (√dim × τ)
 *   loss = (CE_rows + CE_cols) / 2 with labels = I
 *
 * Backward:
 *   Analytical gradients through symmetric CE → similarity → Q/K projections
 *
 * @module shgat-tf/training/batch-contrastive-loss
 */

import type { SHGATConfig } from "../core/types.ts";
import type { HeadParams } from "../initialization/parameters.ts";
import type { KHeadGradientAccumulators } from "./multi-level-trainer-khead.ts";
import * as math from "../utils/math.ts";

// ============================================================================
// Types
// ============================================================================

export interface BatchContrastiveCache {
  /** Per-head Q projections: [numHeads][B][scoringDim] */
  Q_heads: number[][][];
  /** Per-head K projections: [numHeads][B][scoringDim] */
  K_heads: number[][][];
  /** Scaled logits: [B][B] */
  logits: number[][];
  /** Row-wise softmax: [B][B] */
  softmax_rows: number[][];
  /** Col-wise softmax: [B][B] */
  softmax_cols: number[][];
  /** Temperature used */
  temperature: number;
  /** Scale factor 1/√scoringDim */
  scale: number;
  /** Input intentsProjected: [B][inputDim] */
  intentsProjected: number[][];
  /** Input nodeEmbeddings: [B][inputDim] */
  nodeEmbeddings: number[][];
  /** Batch size */
  B: number;
}

// ============================================================================
// Forward
// ============================================================================

/**
 * Batch contrastive forward pass with in-batch negatives.
 *
 * @param intentsProjected [B][inputDim] — already projected by W_intent
 * @param nodeEmbeddings [B][inputDim] — tool embeddings (positive at index i for example i)
 * @param headParams Array of HeadParams (one per head)
 * @param config SHGAT config
 * @param temperature Contrastive temperature τ
 * @returns loss (scalar) and cache for backward
 */
export function batchContrastiveForward(
  intentsProjected: number[][],
  nodeEmbeddings: number[][],
  headParams: HeadParams[],
  config: SHGATConfig,
  temperature: number,
): { loss: number; cache: BatchContrastiveCache } {
  const { numHeads } = config;
  const B = intentsProjected.length;
  const scoringDim = headParams[0].W_q.length;
  const scale = 1.0 / Math.sqrt(scoringDim);

  // 1. Per-head Q/K projections
  const Q_heads: number[][][] = [];
  const K_heads: number[][][] = [];

  for (let h = 0; h < numHeads; h++) {
    const Q_h: number[][] = [];
    const K_h: number[][] = [];
    for (let i = 0; i < B; i++) {
      Q_h.push(math.matVecBlas(headParams[h].W_q, intentsProjected[i]));
      K_h.push(math.matVecBlas(headParams[h].W_k, nodeEmbeddings[i]));
    }
    Q_heads.push(Q_h);
    K_heads.push(K_h);
  }

  // 2. Similarity matrix: sim_avg[i][j] = mean_h( dot(Q_h[i], K_h[j]) )
  const sim: number[][] = Array.from({ length: B }, () => new Array(B).fill(0));
  for (let h = 0; h < numHeads; h++) {
    for (let i = 0; i < B; i++) {
      for (let j = 0; j < B; j++) {
        sim[i][j] += math.dot(Q_heads[h][i], K_heads[h][j]);
      }
    }
  }
  // Average over heads and apply scaling
  const invHeads = 1.0 / numHeads;
  const logits: number[][] = Array.from({ length: B }, () => new Array(B).fill(0));
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < B; j++) {
      logits[i][j] = sim[i][j] * invHeads * scale / temperature;
    }
  }

  // 3. Softmax row-wise and col-wise
  const softmax_rows: number[][] = [];
  for (let i = 0; i < B; i++) {
    softmax_rows.push(math.softmax(logits[i]));
  }

  // For column-wise softmax, transpose, softmax each row, transpose back
  const logits_T: number[][] = Array.from({ length: B }, (_, j) =>
    Array.from({ length: B }, (_, i) => logits[i][j])
  );
  const softmax_cols_T: number[][] = [];
  for (let j = 0; j < B; j++) {
    softmax_cols_T.push(math.softmax(logits_T[j]));
  }
  // softmax_cols[j][i] = probability of row i in column j
  const softmax_cols = softmax_cols_T; // [j][i] indexing

  // 4. Symmetric cross-entropy loss
  // loss1 = -mean_i(log(softmax_rows[i][i]))
  // loss2 = -mean_j(log(softmax_cols[j][j]))
  let loss1 = 0;
  let loss2 = 0;
  for (let i = 0; i < B; i++) {
    loss1 -= Math.log(Math.max(softmax_rows[i][i], 1e-12));
    loss2 -= Math.log(Math.max(softmax_cols[i][i], 1e-12));
  }
  loss1 /= B;
  loss2 /= B;
  const loss = (loss1 + loss2) / 2;

  return {
    loss,
    cache: {
      Q_heads,
      K_heads,
      logits,
      softmax_rows,
      softmax_cols,
      temperature,
      scale,
      intentsProjected,
      nodeEmbeddings,
      B,
    },
  };
}

// ============================================================================
// Backward
// ============================================================================

/**
 * Batch contrastive backward pass.
 *
 * Computes gradients through symmetric CE → logits → Q/K projections.
 * Accumulates dW_q and dW_k into the gradient accumulators.
 *
 * @returns dIntentsProjected and dNodeEmbeddings for upstream backprop
 */
export function batchContrastiveBackward(
  cache: BatchContrastiveCache,
  headParams: HeadParams[],
  grads: KHeadGradientAccumulators,
  config: SHGATConfig,
): { dIntentsProjected: number[][]; dNodeEmbeddings: number[][] } {
  const { numHeads } = config;
  const { Q_heads, K_heads, softmax_rows, softmax_cols, temperature, scale, intentsProjected, nodeEmbeddings, B } = cache;
  const inputDim = intentsProjected[0].length;
  const nodeInputDim = nodeEmbeddings[0].length;

  // 1. dLogits from symmetric CE gradient
  // Row-wise CE: dL_rows/dLogits[i][j] = (softmax_rows[i][j] - delta(i,j)) / B
  // Col-wise CE: dL_cols/dLogits[i][j] = (softmax_cols[j][i] - delta(i,j)) / B
  //   softmax_cols is stored as [j][i], so softmax_cols[j][i] is the prob for column j, row i
  // dLogits = (dLogits_rows + dLogits_cols) / 2
  const dLogits: number[][] = Array.from({ length: B }, () => new Array(B).fill(0));
  for (let i = 0; i < B; i++) {
    for (let j = 0; j < B; j++) {
      const delta = i === j ? 1.0 : 0.0;
      const dRow = (softmax_rows[i][j] - delta) / B;
      const dCol = (softmax_cols[j][i] - delta) / B;
      dLogits[i][j] = (dRow + dCol) / 2;
    }
  }

  // 2. Chain rule through scaling:
  // logits[i][j] = sim[i][j] * (1/numHeads) * scale / temperature
  // where sim[i][j] = sum_h dot(Q_h[i], K_h[j])
  // dLoss/dSim[i][j] = dLoss/dLogits[i][j] * (1/numHeads) * scale / temperature
  const scaleFactor = (1.0 / numHeads) * scale / temperature;

  // 3. Per-head backprop
  const dIntentsProjected: number[][] = Array.from({ length: B }, () => new Array(inputDim).fill(0));
  const dNodeEmbeddings: number[][] = Array.from({ length: B }, () => new Array(nodeInputDim).fill(0));

  for (let h = 0; h < numHeads; h++) {
    const Q_h = Q_heads[h];
    const K_h = K_heads[h];
    const scoringDim = Q_h[0].length;

    // sim[i][j] += dot(Q_h[i], K_h[j])  (per head, before averaging)
    // dQ_h[i] = sum_j dSim[i][j] * K_h[j]  where dSim includes scaleFactor
    // dK_h[j] = sum_i dSim[i][j] * Q_h[i]
    const dQ_h: number[][] = Array.from({ length: B }, () => new Array(scoringDim).fill(0));
    const dK_h: number[][] = Array.from({ length: B }, () => new Array(scoringDim).fill(0));

    for (let i = 0; i < B; i++) {
      for (let j = 0; j < B; j++) {
        const d = dLogits[i][j] * scaleFactor;
        for (let s = 0; s < scoringDim; s++) {
          dQ_h[i][s] += d * K_h[j][s];
          dK_h[j][s] += d * Q_h[i][s];
        }
      }
    }

    // Accumulate dW_q[h] += sum_i outer(dQ_h[i], intentsProjected[i])
    // Accumulate dW_k[h] += sum_i outer(dK_h[i], nodeEmbeddings[i])
    for (let i = 0; i < B; i++) {
      math.outerProductAdd(grads.dW_q[h], dQ_h[i], intentsProjected[i]);
      math.outerProductAdd(grads.dW_k[h], dK_h[i], nodeEmbeddings[i]);
    }

    // Backprop to inputs: dIntent[i] += W_q^T @ dQ_h[i], dNode[i] += W_k^T @ dK_h[i]
    for (let i = 0; i < B; i++) {
      const dI = math.matVecTransposeBlas(headParams[h].W_q, dQ_h[i]);
      const dN = math.matVecTransposeBlas(headParams[h].W_k, dK_h[i]);
      for (let d = 0; d < inputDim; d++) {
        dIntentsProjected[i][d] += dI[d] ?? 0;
      }
      for (let d = 0; d < nodeInputDim; d++) {
        dNodeEmbeddings[i][d] += dN[d] ?? 0;
      }
    }
  }

  return { dIntentsProjected, dNodeEmbeddings };
}
