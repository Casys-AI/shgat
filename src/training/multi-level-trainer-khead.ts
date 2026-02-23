/**
 * SHGAT Multi-Level K-Head Scoring — Forward & Backward
 *
 * K-head attention scoring with multi-level message passing:
 * - Multi-head forward: score = sigmoid(avg(Q·K / √dim))
 * - Multi-head backward (logit path): for InfoNCE contrastive loss
 * - W_intent backward: projects intent embedding into scoring space
 *
 * Gradient application is handled by AdamOptimizer (see adam-optimizer.ts).
 * MP backward is handled by MultiLevelOrchestrator.backwardMultiLevel().
 *
 * @module graphrag/algorithms/shgat/training/multi-level-trainer-khead
 */

import type { SHGATConfig } from "../core/types.ts";
import type { HeadParams } from "../initialization/parameters.ts";
import * as math from "../utils/math.ts";
const { zerosLike2D } = math;
import {
  initMultiLevelGradients,
  type MultiLevelGradientAccumulators,
  resetMultiLevelGradients,
} from "./multi-level-trainer.ts";
import type { LevelParams } from "../core/types.ts";

// ============================================================================
// Types
// ============================================================================

/**
 * Gradient accumulators for K-head parameters
 */
export interface KHeadGradientAccumulators {
  /** Gradients for W_q per head: [numHeads][hiddenDim][inputDim] */
  dW_q: number[][][];
  /** Gradients for W_k per head: [numHeads][hiddenDim][inputDim] */
  dW_k: number[][][];
}

/**
 * Combined gradient accumulators for multi-level + K-head
 */
export interface MultiLevelKHeadGradientAccumulators extends MultiLevelGradientAccumulators {
  /** K-head gradients */
  khead: KHeadGradientAccumulators;
  /** W_intent gradients: [hiddenDim][embeddingDim] */
  dW_intent: number[][];
}

// ============================================================================
// Gradient Initialization
// ============================================================================

/**
 * Initialize gradient accumulators for multi-level K-head training
 *
 * @param levelParams Map of level → LevelParams
 * @param headParams Array of HeadParams (one per head)
 * @param config SHGAT config
 * @returns Initialized gradient accumulators (all zeros)
 */
export function initMultiLevelKHeadGradients(
  levelParams: Map<number, LevelParams>,
  headParams: HeadParams[],
  config: SHGATConfig,
): MultiLevelKHeadGradientAccumulators {
  const base = initMultiLevelGradients(levelParams);
  const { numHeads, hiddenDim, embeddingDim } = config;

  // Initialize K-head gradients
  const dW_q: number[][][] = [];
  const dW_k: number[][][] = [];

  for (let h = 0; h < numHeads; h++) {
    const hp = headParams[h];
    dW_q.push(zerosLike2D(hp.W_q));
    dW_k.push(zerosLike2D(hp.W_k));
  }

  return {
    ...base,
    khead: { dW_q, dW_k },
    dW_intent: Array.from({ length: hiddenDim }, () => Array(embeddingDim).fill(0)),
  };
}

/**
 * Reset gradient accumulators to zero
 */
export function resetMultiLevelKHeadGradients(
  accum: MultiLevelKHeadGradientAccumulators,
  levelParams: Map<number, LevelParams>,
  _headParams: HeadParams[],
  _config: SHGATConfig,
): void {
  resetMultiLevelGradients(accum, levelParams);

  // Zero-fill K-head gradients in place (avoids GC pressure per batch)
  for (const dWq of accum.khead.dW_q) for (const row of dWq) row.fill(0);
  for (const dWk of accum.khead.dW_k) for (const row of dWk) row.fill(0);

  // Zero-fill W_intent gradients in place
  for (const row of accum.dW_intent) row.fill(0);
}

// ============================================================================
// K-Head Scoring Forward (for training)
// ============================================================================

/**
 * Compute K-head score for a single head
 *
 * score = sigmoid(Q · K / √dim)
 * where Q = W_q @ intentProjected, K = W_k @ nodeEmbedding
 *
 * @returns score and intermediates for backprop
 */
export function computeKHeadScoreWithCache(
  intentProjected: number[],
  nodeEmbedding: number[],
  headParams: HeadParams,
  _hiddenDim: number,
): { score: number; logit: number; Q: number[]; K: number[]; dotQK: number } {
  const scoringDim = headParams.W_q.length;

  const Q = math.matVecBlas(headParams.W_q, intentProjected);
  const K = math.matVecBlas(headParams.W_k, nodeEmbedding);

  const dotQK = math.dot(Q, K);
  const scale = Math.sqrt(scoringDim);
  const logit = dotQK / scale;
  const score = math.sigmoid(logit);

  return { score, logit, Q, K, dotQK };
}

/**
 * Compute multi-head K-head scores with cache
 */
export function computeMultiHeadKHeadScoresWithCache(
  intentProjected: number[],
  nodeEmbedding: number[],
  headParams: HeadParams[],
  config: SHGATConfig,
): { scores: number[]; logits: number[]; caches: Array<{ Q: number[]; K: number[]; dotQK: number }> } {
  const scores: number[] = [];
  const logits: number[] = [];
  const caches: Array<{ Q: number[]; K: number[]; dotQK: number }> = [];

  for (let h = 0; h < config.numHeads; h++) {
    const { score, logit, Q, K, dotQK } = computeKHeadScoreWithCache(
      intentProjected,
      nodeEmbedding,
      headParams[h],
      config.hiddenDim,
    );
    scores.push(score);
    logits.push(logit);
    caches.push({ Q, K, dotQK });
  }

  return { scores, logits, caches };
}

// ============================================================================
// K-Head Backward Pass
// ============================================================================

/**
 * Backprop through multi-head K-head scoring using LOGITS (for InfoNCE)
 *
 * avgLogit = (1/numHeads) * Σ headLogits[h]
 * dLoss/dHeadLogit[h] = dLoss/dAvgLogit * (1/numHeads)
 *
 * Chain rule through each head:
 *   logit = Q · K / √dim
 *   dLogit → dQ = K/√dim, dK = Q/√dim
 *   dW_q += dQ ⊗ intent, dW_k += dK ⊗ nodeEmb
 *   dIntent = W_q^T @ dQ, dNodeEmb = W_k^T @ dK
 */
export function backpropMultiHeadKHeadLogit(
  dLoss: number,
  headCaches: Array<{ Q: number[]; K: number[]; dotQK: number }>,
  intentProjected: number[],
  nodeEmbedding: number[],
  headParams: HeadParams[],
  grads: KHeadGradientAccumulators,
  config: SHGATConfig,
): { dIntentProjected: number[]; dNodeEmbedding: number[] } {
  const { numHeads } = config;
  const inputDim = headParams[0]?.W_q[0]?.length ?? intentProjected.length;

  // dLoss/dHeadLogit[h] = dLoss * (1/numHeads) (average fusion)
  const dHeadLogit = dLoss / numHeads;

  const dIntentProjected = new Array(inputDim).fill(0);
  const dNodeEmbedding = new Array(inputDim).fill(0);

  for (let h = 0; h < numHeads; h++) {
    const { Q, K } = headCaches[h];
    const scoringDim = headParams[h].W_q.length;
    const scale = Math.sqrt(scoringDim);

    // Direct gradient: dLoss/d(Q·K) = dHeadLogit / √dim
    const dDotQK = dHeadLogit / scale;

    const dQ = K.map((k) => dDotQK * k);
    const dK = Q.map((q) => dDotQK * q);

    // Accumulate W_q, W_k gradients via outer product (BLAS accelerated)
    math.outerProductAdd(grads.dW_q[h], dQ, intentProjected);
    math.outerProductAdd(grads.dW_k[h], dK, nodeEmbedding);

    // Backprop to inputs via transpose matmul (BLAS accelerated)
    const dI = math.matVecTransposeBlas(headParams[h].W_q, dQ);
    const dC = math.matVecTransposeBlas(headParams[h].W_k, dK);

    for (let j = 0; j < inputDim; j++) {
      dIntentProjected[j] += dI[j] ?? 0;
      dNodeEmbedding[j] += dC[j] ?? 0;
    }
  }

  return { dIntentProjected, dNodeEmbedding };
}

/**
 * Backprop through W_intent projection
 *
 * intentProjected = W_intent @ intentOriginal
 * dW_intent += dIntentProjected @ intentOriginal^T (outer product)
 */
export function backpropWIntent(
  dIntentProjected: number[],
  intentOriginal: number[],
  grads: MultiLevelKHeadGradientAccumulators,
  _config: SHGATConfig,
): void {
  math.outerProductAdd(grads.dW_intent, dIntentProjected, intentOriginal);
}
