/**
 * K-Head Attention Scoring
 *
 * Optimized batch scoring functions for multi-head attention.
 * Extracted from shgat.ts for modularity.
 *
 * Now with tensor-native scoreNodes() for GPU-accelerated scoring.
 *
 * @module shgat/attention/khead-scorer
 */

import * as math from "../utils/math.ts";
import { tf, tidy } from "../tf/backend.ts";
import * as ops from "../tf/ops.ts";
import type { SHGATConfig, AttentionResult, CapabilityNode } from "../core/types.ts";
import type { HeadParams, TensorScoringParams } from "../initialization/index.ts";
import type { ProjectionHeadTFParams } from "../core/projection-head.ts";
import { projectionForward } from "../core/projection-head.ts";

// ==========================================================================
// Intent Projection
// ==========================================================================

/**
 * Project intent embedding via W_intent matrix
 */
export function projectIntent(
  intentEmbedding: number[],
  W_intent: number[][],
): number[] {
  const propagatedDim = W_intent.length;
  const result = new Array(propagatedDim).fill(0);

  for (let i = 0; i < propagatedDim; i++) {
    for (let j = 0; j < intentEmbedding.length; j++) {
      result[i] += W_intent[i][j] * intentEmbedding[j];
    }
  }

  return result;
}

// ==========================================================================
// Q/K Precomputation
// ==========================================================================

/**
 * Pre-compute Q = W_q @ intent for all heads
 *
 * OPTIMIZATION: Q only depends on intent (same for all capabilities)
 * Pre-computing saves numCaps-1 redundant matrix multiplications per head
 *
 * @returns Array of Q vectors, one per head
 */
export function precomputeQForAllHeads(
  intentProjected: number[],
  headParams: HeadParams[],
  config: SHGATConfig,
): number[][] {
  const { numHeads, hiddenDim } = config;
  const precomputedQ: number[][] = [];

  for (let h = 0; h < numHeads; h++) {
    const hp = headParams[h];
    const outputDim = hp.W_q.length;
    const wqCols = hp.W_q[0]?.length || hiddenDim;
    const inputDim = Math.min(intentProjected.length, wqCols);

    const Q = new Array(outputDim).fill(0);
    for (let i = 0; i < outputDim; i++) {
      for (let j = 0; j < inputDim; j++) {
        Q[i] += hp.W_q[i][j] * intentProjected[j];
      }
    }
    precomputedQ.push(Q);
  }

  return precomputedQ;
}

/**
 * Batch compute K vectors for all embeddings in one matmul per head
 *
 * Instead of: 105× (W_k @ cap)  → 105 small matmuls
 * We do:      E @ W_k.T         → 1 large matmul [numCaps×embDim] @ [embDim×hiddenDim]
 *
 * @param E - Capability embeddings matrix [numCaps][embDim]
 * @returns K_all[numHeads][numCaps][hiddenDim] - K vectors for all caps, all heads
 */
export function batchComputeKForAllHeads(
  E: number[][],
  headParams: HeadParams[],
  numHeads: number,
): number[][][] {
  const K_all: number[][][] = [];

  for (let h = 0; h < numHeads; h++) {
    const hp = headParams[h];
    // W_k is [hiddenDim][embDim], we need W_k.T which is [embDim][hiddenDim]
    const W_k_T = math.transpose(hp.W_k);
    // E @ W_k.T: [numCaps×embDim] @ [embDim×hiddenDim] = [numCaps×hiddenDim]
    K_all.push(math.matmul(E, W_k_T));
  }

  return K_all;
}

/**
 * Batch compute scores for all capabilities using precomputed Q and K
 *
 * @param precomputedQ - Q vectors [numHeads][hiddenDim]
 * @param K_all - K vectors [numHeads][numCaps][hiddenDim]
 * @returns scores[numCaps][numHeads]
 */
export function batchComputeScores(
  precomputedQ: number[][],
  K_all: number[][][],
  numHeads: number,
): number[][] {
  const numCaps = K_all[0]?.length || 0;
  const scores: number[][] = new Array(numCaps);

  for (let c = 0; c < numCaps; c++) {
    scores[c] = new Array(numHeads);
    for (let h = 0; h < numHeads; h++) {
      const Q = precomputedQ[h];
      const K = K_all[h][c];
      // Use cosine similarity instead of scaled dot-product
      // BGE embeddings are pre-normalized, but W_q/W_k projections aren't calibrated
      // Cosine keeps only angular (direction) information, ignoring magnitude
      scores[c][h] = math.cosineSimilarity(Q, K);
    }
  }

  return scores;
}

// ==========================================================================
// Single-Item Scoring (Legacy/Fallback)
// ==========================================================================

/**
 * Compute attention score for a single head (v1)
 *
 * Uses Query-Key attention:
 * - Q = W_q @ intentProjected
 * - K = W_k @ capEmbedding
 * - logit = Q·K / √dim (raw, no sigmoid - softmax at discover level)
 */
export function computeHeadScoreV1(
  intentProjected: number[],
  capEmbedding: number[],
  headIdx: number,
  headParams: HeadParams[],
  config: SHGATConfig,
): number {
  const hp = headParams[headIdx];
  const { hiddenDim } = config;

  const outputDim = hp.W_q.length;
  const wqCols = hp.W_q[0]?.length || hiddenDim;
  const inputDim = Math.min(intentProjected.length, capEmbedding.length, wqCols);

  const Q = new Array(outputDim).fill(0);
  const K = new Array(outputDim).fill(0);

  for (let i = 0; i < outputDim; i++) {
    for (let j = 0; j < inputDim; j++) {
      Q[i] += hp.W_q[i][j] * intentProjected[j];
      K[i] += hp.W_k[i][j] * capEmbedding[j];
    }
  }

  // Use cosine similarity instead of scaled dot-product
  return math.cosineSimilarity(Q, K);
}

/**
 * Compute multi-head scores using pre-computed Q vectors (per-item, legacy)
 *
 * @deprecated Use batchComputeKForAllHeads + batchComputeScores for ~30% speedup
 */
export function computeMultiHeadScoresWithPrecomputedQ(
  precomputedQ: number[][],
  capEmbedding: number[],
  headParams: HeadParams[],
  config: SHGATConfig,
): number[] {
  const { numHeads, hiddenDim } = config;
  const scores: number[] = [];

  for (let h = 0; h < numHeads; h++) {
    const hp = headParams[h];
    const Q = precomputedQ[h];
    const outputDim = hp.W_k.length;
    const wkCols = hp.W_k[0]?.length || hiddenDim;
    const inputDim = Math.min(capEmbedding.length, wkCols);

    const K = new Array(outputDim).fill(0);
    for (let i = 0; i < outputDim; i++) {
      for (let j = 0; j < inputDim; j++) {
        K[i] += hp.W_k[i][j] * capEmbedding[j];
      }
    }

    // Use cosine similarity instead of scaled dot-product
    scores.push(math.cosineSimilarity(Q, K));
  }

  return scores;
}

// ==========================================================================
// High-Level Scoring Functions
// ==========================================================================

/**
 * Score all capabilities using K-head attention
 *
 * @param E - Propagated capability embeddings [numCaps][embDim]
 * @param intentEmbedding - User intent embedding
 * @param capabilityNodes - Map of capability nodes
 * @param headParams - K-head parameters
 * @param W_intent - Intent projection matrix
 * @param config - SHGAT config
 * @param getToolAttention - Function to get tool attention for a cap index
 * @returns Sorted array of attention results
 */
export function scoreAllCapabilities(
  E: number[][],
  intentEmbedding: number[],
  capabilityNodes: Map<string, CapabilityNode>,
  headParams: HeadParams[],
  W_intent: number[][],
  config: SHGATConfig,
  getToolAttention?: (capIdx: number) => number[],
): AttentionResult[] {
  const results: AttentionResult[] = [];
  const { numHeads } = config;

  // PreserveDim mode: use raw intent (1024-dim) directly with W_q
  // Standard mode: project intent via W_intent (1024 → hiddenDim)
  const intentForScoring = config.preserveDim
    ? intentEmbedding
    : projectIntent(intentEmbedding, W_intent);

  // 1. Pre-compute Q for all heads
  const precomputedQ = precomputeQForAllHeads(intentForScoring, headParams, config);

  // 2. Batch compute K for all capabilities
  const K_all = batchComputeKForAllHeads(E, headParams, numHeads);

  // 3. Batch compute scores
  const allScores = batchComputeScores(precomputedQ, K_all, numHeads);

  // 4. Build results with capability metadata
  let capIdx = 0;
  for (const [capId, cap] of capabilityNodes) {
    const headScores = allScores[capIdx];

    // Fusion: simple average of head logits
    const avgScore = headScores.reduce((a, b) => a + b, 0) / numHeads;

    // Reliability multiplier based on success rate
    const reliabilityMult = cap.successRate < 0.5 ? 0.5 : (cap.successRate > 0.9 ? 1.2 : 1.0);
    const finalScore = avgScore * reliabilityMult;

    results.push({
      capabilityId: capId,
      score: finalScore,
      headWeights: new Array(numHeads).fill(1 / numHeads),
      headScores,
      recursiveContribution: 0,
      toolAttention: getToolAttention?.(capIdx) ?? [],
    });

    capIdx++;
  }

  return results.sort((a, b) => b.score - a.score);
}

/**
 * Score all tools using K-head attention
 *
 * @param H - Propagated tool embeddings [numTools][embDim]
 * @param intentEmbedding - User intent embedding
 * @param toolIds - Array of tool IDs (in same order as H)
 * @param headParams - K-head parameters
 * @param W_intent - Intent projection matrix
 * @param config - SHGAT config
 * @returns Array of tool scores
 */
export function scoreAllTools(
  H: number[][],
  intentEmbedding: number[],
  toolIds: string[],
  headParams: HeadParams[],
  W_intent: number[][],
  config: SHGATConfig,
): Array<{ toolId: string; score: number; headScores: number[] }> {
  const results: Array<{ toolId: string; score: number; headScores: number[] }> = [];
  const { numHeads } = config;

  const intentForScoring = config.preserveDim
    ? intentEmbedding
    : projectIntent(intentEmbedding, W_intent);

  const precomputedQ = precomputeQForAllHeads(intentForScoring, headParams, config);
  const K_all = batchComputeKForAllHeads(H, headParams, numHeads);
  const allScores = batchComputeScores(precomputedQ, K_all, numHeads);

  for (let i = 0; i < toolIds.length; i++) {
    const headScores = allScores[i];
    const avgScore = headScores.reduce((a, b) => a + b, 0) / numHeads;

    results.push({
      toolId: toolIds[i],
      score: avgScore,
      headScores,
    });
  }

  return results.sort((a, b) => b.score - a.score);
}

// ==========================================================================
// Unified Node Scoring (new API)
// ==========================================================================

/**
 * Result of scoring a node
 */
export interface NodeScore {
  nodeId: string;
  score: number;
  headScores: number[];
  level: number;
}

/**
 * Score nodes using K-head attention (unified API) - TENSOR-NATIVE
 *
 * This is the main scoring function for the unified Node API.
 * It replaces both scoreAllCapabilities and scoreAllTools.
 *
 * OPTIMIZED: Single array→tensor conversion at start, all ops on GPU,
 * single tensor→array conversion at end. Avoids the 20x slowdown from
 * repeated conversions in math.ts functions.
 *
 * @param embeddings - Node embeddings matrix [numNodes][embDim]
 * @param nodeIds - Array of node IDs (same order as embeddings)
 * @param levels - Array of node levels (same order as embeddings)
 * @param intentEmbedding - User intent embedding
 * @param headParams - K-head parameters
 * @param W_intent - Intent projection matrix
 * @param config - SHGAT config
 * @returns Sorted array of node scores
 */
export function scoreNodes(
  embeddings: number[][],
  nodeIds: string[],
  levels: number[],
  intentEmbedding: number[],
  headParams: HeadParams[],
  W_intent: number[][],
  config: SHGATConfig,
): NodeScore[] {
  if (embeddings.length === 0) return [];

  const { numHeads } = config;

  // All computation inside tidy() for automatic tensor cleanup
  const allScores = tidy(() => {
    // 1. Convert inputs to tensors ONCE
    const intentTensor = tf.tensor1d(intentEmbedding);
    const embeddingsTensor = tf.tensor2d(embeddings);
    const W_intentTensor = tf.tensor2d(W_intent);

    // 2. Project intent (if not preserveDim)
    const intentProjected = config.preserveDim
      ? intentTensor
      : ops.matVec(W_intentTensor, intentTensor);

    // 3. Compute Q for all heads: Q[h] = W_q[h] @ intent
    const Qs: tf.Tensor1D[] = [];
    for (let h = 0; h < numHeads; h++) {
      const W_q = tf.tensor2d(headParams[h].W_q);
      const Q_h = ops.matVec(W_q, intentProjected);
      Qs.push(Q_h);
    }
    // Stack into [numHeads, headDim]
    const Q_all = tf.stack(Qs) as tf.Tensor2D;

    // 4. Compute K for all heads and all nodes: K[h] = embeddings @ W_k[h].T
    const Ks: tf.Tensor2D[] = [];
    for (let h = 0; h < numHeads; h++) {
      const W_k = tf.tensor2d(headParams[h].W_k);
      // embeddings @ W_k.T: [numNodes, embDim] @ [embDim, headDim] = [numNodes, headDim]
      const K_h = ops.matmulTranspose(embeddingsTensor, W_k);
      Ks.push(K_h);
    }
    // Stack into [numHeads, numNodes, headDim]
    const K_all = tf.stack(Ks) as tf.Tensor3D;

    // 5. Compute cosine similarity scores for all nodes, all heads
    // Q_all: [numHeads, headDim], K_all: [numHeads, numNodes, headDim]
    // Expand Q for broadcasting: [numHeads, 1, headDim]
    const Q_expanded = Q_all.expandDims(1);

    // Dot product: Q · K
    const dotProduct = tf.sum(tf.mul(Q_expanded, K_all), 2); // [numHeads, numNodes]

    // Norms for cosine similarity
    const normQ = tf.norm(Q_all, 2, 1, true); // [numHeads, 1]
    const normK = tf.norm(K_all, 2, 2); // [numHeads, numNodes]

    // Cosine similarity = dot / (normQ * normK + eps)
    const similarity = tf.div(
      dotProduct,
      tf.add(tf.mul(normQ, normK), 1e-8)
    ); // [numHeads, numNodes]

    // Transpose to [numNodes, numHeads] and compute mean
    const scoresPerHead = tf.transpose(similarity) as tf.Tensor2D; // [numNodes, numHeads]

    return scoresPerHead.arraySync() as number[][];
  });

  // 6. Build results with metadata
  const results: NodeScore[] = [];
  for (let i = 0; i < nodeIds.length; i++) {
    const headScores = allScores[i];
    const avgScore = headScores.reduce((a, b) => a + b, 0) / numHeads;

    results.push({
      nodeId: nodeIds[i],
      score: avgScore,
      headScores,
      level: levels[i],
    });
  }

  return results.sort((a, b) => b.score - a.score);
}

/**
 * Score nodes using K-head attention with pre-initialized tensors (FULLY TENSOR-NATIVE)
 *
 * This is the optimized version that avoids array→tensor conversion on every call.
 * The weights (W_q, W_k, W_intent) are already stored as tf.Variable.
 *
 * PERFORMANCE: ~20x faster than scoreNodes() because:
 * - No array→tensor conversion for weights (done once at init)
 * - Only embeddings and intent are converted per call
 *
 * @param embeddings - Node embeddings matrix [numNodes][embDim]
 * @param nodeIds - Array of node IDs (same order as embeddings)
 * @param levels - Array of node levels (same order as embeddings)
 * @param intentEmbedding - User intent embedding
 * @param tensorParams - Pre-initialized tensor parameters
 * @param config - SHGAT config
 * @returns Sorted array of node scores
 */
export function scoreNodesTensor(
  embeddings: number[][],
  nodeIds: string[],
  levels: number[],
  intentEmbedding: number[],
  tensorParams: TensorScoringParams,
  config: SHGATConfig,
): NodeScore[] {
  if (embeddings.length === 0) return [];

  const { numHeads } = config;

  // All computation inside tidy() for automatic tensor cleanup
  const allScores = tidy(() => {
    // 1. Convert inputs to tensors (only embeddings and intent - weights are already tensors)
    const intentTensor = tf.tensor1d(intentEmbedding);
    const embeddingsTensor = tf.tensor2d(embeddings);

    // 2. Project intent (if not preserveDim)
    // W_intent is already a tf.Variable - cast to Tensor2D for ops functions
    const intentProjected = config.preserveDim
      ? intentTensor
      : ops.matVec(tensorParams.W_intent as tf.Tensor2D, intentTensor);

    // 3. Compute Q for all heads: Q[h] = W_q[h] @ intent
    // W_q is already a tf.Variable - cast to Tensor2D for ops functions
    const Qs: tf.Tensor1D[] = [];
    for (let h = 0; h < numHeads; h++) {
      const Q_h = ops.matVec(tensorParams.headParams[h].W_q as tf.Tensor2D, intentProjected);
      Qs.push(Q_h);
    }
    // Stack into [numHeads, headDim]
    const Q_all = tf.stack(Qs) as tf.Tensor2D;

    // 4. Compute K for all heads and all nodes: K[h] = embeddings @ W_k[h].T
    // W_k is already a tf.Variable - cast to Tensor2D for ops functions
    const Ks: tf.Tensor2D[] = [];
    for (let h = 0; h < numHeads; h++) {
      // embeddings @ W_k.T: [numNodes, embDim] @ [embDim, headDim] = [numNodes, headDim]
      const K_h = ops.matmulTranspose(embeddingsTensor, tensorParams.headParams[h].W_k as tf.Tensor2D);
      Ks.push(K_h);
    }
    // Stack into [numHeads, numNodes, headDim]
    const K_all = tf.stack(Ks) as tf.Tensor3D;

    // 5. Compute cosine similarity scores for all nodes, all heads
    // Q_all: [numHeads, headDim], K_all: [numHeads, numNodes, headDim]
    // Expand Q for broadcasting: [numHeads, 1, headDim]
    const Q_expanded = Q_all.expandDims(1);

    // Dot product: Q · K
    const dotProduct = tf.sum(tf.mul(Q_expanded, K_all), 2); // [numHeads, numNodes]

    // Norms for cosine similarity
    const normQ = tf.norm(Q_all, 2, 1, true); // [numHeads, 1]
    const normK = tf.norm(K_all, 2, 2); // [numHeads, numNodes]

    // Cosine similarity = dot / (normQ * normK + eps)
    const similarity = tf.div(
      dotProduct,
      tf.add(tf.mul(normQ, normK), 1e-8)
    ); // [numHeads, numNodes]

    // Transpose to [numNodes, numHeads]
    const scoresPerHead = tf.transpose(similarity) as tf.Tensor2D; // [numNodes, numHeads]

    return scoresPerHead.arraySync() as number[][];
  });

  // 6. Build results with metadata
  const results: NodeScore[] = [];
  for (let i = 0; i < nodeIds.length; i++) {
    const headScores = allScores[i];
    const avgScore = headScores.reduce((a, b) => a + b, 0) / numHeads;

    results.push({
      nodeId: nodeIds[i],
      score: avgScore,
      headScores,
      level: levels[i],
    });
  }

  return results.sort((a, b) => b.score - a.score);
}

/**
 * Score nodes using K-head attention - DIRECT TENSOR VERSION
 *
 * Same as scoreNodesTensor but accepts tf.Tensor2D directly to avoid
 * array→tensor conversion overhead. Use this when embeddings are already
 * available as tensors (e.g., after tensor-native forward pass).
 *
 * @param embeddingsTensor - Node embeddings as tf.Tensor2D [numNodes, embDim]
 * @param nodeIds - Node IDs corresponding to each row
 * @param levels - Node levels (0=tool, 1=capability)
 * @param intentEmbedding - Intent embedding as array (small, ok to convert)
 * @param tensorParams - Pre-initialized tensor parameters
 * @param config - SHGAT config
 * @returns Sorted array of node scores
 */
export function scoreNodesTensorDirect(
  embeddingsTensor: tf.Tensor2D,
  nodeIds: string[],
  levels: number[],
  intentEmbedding: number[],
  tensorParams: TensorScoringParams,
  config: SHGATConfig,
  projectionParams?: ProjectionHeadTFParams,
): NodeScore[] {
  if (nodeIds.length === 0) return [];

  const { numHeads } = config;
  const useProj = config.useProjectionHead && projectionParams;
  const alpha = config.projectionBlendAlpha ?? 0.5;
  const projTemp = config.projectionTemperature ?? 0.07;

  // All computation inside tidy() for automatic tensor cleanup
  const allScores = tidy(() => {
    // 1. Convert intent to tensor (small, fast)
    const intentTensor = tf.tensor1d(intentEmbedding);

    // 2. Project intent (if not preserveDim)
    const intentProjected = config.preserveDim
      ? intentTensor
      : ops.matVec(tensorParams.W_intent as tf.Tensor2D, intentTensor);

    // 3. Compute Q for all heads: Q[h] = W_q[h] @ intent
    const Qs: tf.Tensor1D[] = [];
    for (let h = 0; h < numHeads; h++) {
      const Q_h = ops.matVec(tensorParams.headParams[h].W_q as tf.Tensor2D, intentProjected);
      Qs.push(Q_h);
    }
    const Q_all = tf.stack(Qs) as tf.Tensor2D;

    // 4. Compute K for all heads: K[h] = embeddings @ W_k[h].T
    const Ks: tf.Tensor2D[] = [];
    for (let h = 0; h < numHeads; h++) {
      const K_h = ops.matmulTranspose(embeddingsTensor, tensorParams.headParams[h].W_k as tf.Tensor2D);
      Ks.push(K_h);
    }
    const K_all = tf.stack(Ks) as tf.Tensor3D;

    // 5. Compute cosine similarity scores
    const Q_expanded = Q_all.expandDims(1);
    const dotProduct = tf.sum(tf.mul(Q_expanded, K_all), 2);
    const normQ = tf.norm(Q_all, 2, 1, true);
    const normK = tf.norm(K_all, 2, 2);
    const similarity = tf.div(
      dotProduct,
      tf.add(tf.mul(normQ, normK), 1e-8)
    );
    const scoresPerHead = tf.transpose(similarity) as tf.Tensor2D;

    // 6. Compute mean K-head score per node → [numNodes]
    const kheadMean = tf.mean(scoresPerHead, 1) as tf.Tensor1D;

    // 7. Optionally blend with projection head scores
    if (useProj) {
      const intentForProj = intentTensor.expandDims(0) as tf.Tensor2D; // [1, embDim]
      const z_intent = projectionForward(intentForProj, projectionParams!);
      const z_nodes = projectionForward(embeddingsTensor, projectionParams!);
      // dot(z_intent, z_nodes.T) / temperature → [1, N] → squeeze → [N]
      const projScores = tf.div(
        tf.matMul(z_intent, z_nodes, false, true),
        projTemp,
      ).squeeze([0]) as tf.Tensor1D;

      // Blend: (1-alpha) * khead + alpha * projection
      const blended = tf.add(
        tf.mul(1 - alpha, kheadMean),
        tf.mul(alpha, projScores),
      ) as tf.Tensor1D;

      // Return [numNodes, numHeads+1] where last col is blended score
      // But for compatibility, return [numNodes, numHeads] (head scores) + blended as separate
      return {
        headScores: scoresPerHead.arraySync() as number[][],
        finalScores: blended.arraySync() as number[],
      };
    }

    return {
      headScores: scoresPerHead.arraySync() as number[][],
      finalScores: kheadMean.arraySync() as number[],
    };
  });

  // Build results with metadata
  const results: NodeScore[] = [];
  for (let i = 0; i < nodeIds.length; i++) {
    results.push({
      nodeId: nodeIds[i],
      score: allScores.finalScores[i],
      headScores: allScores.headScores[i],
      level: levels[i],
    });
  }

  return results.sort((a, b) => b.score - a.score);
}

/**
 * Predict success probability for a path of capabilities
 */
export function predictPathSuccess(
  intentEmbedding: number[],
  path: string[],
  capabilityNodes: Map<string, CapabilityNode>,
  headParams: HeadParams[],
  W_intent: number[][],
  config: SHGATConfig,
  depthDecay: number,
): number {
  if (path.length === 0) return 0;

  const intentForScoring = config.preserveDim
    ? intentEmbedding
    : projectIntent(intentEmbedding, W_intent);

  const precomputedQ = precomputeQForAllHeads(intentForScoring, headParams, config);

  let totalScore = 0;
  let totalWeight = 0;

  for (let i = 0; i < path.length; i++) {
    const cap = capabilityNodes.get(path[i]);
    if (!cap) continue;

    const scores = computeMultiHeadScoresWithPrecomputedQ(
      precomputedQ,
      cap.embedding,
      headParams,
      config,
    );
    const avgScore = scores.reduce((a, b) => a + b, 0) / config.numHeads;
    const weight = Math.pow(depthDecay, i);

    totalScore += avgScore * weight;
    totalWeight += weight;
  }

  return totalWeight > 0 ? totalScore / totalWeight : 0;
}
