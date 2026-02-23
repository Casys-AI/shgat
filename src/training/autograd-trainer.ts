/**
 * SHGAT Autograd Trainer
 *
 * Training with TensorFlow.js automatic differentiation.
 * Replaces 3000+ lines of manual backward passes.
 *
 * NOW WITH FULL MESSAGE PASSING (2026-01-28):
 * - Upward pass: V → E^0 → E^1 → ... → E^L
 * - Downward pass: E^L → ... → E^0 → V
 * - Uses W_up, W_down, a_up, a_down for attention
 *
 * @module shgat-tf/training/autograd-trainer
 */

import { getBackend, supportsAutograd, switchBackend, tf, tidy } from "../tf/backend.ts";
import type { BackendMode } from "../tf/backend.ts";
import * as ops from "../tf/ops.ts";
import type { SHGATConfig, TrainingExample } from "../core/types.ts";
import { initProjectionHeadParams, projectionScore } from "../core/projection-head.ts";

// ============================================================================
// Types
// ============================================================================

/**
 * Trainable parameters as TF.js Variables
 */
export interface TFParams {
  // K-head scoring parameters
  W_k: tf.Variable[]; // [numHeads][embDim, headDim] - Key projection
  W_q?: tf.Variable[]; // [numHeads][embDim, headDim] - Query projection (DEPRECATED: W_k shared for Q and K)
  W_intent: tf.Variable; // [embDim, hiddenDim] - Intent projection

  // Message passing parameters (per level)
  W_up: Map<number, tf.Variable[]>; // level -> [numHeads][embDim, headDim]
  W_down: Map<number, tf.Variable[]>; // level -> [numHeads][embDim, headDim]
  a_up: Map<number, tf.Variable[]>; // level -> [numHeads][2*headDim]
  a_down: Map<number, tf.Variable[]>; // level -> [numHeads][2*headDim]

  // Residual weights (learnable)
  residualWeights?: tf.Variable; // [numLevels]

  // Projection head (optional, enabled by config.useProjectionHead)
  projectionHead?: import("../core/projection-head.ts").ProjectionHeadTFParams;
}

/**
 * Graph structure for message passing
 */
export interface GraphStructure {
  /** Tool → Capability incidence matrix [numTools, numCaps0] */
  toolToCapMatrix: tf.Tensor2D;
  /** Capability → Capability matrices per level [numCapsL-1, numCapsL] */
  capToCapMatrices: Map<number, tf.Tensor2D>;
  /** Tool IDs in order */
  toolIds: string[];
  /** Capability IDs per level */
  capIdsByLevel: Map<number, string[]>;
  /** Maximum hierarchy level */
  maxLevel: number;
}

/**
 * Training configuration
 */
export interface TrainerConfig {
  learningRate: number;
  batchSize: number;
  temperature: number; // InfoNCE temperature (default 0.07)
  gradientClip: number; // Max gradient norm
  l2Lambda: number; // L2 regularization
}

/**
 * Training metrics
 */
export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  gradientNorm: number;
  numExamples: number;
}

/**
 * Default trainer config
 */
export const DEFAULT_TRAINER_CONFIG: TrainerConfig = {
  learningRate: 0.001,
  batchSize: 32,
  temperature: 0.07,
  gradientClip: 1.0,
  l2Lambda: 0.0001,
};

// ============================================================================
// Parameter Initialization
// ============================================================================

/**
 * Initialize trainable parameters as TF.js Variables
 * @param baseSeed - Optional base seed for deterministic initialization. Each parameter
 *   gets a unique seed derived from baseSeed to ensure reproducibility across runs.
 */
export function initTFParams(config: SHGATConfig, maxLevel: number, baseSeed?: number): TFParams {
  const { numHeads, embeddingDim, headDim, hiddenDim } = config;

  // Seed counter: each glorotNormal call gets a unique seed
  let seedCounter = 0;
  const nextSeed = () => baseSeed !== undefined ? baseSeed + (++seedCounter) : undefined;

  // K-head scoring (W_k is shared for Q and K projections, matching old SHGAT behavior)
  const W_k: tf.Variable[] = [];
  for (let h = 0; h < numHeads; h++) {
    W_k.push(tf.variable(ops.glorotNormal([embeddingDim, headDim], nextSeed()), true, `W_k_${h}`));
  }

  const W_intent = tf.variable(
    ops.glorotNormal([embeddingDim, hiddenDim || embeddingDim], nextSeed()),
    true,
    "W_intent",
  );

  // Message passing weights per level
  const W_up = new Map<number, tf.Variable[]>();
  const W_down = new Map<number, tf.Variable[]>();
  const a_up = new Map<number, tf.Variable[]>();
  const a_down = new Map<number, tf.Variable[]>();

  const mpMaxLevel = Math.max(1, maxLevel);
  for (let level = 0; level <= mpMaxLevel; level++) {
    const W_up_level: tf.Variable[] = [];
    const W_down_level: tf.Variable[] = [];
    const a_up_level: tf.Variable[] = [];
    const a_down_level: tf.Variable[] = [];

    for (let h = 0; h < numHeads; h++) {
      W_up_level.push(
        tf.variable(
          ops.glorotNormal([embeddingDim, headDim], nextSeed()),
          true,
          `W_up_${level}_${h}`,
        ),
      );
      W_down_level.push(
        tf.variable(
          ops.glorotNormal([embeddingDim, headDim], nextSeed()),
          true,
          `W_down_${level}_${h}`,
        ),
      );
      a_up_level.push(
        tf.variable(ops.glorotNormal([2 * headDim], nextSeed()), true, `a_up_${level}_${h}`),
      );
      a_down_level.push(
        tf.variable(ops.glorotNormal([2 * headDim], nextSeed()), true, `a_down_${level}_${h}`),
      );
    }

    W_up.set(level, W_up_level);
    W_down.set(level, W_down_level);
    a_up.set(level, a_up_level);
    a_down.set(level, a_down_level);
  }

  // Learnable residual weights
  const residualWeights = tf.variable(
    tf.fill([maxLevel + 1], 0.3), // Default 0.3
    true,
    "residualWeights",
  );

  // Optional projection head
  const projectionHead = config.useProjectionHead
    ? initProjectionHeadParams(
      embeddingDim,
      config.projectionHiddenDim ?? 256,
      config.projectionOutputDim ?? 256,
    )
    : undefined;

  return { W_k, W_intent, W_up, W_down, a_up, a_down, residualWeights, projectionHead };
}

// ============================================================================
// Message Passing (Differentiable)
// ============================================================================

const LEAKY_RELU_SLOPE = 0.2;

/**
 * Single attention head for message passing (differentiable)
 *
 * Computes attention-weighted aggregation from source to target nodes.
 * Uses batched matrix operations to stay within TF.js gradient tracking.
 *
 * Memory optimization: Process in chunks if graph is too large.
 */
/**
 * Attention chunk size threshold: if numTarget * numSource > this,
 * use chunked processing to keep peak memory bounded.
 * Default: 2M elements (~8MB per float32 matrix). Override via env.
 */
const ATTENTION_CHUNK_THRESHOLD = parseInt(
  // deno-lint-ignore no-explicit-any
  (globalThis as any).process?.env?.["SHGAT_ATTN_CHUNK"] || "2000000",
  10,
);

function attentionAggregation(
  sourceEmbs: tf.Tensor2D, // [numSource, embDim]
  targetEmbs: tf.Tensor2D, // [numTarget, embDim]
  connectivity: tf.Tensor2D, // [numSource, numTarget]
  W_source: tf.Variable, // [embDim, headDim]
  W_target: tf.Variable, // [embDim, headDim]
  a: tf.Variable, // [2 * headDim]
): tf.Tensor2D {
  const numTarget = targetEmbs.shape[0] as number;
  const numSource = sourceEmbs.shape[0] as number;

  // Use chunked processing when attention matrix would be too large
  if (numTarget * numSource > ATTENTION_CHUNK_THRESHOLD) {
    return chunkedAttentionAggregation(
      sourceEmbs,
      targetEmbs,
      connectivity,
      W_source,
      W_target,
      a,
    );
  }

  return denseAttentionAggregation(
    sourceEmbs,
    targetEmbs,
    connectivity,
    W_source,
    W_target,
    a,
  );
}

/**
 * Chunked attention: processes targets in blocks to bound peak memory.
 * Mathematically equivalent to dense — softmax is per-row (per-target).
 */
function chunkedAttentionAggregation(
  sourceEmbs: tf.Tensor2D,
  targetEmbs: tf.Tensor2D,
  connectivity: tf.Tensor2D,
  W_source: tf.Variable,
  W_target: tf.Variable,
  a: tf.Variable,
): tf.Tensor2D {
  const numTarget = targetEmbs.shape[0] as number;
  const numSource = sourceEmbs.shape[0] as number;
  const headDim = W_source.shape[1] as number;

  // Chunk size: keep attention matrix ≤ threshold elements
  const chunkSize = Math.max(1, Math.floor(ATTENTION_CHUNK_THRESHOLD / numSource));

  // Pre-compute source-side (shared across all target chunks)
  const srcProj = tf.matMul(sourceEmbs, W_source); // [numSource, headDim]
  const tgtProj = tf.matMul(targetEmbs, W_target); // [numTarget, headDim]

  const a_tgt = a.slice([0], [headDim]);
  const a_src = a.slice([headDim], [headDim]);

  const srcActivated = tf.leakyRelu(srcProj, LEAKY_RELU_SLOPE);
  const tgtActivated = tf.leakyRelu(tgtProj, LEAKY_RELU_SLOPE);

  // scoreSrc is shared: [numSource]
  const scoreSrc = tf.squeeze(tf.matMul(srcActivated, a_src.expandDims(1)));

  // Transpose connectivity once: [numTarget, numSource]
  const connT = tf.transpose(connectivity) as tf.Tensor2D;

  const chunkResults: tf.Tensor2D[] = [];

  for (let start = 0; start < numTarget; start += chunkSize) {
    const end = Math.min(start + chunkSize, numTarget);
    const size = end - start;

    // Slice target chunk
    const tgtChunk = tgtActivated.slice([start, 0], [size, headDim]);
    const scoreTgtChunk = tf.squeeze(tf.matMul(tgtChunk, a_tgt.expandDims(1)));

    // scores_chunk[ct, s] = scoreTgtChunk[ct] + scoreSrc[s]
    const scoresChunk = tf.add(
      size > 1 ? scoreTgtChunk.expandDims(1) : scoreTgtChunk.reshape([1, 1]),
      scoreSrc.expandDims(0),
    );

    // Mask chunk
    const connChunk = connT.slice([start, 0], [size, numSource]);
    const maskChunk = tf.equal(connChunk, 0);
    const maskedChunk = tf.where(maskChunk, tf.fill(scoresChunk.shape, -1e9), scoresChunk);

    // Softmax per target (per row) — independent of other chunks
    const attnChunk = tf.softmax(maskedChunk, -1);

    // Aggregate: [size, headDim]
    const resultChunk = tf.matMul(attnChunk, srcProj);
    chunkResults.push(tf.elu(resultChunk) as tf.Tensor2D);

    // Dispose chunk intermediates
    tgtChunk.dispose();
    if (size > 1) scoreTgtChunk.dispose();
    scoresChunk.dispose();
    connChunk.dispose();
    maskedChunk.dispose();
    attnChunk.dispose();
    resultChunk.dispose();
  }

  // Dispose shared intermediates
  srcProj.dispose();
  tgtProj.dispose();
  srcActivated.dispose();
  tgtActivated.dispose();
  scoreSrc.dispose();
  connT.dispose();

  // Concatenate all chunks: [numTarget, headDim]
  const result = chunkResults.length === 1
    ? chunkResults[0]
    : tf.concat(chunkResults, 0) as tf.Tensor2D;

  // Dispose individual chunks (if concatenated)
  if (chunkResults.length > 1) {
    for (const chunk of chunkResults) chunk.dispose();
  }

  return result;
}

/**
 * Dense attention aggregation (original implementation, fast for small matrices)
 */
function denseAttentionAggregation(
  sourceEmbs: tf.Tensor2D,
  targetEmbs: tf.Tensor2D,
  connectivity: tf.Tensor2D,
  W_source: tf.Variable,
  W_target: tf.Variable,
  a: tf.Variable,
): tf.Tensor2D {
  // Project embeddings (efficient matmul)
  const srcProj = tf.matMul(sourceEmbs, W_source); // [numSource, headDim]
  const tgtProj = tf.matMul(targetEmbs, W_target); // [numTarget, headDim]

  // Compute attention scores using einsum-style operation
  // For each (t, s) pair: score = a^T @ LeakyReLU([tgtProj[t], srcProj[s]])
  //
  // Optimization: Split attention vector a = [a_tgt, a_src] where each is [headDim]
  // Then: score = a_tgt @ LeakyReLU(tgtProj[t]) + a_src @ LeakyReLU(srcProj[s])
  // This avoids creating the [numTgt, numSrc, 2*headDim] tensor!

  const headDim = W_source.shape[1] as number;
  const a_tgt = a.slice([0], [headDim]); // First half of attention vector
  const a_src = a.slice([headDim], [headDim]); // Second half

  // Compute per-target and per-source contributions separately
  const tgtActivated = tf.leakyRelu(tgtProj, LEAKY_RELU_SLOPE); // [numTarget, headDim]
  const srcActivated = tf.leakyRelu(srcProj, LEAKY_RELU_SLOPE); // [numSource, headDim]

  // score_tgt[t] = tgtActivated[t] @ a_tgt  -> [numTarget]
  const scoreTgt = tf.squeeze(tf.matMul(tgtActivated, a_tgt.expandDims(1)));

  // score_src[s] = srcActivated[s] @ a_src  -> [numSource]
  const scoreSrc = tf.squeeze(tf.matMul(srcActivated, a_src.expandDims(1)));

  // Full scores: scores[t, s] = score_tgt[t] + score_src[s]
  // Use broadcasting: [numTarget, 1] + [1, numSource] -> [numTarget, numSource]
  const scores = tf.add(
    scoreTgt.expandDims(1), // [numTarget, 1]
    scoreSrc.expandDims(0), // [1, numSource]
  ); // [numTarget, numSource]

  // Mask non-connected pairs (transpose connectivity to [numTarget, numSource])
  const connT = tf.transpose(connectivity) as tf.Tensor2D;
  const mask = tf.equal(connT, 0);
  const maskedScores = tf.where(mask, tf.fill(scores.shape, -1e9), scores);

  // Softmax over sources (last dim)
  const attention = tf.softmax(maskedScores, -1); // [numTarget, numSource]

  // Aggregate: result[t] = sum_s(attention[t,s] * srcProj[s])
  const result = tf.matMul(attention, srcProj); // [numTarget, headDim]

  return tf.elu(result) as tf.Tensor2D;
}

/**
 * Multi-head message passing phase (differentiable)
 */
function multiHeadMessagePassing(
  sourceEmbs: tf.Tensor2D,
  targetEmbs: tf.Tensor2D,
  connectivity: tf.Tensor2D,
  W_source: tf.Variable[],
  W_target: tf.Variable[],
  a: tf.Variable[],
  numHeads: number,
): tf.Tensor2D {
  const headResults: tf.Tensor2D[] = [];

  for (let h = 0; h < numHeads; h++) {
    const headResult = attentionAggregation(
      sourceEmbs,
      targetEmbs,
      connectivity,
      W_source[h],
      W_target[h],
      a[h],
    );
    headResults.push(headResult);
  }

  // Concatenate heads: [numTarget, numHeads * headDim]
  return tf.concat(headResults, 1) as tf.Tensor2D;
}

/**
 * Full message passing forward (differentiable)
 *
 * Upward: V → E^0 → E^1 → ... → E^L
 * Downward: E^L → ... → E^0 → V
 */
export function messagePassingForward(
  H_init: tf.Tensor2D, // [numTools, embDim]
  E_init: Map<number, tf.Tensor2D>, // level -> [numCaps, embDim]
  graph: GraphStructure,
  params: TFParams,
  config: SHGATConfig,
): { H: tf.Tensor2D; E: Map<number, tf.Tensor2D> } {
  const { numHeads } = config;
  const maxLevel = graph.maxLevel;

  // Clone initial embeddings
  const E = new Map<number, tf.Tensor2D>();
  for (const [level, tensor] of E_init) {
    E.set(level, tensor.clone());
  }
  let H = H_init.clone();

  // ========================================================================
  // UPWARD PASS: V → E^0 → E^1 → ... → E^L
  // ========================================================================

  for (let level = 0; level <= maxLevel; level++) {
    const W_up = params.W_up.get(level) || params.W_up.get(1);
    const a_up = params.a_up.get(level) || params.a_up.get(1);
    if (!W_up || !a_up) continue;

    const capsAtLevel = E.get(level);
    if (!capsAtLevel) continue;

    if (level === 0) {
      // V → E^0: Tools aggregate to level-0 capabilities
      const E_new = multiHeadMessagePassing(
        H, // source: tools
        capsAtLevel, // target: caps level 0
        graph.toolToCapMatrix, // connectivity
        W_up, // W_source (for tools)
        W_up, // W_target (same weights for simplicity)
        a_up,
        numHeads,
      );
      E.get(level)?.dispose();
      E.set(level, E_new);
    } else {
      // E^(k-1) → E^k
      const E_prev = E.get(level - 1);
      const connectivity = graph.capToCapMatrices.get(level);
      if (!E_prev || !connectivity) continue;

      const E_new = multiHeadMessagePassing(
        E_prev,
        capsAtLevel,
        connectivity,
        W_up,
        W_up,
        a_up,
        numHeads,
      );
      E.get(level)?.dispose();
      E.set(level, E_new);
    }
  }

  // ========================================================================
  // DOWNWARD PASS: E^L → ... → E^0 → V
  // Uses config.downwardResidual for blend: (1-dr)*propagated + dr*original
  // ========================================================================

  const dr = config.downwardResidual ?? 0;

  for (let level = maxLevel - 1; level >= 0; level--) {
    const W_down = params.W_down.get(level + 1) || params.W_down.get(1);
    const a_down = params.a_down.get(level + 1) || params.a_down.get(1);
    if (!W_down || !a_down) continue;

    const capsAtLevel = E.get(level);
    const capsAtParent = E.get(level + 1);
    if (!capsAtLevel || !capsAtParent) continue;

    const forwardConn = graph.capToCapMatrices.get(level + 1);
    if (!forwardConn) continue;

    // Transpose for downward pass
    const reverseConn = tf.transpose(forwardConn) as tf.Tensor2D;

    const E_propagated = multiHeadMessagePassing(
      capsAtParent, // source: parent caps
      capsAtLevel, // target: child caps
      reverseConn,
      W_down,
      W_down,
      a_down,
      numHeads,
    );

    reverseConn.dispose();

    if (dr > 0) {
      // Blend: (1-dr)*propagated + dr*pre_downward
      const E_blended = tf.add(
        tf.mul(E_propagated, 1 - dr),
        tf.mul(capsAtLevel, dr),
      ) as tf.Tensor2D;
      E_propagated.dispose();
      E.get(level)?.dispose();
      E.set(level, E_blended);
    } else {
      E.get(level)?.dispose();
      E.set(level, E_propagated);
    }
  }

  // Final: E^0 → V
  const E_level0 = E.get(0);
  const W_down_final = params.W_down.get(1);
  const a_down_final = params.a_down.get(1);
  if (E_level0 && W_down_final && a_down_final) {
    // Transpose toolToCapMatrix for E→V direction
    const reverseConn = tf.transpose(graph.toolToCapMatrix) as tf.Tensor2D;

    const H_propagated = multiHeadMessagePassing(
      E_level0, // source: caps
      H, // target: tools
      reverseConn,
      W_down_final,
      W_down_final,
      a_down_final,
      numHeads,
    );

    reverseConn.dispose();

    if (dr > 0) {
      // Blend: (1-dr)*propagated + dr*pre_downward
      const H_blended = tf.add(
        tf.mul(H_propagated, 1 - dr),
        tf.mul(H, dr),
      ) as tf.Tensor2D;
      H_propagated.dispose();
      H.dispose();
      H = H_blended;
    } else {
      H.dispose();
      H = H_propagated;
    }
  }

  // ========================================================================
  // POST-MP RESIDUAL: blend with initial embeddings
  // Uses config.preserveDimResiduals (per-level) or config.preserveDimResidual (global)
  // ========================================================================

  const pdr = config.preserveDimResiduals;
  const globalResidual = config.preserveDimResidual ?? 0.3;

  // Tools (H) — graph level 0 → preserveDimResiduals[0]
  const toolAlpha = pdr?.[0] ?? globalResidual;
  const H_residual = tf.add(
    tf.mul(H, 1 - toolAlpha),
    tf.mul(H_init, toolAlpha),
  ) as tf.Tensor2D;
  H.dispose();
  H = H_residual;

  // Capabilities (E) — graph level = capLevel + 1
  for (const [capLevel, E_tensor] of E) {
    const E_initLevel = E_init.get(capLevel);
    if (E_initLevel) {
      const graphLevel = capLevel + 1; // caps at E level 0 are graph level 1
      const capAlpha = pdr?.[graphLevel] ?? globalResidual;
      const E_residual = tf.add(
        tf.mul(E_tensor, 1 - capAlpha),
        tf.mul(E_initLevel, capAlpha),
      ) as tf.Tensor2D;
      E_tensor.dispose();
      E.set(capLevel, E_residual);
    }
  }

  return { H, E };
}

// ============================================================================
// Forward Pass (Differentiable)
// ============================================================================

/**
 * K-head attention scoring (differentiable)
 *
 * @param intentProj - Projected intent [hiddenDim]
 * @param nodeEmbs - Node embeddings [numNodes, embDim]
 * @param params - Trainable parameters
 * @param config - SHGAT config
 * @returns Scores [numNodes]
 */
export function kHeadScoring(
  intentProj: tf.Tensor1D,
  nodeEmbs: tf.Tensor2D,
  params: TFParams,
  config: SHGATConfig,
): tf.Tensor1D {
  const { numHeads } = config;

  // Compute attention scores per head
  const headScores: tf.Tensor2D[] = [];

  for (let h = 0; h < numHeads; h++) {
    // Project nodes: K = nodeEmbs @ W_k[h]  [numNodes, headDim]
    const K = tf.matMul(nodeEmbs, params.W_k[h]);

    // Project intent: Q = intentProj @ W_k[h]  [headDim]
    // Uses same W_k for both Q and K (shared projection, matching old SHGAT behavior)
    const Q = tf.squeeze(tf.matMul(intentProj.expandDims(0), params.W_k[h]));

    // Cosine similarity: dot(K, Q) / (||K|| * ||Q|| + eps)
    // Matches production scorer (khead-scorer.ts:scoreNodes)
    const dotProduct = tf.squeeze(tf.matMul(K, Q.expandDims(1))); // [numNodes]
    const normK = tf.norm(K, 2, 1); // [numNodes]
    const normQ = tf.norm(Q); // scalar
    const scores = dotProduct.div(tf.add(tf.mul(normK, normQ), 1e-8));

    headScores.push(scores.expandDims(1) as tf.Tensor2D);
  }

  // Concatenate and average heads [numNodes, numHeads] -> [numNodes]
  const allScores = tf.concat(headScores, 1);
  const finalScores = tf.mean(allScores, 1) as tf.Tensor1D;

  return finalScores;
}

/**
 * Full forward pass for scoring
 */
export function forwardScoring(
  intentEmb: tf.Tensor1D,
  nodeEmbs: tf.Tensor2D,
  params: TFParams,
  config: SHGATConfig,
): tf.Tensor1D {
  // Project intent (skip W_intent when preserveDim=true to preserve BGE-M3 structure)
  let intentProj: tf.Tensor1D;
  if (config.preserveDim) {
    intentProj = intentEmb;
  } else {
    intentProj = tf.squeeze(
      tf.matMul(intentEmb.expandDims(0), params.W_intent),
    ) as tf.Tensor1D;
  }

  // K-head scoring
  const kheadScores = kHeadScoring(intentProj, nodeEmbs, params, config);

  // Blend with projection head if enabled
  if (config.useProjectionHead && params.projectionHead) {
    const alpha = config.projectionBlendAlpha ?? 0.5;
    const temp = config.projectionTemperature ?? 0.07;
    const projScores = projectionScore(
      intentEmb.expandDims(0) as tf.Tensor2D,
      nodeEmbs,
      params.projectionHead,
      temp,
    );
    // final = (1-alpha) * khead + alpha * projection
    return tf.add(
      tf.mul(1 - alpha, kheadScores),
      tf.mul(alpha, projScores),
    ) as tf.Tensor1D;
  }

  return kheadScores;
}

// ============================================================================
// Loss Functions
// ============================================================================

/**
 * InfoNCE contrastive loss
 *
 * @param anchorScores - Scores for anchor (positive) [1]
 * @param negativeScores - Scores for negatives [numNegatives]
 * @param temperature - Temperature parameter
 */
export function infoNCELoss(
  anchorScore: tf.Scalar,
  negativeScores: tf.Tensor1D,
  temperature: number,
): tf.Scalar {
  // Logits: [anchor, neg1, neg2, ...]
  const allScores = tf.concat([anchorScore.expandDims(0), negativeScores]);
  const logits = tf.div(allScores, temperature);

  // Cross-entropy with label 0 (anchor is correct)
  const labels = tf.oneHot(0, allScores.shape[0]);
  return tf.losses.softmaxCrossEntropy(
    labels.expandDims(0),
    logits.expandDims(0),
  ) as tf.Scalar;
}

/**
 * Batch contrastive loss with in-batch negatives
 */
export function batchContrastiveLoss(
  intentEmbs: tf.Tensor2D, // [batchSize, embDim]
  positiveEmbs: tf.Tensor2D, // [batchSize, embDim]
  params: TFParams,
  config: SHGATConfig,
  temperature: number,
): tf.Scalar {
  return tidy(() => {
    const batchSize = intentEmbs.shape[0];
    const numHeads = config.numHeads;

    // Project intents [batchSize, hiddenDim]
    const intentProj = tf.matMul(intentEmbs, params.W_intent);

    // Project positives using ALL K-heads and average (multi-head fusion)
    const headProjections: tf.Tensor2D[] = [];
    for (let h = 0; h < numHeads; h++) {
      headProjections.push(tf.matMul(positiveEmbs, params.W_k[h]) as tf.Tensor2D);
    }
    // Mean pooling across heads: [batchSize, headDim]
    const stacked = tf.stack(headProjections);
    const positiveProj = tf.mean(stacked, 0) as tf.Tensor2D;

    // Normalize
    const intentNorm = ops.l2Normalize(intentProj, 1) as tf.Tensor2D;
    const positiveNorm = ops.l2Normalize(positiveProj, 1) as tf.Tensor2D;

    // Similarity matrix [batchSize, batchSize]
    const similarity = tf.div(
      ops.matmulTranspose(intentNorm, positiveNorm),
      temperature,
    );

    // Labels: diagonal (i matches i)
    const labels = tf.eye(batchSize);

    // Symmetric cross-entropy
    const loss1 = tf.losses.softmaxCrossEntropy(labels, similarity);
    const loss2 = tf.losses.softmaxCrossEntropy(labels, tf.transpose(similarity));

    return tf.div(tf.add(loss1, loss2), 2) as tf.Scalar;
  });
}

// ============================================================================
// KL Divergence Loss (n8n soft targets)
// ============================================================================

/**
 * KL divergence metrics
 */
export interface KLTrainingMetrics {
  klLoss: number;
  gradientNorm: number;
  numExamples: number;
}

/**
 * KL divergence loss between K-head scores and soft target distribution.
 *
 * loss = KL(target || softmax(scores / temperature))
 *      = Σ target_i * (log(target_i) - log(pred_i))
 *
 * Only non-zero entries of softTargetSparse contribute to the gradient,
 * making this efficient with sparse top-K targets.
 *
 * @param scores - K-head scores for ALL tools [vocabSize]
 * @param softTargetSparse - Sparse soft target: [[toolIndex, probability], ...]
 * @param vocabSize - Number of tools in vocabulary
 * @param temperature - Temperature for softmax over scores
 */
export function klDivergenceLoss(
  scores: tf.Tensor1D,
  softTargetSparse: [number, number][],
  vocabSize: number,
  temperature: number,
): tf.Scalar {
  // Reconstruct dense target from sparse
  const denseTarget = new Float32Array(vocabSize);
  for (const [idx, prob] of softTargetSparse) {
    denseTarget[idx] = prob;
  }
  const targetTensor = tf.tensor1d(denseTarget);

  // Predicted distribution: softmax(scores / temperature)
  const logits = tf.div(scores, temperature);
  const predicted = tf.softmax(logits);

  // KL(target || predicted) = Σ target * (log(target + eps) - log(predicted + eps))
  const eps = 1e-10;
  const logTarget = tf.log(tf.add(targetTensor, eps));
  const logPredicted = tf.log(tf.add(predicted, eps));
  const kl = tf.sum(tf.mul(targetTensor, tf.sub(logTarget, logPredicted)));

  return kl as tf.Scalar;
}

/**
 * KL training step for soft target examples.
 *
 * Unlike trainStep (InfoNCE), this scores against ALL tools in the vocabulary
 * since KL divergence is a full-distribution loss.
 *
 * @param klExamples - Soft target examples
 * @param allToolEmbsTensor - Pre-computed [numTools, embDim] tensor of all tool embeddings
 * @param vocabSize - Number of tools
 * @param params - Trainable parameters
 * @param config - SHGAT configuration
 * @param trainerConfig - Training configuration
 * @param optimizer - TF.js optimizer
 * @param klTemperature - Temperature for KL softmax (separate from InfoNCE temperature)
 */
/**
 * Batched K-head scoring for KL training.
 *
 * Pre-computes K = nodeEmbs @ W_k[h] once per head (instead of per example),
 * then batch-projects all intents. ~50x faster than per-example forwardScoring.
 *
 * @returns scores [batchSize, numTools]
 */
function batchedKHeadForward(
  intentsBatch: tf.Tensor2D, // [batchSize, embDim]
  nodeEmbs: tf.Tensor2D, // [numTools, embDim]
  params: TFParams,
  config: import("../core/types.ts").SHGATConfig,
): tf.Tensor2D {
  const { numHeads } = config;

  // Project intents through W_intent if needed
  let intentsProj: tf.Tensor2D;
  if (config.preserveDim) {
    intentsProj = intentsBatch;
  } else {
    intentsProj = tf.matMul(intentsBatch, params.W_intent) as tf.Tensor2D; // [B, embDim]
  }

  const headScores: tf.Tensor2D[] = [];

  for (let h = 0; h < numHeads; h++) {
    // Pre-compute K for ALL tools ONCE: [numTools, headDim]
    const K = tf.matMul(nodeEmbs, params.W_k[h]);
    // Batch Q for ALL intents: [batchSize, headDim]
    const Q = tf.matMul(intentsProj, params.W_k[h]);

    // Batch cosine similarity: [batchSize, numTools]
    // dot = Q @ K^T  →  [batchSize, numTools]
    const dot = tf.matMul(Q, K, false, true);
    // Norms: ||K|| [numTools], ||Q|| [batchSize]
    const normK = tf.norm(K, 2, 1); // [numTools]
    const normQ = tf.norm(Q, 2, 1); // [batchSize]
    // denominator: normQ[:, None] * normK[None, :] + eps  →  [batchSize, numTools]
    const denom = tf.add(
      tf.matMul(normQ.expandDims(1), normK.expandDims(0)),
      1e-8,
    );
    const scores = tf.div(dot, denom) as tf.Tensor2D;
    headScores.push(scores);
  }

  // Average across heads: stack [numHeads, batchSize, numTools] → mean → [batchSize, numTools]
  const stacked = tf.stack(headScores); // [numHeads, B, numTools]
  const averaged = tf.mean(stacked, 0) as tf.Tensor2D; // [B, numTools]

  return averaged;
}

/**
 * Batched KL divergence loss.
 *
 * Computes KL(target || softmax(scores/T)) for all examples at once.
 *
 * @param scores [batchSize, numTools]
 * @param klExamples examples with sparse soft targets
 * @param vocabSize number of tools
 * @param temperature softmax temperature
 * @returns scalar loss (averaged over batch)
 */
function batchedKLLoss(
  scores: tf.Tensor2D,
  klExamples: import("../core/types.ts").SoftTargetExample[],
  vocabSize: number,
  temperature: number,
): tf.Scalar {
  const batchSize = klExamples.length;

  // Build dense target matrix [batchSize, vocabSize]
  const targetData = new Float32Array(batchSize * vocabSize);
  for (let i = 0; i < batchSize; i++) {
    for (const [idx, prob] of klExamples[i].softTargetSparse) {
      targetData[i * vocabSize + idx] = prob;
    }
  }
  const targetTensor = tf.tensor2d(targetData, [batchSize, vocabSize]);

  // Predicted: softmax(scores / T) per row
  const logits = tf.div(scores, temperature);
  const predicted = tf.softmax(logits, 1); // along axis=1 (tools)

  // KL(target || predicted) = Σ target * (log(target+eps) - log(pred+eps))
  const eps = 1e-10;
  const logTarget = tf.log(tf.add(targetTensor, eps));
  const logPredicted = tf.log(tf.add(predicted, eps));
  const klPerExample = tf.sum(tf.mul(targetTensor, tf.sub(logTarget, logPredicted)), 1); // [batchSize]
  const klMean = tf.mean(klPerExample) as tf.Scalar;

  return klMean;
}

export function trainStepKL(
  klExamples: import("../core/types.ts").SoftTargetExample[],
  allToolEmbsTensor: tf.Tensor2D,
  vocabSize: number,
  params: TFParams,
  config: import("../core/types.ts").SHGATConfig,
  trainerConfig: TrainerConfig,
  optimizer: tf.Optimizer,
  klTemperature: number,
  klWeight: number = 1.0,
): KLTrainingMetrics {
  let totalLoss = 0;

  const { grads, value: batchLoss } = tf.variableGrads(() => {
    // Stack all intent embeddings into a matrix [batchSize, embDim]
    const embDim = config.embeddingDim;
    const intentData = new Float32Array(klExamples.length * embDim);
    for (let i = 0; i < klExamples.length; i++) {
      const ie = klExamples[i].intentEmbedding;
      if (ie.length !== embDim) {
        throw new Error(
          `[trainStepKL] intentEmbedding[${i}].length=${ie.length} != embeddingDim=${embDim}`,
        );
      }
      intentData.set(ie, i * embDim);
    }
    const intentsBatch = tf.tensor2d(intentData, [klExamples.length, config.embeddingDim]);

    // Batched K-head scoring: [batchSize, numTools]
    const scores = batchedKHeadForward(intentsBatch, allToolEmbsTensor, params, config);

    // Batched KL divergence (weighted by klWeight to control gradient dominance)
    let klLoss = batchedKLLoss(scores, klExamples, vocabSize, klTemperature);
    if (klWeight !== 1.0) {
      klLoss = klLoss.mul(klWeight) as tf.Scalar;
    }

    // L2 regularization
    let l2Loss = tf.scalar(0);
    for (const W of params.W_k) {
      l2Loss = l2Loss.add(tf.sum(tf.square(W)));
    }
    l2Loss = l2Loss.add(tf.sum(tf.square(params.W_intent)));
    l2Loss = l2Loss.mul(trainerConfig.l2Lambda);

    return klLoss.add(l2Loss) as tf.Scalar;
  });

  // Extract loss value AFTER the tape (avoids arraySync inside variableGrads — C3 fix)
  totalLoss = (batchLoss as tf.Tensor).dataSync()[0];

  // Compute gradient norm (dispose intermediates to prevent WASM OOM)
  let gradNormSquared = 0;
  for (const g of Object.values(grads)) {
    const sq = tf.square(g);
    const s = tf.sum(sq);
    gradNormSquared += s.dataSync()[0];
    sq.dispose();
    s.dispose();
  }
  const gradientNorm = Math.sqrt(gradNormSquared);

  // Clip gradients
  if (gradientNorm > trainerConfig.gradientClip) {
    const scale = trainerConfig.gradientClip / gradientNorm;
    for (const key of Object.keys(grads)) {
      const old = grads[key];
      grads[key] = old.mul(scale);
      old.dispose();
    }
  }

  // Apply gradients
  optimizer.applyGradients(grads);

  // Cleanup
  Object.values(grads).forEach((g) => g.dispose());
  (batchLoss as tf.Tensor).dispose();

  return {
    klLoss: totalLoss,
    gradientNorm,
    numExamples: klExamples.length,
  };
}

// ============================================================================
// Training Step
// ============================================================================

/**
 * Message passing context for training.
 * Pre-computed once per batch to avoid recomputation per example.
 */
export interface MessagePassingContext {
  graph: GraphStructure;
  H_init: tf.Tensor2D;
  E_init: Map<number, tf.Tensor2D>;
}

/**
 * Single training step with autograd
 *
 * If mpContext is provided, message passing is applied to enrich embeddings
 * BEFORE scoring, allowing gradients to flow through W_up, W_down, a_up, a_down.
 *
 * Uses dense TF.js autograd for all operations.
 * Requires CPU or WebGPU backend (WASM lacks UnsortedSegmentSum kernel for tf.gather gradients).
 * Provides correct gradients for ALL parameters including message passing weights.
 *
 * @param examples - Training examples
 * @param nodeEmbeddings - Raw node embeddings
 * @param params - Trainable parameters (including message passing weights)
 * @param config - SHGAT configuration
 * @param trainerConfig - Training configuration
 * @param optimizer - TF.js optimizer
 * @param mpContext - Optional message passing context for enriched embeddings
 */
export function trainStep(
  examples: TrainingExample[],
  nodeEmbeddings: Map<string, number[]>,
  params: TFParams,
  config: SHGATConfig,
  trainerConfig: TrainerConfig,
  optimizer: tf.Optimizer,
  mpContext?: MessagePassingContext,
): TrainingMetrics {
  let totalLoss = 0;
  let totalCorrect = 0;

  const { grads, value: batchLoss } = tf.variableGrads(() => {
    let loss = tf.scalar(0);

    // ========================================================================
    // OPTIMIZATION: Run message passing ONCE for the entire batch
    // This creates enriched embeddings that include gradients for W_up, W_down, etc.
    // ========================================================================
    let enrichedToolEmbs: tf.Tensor2D | null = null;
    let enrichedCapEmbsByLevel: Map<number, tf.Tensor2D> | null = null;
    let toolIdToIdx: Map<string, number> | null = null;
    let capIdToLevelIdx: Map<string, { level: number; idx: number }> | null = null;

    if (mpContext) {
      // Run message passing - gradients will flow through this!
      const { H, E } = messagePassingForward(
        mpContext.H_init,
        mpContext.E_init,
        mpContext.graph,
        params,
        config,
      );

      enrichedToolEmbs = H;
      enrichedCapEmbsByLevel = E;

      // Build lookup maps for fast access
      toolIdToIdx = new Map<string, number>();
      for (let i = 0; i < mpContext.graph.toolIds.length; i++) {
        toolIdToIdx.set(mpContext.graph.toolIds[i], i);
      }

      capIdToLevelIdx = new Map<string, { level: number; idx: number }>();
      for (const [level, capIds] of mpContext.graph.capIdsByLevel) {
        for (let i = 0; i < capIds.length; i++) {
          capIdToLevelIdx.set(capIds[i], { level, idx: i });
        }
      }
    }

    // Collect ALL unique node IDs needed for this batch FIRST
    const allNodeIds = new Set<string>();
    for (const ex of examples) {
      allNodeIds.add(ex.candidateId);
      for (const negId of ex.negativeCapIds || []) {
        if (nodeEmbeddings.has(negId)) {
          allNodeIds.add(negId);
        }
      }
    }
    const uniqueNodeIds = Array.from(allNodeIds);

    // Build a single tensor of ALL embeddings needed (using tf.gather for differentiability)
    // Map nodeId -> index in the batch tensor
    const nodeIdToIdx = new Map<string, number>();

    // Collect indices for gathering from enriched tensors
    const toolGatherIndices: number[] = [];
    const toolGatherNodeIds: string[] = [];
    const capGatherIndicesByLevel = new Map<number, { indices: number[]; nodeIds: string[] }>();
    const rawEmbNodeIds: string[] = [];

    for (const nodeId of uniqueNodeIds) {
      if (enrichedToolEmbs && toolIdToIdx?.has(nodeId)) {
        toolGatherIndices.push(toolIdToIdx.get(nodeId)!);
        toolGatherNodeIds.push(nodeId);
      } else if (enrichedCapEmbsByLevel && capIdToLevelIdx?.has(nodeId)) {
        const { level, idx } = capIdToLevelIdx.get(nodeId)!;
        if (!capGatherIndicesByLevel.has(level)) {
          capGatherIndicesByLevel.set(level, { indices: [], nodeIds: [] });
        }
        capGatherIndicesByLevel.get(level)!.indices.push(idx);
        capGatherIndicesByLevel.get(level)!.nodeIds.push(nodeId);
      } else {
        rawEmbNodeIds.push(nodeId);
      }
    }

    // Gather enriched tool embeddings (differentiable!)
    const gatheredParts: tf.Tensor2D[] = [];
    const gatheredNodeIds: string[] = [];

    if (enrichedToolEmbs && toolGatherIndices.length > 0) {
      const gathered = tf.gather(enrichedToolEmbs, toolGatherIndices);
      gatheredParts.push(gathered as tf.Tensor2D);
      gatheredNodeIds.push(...toolGatherNodeIds);
    }

    // Gather enriched cap embeddings per level (differentiable!)
    if (enrichedCapEmbsByLevel) {
      for (const [level, { indices, nodeIds: ids }] of capGatherIndicesByLevel) {
        const levelEmbs = enrichedCapEmbsByLevel.get(level);
        if (levelEmbs && indices.length > 0) {
          const gathered = tf.gather(levelEmbs, indices);
          gatheredParts.push(gathered as tf.Tensor2D);
          gatheredNodeIds.push(...ids);
        }
      }
    }

    // Add raw embeddings for nodes not in enriched tensors
    if (rawEmbNodeIds.length > 0) {
      const rawEmbs = rawEmbNodeIds.map((id) =>
        nodeEmbeddings.get(id) || new Array(config.embeddingDim).fill(0)
      );
      gatheredParts.push(ops.toTensor(rawEmbs));
      gatheredNodeIds.push(...rawEmbNodeIds);
    }

    // Concatenate all gathered embeddings
    const allEmbsTensor = gatheredParts.length > 0
      ? tf.concat(gatheredParts, 0) as tf.Tensor2D
      : ops.toTensor([new Array(config.embeddingDim).fill(0)]);

    // Build final nodeId -> index mapping
    for (let i = 0; i < gatheredNodeIds.length; i++) {
      nodeIdToIdx.set(gatheredNodeIds[i], i);
    }

    for (const ex of examples) {
      // Collect node IDs for this example
      const nodeIds: string[] = [ex.candidateId];
      for (const negId of ex.negativeCapIds || []) {
        if (nodeEmbeddings.has(negId)) {
          nodeIds.push(negId);
        }
      }

      if (nodeIds.length < 2) {
        // Skip if no negatives
        continue;
      }

      // Gather embeddings for this example from the pre-computed tensor
      const indices = nodeIds.map((id) => nodeIdToIdx.get(id) ?? 0);
      const nodeEmbsTensor = tf.gather(allEmbsTensor, indices) as tf.Tensor2D;

      // Get intent embedding
      const intentEmb = ops.toTensor(ex.intentEmbedding);

      const scores = forwardScoring(intentEmb, nodeEmbsTensor, params, config);

      // Positive is at index 0
      const positiveScore = scores.slice([0], [1]).squeeze() as tf.Scalar;
      const negativeScores = scores.slice([1], [nodeIds.length - 1]) as tf.Tensor1D;

      // InfoNCE loss
      const exampleLoss = infoNCELoss(positiveScore, negativeScores, trainerConfig.temperature);
      loss = loss.add(exampleLoss);

      // Track accuracy (is positive score highest?)
      // Use tf.argMax → dataSync on a single scalar instead of arraySync on the full
      // scores vector. arraySync inside variableGrads forces GPU→CPU sync and causes
      // the tape to retain extra copies of the scores tensor (~1-2 GB waste).
      if (tf.argMax(scores).dataSync()[0] === 0) totalCorrect++;
    }

    // Average loss
    const avgLoss = loss.div(examples.length);

    // L2 regularization on ALL parameters including message passing
    let l2Loss = tf.scalar(0);

    // K-head scoring parameters (W_k shared for Q and K)
    for (const W of params.W_k) {
      l2Loss = l2Loss.add(tf.sum(tf.square(W)));
    }
    l2Loss = l2Loss.add(tf.sum(tf.square(params.W_intent)));

    // Message passing parameters (important for regularization!)
    if (mpContext) {
      for (const [, weights] of params.W_up) {
        for (const W of weights) {
          l2Loss = l2Loss.add(tf.sum(tf.square(W)));
        }
      }
      for (const [, weights] of params.W_down) {
        for (const W of weights) {
          l2Loss = l2Loss.add(tf.sum(tf.square(W)));
        }
      }
      for (const [, weights] of params.a_up) {
        for (const a of weights) {
          l2Loss = l2Loss.add(tf.sum(tf.square(a)));
        }
      }
      for (const [, weights] of params.a_down) {
        for (const a of weights) {
          l2Loss = l2Loss.add(tf.sum(tf.square(a)));
        }
      }
    }

    // Projection head parameters (stronger L2 to prevent overfitting)
    if (params.projectionHead) {
      const projL2Scale = 10;
      l2Loss = l2Loss.add(tf.sum(tf.square(params.projectionHead.W1)).mul(projL2Scale));
      l2Loss = l2Loss.add(tf.sum(tf.square(params.projectionHead.W2)).mul(projL2Scale));
    }

    l2Loss = l2Loss.mul(trainerConfig.l2Lambda);

    return avgLoss.add(l2Loss) as tf.Scalar;
  });

  // Extract loss value AFTER the tape (avoids arraySync inside variableGrads)
  totalLoss = (batchLoss as tf.Tensor).dataSync()[0];

  // Compute gradient norm (dispose intermediates to prevent WASM OOM)
  let gradNormSquared = 0;
  for (const g of Object.values(grads)) {
    const sq = tf.square(g);
    const s = tf.sum(sq);
    gradNormSquared += s.arraySync() as number;
    sq.dispose();
    s.dispose();
  }
  const gradientNorm = Math.sqrt(gradNormSquared);

  // Clip gradients if needed (dispose old tensor before replacing)
  if (gradientNorm > trainerConfig.gradientClip) {
    const scale = trainerConfig.gradientClip / gradientNorm;
    for (const key of Object.keys(grads)) {
      const old = grads[key];
      grads[key] = old.mul(scale);
      old.dispose();
    }
  }

  // Scale MP gradients by mpLearningRateScale (amplify W_up/W_down/a_up/a_down learning).
  // These params get smaller gradients due to the long computation chain through attention.
  // Scaling by 50-100x compensates without needing a separate optimizer.
  const mpScale = config.mpLearningRateScale ?? 1;
  if (mpScale !== 1 && mpContext) {
    const mpPrefixes = ["W_up_", "W_down_", "a_up_", "a_down_"];
    for (const key of Object.keys(grads)) {
      if (mpPrefixes.some((p) => key.startsWith(p))) {
        const old = grads[key];
        grads[key] = old.mul(mpScale);
        old.dispose();
      }
    }
  }

  // Apply gradients
  optimizer.applyGradients(grads);

  // Cleanup
  Object.values(grads).forEach((g) => g.dispose());
  (batchLoss as tf.Tensor).dispose();

  return {
    loss: totalLoss,
    accuracy: totalCorrect / examples.length,
    gradientNorm,
    numExamples: examples.length,
  };
}

// ============================================================================
// Graph Structure Builder
// ============================================================================

/**
 * Capability info for graph building
 */
export interface CapabilityInfo {
  id: string;
  toolsUsed: string[];
  parents?: string[];
  children?: string[];
}

/**
 * Build graph structure from capabilities and tools
 *
 * @param capabilities - Array of capability info with toolsUsed and hierarchy
 * @param toolIds - Array of all tool IDs
 * @returns Graph structure for message passing
 */
export function buildGraphStructure(
  capabilities: CapabilityInfo[],
  toolIds: string[],
): GraphStructure {
  // Group capabilities by hierarchy level
  // Level 0 = leaves (no children), Level 1+ = parents with children
  // This gives deeper hierarchy (maxLevel=2 instead of 1)
  const capIdToLevel = new Map<string, number>();
  const capIdToInfo = new Map<string, CapabilityInfo>();

  for (const cap of capabilities) {
    capIdToInfo.set(cap.id, cap);
  }

  // Recursive level computation: level = 0 if leaf, else 1 + max(children levels)
  const computeLevel = (capId: string): number => {
    const cached = capIdToLevel.get(capId);
    if (cached !== undefined) return cached;

    const cap = capIdToInfo.get(capId);
    if (!cap) {
      capIdToLevel.set(capId, 0);
      return 0;
    }

    const validChildren = (cap.children || []).filter((id) => capIdToInfo.has(id));
    let level: number;
    if (validChildren.length === 0) {
      level = 0; // Leaf
    } else {
      const childLevels = validChildren.map((childId) => computeLevel(childId));
      level = 1 + Math.max(...childLevels);
    }

    capIdToLevel.set(capId, level);
    return level;
  };

  for (const cap of capabilities) {
    computeLevel(cap.id);
  }

  // Group by level
  const capIdsByLevel = new Map<number, string[]>();
  let maxLevel = 0;
  for (const [capId, level] of capIdToLevel) {
    if (!capIdsByLevel.has(level)) {
      capIdsByLevel.set(level, []);
    }
    capIdsByLevel.get(level)!.push(capId);
    maxLevel = Math.max(maxLevel, level);
  }

  // Build tool→cap matrix [numTools, numCaps0]
  const caps0 = capIdsByLevel.get(0) || [];
  const toolToCapData: number[][] = [];
  for (const toolId of toolIds) {
    const row: number[] = [];
    for (const capId of caps0) {
      const cap = capIdToInfo.get(capId);
      const connected = cap?.toolsUsed?.includes(toolId) ? 1 : 0;
      row.push(connected);
    }
    toolToCapData.push(row);
  }
  const toolToCapMatrix = tf.tensor2d(toolToCapData);

  // Build cap→cap matrices per level
  // Convention: [numChildrenAtLevelMinus1, numParentsAtLevel] = [source, target]
  // Same as toolToCapMatrix which is [numTools, numCaps0] = [source, target]
  const capToCapMatrices = new Map<number, tf.Tensor2D>();
  for (let level = 1; level <= maxLevel; level++) {
    const parentCaps = capIdsByLevel.get(level) || []; // Parents (targets in upward)
    const childCaps = capIdsByLevel.get(level - 1) || []; // Children (sources in upward)

    const matrixData: number[][] = [];
    for (const childId of childCaps) {
      const row: number[] = [];
      for (const parentId of parentCaps) {
        const parentInfo = capIdToInfo.get(parentId);
        const connected = parentInfo?.children?.includes(childId) ? 1 : 0;
        row.push(connected);
      }
      matrixData.push(row);
    }
    if (matrixData.length > 0 && matrixData[0].length > 0) {
      capToCapMatrices.set(level, tf.tensor2d(matrixData));
    }
  }

  return {
    toolToCapMatrix,
    capToCapMatrices,
    toolIds,
    capIdsByLevel,
    maxLevel,
  };
}

/**
 * Dispose graph structure tensors
 */
export function disposeGraphStructure(graph: GraphStructure): void {
  graph.toolToCapMatrix.dispose();
  for (const [, tensor] of graph.capToCapMatrices) {
    tensor.dispose();
  }
}

// ============================================================================
// Subgraph Sampling (Ancestral Path Sampling)
// ============================================================================

/**
 * Cached JS representation of the adjacency structure.
 * Built once from the full graph to avoid repeated arraySync() calls.
 */
export interface AdjacencyCache {
  /** For each tool index: list of connected cap indices at level 0 */
  toolToCaps: number[][];
  /** For each cap index at level 0: list of connected tool indices */
  capToTools: number[][];
  /** For each level L (1..maxLevel): childIdx → list of parent indices at level L */
  childToParents: Map<number, number[][]>;
  /** For each level L (1..maxLevel): parentIdx → list of child indices at level L-1 */
  parentToChildren: Map<number, number[][]>;
  /** Maximum hierarchy level in the graph */
  maxLevel: number;
}

/**
 * Build adjacency cache from the full GraphStructure.
 * Call once after graph construction; reuse across all batches.
 */
export function buildAdjacencyCache(graph: GraphStructure): AdjacencyCache {
  const numTools = graph.toolIds.length;
  const caps0 = graph.capIdsByLevel.get(0) || [];
  const numCaps = caps0.length;

  // Read the dense matrix once
  const dense = graph.toolToCapMatrix.arraySync() as number[][];

  const toolToCaps: number[][] = Array.from({ length: numTools }, () => []);
  const capToTools: number[][] = Array.from({ length: numCaps }, () => []);

  for (let t = 0; t < numTools; t++) {
    for (let c = 0; c < numCaps; c++) {
      if (dense[t][c] > 0) {
        toolToCaps[t].push(c);
        capToTools[c].push(t);
      }
    }
  }

  // Build multi-level cap-to-cap adjacency lists
  // capToCapMatrices[level] is [numChildren, numParents] = [source, target] convention
  const childToParents = new Map<number, number[][]>();
  const parentToChildren = new Map<number, number[][]>();

  for (let level = 1; level <= graph.maxLevel; level++) {
    const matrix = graph.capToCapMatrices.get(level);
    if (!matrix) continue;

    const denseCC = matrix.arraySync() as number[][];
    const numChildren = denseCC.length;
    const numParents = denseCC[0]?.length || 0;

    const c2p: number[][] = Array.from({ length: numChildren }, () => []);
    const p2c: number[][] = Array.from({ length: numParents }, () => []);

    for (let c = 0; c < numChildren; c++) {
      for (let p = 0; p < numParents; p++) {
        if (denseCC[c][p] > 0) {
          c2p[c].push(p);
          p2c[p].push(c);
        }
      }
    }

    childToParents.set(level, c2p);
    parentToChildren.set(level, p2c);
  }

  return { toolToCaps, capToTools, childToParents, parentToChildren, maxLevel: graph.maxLevel };
}

/**
 * Sample K items from an array using Fisher-Yates partial shuffle.
 * Returns all items if array.length <= K.
 */
function sampleK(items: number[], K: number, random: () => number): number[] {
  if (items.length <= K) return items;
  const arr = [...items];
  const n = arr.length;
  for (let i = n - 1; i >= n - K; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr.slice(n - K);
}

/**
 * Sample a multi-level subgraph around batch tool IDs (Ancestral Path Sampling).
 *
 * Algorithm:
 * 1. Map batch tool IDs to indices in the full graph
 * 2. For each batch tool, find connected caps (level 0), sample K max
 * 3. For each sampled cap at level L, find parents at level L+1, sample K max
 *    → repeat until root level (Ancestral Path Sampling)
 * 4. For each sampled cap at L0, find sibling tools, sample K max
 * 5. Build mini GraphStructure with tool→cap and cap→cap matrices at each level
 *
 * This ensures W_up/W_down at ALL levels receive gradients through the tape.
 *
 * @param batchToolIds - Tool IDs in the current batch (positive + negatives)
 * @param graph - Full graph structure
 * @param adjCache - Pre-computed adjacency lists (multi-level)
 * @param K - Max neighbors to sample per node (default 8)
 * @param rng - Seeded RNG for reproducible sampling
 * @returns Mini GraphStructure with full hierarchy for sampled subgraph
 */
export function sampleSubgraph(
  batchToolIds: Set<string>,
  graph: GraphStructure,
  adjCache: AdjacencyCache,
  K: number = 8,
  rng?: () => number,
): GraphStructure {
  const random = rng || Math.random;

  // 1. Map batch tool IDs to global indices
  const toolIdToGlobalIdx = new Map<string, number>();
  for (let i = 0; i < graph.toolIds.length; i++) {
    toolIdToGlobalIdx.set(graph.toolIds[i], i);
  }

  const batchToolIndices = new Set<number>();
  for (const toolId of batchToolIds) {
    const idx = toolIdToGlobalIdx.get(toolId);
    if (idx !== undefined) batchToolIndices.add(idx);
  }

  // 2. Sample L0 caps from batch tools (tool → cap L0)
  const sampledCapsByLevel = new Map<number, Set<number>>();
  sampledCapsByLevel.set(0, new Set<number>());

  for (const toolIdx of batchToolIndices) {
    const connectedCaps = adjCache.toolToCaps[toolIdx];
    if (!connectedCaps || connectedCaps.length === 0) continue;
    for (const c of sampleK(connectedCaps, K, random)) {
      sampledCapsByLevel.get(0)!.add(c);
    }
  }

  // 3. Ancestral Path Sampling: for each sampled cap at level L,
  //    find parents at level L+1 and sample K. Repeat up to maxLevel.
  for (let level = 1; level <= adjCache.maxLevel; level++) {
    const childToParents = adjCache.childToParents.get(level);
    if (!childToParents) break;

    const childCaps = sampledCapsByLevel.get(level - 1);
    if (!childCaps || childCaps.size === 0) break;

    const parentSet = new Set<number>();
    for (const childIdx of childCaps) {
      const parents = childToParents[childIdx];
      if (!parents || parents.length === 0) continue;
      for (const p of sampleK(parents, K, random)) {
        parentSet.add(p);
      }
    }

    if (parentSet.size === 0) break;
    sampledCapsByLevel.set(level, parentSet);
  }

  // 4. Find sibling tools from sampled L0 caps
  const expandedToolIndices = new Set(batchToolIndices);
  const caps0Set = sampledCapsByLevel.get(0)!;
  for (const capIdx of caps0Set) {
    const connectedTools = adjCache.capToTools[capIdx];
    if (!connectedTools || connectedTools.length === 0) continue;
    for (const t of sampleK(connectedTools, K, random)) {
      expandedToolIndices.add(t);
    }
  }

  // 5. Build sorted index arrays per level
  const subToolGlobalIndices = [...expandedToolIndices].sort((a, b) => a - b);

  // Tool global→local map
  const toolGlobalToLocal = new Map<number, number>();
  for (let i = 0; i < subToolGlobalIndices.length; i++) {
    toolGlobalToLocal.set(subToolGlobalIndices[i], i);
  }

  // Cap global→local maps per level
  const capGlobalToLocalByLevel = new Map<number, Map<number, number>>();
  const sortedCapsByLevel = new Map<number, number[]>();
  let subMaxLevel = 0;
  for (const [level, capSet] of sampledCapsByLevel) {
    if (capSet.size === 0) continue;
    const sorted = [...capSet].sort((a, b) => a - b);
    sortedCapsByLevel.set(level, sorted);
    const g2l = new Map<number, number>();
    for (let i = 0; i < sorted.length; i++) {
      g2l.set(sorted[i], i);
    }
    capGlobalToLocalByLevel.set(level, g2l);
    subMaxLevel = Math.max(subMaxLevel, level);
  }

  // 6. Build mini tool→cap incidence matrix [numSubTools, numSubCapsL0]
  const numSubTools = subToolGlobalIndices.length;
  const caps0Sorted = sortedCapsByLevel.get(0) || [];
  const numSubCaps0 = caps0Sorted.length;
  const cap0G2L = capGlobalToLocalByLevel.get(0) || new Map();

  const miniMatrixData: number[][] = [];
  for (const tGlobal of subToolGlobalIndices) {
    const row = new Array(numSubCaps0).fill(0);
    for (const cGlobal of adjCache.toolToCaps[tGlobal]) {
      const cLocal = cap0G2L.get(cGlobal);
      if (cLocal !== undefined) row[cLocal] = 1;
    }
    miniMatrixData.push(row);
  }

  const toolToCapMatrix = numSubTools > 0 && numSubCaps0 > 0
    ? tf.tensor2d(miniMatrixData)
    : tf.zeros([Math.max(numSubTools, 1), Math.max(numSubCaps0, 1)]) as tf.Tensor2D;

  // 7. Build mini cap→cap matrices per level
  // 7. Build mini cap→cap matrices per level
  // Convention: [numChildren, numParents] = [source, target] (same as toolToCapMatrix)
  const capToCapMatrices = new Map<number, tf.Tensor2D>();
  for (let level = 1; level <= subMaxLevel; level++) {
    const parentsSorted = sortedCapsByLevel.get(level);
    const childrenSorted = sortedCapsByLevel.get(level - 1);
    if (!parentsSorted || !childrenSorted) continue;

    const parentG2L = capGlobalToLocalByLevel.get(level)!;
    const parentToChildrenForLevel = adjCache.parentToChildren.get(level);
    if (!parentToChildrenForLevel) continue;

    // Build [numChildren, numParents] matrix (children = rows, parents = cols)
    const matrixData: number[][] = [];
    for (const _cGlobal of childrenSorted) {
      matrixData.push(new Array(parentsSorted.length).fill(0));
    }
    const childG2L = capGlobalToLocalByLevel.get(level - 1)!;
    for (const pGlobal of parentsSorted) {
      const pLocal = parentG2L.get(pGlobal);
      if (pLocal === undefined) continue;
      const children = parentToChildrenForLevel[pGlobal];
      if (children) {
        for (const cGlobal of children) {
          const cLocal = childG2L.get(cGlobal);
          if (cLocal !== undefined) matrixData[cLocal][pLocal] = 1;
        }
      }
    }

    if (matrixData.length > 0 && matrixData[0].length > 0) {
      capToCapMatrices.set(level, tf.tensor2d(matrixData));
    }
  }

  // 8. Build subgraph ID arrays
  const subToolIds = subToolGlobalIndices.map((i) => graph.toolIds[i]);
  const capIdsByLevel = new Map<number, string[]>();
  for (const [level, sorted] of sortedCapsByLevel) {
    const fullLevelCaps = graph.capIdsByLevel.get(level) || [];
    capIdsByLevel.set(level, sorted.map((i) => fullLevelCaps[i]));
  }

  return {
    toolToCapMatrix,
    capToCapMatrices,
    toolIds: subToolIds,
    capIdsByLevel,
    maxLevel: subMaxLevel,
  };
}

// ============================================================================
// Trainer Class
// ============================================================================

/**
 * SHGAT Trainer with TensorFlow.js autograd
 *
 * Uses dense TF.js autograd for all operations (message passing + K-head scoring).
 * Requires CPU or WebGPU backend (WASM lacks UnsortedSegmentSum for tf.gather gradients).
 */
export class AutogradTrainer {
  private params: TFParams;
  private optimizer: tf.Optimizer;
  private config: SHGATConfig;
  private trainerConfig: TrainerConfig;
  private nodeEmbeddings: Map<string, number[]> = new Map();
  private enrichedNodeEmbeddings: Map<string, number[]> | null = null;
  private graph: GraphStructure | null = null;
  private useMessagePassing: boolean = false;
  /** Cached adjacency lists for subgraph sampling (built once in setGraph) */
  private adjCache: AdjacencyCache | null = null;
  /** Subgraph neighbor sample size (K parameter for Ancestral Path Sampling) */
  private subgraphK: number = 8;
  /** Seeded RNG for reproducible subgraph sampling */
  private subgraphRng: (() => number) | undefined;

  constructor(
    config: SHGATConfig,
    trainerConfig: Partial<TrainerConfig> = {},
    maxLevel = 3,
    baseSeed?: number,
  ) {
    this.config = config;
    this.trainerConfig = { ...DEFAULT_TRAINER_CONFIG, ...trainerConfig };
    this.params = initTFParams(config, maxLevel, baseSeed);
    this.optimizer = tf.train.adam(this.trainerConfig.learningRate);
    // Initialize seeded RNG for reproducible subgraph sampling
    if (baseSeed !== undefined) {
      this.subgraphRng = AutogradTrainer.createSeededRng(baseSeed + 99999);
    }
  }

  /** Create a mulberry32-based seeded RNG */
  private static createSeededRng(seed: number): () => number {
    let s = seed | 0;
    return () => {
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /**
   * Configure subgraph neighbor sampling K parameter.
   * @param K - Max neighbors per node (default 8, recommended 16)
   */
  setSubgraphK(K: number): void {
    this.subgraphK = K;
  }

  /**
   * Update temperature for InfoNCE loss (for annealing during training)
   */
  setTemperature(temperature: number): void {
    this.trainerConfig.temperature = temperature;
  }

  /**
   * Update learning rate (for warmup + cosine decay scheduling).
   * Recreates the Adam optimizer with the new LR since TF.js Adam
   * doesn't expose a mutable learningRate property.
   * Adam momentum buffers (m, v) are reset — acceptable for smooth LR schedules.
   */
  setLearningRate(lr: number): void {
    if (Math.abs(lr - this.trainerConfig.learningRate) < 1e-10) return;
    this.trainerConfig.learningRate = lr;
    this.optimizer.dispose();
    this.optimizer = tf.train.adam(lr);
  }

  /**
   * Set node embeddings (tools + capabilities)
   */
  setNodeEmbeddings(embeddings: Map<string, number[]>): void {
    this.nodeEmbeddings = embeddings;
  }

  /**
   * Set graph structure for message passing
   *
   * When set, scoring and training will use full message passing.
   * Without graph, only K-head attention is used (faster but less accurate).
   *
   * Also builds the adjacency cache for subgraph sampling (one-time cost).
   */
  setGraph(graph: GraphStructure): void {
    this.graph = graph;
    this.useMessagePassing = true;
    // Build adjacency cache for efficient subgraph sampling in trainBatch
    const t0 = Date.now();
    this.adjCache = buildAdjacencyCache(graph);
    const levelSummary = [];
    for (let l = 0; l <= graph.maxLevel; l++) {
      const n = (graph.capIdsByLevel.get(l) || []).length;
      if (n > 0) levelSummary.push(`L${l}:${n}`);
    }
    console.log(
      `  [AdjCache] Built in ${Date.now() - t0}ms: ` +
        `${graph.toolIds.length} tools, caps=[${levelSummary.join(", ")}], maxLevel=${graph.maxLevel}`,
    );
  }

  /**
   * Configure subgraph sampling parameters for mini-batch MP.
   *
   * @param K - Max neighbors per node (default 8). Higher = more accurate but more memory.
   * @param rng - Optional seeded RNG for reproducible sampling.
   */
  setSubgraphSampling(K: number, rng?: () => number): void {
    this.subgraphK = K;
    this.subgraphRng = rng;
  }

  /**
   * Check if message passing is enabled
   */
  hasMessagePassing(): boolean {
    return this.useMessagePassing && this.graph !== null;
  }

  /**
   * Pre-compute enriched embeddings via message passing OUTSIDE the gradient tape.
   *
   * Standard GNN optimization (mini-batch sampling, ClusterGCN): run MP once per epoch,
   * store enriched embeddings as plain JS arrays, then use them as frozen inputs
   * for per-batch scoring. This reduces autograd tape memory from ~3GB to ~50MB.
   *
   * Call this once per epoch before trainBatch/trainBatchKL.
   * W_up/W_down params still participate: each epoch re-runs MP with updated params.
   */
  precomputeEnrichedEmbeddings(): void {
    if (!this.useMessagePassing || !this.graph) {
      this.enrichedNodeEmbeddings = null;
      return;
    }

    const t0 = Date.now();

    // Build initial embedding tensors
    const toolEmbs: number[][] = [];
    for (const toolId of this.graph.toolIds) {
      toolEmbs.push(
        this.nodeEmbeddings.get(toolId) || new Array(this.config.embeddingDim).fill(0),
      );
    }
    const H_init = ops.toTensor(toolEmbs);

    const E_init = new Map<number, tf.Tensor2D>();
    for (const [level, capIds] of this.graph.capIdsByLevel) {
      const capEmbs: number[][] = [];
      for (const capId of capIds) {
        capEmbs.push(
          this.nodeEmbeddings.get(capId) || new Array(this.config.embeddingDim).fill(0),
        );
      }
      E_init.set(level, ops.toTensor(capEmbs));
    }

    // Run message passing forward inside tidy to dispose ALL intermediates
    // (attention matrices, projections, residuals, etc.)
    // Only H and E survive the tidy scope via tf.keep()
    const { H, E } = tf.tidy(() => {
      const result = messagePassingForward(H_init, E_init, this.graph!, this.params, this.config);
      tf.keep(result.H);
      for (const [, tensor] of result.E) tf.keep(tensor);
      return result;
    });

    // H_init and E_init were consumed inside tidy — dispose them
    H_init.dispose();
    for (const [, tensor] of E_init) tensor.dispose();

    // Extract to plain JS arrays and store
    const enriched = new Map<string, number[]>();
    const H_arr = H.arraySync() as number[][];
    for (let i = 0; i < this.graph.toolIds.length; i++) {
      enriched.set(this.graph.toolIds[i], H_arr[i]);
    }
    for (const [level, capIds] of this.graph.capIdsByLevel) {
      const E_tensor = E.get(level);
      if (E_tensor) {
        const E_arr = E_tensor.arraySync() as number[][];
        for (let i = 0; i < capIds.length; i++) {
          enriched.set(capIds[i], E_arr[i]);
        }
      }
    }

    // Dispose output tensors — enriched data is now in JS arrays
    H.dispose();
    for (const [, tensor] of E) tensor.dispose();

    this.enrichedNodeEmbeddings = enriched;
    console.log(`  [MP] Pre-computed ${enriched.size} enriched embeddings in ${Date.now() - t0}ms`);
  }

  /**
   * Get the effective embeddings (enriched if available, else raw)
   */
  private getEffectiveEmbeddings(): Map<string, number[]> {
    return this.enrichedNodeEmbeddings ?? this.nodeEmbeddings;
  }

  /**
   * Build a mini-batch MP context using Ancestral Path Sampling.
   *
   * Instead of running MP on the full graph [1901 tools, 7083 caps],
   * samples a multi-level subgraph around the batch's tools (~500 tools, ~200 caps/level).
   * This keeps the gradient tape memory under ~1 GB while allowing
   * W_up, W_down, a_up, a_down at ALL levels to receive gradients.
   */
  private buildSubgraphContext(
    examples: TrainingExample[],
  ): MessagePassingContext | undefined {
    if (!this.useMessagePassing || !this.graph || !this.adjCache) {
      return undefined;
    }

    // Collect all tool IDs referenced in this batch (candidate + negatives)
    const batchToolIds = new Set<string>();
    for (const ex of examples) {
      batchToolIds.add(ex.candidateId);
      if (ex.negativeCapIds) {
        for (const negId of ex.negativeCapIds) {
          // Only include IDs that are actual tools (exist in nodeEmbeddings)
          if (this.nodeEmbeddings.has(negId)) {
            batchToolIds.add(negId);
          }
        }
      }
    }

    // Sample subgraph around batch tools
    const subgraph = sampleSubgraph(
      batchToolIds,
      this.graph,
      this.adjCache,
      this.subgraphK,
      this.subgraphRng,
    );

    // Build initial tool embedding tensor [numSubTools, embDim]
    const H_init = ops.toTensor(
      subgraph.toolIds.map(
        (id) =>
          this.nodeEmbeddings.get(id) ||
          new Array(this.config.embeddingDim).fill(0),
      ),
    );

    // Build capability embeddings per level (multi-level via Ancestral Path Sampling)
    const E_init = new Map<number, tf.Tensor2D>();
    for (const [level, capIds] of subgraph.capIdsByLevel) {
      E_init.set(
        level,
        ops.toTensor(
          capIds.map(
            (id) =>
              this.nodeEmbeddings.get(id) ||
              new Array(this.config.embeddingDim).fill(0),
          ),
        ),
      );
    }

    return {
      graph: subgraph,
      H_init,
      E_init,
    };
  }

  /**
   * Train on a batch of examples.
   *
   * Uses full MP inside the gradient tape so W_up/W_down/a_up/a_down receive gradients.
   * Pre-computed enriched embeddings are only for KL batches and eval/score.
   */
  async trainBatch(examples: TrainingExample[]): Promise<TrainingMetrics> {
    // Ensure backend supports autograd
    if (!supportsAutograd()) {
      const prevBackend = getBackend();
      const newBackend = await switchBackend("training" as BackendMode);
      console.error(
        `[AutogradTrainer] Switched backend from ${prevBackend} to ${newBackend} for training (autograd required)`,
      );
    }

    // Build subgraph MP context: samples multi-level neighborhood around batch tools
    // via Ancestral Path Sampling so W_up, W_down at ALL levels receive gradients.
    // Mini graph (~500 tools, ~200 caps/level) instead of full graph (1901, 7083).
    const mpContext = this.buildSubgraphContext(examples);

    const metrics = trainStep(
      examples,
      this.nodeEmbeddings,
      this.params,
      this.config,
      this.trainerConfig,
      this.optimizer,
      mpContext,
    );

    // Clean up subgraph tensors (H_init/E_init consumed by trainStep, but
    // we still own the mini incidence matrix from the subgraph)
    if (mpContext) {
      mpContext.H_init.dispose();
      for (const [, tensor] of mpContext.E_init) {
        tensor.dispose();
      }
      // Dispose the mini incidence matrix (separate from the full graph)
      if (mpContext.graph !== this.graph) {
        disposeGraphStructure(mpContext.graph);
      }
    }

    return metrics;
  }

  /**
   * Train on a batch of KL soft target examples (n8n augmentation).
   *
   * Scores each intent against ALL tools in the vocabulary.
   * Uses KL divergence instead of InfoNCE.
   *
   * @param examples - Soft target examples
   * @param toolIds - Ordered list of tool IDs (must match soft target indices)
   * @param klTemperature - Temperature for softmax over K-head scores
   */
  async trainBatchKL(
    examples: import("../core/types.ts").SoftTargetExample[],
    toolIds: string[],
    klTemperature: number,
    klWeight: number = 1.0,
  ): Promise<KLTrainingMetrics> {
    // Ensure backend supports autograd
    if (!supportsAutograd()) {
      const prevBackend = getBackend();
      const newBackend = await switchBackend("training" as BackendMode);
      console.error(
        `[AutogradTrainer] Switched backend from ${prevBackend} to ${newBackend} for training (autograd required)`,
      );
    }

    // Build all-tools embedding tensor [numTools, embDim]
    // Use enriched embeddings if available (pre-computed MP)
    const effectiveEmbs = this.getEffectiveEmbeddings();
    const toolEmbs: number[][] = [];
    for (const toolId of toolIds) {
      toolEmbs.push(
        effectiveEmbs.get(toolId) || new Array(this.config.embeddingDim).fill(0),
      );
    }
    const allToolEmbsTensor = ops.toTensor(toolEmbs);

    const metrics = trainStepKL(
      examples,
      allToolEmbsTensor,
      toolIds.length,
      this.params,
      this.config,
      this.trainerConfig,
      this.optimizer,
      klTemperature,
      klWeight,
    );

    allToolEmbsTensor.dispose();

    return metrics;
  }

  /**
   * Score nodes for an intent
   *
   * If graph is set, uses dense TF.js message passing for enriched embeddings.
   * Otherwise, uses direct K-head attention (faster but less accurate).
   */
  score(intentEmb: number[], nodeIds: string[]): number[] {
    return tidy(() => {
      const intentTensor = ops.toTensor(intentEmb);

      // Use pre-computed enriched embeddings if available (fast path)
      if (this.enrichedNodeEmbeddings) {
        const enriched: number[][] = nodeIds.map(
          (id) =>
            this.enrichedNodeEmbeddings!.get(id) || this.nodeEmbeddings.get(id) ||
            new Array(this.config.embeddingDim).fill(0),
        );
        const nodesTensor = ops.toTensor(enriched);
        const scores = forwardScoring(intentTensor, nodesTensor, this.params, this.config);
        return scores.arraySync() as number[];
      }

      // Get raw embeddings
      const nodeEmbs: number[][] = nodeIds.map(
        (id) => this.nodeEmbeddings.get(id) || new Array(this.config.embeddingDim).fill(0),
      );
      let nodesTensor = ops.toTensor(nodeEmbs);

      // Apply message passing if graph is available
      if (this.useMessagePassing && this.graph) {
        // Build initial embeddings
        const toolEmbs: number[][] = [];
        for (const toolId of this.graph.toolIds) {
          toolEmbs.push(
            this.nodeEmbeddings.get(toolId) || new Array(this.config.embeddingDim).fill(0),
          );
        }

        const H_init = ops.toTensor(toolEmbs);
        const E_init = new Map<number, tf.Tensor2D>();
        for (const [level, capIds] of this.graph.capIdsByLevel) {
          const capEmbs: number[][] = [];
          for (const capId of capIds) {
            capEmbs.push(
              this.nodeEmbeddings.get(capId) || new Array(this.config.embeddingDim).fill(0),
            );
          }
          E_init.set(level, ops.toTensor(capEmbs));
        }

        // Dense message passing
        const { H: H_mp, E: E_mp } = messagePassingForward(
          H_init,
          E_init,
          this.graph,
          this.params,
          this.config,
        );

        const H_arr = H_mp.arraySync() as number[][];
        const E_arrs = new Map<number, number[][]>();
        for (const [level, tensor] of E_mp) {
          E_arrs.set(level, tensor.arraySync() as number[][]);
        }

        // Cleanup dense tensors
        H_init.dispose();
        for (const [, tensor] of E_init) tensor.dispose();
        H_mp.dispose();
        for (const [, tensor] of E_mp) tensor.dispose();

        // Map nodeIds to their enriched embeddings
        const enriched: number[][] = [];
        for (const nodeId of nodeIds) {
          const toolIdx = this.graph.toolIds.indexOf(nodeId);
          if (toolIdx >= 0) {
            enriched.push(H_arr[toolIdx]);
            continue;
          }

          let found = false;
          for (const [level, capIds] of this.graph.capIdsByLevel) {
            const capIdx = capIds.indexOf(nodeId);
            if (capIdx >= 0) {
              const E_arr = E_arrs.get(level);
              if (E_arr && capIdx < E_arr.length) {
                enriched.push(E_arr[capIdx]);
                found = true;
                break;
              }
            }
          }

          if (!found) {
            enriched.push(
              this.nodeEmbeddings.get(nodeId) || new Array(this.config.embeddingDim).fill(0),
            );
          }
        }

        nodesTensor = ops.toTensor(enriched);
      }

      const scores = forwardScoring(intentTensor, nodesTensor, this.params, this.config);
      return scores.arraySync() as number[];
    });
  }

  /**
   * Get parameters for serialization
   */
  getParams(): TFParams {
    return this.params;
  }

  /**
   * Export projection head parameters as arrays for persistence.
   * Returns undefined if projection head is not enabled.
   */
  exportProjectionHeadToArray():
    | import("../core/projection-head.ts").ProjectionHeadArrayParams
    | undefined {
    if (!this.params.projectionHead) return undefined;
    return {
      W1: this.params.projectionHead.W1.arraySync() as number[][],
      b1: this.params.projectionHead.b1.arraySync() as number[],
      W2: this.params.projectionHead.W2.arraySync() as number[][],
      b2: this.params.projectionHead.b2.arraySync() as number[],
    };
  }

  /**
   * Dispose all tensors
   */
  dispose(): void {
    for (const W of this.params.W_k) W.dispose();
    if (this.params.W_q) {
      for (const W of this.params.W_q) W.dispose();
    }
    this.params.W_intent.dispose();
    this.params.residualWeights?.dispose();

    for (const [_, weights] of this.params.W_up) {
      weights.forEach((w) => w.dispose());
    }
    for (const [_, weights] of this.params.W_down) {
      weights.forEach((w) => w.dispose());
    }
    for (const [_, weights] of this.params.a_up) {
      weights.forEach((w) => w.dispose());
    }
    for (const [_, weights] of this.params.a_down) {
      weights.forEach((w) => w.dispose());
    }

    if (this.params.projectionHead) {
      this.params.projectionHead.W1.dispose();
      this.params.projectionHead.b1.dispose();
      this.params.projectionHead.W2.dispose();
      this.params.projectionHead.b2.dispose();
    }
  }
}
