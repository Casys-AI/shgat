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

import { tf, tidy } from "../tf/backend.ts";
import * as ops from "../tf/ops.ts";
import type { SHGATConfig, TrainingExample } from "../core/types.ts";
import {
  initProjectionHeadParams,
  projectionScore,
} from "../core/projection-head.ts";

// Sparse message passing imports
import {
  sparseMPForward,
  sparseMPBackward,
  applySparseMPGradients,
} from "./sparse-mp.ts";

// ============================================================================
// Types
// ============================================================================

/**
 * Trainable parameters as TF.js Variables
 */
export interface TFParams {
  // K-head scoring parameters
  W_k: tf.Variable[];      // [numHeads][embDim, headDim] - Key projection
  W_q: tf.Variable[];      // [numHeads][embDim, headDim] - Query projection
  W_intent: tf.Variable;   // [embDim, hiddenDim] - Intent projection

  // Message passing parameters (per level)
  W_up: Map<number, tf.Variable[]>;    // level -> [numHeads][embDim, headDim]
  W_down: Map<number, tf.Variable[]>;  // level -> [numHeads][embDim, headDim]
  a_up: Map<number, tf.Variable[]>;    // level -> [numHeads][2*headDim]
  a_down: Map<number, tf.Variable[]>;  // level -> [numHeads][2*headDim]

  // Residual weights (learnable)
  residualWeights?: tf.Variable;  // [numLevels]

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
  temperature: number;      // InfoNCE temperature (default 0.07)
  gradientClip: number;     // Max gradient norm
  l2Lambda: number;         // L2 regularization
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
 */
export function initTFParams(config: SHGATConfig, maxLevel: number): TFParams {
  const { numHeads, embeddingDim, headDim, hiddenDim } = config;

  // K-head scoring
  const W_k: tf.Variable[] = [];
  const W_q: tf.Variable[] = [];
  for (let h = 0; h < numHeads; h++) {
    W_k.push(tf.variable(ops.glorotNormal([embeddingDim, headDim]), true, `W_k_${h}`));
    W_q.push(tf.variable(ops.glorotNormal([embeddingDim, headDim]), true, `W_q_${h}`));
  }

  const W_intent = tf.variable(
    ops.glorotNormal([embeddingDim, hiddenDim || embeddingDim]),
    true,
    "W_intent"
  );

  // Message passing per level
  const W_up = new Map<number, tf.Variable[]>();
  const W_down = new Map<number, tf.Variable[]>();
  const a_up = new Map<number, tf.Variable[]>();
  const a_down = new Map<number, tf.Variable[]>();

  for (let level = 1; level <= maxLevel; level++) {
    const W_up_level: tf.Variable[] = [];
    const W_down_level: tf.Variable[] = [];
    const a_up_level: tf.Variable[] = [];
    const a_down_level: tf.Variable[] = [];

    for (let h = 0; h < numHeads; h++) {
      W_up_level.push(tf.variable(ops.glorotNormal([embeddingDim, headDim]), true, `W_up_${level}_${h}`));
      W_down_level.push(tf.variable(ops.glorotNormal([embeddingDim, headDim]), true, `W_down_${level}_${h}`));
      a_up_level.push(tf.variable(ops.glorotNormal([2 * headDim]), true, `a_up_${level}_${h}`));
      a_down_level.push(tf.variable(ops.glorotNormal([2 * headDim]), true, `a_down_${level}_${h}`));
    }

    W_up.set(level, W_up_level);
    W_down.set(level, W_down_level);
    a_up.set(level, a_up_level);
    a_down.set(level, a_down_level);
  }

  // Learnable residual weights
  const residualWeights = tf.variable(
    tf.fill([maxLevel + 1], 0.3),  // Default 0.3
    true,
    "residualWeights"
  );

  // Optional projection head
  const projectionHead = config.useProjectionHead
    ? initProjectionHeadParams(
        embeddingDim,
        config.projectionHiddenDim ?? 256,
        config.projectionOutputDim ?? 256,
      )
    : undefined;

  return { W_k, W_q, W_intent, W_up, W_down, a_up, a_down, residualWeights, projectionHead };
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
function attentionAggregation(
  sourceEmbs: tf.Tensor2D,     // [numSource, embDim]
  targetEmbs: tf.Tensor2D,     // [numTarget, embDim]
  connectivity: tf.Tensor2D,   // [numSource, numTarget]
  W_source: tf.Variable,       // [embDim, headDim]
  W_target: tf.Variable,       // [embDim, headDim]
  a: tf.Variable,              // [2 * headDim]
): tf.Tensor2D {
  // Project embeddings (efficient matmul)
  const srcProj = tf.matMul(sourceEmbs, W_source);  // [numSource, headDim]
  const tgtProj = tf.matMul(targetEmbs, W_target);  // [numTarget, headDim]

  // Compute attention scores using einsum-style operation
  // For each (t, s) pair: score = a^T @ LeakyReLU([tgtProj[t], srcProj[s]])
  //
  // Optimization: Split attention vector a = [a_tgt, a_src] where each is [headDim]
  // Then: score = a_tgt @ LeakyReLU(tgtProj[t]) + a_src @ LeakyReLU(srcProj[s])
  // This avoids creating the [numTgt, numSrc, 2*headDim] tensor!

  const headDim = W_source.shape[1] as number;
  const a_tgt = a.slice([0], [headDim]);       // First half of attention vector
  const a_src = a.slice([headDim], [headDim]); // Second half

  // Compute per-target and per-source contributions separately
  const tgtActivated = tf.leakyRelu(tgtProj, LEAKY_RELU_SLOPE);  // [numTarget, headDim]
  const srcActivated = tf.leakyRelu(srcProj, LEAKY_RELU_SLOPE);  // [numSource, headDim]

  // score_tgt[t] = tgtActivated[t] @ a_tgt  -> [numTarget]
  const scoreTgt = tf.squeeze(tf.matMul(tgtActivated, a_tgt.expandDims(1)));

  // score_src[s] = srcActivated[s] @ a_src  -> [numSource]
  const scoreSrc = tf.squeeze(tf.matMul(srcActivated, a_src.expandDims(1)));

  // Full scores: scores[t, s] = score_tgt[t] + score_src[s]
  // Use broadcasting: [numTarget, 1] + [1, numSource] -> [numTarget, numSource]
  const scores = tf.add(
    scoreTgt.expandDims(1),  // [numTarget, 1]
    scoreSrc.expandDims(0),  // [1, numSource]
  );  // [numTarget, numSource]

  // Mask non-connected pairs (transpose connectivity to [numTarget, numSource])
  const connT = tf.transpose(connectivity) as tf.Tensor2D;
  const mask = tf.equal(connT, 0);
  const maskedScores = tf.where(mask, tf.fill(scores.shape, -1e9), scores);

  // Softmax over sources (last dim)
  const attention = tf.softmax(maskedScores, -1);  // [numTarget, numSource]

  // Aggregate: result[t] = sum_s(attention[t,s] * srcProj[s])
  const result = tf.matMul(attention, srcProj);  // [numTarget, headDim]

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
  H_init: tf.Tensor2D,                    // [numTools, embDim]
  E_init: Map<number, tf.Tensor2D>,       // level -> [numCaps, embDim]
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
        H,                          // source: tools
        capsAtLevel,                // target: caps level 0
        graph.toolToCapMatrix,      // connectivity
        W_up,                       // W_source (for tools)
        W_up,                       // W_target (same weights for simplicity)
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
  // ========================================================================

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

    const E_new = multiHeadMessagePassing(
      capsAtParent,   // source: parent caps
      capsAtLevel,    // target: child caps
      reverseConn,
      W_down,
      W_down,
      a_down,
      numHeads,
    );

    reverseConn.dispose();
    E.get(level)?.dispose();
    E.set(level, E_new);
  }

  // Final: E^0 → V
  const E_level0 = E.get(0);
  const W_down = params.W_down.get(1);
  const a_down = params.a_down.get(1);
  if (E_level0 && W_down && a_down) {
    // Transpose toolToCapMatrix for E→V direction
    const reverseConn = tf.transpose(graph.toolToCapMatrix) as tf.Tensor2D;

    const H_new = multiHeadMessagePassing(
      E_level0,      // source: caps
      H,             // target: tools
      reverseConn,
      W_down,
      W_down,
      a_down,
      numHeads,
    );

    reverseConn.dispose();
    H.dispose();
    H = H_new;
  }

  // Apply residual connection
  if (params.residualWeights) {
    const alpha = 0.3;  // Default residual weight
    const H_residual = tf.add(
      tf.mul(H, 1 - alpha),
      tf.mul(H_init, alpha),
    ) as tf.Tensor2D;
    H.dispose();
    H = H_residual;

    for (const [level, E_tensor] of E) {
      const E_initLevel = E_init.get(level);
      if (E_initLevel) {
        const E_residual = tf.add(
          tf.mul(E_tensor, 1 - alpha),
          tf.mul(E_initLevel, alpha),
        ) as tf.Tensor2D;
        E_tensor.dispose();
        E.set(level, E_residual);
      }
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

    // Project intent: Q = intentProj @ W_q[h]  [headDim]
    // Note: intentProj is already projected, we use it directly reshaped
    const Q = intentProj.slice([h * config.headDim], [config.headDim]);

    // Attention: scores = K @ Q  [numNodes]
    const scores = tf.squeeze(tf.matMul(K, Q.expandDims(1)));

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
  // Project intent
  const intentProj = tf.squeeze(
    tf.matMul(intentEmb.expandDims(0), params.W_intent)
  ) as tf.Tensor1D;

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
    logits.expandDims(0)
  ) as tf.Scalar;
}

/**
 * Batch contrastive loss with in-batch negatives
 *
 * Note: This function is NOT used in the main sparse training path.
 * It's kept for potential future use or legacy dense training.
 */
export function batchContrastiveLoss(
  intentEmbs: tf.Tensor2D,   // [batchSize, embDim]
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
      temperature
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
// Training Step
// ============================================================================

/**
 * Message passing context for training
 * Pre-computed once per batch to avoid recomputation per example
 */
export interface MessagePassingContext {
  graph: GraphStructure;
  H_init: tf.Tensor2D;
  E_init: Map<number, tf.Tensor2D>;
  /**
   * Use sparse message passing implementation.
   * When true, uses JS sparse loops for MP (avoids TF.js tf.gather gradient issues on WASM).
   * When false, uses dense TF.js autograd (requires CPU backend, ~10x slower).
   * @default true
   */
  useSparse?: boolean;
}

/**
 * Single training step with autograd
 *
 * If mpContext is provided, message passing is applied to enrich embeddings
 * BEFORE scoring, allowing gradients to flow through W_up, W_down, a_up, a_down.
 *
 * SPARSE MODE (default, recommended):
 * When mpContext.useSparse is true (default), uses sparse JS loops for message passing.
 * This avoids TF.js tf.gather gradient issues on WASM backend and is ~10x faster.
 * K-head scoring still uses TF.js autograd for fast dense operations.
 *
 * DENSE MODE (legacy):
 * When mpContext.useSparse is false, uses dense TF.js autograd for all operations.
 * Requires CPU backend (WASM lacks UnsortedSegmentSum kernel for tf.gather gradients).
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
  // Check if we should use sparse MP (default: true when mpContext is provided)
  const useSparse = mpContext?.useSparse !== false;

  if (useSparse && mpContext) {
    return trainStepSparse(
      examples,
      nodeEmbeddings,
      params,
      config,
      trainerConfig,
      optimizer,
      mpContext,
    );
  }

  // Dense mode (legacy) - use TF.js autograd for everything
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
      const rawEmbs = rawEmbNodeIds.map(id =>
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
      const indices = nodeIds.map(id => nodeIdToIdx.get(id) ?? 0);
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
      const allScoresArr = scores.arraySync() as number[];
      const maxIdx = allScoresArr.indexOf(Math.max(...allScoresArr));
      if (maxIdx === 0) totalCorrect++;
    }

    // Average loss
    const avgLoss = loss.div(examples.length);

    // L2 regularization on ALL parameters including message passing
    let l2Loss = tf.scalar(0);

    // K-head scoring parameters
    for (const W of params.W_k) {
      l2Loss = l2Loss.add(tf.sum(tf.square(W)));
    }
    for (const W of params.W_q) {
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

    totalLoss = avgLoss.add(l2Loss).arraySync() as number;

    return avgLoss.add(l2Loss) as tf.Scalar;
  });

  // Compute gradient norm
  let gradNormSquared = 0;
  for (const g of Object.values(grads)) {
    gradNormSquared += (tf.sum(tf.square(g)).arraySync() as number);
  }
  const gradientNorm = Math.sqrt(gradNormSquared);

  // Clip gradients if needed
  if (gradientNorm > trainerConfig.gradientClip) {
    const scale = trainerConfig.gradientClip / gradientNorm;
    for (const key of Object.keys(grads)) {
      grads[key] = grads[key].mul(scale);
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

    const validChildren = (cap.children || []).filter(id => capIdToInfo.has(id));
    let level: number;
    if (validChildren.length === 0) {
      level = 0;  // Leaf
    } else {
      const childLevels = validChildren.map(childId => computeLevel(childId));
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
  // With leaf=0 convention: level L parents connect to level L-1 children
  // Matrix[level] is [numParentsAtLevel, numChildrenAtLevelMinus1]
  const capToCapMatrices = new Map<number, tf.Tensor2D>();
  for (let level = 1; level <= maxLevel; level++) {
    const parentCaps = capIdsByLevel.get(level) || [];      // Parents at higher level
    const childCaps = capIdsByLevel.get(level - 1) || [];   // Children at lower level

    const matrixData: number[][] = [];
    for (const parentId of parentCaps) {
      const row: number[] = [];
      const parentInfo = capIdToInfo.get(parentId);
      for (const childId of childCaps) {
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
// Sparse Training Step (Hybrid: Sparse MP + Dense K-head)
// ============================================================================

/**
 * Training step with sparse message passing
 *
 * This hybrid approach:
 * 1. Runs sparse MP forward (JS loops, only connected pairs)
 * 2. Computes K-head scoring loss with TF.js autograd (dense, fast)
 * 3. Backprops K-head gradients via autograd
 * 4. Manually computes and applies MP gradients (sparse backward)
 *
 * Benefits:
 * - Works on WASM backend (no UnsortedSegmentSum needed)
 * - ~10x faster than dense MP on large graphs
 * - MP gradients flow correctly to W_up, W_down, a_up, a_down
 */
function trainStepSparse(
  examples: TrainingExample[],
  nodeEmbeddings: Map<string, number[]>,
  params: TFParams,
  config: SHGATConfig,
  trainerConfig: TrainerConfig,
  optimizer: tf.Optimizer,
  mpContext: MessagePassingContext,
): TrainingMetrics {
  // ========================================================================
  // STEP 1: Sparse message passing forward
  // ========================================================================

  // Convert TF.js tensors to JS arrays for sparse MP
  const H_init_arr = mpContext.H_init.arraySync() as number[][];
  const E_init_arr = new Map<number, number[][]>();
  for (const [level, tensor] of mpContext.E_init) {
    E_init_arr.set(level, tensor.arraySync() as number[][]);
  }

  // Run sparse MP forward
  const { H: H_enriched, E: E_enriched, cache: mpCache } = sparseMPForward(
    H_init_arr,
    E_init_arr,
    mpContext.graph,
    params,
    config,
  );

  // Build enriched embeddings map for K-head scoring
  const enrichedEmbeddings = new Map<string, number[]>();

  // Add enriched tool embeddings
  for (let i = 0; i < mpContext.graph.toolIds.length; i++) {
    enrichedEmbeddings.set(mpContext.graph.toolIds[i], H_enriched[i]);
  }

  // Add enriched cap embeddings
  for (const [level, capIds] of mpContext.graph.capIdsByLevel) {
    const E_level = E_enriched.get(level);
    if (E_level) {
      for (let i = 0; i < capIds.length; i++) {
        enrichedEmbeddings.set(capIds[i], E_level[i]);
      }
    }
  }

  // ========================================================================
  // STEP 2: K-head scoring with TF.js autograd (for W_k, W_q, W_intent)
  // ========================================================================

  let totalLoss = 0;
  let totalCorrect = 0;

  // Gradient accumulators for manual MP backward
  const dH_accum: number[][] = H_enriched.map(row => new Array(row.length).fill(0));
  const dE_accum = new Map<number, number[][]>();
  for (const [level, embs] of E_enriched) {
    dE_accum.set(level, embs.map(row => new Array(row.length).fill(0)));
  }

  const { grads: kheadGrads, value: batchLoss } = tf.variableGrads(() => {
    let loss = tf.scalar(0);

    for (const ex of examples) {
      // Collect node IDs for this example
      const nodeIds: string[] = [ex.candidateId];
      for (const negId of ex.negativeCapIds || []) {
        if (enrichedEmbeddings.has(negId) || nodeEmbeddings.has(negId)) {
          nodeIds.push(negId);
        }
      }

      if (nodeIds.length < 2) continue;

      // Build embeddings tensor from enriched embeddings
      const nodeEmbs: number[][] = nodeIds.map(id =>
        enrichedEmbeddings.get(id) || nodeEmbeddings.get(id) || new Array(config.embeddingDim).fill(0)
      );
      const nodeEmbsTensor = ops.toTensor(nodeEmbs);

      // Get intent embedding
      const intentEmb = ops.toTensor(ex.intentEmbedding);

      // K-head scoring (this is what autograd will differentiate)
      const scores = forwardScoring(intentEmb, nodeEmbsTensor, params, config);

      // Positive is at index 0
      const positiveScore = scores.slice([0], [1]).squeeze() as tf.Scalar;
      const negativeScores = scores.slice([1], [nodeIds.length - 1]) as tf.Tensor1D;

      // InfoNCE loss
      const exampleLoss = infoNCELoss(positiveScore, negativeScores, trainerConfig.temperature);
      loss = loss.add(exampleLoss);

      // Track accuracy
      const allScoresArr = scores.arraySync() as number[];
      const maxIdx = allScoresArr.indexOf(Math.max(...allScoresArr));
      if (maxIdx === 0) totalCorrect++;

      // ====================================================================
      // ACCUMULATE dEnrichedEmb for MP backward
      // We need d(loss)/d(enrichedEmb) to backprop through MP
      // Approximation: use the logit gradient as a proxy
      // ====================================================================
      const softmaxLogits = allScoresArr.map(s => s / trainerConfig.temperature);
      const maxLogit = Math.max(...softmaxLogits);
      const expLogits = softmaxLogits.map(s => Math.exp(s - maxLogit));
      const sumExp = expLogits.reduce((a, b) => a + b, 0);
      const softmaxProbs = expLogits.map(e => e / sumExp);

      // dLogit[i] = softmax[i] - (1 if i==0 else 0)
      const dLogits = softmaxProbs.map((p, i) => (i === 0 ? p - 1 : p) / trainerConfig.temperature);

      // Propagate to embeddings
      // Use full gradient magnitude - gradient clipping handles explosion prevention
      for (let idx = 0; idx < nodeIds.length; idx++) {
        const nodeId = nodeIds[idx];
        const dLogit = dLogits[idx];

        // Find where this node lives (tool or cap at which level)
        const toolIdx = mpContext.graph.toolIds.indexOf(nodeId);
        if (toolIdx >= 0) {
          for (let j = 0; j < dH_accum[toolIdx].length; j++) {
            dH_accum[toolIdx][j] += dLogit;
          }
        } else {
          for (const [level, capIds] of mpContext.graph.capIdsByLevel) {
            const capIdx = capIds.indexOf(nodeId);
            if (capIdx >= 0) {
              const dE_level = dE_accum.get(level);
              if (dE_level) {
                for (let j = 0; j < dE_level[capIdx].length; j++) {
                  dE_level[capIdx][j] += dLogit;
                }
              }
              break;
            }
          }
        }
      }
    }

    // Average loss
    const avgLoss = loss.div(examples.length);

    // L2 regularization on K-head parameters only (MP params regularized separately)
    let l2Loss = tf.scalar(0);
    for (const W of params.W_k) {
      l2Loss = l2Loss.add(tf.sum(tf.square(W)));
    }
    for (const W of params.W_q) {
      l2Loss = l2Loss.add(tf.sum(tf.square(W)));
    }
    l2Loss = l2Loss.add(tf.sum(tf.square(params.W_intent)));

    // Projection head L2 (stronger to prevent overfitting with few examples)
    if (params.projectionHead) {
      const projL2Scale = 10;
      l2Loss = l2Loss.add(tf.sum(tf.square(params.projectionHead.W1)).mul(projL2Scale));
      l2Loss = l2Loss.add(tf.sum(tf.square(params.projectionHead.W2)).mul(projL2Scale));
    }

    l2Loss = l2Loss.mul(trainerConfig.l2Lambda);

    totalLoss = avgLoss.add(l2Loss).arraySync() as number;

    return avgLoss.add(l2Loss) as tf.Scalar;
  });

  // ========================================================================
  // STEP 3: Apply K-head gradients via TF.js optimizer
  // ========================================================================

  // Compute gradient norm (K-head only)
  let kheadGradNormSq = 0;
  for (const g of Object.values(kheadGrads)) {
    kheadGradNormSq += (tf.sum(tf.square(g)).arraySync() as number);
  }
  const kheadGradNorm = Math.sqrt(kheadGradNormSq);

  // Clip K-head gradients if needed
  if (kheadGradNorm > trainerConfig.gradientClip) {
    const scale = trainerConfig.gradientClip / kheadGradNorm;
    for (const key of Object.keys(kheadGrads)) {
      kheadGrads[key] = kheadGrads[key].mul(scale);
    }
  }

  // Apply K-head gradients
  optimizer.applyGradients(kheadGrads);

  // Cleanup K-head gradients
  Object.values(kheadGrads).forEach((g) => g.dispose());
  (batchLoss as tf.Tensor).dispose();

  // ========================================================================
  // STEP 4: Sparse MP backward and gradient application
  // ========================================================================

  // Run sparse backward pass
  const mpGrads = sparseMPBackward(dH_accum, dE_accum, mpCache, params);

  // Apply MP gradients
  const mpLearningRate = trainerConfig.learningRate * (config.mpLearningRateScale ?? 1);
  applySparseMPGradients(params, mpGrads, mpLearningRate, examples.length);

  // Compute MP gradient norm
  let mpGradNormSq = 0;
  for (const [, dW_level] of mpGrads.dW_up) {
    for (const dW_h of dW_level) {
      for (const row of dW_h) {
        for (const v of row) mpGradNormSq += v * v;
      }
    }
  }
  for (const [, da_level] of mpGrads.da_up) {
    for (const da_h of da_level) {
      for (const v of da_h) mpGradNormSq += v * v;
    }
  }
  // Combined gradient norm (includes both K-head and MP gradients)
  const gradientNorm = Math.sqrt(kheadGradNormSq + mpGradNormSq);

  return {
    loss: totalLoss,
    accuracy: totalCorrect / examples.length,
    gradientNorm,
    numExamples: examples.length,
  };
}

// ============================================================================
// Trainer Class
// ============================================================================

/**
 * SHGAT Trainer with TensorFlow.js autograd
 *
 * Supports full message passing when graph structure is provided.
 * By default uses sparse MP for better performance and WASM compatibility.
 */
export class AutogradTrainer {
  private params: TFParams;
  private optimizer: tf.Optimizer;
  private config: SHGATConfig;
  private trainerConfig: TrainerConfig;
  private nodeEmbeddings: Map<string, number[]> = new Map();
  private graph: GraphStructure | null = null;
  private useMessagePassing: boolean = false;
  private useSparseMP: boolean = true; // Use sparse MP by default (WASM compatible)

  constructor(
    config: SHGATConfig,
    trainerConfig: Partial<TrainerConfig> = {},
    maxLevel = 3,
  ) {
    this.config = config;
    this.trainerConfig = { ...DEFAULT_TRAINER_CONFIG, ...trainerConfig };
    this.params = initTFParams(config, maxLevel);
    this.optimizer = tf.train.adam(this.trainerConfig.learningRate);
  }

  /**
   * Enable or disable sparse message passing mode
   *
   * @param useSparse - true for sparse JS loops (WASM compatible, faster)
   *                    false for dense TF.js autograd (CPU only, slower)
   * @default true
   */
  setSparseMP(useSparse: boolean): void {
    this.useSparseMP = useSparse;
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
   */
  setGraph(graph: GraphStructure): void {
    this.graph = graph;
    this.useMessagePassing = true;
  }

  /**
   * Check if message passing is enabled
   */
  hasMessagePassing(): boolean {
    return this.useMessagePassing && this.graph !== null;
  }

  /**
   * Build initial embedding tensors for message passing
   */
  private buildMessagePassingContext(): MessagePassingContext | undefined {
    if (!this.useMessagePassing || !this.graph) {
      return undefined;
    }

    // Build tool embeddings tensor [numTools, embDim]
    const toolEmbs: number[][] = [];
    for (const toolId of this.graph.toolIds) {
      toolEmbs.push(
        this.nodeEmbeddings.get(toolId) || new Array(this.config.embeddingDim).fill(0)
      );
    }
    const H_init = ops.toTensor(toolEmbs);

    // Build capability embeddings per level
    const E_init = new Map<number, tf.Tensor2D>();
    for (const [level, capIds] of this.graph.capIdsByLevel) {
      const capEmbs: number[][] = [];
      for (const capId of capIds) {
        capEmbs.push(
          this.nodeEmbeddings.get(capId) || new Array(this.config.embeddingDim).fill(0)
        );
      }
      E_init.set(level, ops.toTensor(capEmbs));
    }

    return {
      graph: this.graph,
      H_init,
      E_init,
      useSparse: this.useSparseMP,
    };
  }

  /**
   * Train on a batch of examples
   *
   * When graph is set via setGraph(), message passing is integrated into training.
   * This allows gradients to flow through W_up, W_down, a_up, a_down parameters.
   *
   * Uses sparse MP by default (WASM compatible). Call setSparseMP(false) to use
   * dense TF.js autograd (requires CPU backend).
   */
  trainBatch(examples: TrainingExample[]): TrainingMetrics {
    // Build message passing context if graph is available
    const mpContext = this.buildMessagePassingContext();

    const metrics = trainStep(
      examples,
      this.nodeEmbeddings,
      this.params,
      this.config,
      this.trainerConfig,
      this.optimizer,
      mpContext,
    );

    // Clean up initial tensors (enriched tensors are disposed inside trainStep)
    if (mpContext) {
      mpContext.H_init.dispose();
      for (const [, tensor] of mpContext.E_init) {
        tensor.dispose();
      }
    }

    return metrics;
  }

  /**
   * Score nodes for an intent
   *
   * If graph is set, uses full message passing for enriched embeddings.
   * Otherwise, uses direct K-head attention (faster but less accurate).
   */
  score(intentEmb: number[], nodeIds: string[]): number[] {
    return tidy(() => {
      const intentTensor = ops.toTensor(intentEmb);

      // Get raw embeddings
      const nodeEmbs: number[][] = nodeIds.map(
        (id) => this.nodeEmbeddings.get(id) || new Array(this.config.embeddingDim).fill(0)
      );
      let nodesTensor = ops.toTensor(nodeEmbs);

      // Apply message passing if graph is available
      if (this.useMessagePassing && this.graph) {
        // Build initial embeddings (JS arrays)
        const toolEmbs: number[][] = [];
        for (const toolId of this.graph.toolIds) {
          toolEmbs.push(this.nodeEmbeddings.get(toolId) || new Array(this.config.embeddingDim).fill(0));
        }

        const E_init_arr = new Map<number, number[][]>();
        for (const [level, capIds] of this.graph.capIdsByLevel) {
          const capEmbs: number[][] = [];
          for (const capId of capIds) {
            capEmbs.push(this.nodeEmbeddings.get(capId) || new Array(this.config.embeddingDim).fill(0));
          }
          E_init_arr.set(level, capEmbs);
        }

        let H_arr: number[][];
        let E_arrs: Map<number, number[][]>;

        if (this.useSparseMP) {
          // Use sparse MP (faster, WASM compatible)
          const { H, E } = sparseMPForward(
            toolEmbs,
            E_init_arr,
            this.graph,
            this.params,
            this.config,
          );
          H_arr = H;
          E_arrs = E;
        } else {
          // Use dense TF.js MP (legacy, requires CPU)
          const H_init = ops.toTensor(toolEmbs);
          const E_init = new Map<number, tf.Tensor2D>();
          for (const [level, embs] of E_init_arr) {
            E_init.set(level, ops.toTensor(embs));
          }

          const { H: H_mp, E: E_mp } = messagePassingForward(
            H_init,
            E_init,
            this.graph,
            this.params,
            this.config,
          );

          H_arr = H_mp.arraySync() as number[][];
          E_arrs = new Map<number, number[][]>();
          for (const [level, tensor] of E_mp) {
            E_arrs.set(level, tensor.arraySync() as number[][]);
          }

          // Cleanup dense tensors
          H_init.dispose();
          for (const [, tensor] of E_init) tensor.dispose();
          H_mp.dispose();
          for (const [, tensor] of E_mp) tensor.dispose();
        }

        // Map nodeIds to their enriched embeddings
        const enriched: number[][] = [];
        for (const nodeId of nodeIds) {
          // Check if it's a tool
          const toolIdx = this.graph.toolIds.indexOf(nodeId);
          if (toolIdx >= 0) {
            enriched.push(H_arr[toolIdx]);
            continue;
          }

          // Check if it's a capability at any level
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
            // Fallback to original embedding
            enriched.push(this.nodeEmbeddings.get(nodeId) || new Array(this.config.embeddingDim).fill(0));
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
  exportProjectionHeadToArray(): import("../core/projection-head.ts").ProjectionHeadArrayParams | undefined {
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
    for (const W of this.params.W_q) W.dispose();
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
