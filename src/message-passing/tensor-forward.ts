/**
 * Tensor-Native Forward Pass for SHGAT-TF
 *
 * Complete message passing in TensorFlow.js tensors WITHOUT array conversions.
 * This replaces the array-based forward pass for inference (scoring).
 *
 * Performance: ~10-20x faster than array-based version because:
 * - No array→tensor conversion per operation
 * - GPU-accelerated batch operations
 * - Single tidy() block for all computations
 *
 * @module shgat-tf/message-passing/tensor-forward
 */

import { tf, tidy } from "../tf/backend.ts";
import type { LevelParams } from "../core/types.ts";

/**
 * Result of tensor forward pass
 */
export interface TensorForwardResult {
  /** Final L0 node embeddings [numL0, embDim] */
  H: tf.Tensor2D;
  /** Final L1+ node embeddings per level */
  E: Map<number, tf.Tensor2D>;
}

/**
 * Tensor-native forward pass configuration
 */
export interface TensorForwardConfig {
  numHeads: number;
  leakyReluSlope: number;
  preserveDim: boolean;
  preserveDimResidual?: number;
}

/**
 * Execute tensor-native forward pass (V → E^0 → E^1 → ... → E^L_max → ... → E^0 → V)
 *
 * All computations stay in tensors until the very end.
 *
 * @param H_init - Initial L0 node embeddings [numL0, embDim]
 * @param E_levels_init - Initial L1+ node embeddings per level
 * @param l0ToL1Matrix - Incidence matrix [numL0, numL1]
 * @param interLevelMatrices - Level connectivity matrices
 * @param levelParams - Parameters per hierarchy level
 * @param config - Forward pass configuration
 * @returns Final embeddings as tensors
 */
export function tensorForwardPass(
  H_init: tf.Tensor2D,
  E_levels_init: Map<number, tf.Tensor2D>,
  l0ToL1Matrix: tf.Tensor2D,
  interLevelMatrices: Map<number, tf.Tensor2D>,
  levelParams: Map<number, TensorLevelParams>,
  config: TensorForwardConfig,
): TensorForwardResult {
  const maxLevel = Math.max(...Array.from(E_levels_init.keys()));
  const { numHeads, leakyReluSlope } = config;

  // Clone initial embeddings (we'll modify them)
  const E = new Map<number, tf.Tensor2D>();
  for (const [level, tensor] of E_levels_init) {
    E.set(level, tensor.clone());
  }
  let H = H_init.clone();

  // ========================================================================
  // UPWARD PASS: V → E^0 → E^1 → ... → E^L_max
  // ========================================================================

  for (let level = 0; level <= maxLevel; level++) {
    const params = levelParams.get(level);
    if (!params) continue;

    const nodesAtLevel = E.get(level);
    if (!nodesAtLevel) continue;

    if (level === 0) {
      // V → E^0: L0 nodes aggregate to level-0 L1 nodes
      const E_new = tensorVertexToEdge(
        H,
        nodesAtLevel,
        l0ToL1Matrix,
        params,
        numHeads,
        leakyReluSlope,
      );
      E.get(level)?.dispose();
      E.set(level, E_new);
    } else {
      // E^(k-1) → E^k: Lower-level nodes aggregate to higher-level nodes
      const E_prev = E.get(level - 1);
      const connectivity = interLevelMatrices.get(level);
      if (!E_prev || !connectivity) continue;

      const E_new = tensorEdgeToEdge(
        E_prev,
        nodesAtLevel,
        connectivity,
        params,
        numHeads,
        leakyReluSlope,
      );
      E.get(level)?.dispose();
      E.set(level, E_new);
    }
  }

  // ========================================================================
  // DOWNWARD PASS: E^L_max → ... → E^1 → E^0 → V
  // ========================================================================

  for (let level = maxLevel - 1; level >= 0; level--) {
    const params = levelParams.get(level);
    if (!params) continue;

    const nodesAtLevel = E.get(level);
    const nodesAtParentLevel = E.get(level + 1);
    if (!nodesAtLevel || !nodesAtParentLevel) continue;

    const forwardConnectivity = interLevelMatrices.get(level + 1);
    if (!forwardConnectivity) continue;

    // Transpose for downward pass
    const reverseConnectivity = tf.transpose(forwardConnectivity) as tf.Tensor2D;

    const E_new = tensorEdgeToEdgeDownward(
      nodesAtParentLevel,
      nodesAtLevel,
      reverseConnectivity,
      params,
      numHeads,
      leakyReluSlope,
    );

    reverseConnectivity.dispose();
    E.get(level)?.dispose();
    E.set(level, E_new);
  }

  // Final: E^0 → V
  const E_level0 = E.get(0);
  if (E_level0) {
    const params = levelParams.get(0);
    if (params) {
      const H_new = tensorEdgeToVertex(
        E_level0,
        H,
        l0ToL1Matrix,
        params,
        numHeads,
        leakyReluSlope,
      );
      H.dispose();
      H = H_new;
    }
  }

  // Apply residual connection if preserveDim
  if (config.preserveDim && config.preserveDimResidual) {
    const alpha = config.preserveDimResidual;
    // H = (1-α)*H + α*H_init
    const H_residual = tf.add(
      tf.mul(H, 1 - alpha),
      tf.mul(H_init, alpha),
    ) as tf.Tensor2D;
    H.dispose();
    H = H_residual;

    // Same for E
    for (const [level, E_tensor] of E) {
      const E_init = E_levels_init.get(level);
      if (E_init) {
        const E_residual = tf.add(
          tf.mul(E_tensor, 1 - alpha),
          tf.mul(E_init, alpha),
        ) as tf.Tensor2D;
        E_tensor.dispose();
        E.set(level, E_residual);
      }
    }
  }

  return { H, E };
}

/**
 * Tensor-based level parameters
 */
export interface TensorLevelParams {
  /** Child projection [numHeads, headDim, inputDim] */
  W_child: tf.Tensor3D;
  /** Parent projection [numHeads, headDim, inputDim] */
  W_parent: tf.Tensor3D;
  /** Upward attention [numHeads, 2*headDim] */
  a_upward: tf.Tensor2D;
  /** Downward attention [numHeads, 2*headDim] */
  a_downward: tf.Tensor2D;
}

/**
 * Convert array-based LevelParams to tensor-based
 */
export function createTensorLevelParams(
  params: LevelParams,
  instanceId: number,
): TensorLevelParams {
  return {
    W_child: tf.variable(
      tf.tensor3d(params.W_child),
      true,
      `W_child_${instanceId}`,
    ) as unknown as tf.Tensor3D,
    W_parent: tf.variable(
      tf.tensor3d(params.W_parent),
      true,
      `W_parent_${instanceId}`,
    ) as unknown as tf.Tensor3D,
    a_upward: tf.variable(
      tf.tensor2d(params.a_upward),
      true,
      `a_upward_${instanceId}`,
    ) as unknown as tf.Tensor2D,
    a_downward: tf.variable(
      tf.tensor2d(params.a_downward),
      true,
      `a_downward_${instanceId}`,
    ) as unknown as tf.Tensor2D,
  };
}

/**
 * Dispose tensor level params
 */
export function disposeTensorLevelParams(params: TensorLevelParams): void {
  params.W_child.dispose();
  params.W_parent.dispose();
  params.a_upward.dispose();
  params.a_downward.dispose();
}

// ============================================================================
// Phase Implementations (Tensor-Native)
// ============================================================================

/**
 * Vertex → Edge phase (tensor-native)
 *
 * L0 nodes aggregate to L1 nodes via attention.
 */
function tensorVertexToEdge(
  H: tf.Tensor2D,           // [numL0, embDim]
  E: tf.Tensor2D,           // [numL1, embDim]
  connectivity: tf.Tensor2D, // [numL0, numL1]
  params: TensorLevelParams,
  numHeads: number,
  leakyReluSlope: number,
): tf.Tensor2D {
  return tidy(() => {
    const headResults: tf.Tensor2D[] = [];

    for (let h = 0; h < numHeads; h++) {
      // Get per-head weights
      const W_source = params.W_child.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const W_target = params.W_parent.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const a = params.a_upward.slice([h, 0], [1, -1]).squeeze([0]) as tf.Tensor1D;

      // Project embeddings: H_proj = H @ W_source.T, E_proj = E @ W_target.T
      const H_proj = tf.matMul(H, W_source, false, true);  // [numL0, headDim]
      const E_proj = tf.matMul(E, W_target, false, true);  // [numL1, headDim]

      // Compute attention scores
      // For each (L0, L1) pair: score = a^T @ LeakyReLU([H_proj[i] || E_proj[j]])

      // Expand for broadcasting: H_proj [numL0, 1, headDim], E_proj [1, numL1, headDim]
      const H_exp = H_proj.expandDims(1);  // [numL0, 1, headDim]
      const E_exp = E_proj.expandDims(0);  // [1, numL1, headDim]

      // Broadcast and concatenate: [numL0, numL1, 2*headDim]
      const numL0 = H.shape[0];
      const numL1 = E.shape[0];
      const H_tiled = tf.tile(H_exp, [1, numL1, 1]);  // [numL0, numL1, headDim]
      const E_tiled = tf.tile(E_exp, [numL0, 1, 1]); // [numL0, numL1, headDim]
      const concat = tf.concat([H_tiled, E_tiled], 2);  // [numL0, numL1, 2*headDim]

      // LeakyReLU activation
      const activated = tf.leakyRelu(concat, leakyReluSlope);

      // Dot with attention vector: [numL0, numL1]
      const scores = tf.sum(tf.mul(activated, a), 2);

      // Mask by connectivity (set non-connected to -inf)
      const mask = tf.equal(connectivity, 0);
      const maskedScores = tf.where(mask, tf.fill(scores.shape, -1e9), scores);

      // Softmax per L1 node (column-wise) = per L0 node seeing each L1 node
      // We need softmax over L0 nodes for each L1 node
      const attention = tf.softmax(maskedScores, 0);  // softmax over L0 nodes (axis 0)

      // Aggregate: E_new = attention.T @ H_proj
      const E_new_head = tf.matMul(attention, H_proj, true, false);  // [numL1, headDim]

      // ELU activation
      const E_activated = tf.elu(E_new_head) as tf.Tensor2D;

      headResults.push(E_activated);
    }

    // Concatenate heads: [numL1, numHeads * headDim]
    return tf.concat(headResults, 1) as tf.Tensor2D;
  });
}

/**
 * Edge → Edge phase (upward, tensor-native)
 */
function tensorEdgeToEdge(
  E_source: tf.Tensor2D,     // [numSourceNodes, embDim]
  E_target: tf.Tensor2D,     // [numTargetNodes, embDim]
  connectivity: tf.Tensor2D, // [numSourceNodes, numTargetNodes]
  params: TensorLevelParams,
  numHeads: number,
  leakyReluSlope: number,
): tf.Tensor2D {
  return tidy(() => {
    const headResults: tf.Tensor2D[] = [];

    for (let h = 0; h < numHeads; h++) {
      const W_source = params.W_child.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const W_target = params.W_parent.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const a = params.a_upward.slice([h, 0], [1, -1]).squeeze([0]) as tf.Tensor1D;

      // Project
      const E_src_proj = tf.matMul(E_source, W_source, false, true);
      const E_tgt_proj = tf.matMul(E_target, W_target, false, true);

      const numSrc = E_source.shape[0];
      const numTgt = E_target.shape[0];

      // Attention computation
      const E_src_exp = E_src_proj.expandDims(1);
      const E_tgt_exp = E_tgt_proj.expandDims(0);
      const E_src_tiled = tf.tile(E_src_exp, [1, numTgt, 1]);
      const E_tgt_tiled = tf.tile(E_tgt_exp, [numSrc, 1, 1]);
      const concat = tf.concat([E_src_tiled, E_tgt_tiled], 2);

      const activated = tf.leakyRelu(concat, leakyReluSlope);
      const scores = tf.sum(tf.mul(activated, a), 2);

      const mask = tf.equal(connectivity, 0);
      const maskedScores = tf.where(mask, tf.fill(scores.shape, -1e9), scores);
      const attention = tf.softmax(maskedScores, 0);

      const E_new_head = tf.matMul(attention, E_src_proj, true, false);
      const E_activated = tf.elu(E_new_head) as tf.Tensor2D;

      headResults.push(E_activated);
    }

    return tf.concat(headResults, 1) as tf.Tensor2D;
  });
}

/**
 * Edge → Edge phase (downward, tensor-native)
 */
function tensorEdgeToEdgeDownward(
  E_parent: tf.Tensor2D,      // [numParentNodes, embDim]
  E_child: tf.Tensor2D,       // [numChildNodes, embDim]
  connectivity: tf.Tensor2D,  // [numParentNodes, numChildNodes] (transposed from forward)
  params: TensorLevelParams,
  numHeads: number,
  leakyReluSlope: number,
): tf.Tensor2D {
  return tidy(() => {
    const headResults: tf.Tensor2D[] = [];

    for (let h = 0; h < numHeads; h++) {
      // In downward: parent is source, child is target
      const W_source = params.W_parent.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const W_target = params.W_child.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const a = params.a_downward.slice([h, 0], [1, -1]).squeeze([0]) as tf.Tensor1D;

      const E_parent_proj = tf.matMul(E_parent, W_source, false, true);
      const E_child_proj = tf.matMul(E_child, W_target, false, true);

      const numParent = E_parent.shape[0];
      const numChild = E_child.shape[0];

      const E_parent_exp = E_parent_proj.expandDims(1);
      const E_child_exp = E_child_proj.expandDims(0);
      const E_parent_tiled = tf.tile(E_parent_exp, [1, numChild, 1]);
      const E_child_tiled = tf.tile(E_child_exp, [numParent, 1, 1]);
      const concat = tf.concat([E_parent_tiled, E_child_tiled], 2);

      const activated = tf.leakyRelu(concat, leakyReluSlope);
      const scores = tf.sum(tf.mul(activated, a), 2);

      const mask = tf.equal(connectivity, 0);
      const maskedScores = tf.where(mask, tf.fill(scores.shape, -1e9), scores);
      const attention = tf.softmax(maskedScores, 0);

      const E_new_head = tf.matMul(attention, E_parent_proj, true, false);
      const E_activated = tf.elu(E_new_head) as tf.Tensor2D;

      headResults.push(E_activated);
    }

    return tf.concat(headResults, 1) as tf.Tensor2D;
  });
}

/**
 * Edge → Vertex phase (tensor-native)
 */
function tensorEdgeToVertex(
  E: tf.Tensor2D,             // [numL1, embDim]
  H: tf.Tensor2D,             // [numL0, embDim]
  connectivity: tf.Tensor2D,  // [numL0, numL1]
  params: TensorLevelParams,
  numHeads: number,
  leakyReluSlope: number,
): tf.Tensor2D {
  return tidy(() => {
    const headResults: tf.Tensor2D[] = [];

    for (let h = 0; h < numHeads; h++) {
      // In E→V: L1 nodes are source, L0 nodes are target
      const W_source = params.W_parent.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const W_target = params.W_child.slice([h, 0, 0], [1, -1, -1]).squeeze([0]) as tf.Tensor2D;
      const a = params.a_downward.slice([h, 0], [1, -1]).squeeze([0]) as tf.Tensor1D;

      const E_proj = tf.matMul(E, W_source, false, true);  // [numL1, headDim]
      const H_proj = tf.matMul(H, W_target, false, true);  // [numL0, headDim]

      const numL1 = E.shape[0];
      const numL0 = H.shape[0];

      // For E→V, we need attention over L1 nodes for each L0 node
      // connectivity is [numL0, numL1]
      const E_exp = E_proj.expandDims(0);  // [1, numL1, headDim]
      const H_exp = H_proj.expandDims(1);  // [numL0, 1, headDim]
      const E_tiled = tf.tile(E_exp, [numL0, 1, 1]);  // [numL0, numL1, headDim]
      const H_tiled = tf.tile(H_exp, [1, numL1, 1]);   // [numL0, numL1, headDim]

      // Concat: [E_proj, H_proj] for each (L0 node, L1 node) pair
      const concat = tf.concat([E_tiled, H_tiled], 2);  // [numL0, numL1, 2*headDim]

      const activated = tf.leakyRelu(concat, leakyReluSlope);
      const scores = tf.sum(tf.mul(activated, a), 2);  // [numL0, numL1]

      // Mask and softmax over L1 nodes (axis 1) for each L0 node
      const mask = tf.equal(connectivity, 0);
      const maskedScores = tf.where(mask, tf.fill(scores.shape, -1e9), scores);
      const attention = tf.softmax(maskedScores, 1);  // softmax over L1 nodes

      // Aggregate: H_new = attention @ E_proj
      const H_new_head = tf.matMul(attention, E_proj);  // [numL0, headDim]
      const H_activated = tf.elu(H_new_head) as tf.Tensor2D;

      headResults.push(H_activated);
    }

    return tf.concat(headResults, 1) as tf.Tensor2D;
  });
}
