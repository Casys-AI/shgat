/**
 * Learned Projection Head for SHGAT-TF
 *
 * Post-MP non-linear projection that maps enriched embeddings (1024D) to a
 * compact contrastive space (256D) where semantically similar but functionally
 * distinct tools are separated (SimCLR/CLIP pattern).
 *
 * Architecture:
 *   enrichedEmb [N, 1024]
 *     → Linear(1024, hiddenDim) + ReLU
 *     → Linear(hiddenDim, outputDim) + L2 normalize
 *   = projected [N, outputDim]
 *
 * Scoring:
 *   score_proj = dot(proj_intent, proj_nodes) / temperature
 *   final = (1-alpha) * khead_score + alpha * proj_score
 *
 * @module shgat-tf/core/projection-head
 */

import { tf } from "../tf/backend.ts";
import * as ops from "../tf/ops.ts";

// ============================================================================
// Types
// ============================================================================

/**
 * Projection head TF.js variables (trainable)
 */
export interface ProjectionHeadTFParams {
  W1: tf.Variable;   // [inputDim, hiddenDim]
  b1: tf.Variable;   // [hiddenDim]
  W2: tf.Variable;   // [hiddenDim, outputDim]
  b2: tf.Variable;   // [outputDim]
}

/**
 * Projection head array params (for serialization)
 */
export interface ProjectionHeadArrayParams {
  W1: number[][];    // [inputDim, hiddenDim]
  b1: number[];      // [hiddenDim]
  W2: number[][];    // [hiddenDim, outputDim]
  b2: number[];      // [outputDim]
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Initialize projection head parameters.
 * Uses Glorot for W1, autoencoder-like init for W2 (W2 ≈ W1.T scaled).
 */
export function initProjectionHeadParams(
  inputDim: number,
  hiddenDim: number,
  outputDim: number,
): ProjectionHeadTFParams {
  const W1 = tf.variable(
    ops.glorotNormal([inputDim, hiddenDim]),
    true,
    "proj_W1"
  );
  const b1 = tf.variable(
    tf.zeros([hiddenDim]),
    true,
    "proj_b1"
  );
  const W2 = tf.variable(
    ops.glorotNormal([hiddenDim, outputDim]),
    true,
    "proj_W2"
  );
  const b2 = tf.variable(
    tf.zeros([outputDim]),
    true,
    "proj_b2"
  );

  return { W1, b1, W2, b2 };
}

// ============================================================================
// Forward Pass
// ============================================================================

/**
 * Project embeddings through the projection head (differentiable).
 *
 * Must be called inside tf.tidy() or tf.variableGrads() scope.
 *
 * @param embeddings - [numNodes, inputDim] enriched embeddings
 * @param params - Projection head variables
 * @returns [numNodes, outputDim] L2-normalized projected embeddings
 */
export function projectionForward(
  embeddings: tf.Tensor2D,
  params: ProjectionHeadTFParams,
): tf.Tensor2D {
  // h = relu(emb @ W1 + b1)
  const h1 = tf.add(
    tf.matMul(embeddings, params.W1 as tf.Tensor2D),
    params.b1
  );
  const h1_relu = tf.relu(h1);

  // z = emb @ W2 + b2
  const z = tf.add(
    tf.matMul(h1_relu, params.W2 as tf.Tensor2D),
    params.b2
  );

  // L2 normalize along last axis
  const norm = tf.norm(z, 2, -1, true); // [N, 1]
  const z_normalized = tf.div(z, tf.add(norm, 1e-8)) as tf.Tensor2D;

  return z_normalized;
}

/**
 * Compute projection-based scores for nodes given an intent.
 *
 * Must be called inside tf.tidy() or tf.variableGrads() scope.
 *
 * @param intentEmbedding - [1, inputDim] intent embedding
 * @param nodeEmbeddings - [numNodes, inputDim] enriched node embeddings
 * @param params - Projection head variables
 * @param temperature - Softmax temperature for scaling
 * @returns [numNodes] scores (higher = more relevant)
 */
export function projectionScore(
  intentEmbedding: tf.Tensor2D,
  nodeEmbeddings: tf.Tensor2D,
  params: ProjectionHeadTFParams,
  temperature: number,
): tf.Tensor1D {
  // Project both intent and nodes
  const z_intent = projectionForward(intentEmbedding, params); // [1, outputDim]
  const z_nodes = projectionForward(nodeEmbeddings, params);   // [N, outputDim]

  // Dot product: z_intent @ z_nodes.T / temperature → [1, N]
  const scores2d = tf.div(
    tf.matMul(z_intent, z_nodes, false, true),
    temperature
  );

  // Squeeze to [N]
  return scores2d.squeeze([0]) as tf.Tensor1D;
}

// ============================================================================
// Serialization
// ============================================================================

/**
 * Export projection head params to arrays for serialization.
 */
export function exportProjectionHeadParams(
  params: ProjectionHeadTFParams,
): ProjectionHeadArrayParams {
  return {
    W1: params.W1.arraySync() as number[][],
    b1: params.b1.arraySync() as number[],
    W2: params.W2.arraySync() as number[][],
    b2: params.b2.arraySync() as number[],
  };
}

/**
 * Import projection head params from arrays.
 */
export function importProjectionHeadParams(
  data: ProjectionHeadArrayParams,
): ProjectionHeadTFParams {
  return {
    W1: tf.variable(tf.tensor2d(data.W1), true, "proj_W1"),
    b1: tf.variable(tf.tensor1d(data.b1), true, "proj_b1"),
    W2: tf.variable(tf.tensor2d(data.W2), true, "proj_W2"),
    b2: tf.variable(tf.tensor1d(data.b2), true, "proj_b2"),
  };
}

/**
 * Get all trainable variables from projection head (for variableGrads).
 */
export function getProjectionTrainableVars(
  params: ProjectionHeadTFParams,
): tf.Variable[] {
  return [params.W1, params.b1, params.W2, params.b2];
}

/**
 * Dispose all projection head tensors.
 */
export function disposeProjectionHead(params: ProjectionHeadTFParams): void {
  params.W1.dispose();
  params.b1.dispose();
  params.W2.dispose();
  params.b2.dispose();
}
