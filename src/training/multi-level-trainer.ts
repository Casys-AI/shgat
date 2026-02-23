/**
 * SHGAT Multi-Level Training Types & Gradient Initialization
 *
 * Provides types and gradient accumulator utilities used by:
 * - multi-level-trainer-khead.ts (K-head scoring forward/backward)
 * - multi-level-orchestrator.ts (MP forward/backward with per-phase caches)
 * - train-ob.ts (top-level training script)
 *
 * The actual backward pass lives in MultiLevelOrchestrator.backwardMultiLevel()
 * which handles both dH (tools, via E→V backward) and dE (capabilities).
 *
 * @module graphrag/algorithms/shgat/training/multi-level-trainer
 */

import type { LevelParams, MultiLevelForwardCache } from "../core/types.ts";
import * as math from "../utils/math.ts";
const { zerosLike3D } = math;

// ============================================================================
// Types
// ============================================================================

/**
 * Gradient accumulators for multi-level parameters
 *
 * Stores gradients for each hierarchy level's learnable parameters.
 */
export interface MultiLevelGradientAccumulators {
  /** Gradients per level: level → LevelGradients */
  levelGradients: Map<number, LevelGradients>;
}

/**
 * Gradients for a single hierarchy level
 */
export interface LevelGradients {
  /** Gradient for W_child: [numHeads][headDim][inputDim] */
  dW_child: number[][][];
  /** Gradient for W_parent: [numHeads][headDim][inputDim] */
  dW_parent: number[][][];
  /** Gradient for a_upward: [numHeads][2*headDim] */
  da_upward: number[][];
  /** Gradient for a_downward: [numHeads][2*headDim] */
  da_downward: number[][];
}

/**
 * Intermediate activations for gradient computation (per level)
 */
export interface LevelIntermediates {
  /** Child projections per head: [head][numChildren][headDim] — Float32 for RAM */
  childProj: Float32Array[][];
  /** Parent projections per head: [head][numParents][headDim] — Float32 for RAM */
  parentProj: Float32Array[][];
  /** Pre-softmax attention scores: [head][numChildren][numParents] */
  scores: number[][][];
  /** Post-softmax attention weights: [head][numChildren][numParents] */
  attention: number[][][];
}

/**
 * Extended forward cache for multi-level backpropagation
 */
export interface ExtendedMultiLevelForwardCache extends MultiLevelForwardCache {
  /** Intermediate activations for upward pass: level → LevelIntermediates */
  intermediateUpwardActivations: Map<number, LevelIntermediates>;
  /** Intermediate activations for downward pass: level → LevelIntermediates */
  intermediateDownwardActivations: Map<number, LevelIntermediates>;
}

// ============================================================================
// Gradient Initialization
// ============================================================================

/**
 * Initialize gradient accumulators for multi-level training
 *
 * @param levelParams Map of level → LevelParams
 * @returns Initialized gradient accumulators (all zeros)
 */
export function initMultiLevelGradients(
  levelParams: Map<number, LevelParams>,
): MultiLevelGradientAccumulators {
  const grads = new Map<number, LevelGradients>();

  for (const [level, params] of levelParams) {
    grads.set(level, {
      dW_child: zerosLike3D(params.W_child),
      dW_parent: zerosLike3D(params.W_parent),
      da_upward: params.a_upward.map((row) => row.map(() => 0)),
      da_downward: params.a_downward.map((row) => row.map(() => 0)),
    });
  }

  return { levelGradients: grads };
}

/**
 * Reset gradient accumulators to zero
 */
export function resetMultiLevelGradients(
  accum: MultiLevelGradientAccumulators,
  levelParams: Map<number, LevelParams>,
): void {
  for (const [level] of levelParams) {
    const grads = accum.levelGradients.get(level);
    if (grads) {
      // Zero-fill in place instead of reallocating (avoids GC pressure per batch)
      for (const head of grads.dW_child) for (const row of head) row.fill(0);
      for (const head of grads.dW_parent) for (const row of head) row.fill(0);
      for (const row of grads.da_upward) row.fill(0);
      for (const row of grads.da_downward) row.fill(0);
    }
  }
}
