/**
 * Hyperedge → Hyperedge Message Passing Phase (Multi-Level)
 *
 * Phase for multi-level n-SuperHyperGraph: Level-k nodes send
 * messages to level-(k+1) nodes that contain them.
 *
 * Delegates to the shared phaseForward/phaseBackward implementation.
 * E→E mapping: source=E_k(child), target=E_{k+1}(parent),
 *              neighborMap=conn.targetToSources.
 *
 * @module graphrag/algorithms/shgat/message-passing/edge-to-edge-phase
 */

import type {
  MessagePassingPhase,
  PhaseForwardCache,
  PhaseGradients,
  PhaseParameters,
  PhaseResult,
  PhaseResultWithCache,
  SparseConnectivity,
} from "./phase-interface.ts";
import { phaseBackward, phaseForward } from "./phase-shared.ts";

// Backward-compatible type aliases
export type EEForwardCache = PhaseForwardCache;
export type EEPhaseResultWithCache = PhaseResultWithCache;

export interface EEGradients {
  dW_source: number[][];
  dW_target: number[][];
  da_attention: number[];
  /** Gradient for child node embeddings, level k (= source) */
  dE_k: number[][];
  /** Gradient for parent node embeddings, level k+1 (= target) */
  dE_kPlus1: number[][];
}

/**
 * Hyperedge → Hyperedge message passing implementation
 *
 * Used for hierarchical node levels where higher-level nodes contain
 * lower-level nodes (n-SuperHyperGraph structure).
 *
 * Containment semantics: source=child, target=parent.
 * conn.targetToSources.get(p) gives all children in parent p.
 */
export class EdgeToEdgePhase implements MessagePassingPhase {
  private readonly levelK: number;
  private readonly levelKPlus1: number;

  constructor(levelK: number, levelKPlus1: number) {
    this.levelK = levelK;
    this.levelKPlus1 = levelKPlus1;
  }

  getName(): string {
    return `Edge^${this.levelK}→Edge^${this.levelKPlus1}`;
  }

  forward(
    E_k: number[][],
    E_kPlus1: number[][],
    connectivity: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): PhaseResult {
    const result = this.forwardWithCache(E_k, E_kPlus1, connectivity, params, config);
    return { embeddings: result.embeddings, attention: result.attention };
  }

  /**
   * Forward pass with cache for backward
   */
  forwardWithCache(
    E_k: number[][],
    E_kPlus1: number[][],
    conn: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): EEPhaseResultWithCache {
    // E→E: target=parent(E_kPlus1), sources=children(E_k) → iterate conn.targetToSources
    return phaseForward(E_k, E_kPlus1, conn.targetToSources, E_kPlus1.length, params, config);
  }

  /**
   * Backward pass: compute gradients for W_source, W_target, a_attention
   */
  backward(
    dE_kPlus1_new: number[][],
    cache: EEForwardCache,
    params: PhaseParameters,
  ): EEGradients {
    const grads: PhaseGradients = phaseBackward(dE_kPlus1_new, cache, params);
    return {
      dW_source: grads.dW_source,
      dW_target: grads.dW_target,
      da_attention: grads.da_attention,
      dE_k: grads.dSource,
      dE_kPlus1: grads.dTarget,
    };
  }
}
