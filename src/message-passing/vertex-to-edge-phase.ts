/**
 * Vertex → Hyperedge Message Passing Phase
 *
 * Phase 1 of SHGAT message passing: L0 nodes send messages to
 * L1+ nodes (groups) they participate in.
 *
 * Delegates to the shared phaseForward/phaseBackward implementation.
 * V→E mapping: source=H(L0), target=E(L1+), neighborMap=conn.targetToSources.
 *
 * @module graphrag/algorithms/shgat/message-passing/vertex-to-edge-phase
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
export type VEForwardCache = PhaseForwardCache;
export type VEPhaseResultWithCache = PhaseResultWithCache;

export interface VEGradients {
  dW_source: number[][];
  dW_target: number[][];
  da_attention: number[];
  /** Gradient for L0 node embeddings (= source) */
  dH: number[][];
  /** Gradient for L1+ node embeddings (= target) */
  dE: number[][];
}

/**
 * Vertex → Hyperedge message passing implementation
 */
export class VertexToEdgePhase implements MessagePassingPhase {
  getName(): string {
    return "Vertex→Edge";
  }

  forward(
    H: number[][],
    E: number[][],
    connectivity: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): PhaseResult {
    const result = this.forwardWithCache(H, E, connectivity, params, config);
    return { embeddings: result.embeddings, attention: result.attention };
  }

  /**
   * Forward pass with cache for backward
   */
  forwardWithCache(
    H: number[][],
    E: number[][],
    conn: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): VEPhaseResultWithCache {
    // V→E: target=L1+(E), sources=L0(H) → iterate conn.targetToSources
    return phaseForward(H, E, conn.targetToSources, E.length, params, config);
  }

  /**
   * Backward pass: compute gradients for W_source, W_target, a_attention
   */
  backward(
    dE_new: number[][],
    cache: VEForwardCache,
    params: PhaseParameters,
  ): VEGradients {
    const grads: PhaseGradients = phaseBackward(dE_new, cache, params);
    return {
      dW_source: grads.dW_source,
      dW_target: grads.dW_target,
      da_attention: grads.da_attention,
      dH: grads.dSource,
      dE: grads.dTarget,
    };
  }
}
