/**
 * Hyperedge → Vertex Message Passing Phase
 *
 * Phase 2 of SHGAT message passing: L1+ nodes (groups) send messages
 * back to the L0 nodes they contain.
 *
 * Delegates to the shared phaseForward/phaseBackward implementation.
 * E→V mapping: source=E(L1+), target=H(L0), neighborMap=conn.sourceToTargets.
 *
 * @module graphrag/algorithms/shgat/message-passing/edge-to-vertex-phase
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
export type EVForwardCache = PhaseForwardCache;
export type EVPhaseResultWithCache = PhaseResultWithCache;

export interface EVGradients {
  dW_source: number[][];
  dW_target: number[][];
  da_attention: number[];
  /** Gradient for L1+ node embeddings (= source) */
  dE: number[][];
  /** Gradient for L0 node embeddings (= target) */
  dH: number[][];
}

/**
 * Hyperedge → Vertex message passing implementation
 */
export class EdgeToVertexPhase implements MessagePassingPhase {
  getName(): string {
    return "Edge→Vertex";
  }

  forward(
    E: number[][],
    H: number[][],
    connectivity: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): PhaseResult {
    const result = this.forwardWithCache(E, H, connectivity, params, config);
    return { embeddings: result.embeddings, attention: result.attention };
  }

  /**
   * Forward pass with cache for backward
   */
  forwardWithCache(
    E: number[][],
    H: number[][],
    conn: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): EVPhaseResultWithCache {
    // E→V: source=E(L1+), target=H(L0)
    // conn.sourceToTargets maps L0 → L1+, but in E→V semantics:
    //   key = L0 node (= aggregation target), values = L1+ nodes (= message sources)
    return phaseForward(E, H, conn.sourceToTargets, H.length, params, config);
  }

  /**
   * Backward pass: compute gradients for W_source, W_target, a_attention
   */
  backward(
    dH_new: number[][],
    cache: EVForwardCache,
    params: PhaseParameters,
  ): EVGradients {
    const grads: PhaseGradients = phaseBackward(dH_new, cache, params);
    return {
      dW_source: grads.dW_source,
      dW_target: grads.dW_target,
      da_attention: grads.da_attention,
      dE: grads.dSource,
      dH: grads.dTarget,
    };
  }
}
