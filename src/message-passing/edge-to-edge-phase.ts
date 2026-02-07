/**
 * Hyperedge → Hyperedge Message Passing Phase (Multi-Level)
 *
 * Phase for multi-level n-SuperHyperGraph: Capabilities at level k send
 * messages to capabilities at level k+1 that contain them.
 *
 * This is the KEY phase for hierarchical message passing:
 *   V → E^0 → E^1 → E^2 → ... → E^n → ... → V
 *
 * Algorithm (E^k → E^(k+1)):
 *   1. Project child capability embeddings: E^k' = E^k · W_source^T
 *   2. Project parent capability embeddings: E^(k+1)' = E^(k+1) · W_target^T
 *   3. Compute attention scores: score(c_k, c_{k+1}) = a^T · LeakyReLU([E^k'_c || E^(k+1)'_p])
 *      (masked by containment matrix: only compute for c_k ∈ c_{k+1})
 *   4. Normalize per parent: α_p = softmax({score(c, p) | c ∈ p})
 *   5. Aggregate: E^(k+1)^new_p = ELU(Σ_c α_cp · E^k'_c)
 *
 * This is identical to VertexToEdgePhase but operates on E^k instead of V.
 *
 * @module graphrag/algorithms/shgat/message-passing/edge-to-edge-phase
 */

import * as math from "../utils/math.ts";
import type { MessagePassingPhase, PhaseParameters, PhaseResult } from "./phase-interface.ts";

/**
 * Cache for backward pass
 */
export interface EEForwardCache {
  /** Child capability embeddings [numChild][embDim] */
  E_k: number[][];
  /** Parent capability embeddings [numParent][embDim] */
  E_kPlus1: number[][];
  /** Projected child embeddings [numChild][headDim] */
  E_k_proj: number[][];
  /** Projected parent embeddings [numParent][headDim] */
  E_kPlus1_proj: number[][];
  /** Pre-activation concatenated vectors for each (c,p) pair */
  concatPreAct: Map<string, number[]>;
  /** Aggregated values before ELU [numParent][headDim] */
  aggregated: number[][];
  /** Attention weights [numChild][numParent] */
  attention: number[][];
  /** Containment matrix [numChild][numParent] */
  containment: number[][];
  /** LeakyReLU slope */
  leakyReluSlope: number;
}

/**
 * Gradients from backward pass
 */
export interface EEGradients {
  /** Gradient for W_source [headDim][embDim] */
  dW_source: number[][];
  /** Gradient for W_target [headDim][embDim] */
  dW_target: number[][];
  /** Gradient for a_attention [2*headDim] */
  da_attention: number[];
  /** Gradient for input E_k [numChild][embDim] */
  dE_k: number[][];
  /** Gradient for input E_kPlus1 [numParent][embDim] */
  dE_kPlus1: number[][];
}

/**
 * Extended result with cache
 */
export interface EEPhaseResultWithCache extends PhaseResult {
  cache: EEForwardCache;
}

/**
 * Hyperedge → Hyperedge message passing implementation
 *
 * Used for hierarchical capabilities where capabilities can contain
 * other capabilities (n-SuperHyperGraph structure).
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
    containment: number[][],
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): PhaseResult {
    const result = this.forwardWithCache(E_k, E_kPlus1, containment, params, config);
    return { embeddings: result.embeddings, attention: result.attention };
  }

  /**
   * Forward pass with cache for backward
   */
  forwardWithCache(
    E_k: number[][],
    E_kPlus1: number[][],
    containment: number[][],
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): EEPhaseResultWithCache {
    const numChildCaps = E_k.length;
    const numParentCaps = E_kPlus1.length;

    // Project embeddings
    const E_k_proj = math.matmulTranspose(E_k, params.W_source);
    const E_kPlus1_proj = math.matmulTranspose(E_kPlus1, params.W_target);

    // Cache for backward: pre-activation concat values
    const concatPreAct = new Map<string, number[]>();

    // Compute attention scores (masked by containment matrix)
    const attentionScores: number[][] = Array.from(
      { length: numChildCaps },
      () => Array(numParentCaps).fill(-Infinity),
    );

    for (let c = 0; c < numChildCaps; c++) {
      for (let p = 0; p < numParentCaps; p++) {
        if (containment[c][p] === 1) {
          // Concatenate projected embeddings
          const concat = [...E_k_proj[c], ...E_kPlus1_proj[p]];
          concatPreAct.set(`${c}:${p}`, concat); // Cache pre-activation
          const activated = concat.map((x) => math.leakyRelu(x, config.leakyReluSlope));
          attentionScores[c][p] = math.dot(params.a_attention, activated);
        }
      }
    }

    // Softmax per parent capability (column-wise)
    const attentionCE: number[][] = Array.from(
      { length: numChildCaps },
      () => Array(numParentCaps).fill(0),
    );

    for (let p = 0; p < numParentCaps; p++) {
      // Find all child capabilities in this parent
      const childrenInParent: number[] = [];
      for (let c = 0; c < numChildCaps; c++) {
        if (containment[c][p] === 1) {
          childrenInParent.push(c);
        }
      }

      if (childrenInParent.length === 0) continue;

      // Normalize attention weights for this parent
      const scores = childrenInParent.map((c) => attentionScores[c][p]);
      const softmaxed = math.softmax(scores);

      for (let i = 0; i < childrenInParent.length; i++) {
        attentionCE[childrenInParent[i]][p] = softmaxed[i];
      }
    }

    // Aggregate: E^(k+1)_new = σ(A'^T · E^k_proj)
    const E_kPlus1_new: number[][] = [];
    const aggregated: number[][] = [];
    const hiddenDim = E_k_proj[0]?.length ?? 0;

    for (let p = 0; p < numParentCaps; p++) {
      const agg = Array(hiddenDim).fill(0);

      // Weighted sum of child capability embeddings
      for (let c = 0; c < numChildCaps; c++) {
        if (attentionCE[c][p] > 0) {
          for (let d = 0; d < hiddenDim; d++) {
            agg[d] += attentionCE[c][p] * E_k_proj[c][d];
          }
        }
      }

      aggregated.push(agg);
      // Apply ELU activation
      E_kPlus1_new.push(agg.map((x) => math.elu(x)));
    }

    const cache: EEForwardCache = {
      E_k,
      E_kPlus1,
      E_k_proj,
      E_kPlus1_proj,
      concatPreAct,
      aggregated,
      attention: attentionCE,
      containment,
      leakyReluSlope: config.leakyReluSlope,
    };

    return { embeddings: E_kPlus1_new, attention: attentionCE, cache };
  }

  /**
   * Backward pass: compute gradients for W_source, W_target, a_attention
   *
   * @param dE_kPlus1_new - Gradient from next layer [numParent][headDim]
   * @param cache - Forward pass cache
   * @param params - Phase parameters (needed for chain rule)
   * @returns Gradients for all parameters and inputs
   */
  backward(
    dE_kPlus1_new: number[][],
    cache: EEForwardCache,
    params: PhaseParameters,
  ): EEGradients {
    const { E_k, E_kPlus1, E_k_proj, concatPreAct, aggregated, attention, containment, leakyReluSlope } = cache;
    const numChildCaps = E_k.length;
    const numParentCaps = E_kPlus1.length;
    const headDim = E_k_proj[0]?.length ?? 0;
    const embDim = E_k[0]?.length ?? 0;

    // Initialize gradients
    const dW_source: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
    const dW_target: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
    const da_attention: number[] = Array(2 * headDim).fill(0);
    const dE_k: number[][] = Array.from({ length: numChildCaps }, () => Array(embDim).fill(0));
    const dE_kPlus1: number[][] = Array.from({ length: numParentCaps }, () => Array(embDim).fill(0));

    // Intermediate gradients
    const dE_k_proj: number[][] = Array.from({ length: numChildCaps }, () => Array(headDim).fill(0));
    const dE_kPlus1_proj: number[][] = Array.from({ length: numParentCaps }, () => Array(headDim).fill(0));

    // Step 1: Through ELU activation
    const dAggregated: number[][] = [];
    for (let p = 0; p < numParentCaps; p++) {
      const dAgg = dE_kPlus1_new[p].map((grad, d) => {
        const x = aggregated[p][d];
        const eluDeriv = x >= 0 ? 1 : Math.exp(x);
        return grad * eluDeriv;
      });
      dAggregated.push(dAgg);
    }

    // Step 2: Through aggregation
    const dAttention: number[][] = Array.from({ length: numChildCaps }, () => Array(numParentCaps).fill(0));

    for (let p = 0; p < numParentCaps; p++) {
      for (let c = 0; c < numChildCaps; c++) {
        if (attention[c][p] > 0) {
          dAttention[c][p] = math.dot(dAggregated[p], E_k_proj[c]);
          for (let d = 0; d < headDim; d++) {
            dE_k_proj[c][d] += attention[c][p] * dAggregated[p][d];
          }
        }
      }
    }

    // Step 3: Through softmax (per parent)
    const dScore: number[][] = Array.from({ length: numChildCaps }, () => Array(numParentCaps).fill(0));

    for (let p = 0; p < numParentCaps; p++) {
      const childrenInParent: number[] = [];
      for (let c = 0; c < numChildCaps; c++) {
        if (containment[c][p] === 1) {
          childrenInParent.push(c);
        }
      }

      if (childrenInParent.length === 0) continue;

      let sumAttnDAttn = 0;
      for (const c of childrenInParent) {
        sumAttnDAttn += attention[c][p] * dAttention[c][p];
      }

      for (const c of childrenInParent) {
        dScore[c][p] = attention[c][p] * (dAttention[c][p] - sumAttnDAttn);
      }
    }

    // Step 4 & 5: Through attention computation and LeakyReLU
    for (let c = 0; c < numChildCaps; c++) {
      for (let p = 0; p < numParentCaps; p++) {
        if (containment[c][p] !== 1) continue;

        const concat = concatPreAct.get(`${c}:${p}`);
        if (!concat) continue;

        const score_grad = dScore[c][p];
        const activated = concat.map((x) => math.leakyRelu(x, leakyReluSlope));

        for (let i = 0; i < activated.length; i++) {
          da_attention[i] += activated[i] * score_grad;
        }

        const dConcat = concat.map((x, i) => {
          const leakyDeriv = x > 0 ? 1 : leakyReluSlope;
          return score_grad * params.a_attention[i] * leakyDeriv;
        });

        for (let d = 0; d < headDim; d++) {
          dE_k_proj[c][d] += dConcat[d];
          dE_kPlus1_proj[p][d] += dConcat[headDim + d];
        }
      }
    }

    // Step 6: Through projection matrices (BLAS-accelerated matrix multiplications)
    // dW_source = dE_k_proj.T @ E_k (BLAS-accelerated)
    const dW_source_contrib = math.matmulTranspose(math.transpose(dE_k_proj), E_k);
    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < embDim; j++) {
        dW_source[i][j] += dW_source_contrib[i]?.[j] ?? 0;
      }
    }

    // dW_target = dE_kPlus1_proj.T @ E_kPlus1 (BLAS-accelerated)
    const dW_target_contrib = math.matmulTranspose(math.transpose(dE_kPlus1_proj), E_kPlus1);
    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < embDim; j++) {
        dW_target[i][j] += dW_target_contrib[i]?.[j] ?? 0;
      }
    }

    // dE_k = dE_k_proj @ W_source (BLAS-accelerated)
    const dE_k_contrib = math.matmul(dE_k_proj, params.W_source);
    for (let c = 0; c < numChildCaps; c++) {
      for (let j = 0; j < embDim; j++) {
        dE_k[c][j] += dE_k_contrib[c]?.[j] ?? 0;
      }
    }

    // dE_kPlus1 = dE_kPlus1_proj @ W_target (BLAS-accelerated)
    const dE_kPlus1_contrib = math.matmul(dE_kPlus1_proj, params.W_target);
    for (let p = 0; p < numParentCaps; p++) {
      for (let j = 0; j < embDim; j++) {
        dE_kPlus1[p][j] += dE_kPlus1_contrib[p]?.[j] ?? 0;
      }
    }

    return { dW_source, dW_target, da_attention, dE_k, dE_kPlus1 };
  }
}
