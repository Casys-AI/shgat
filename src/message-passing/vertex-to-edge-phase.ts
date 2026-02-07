/**
 * Vertex → Hyperedge Message Passing Phase
 *
 * Phase 1 of SHGAT message passing: Tools (vertices) send messages to
 * capabilities (hyperedges) they participate in.
 *
 * Algorithm:
 *   1. Project tool embeddings: H' = H · W_v^T
 *   2. Project capability embeddings: E' = E · W_e^T
 *   3. Compute attention scores: score(t, c) = a^T · LeakyReLU([H'_t || E'_c])
 *      (masked by incidence matrix: only compute for tools in capability)
 *   4. Normalize per capability: α_c = softmax({score(t, c) | t ∈ c})
 *   5. Aggregate: E^new_c = ELU(Σ_t α_tc · H'_t)
 *
 * @module graphrag/algorithms/shgat/message-passing/vertex-to-edge-phase
 */

import * as math from "../utils/math.ts";
import type { MessagePassingPhase, PhaseParameters, PhaseResult } from "./phase-interface.ts";

/**
 * Cache for backward pass
 */
export interface VEForwardCache {
  /** Original tool embeddings [numTools][embDim] */
  H: number[][];
  /** Original capability embeddings [numCaps][embDim] */
  E: number[][];
  /** Projected tool embeddings [numTools][headDim] */
  H_proj: number[][];
  /** Projected capability embeddings [numCaps][headDim] */
  E_proj: number[][];
  /** Pre-activation concatenated vectors for each (t,c) pair */
  concatPreAct: Map<string, number[]>;
  /** Aggregated values before ELU [numCaps][headDim] */
  aggregated: number[][];
  /** Attention weights [numTools][numCaps] */
  attention: number[][];
  /** Connectivity matrix */
  connectivity: number[][];
  /** LeakyReLU slope */
  leakyReluSlope: number;
}

/**
 * Gradients from backward pass
 */
export interface VEGradients {
  /** Gradient for W_source [headDim][embDim] */
  dW_source: number[][];
  /** Gradient for W_target [headDim][embDim] */
  dW_target: number[][];
  /** Gradient for a_attention [2*headDim] */
  da_attention: number[];
  /** Gradient for input H [numTools][embDim] */
  dH: number[][];
  /** Gradient for input E [numCaps][embDim] */
  dE: number[][];
}

/**
 * Extended result with cache
 */
export interface VEPhaseResultWithCache extends PhaseResult {
  cache: VEForwardCache;
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
    connectivity: number[][],
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
    connectivity: number[][],
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): VEPhaseResultWithCache {
    const numTools = H.length;
    const numCaps = E.length;

    // Project embeddings
    const H_proj = math.matmulTranspose(H, params.W_source);
    const E_proj = math.matmulTranspose(E, params.W_target);

    // Cache for backward: pre-activation concat values
    const concatPreAct = new Map<string, number[]>();

    // Compute attention scores (masked by incidence matrix)
    const attentionScores: number[][] = Array.from(
      { length: numTools },
      () => Array(numCaps).fill(-Infinity),
    );

    for (let t = 0; t < numTools; t++) {
      for (let c = 0; c < numCaps; c++) {
        if (connectivity[t][c] === 1) {
          // Concatenate projected embeddings
          const concat = [...H_proj[t], ...E_proj[c]];
          concatPreAct.set(`${t}:${c}`, concat); // Cache pre-activation
          const activated = concat.map((x) => math.leakyRelu(x, config.leakyReluSlope));
          attentionScores[t][c] = math.dot(params.a_attention, activated);
        }
      }
    }

    // Softmax per capability (column-wise)
    const attentionVE: number[][] = Array.from({ length: numTools }, () => Array(numCaps).fill(0));

    for (let c = 0; c < numCaps; c++) {
      // Find all tools in this capability
      const toolsInCap: number[] = [];
      for (let t = 0; t < numTools; t++) {
        if (connectivity[t][c] === 1) {
          toolsInCap.push(t);
        }
      }

      if (toolsInCap.length === 0) continue;

      // Normalize attention weights for this capability
      const scores = toolsInCap.map((t) => attentionScores[t][c]);
      const softmaxed = math.softmax(scores);

      for (let i = 0; i < toolsInCap.length; i++) {
        attentionVE[toolsInCap[i]][c] = softmaxed[i];
      }
    }

    // Aggregate: E_new = σ(A'^T · H_proj)
    const E_new: number[][] = [];
    const aggregated: number[][] = [];
    const hiddenDim = H_proj[0]?.length ?? 0;

    for (let c = 0; c < numCaps; c++) {
      const agg = Array(hiddenDim).fill(0);

      // Weighted sum of tool embeddings
      for (let t = 0; t < numTools; t++) {
        if (attentionVE[t][c] > 0) {
          for (let d = 0; d < hiddenDim; d++) {
            agg[d] += attentionVE[t][c] * H_proj[t][d];
          }
        }
      }

      aggregated.push(agg);
      // Apply ELU activation
      E_new.push(agg.map((x) => math.elu(x)));
    }

    const cache: VEForwardCache = {
      H,
      E,
      H_proj,
      E_proj,
      concatPreAct,
      aggregated,
      attention: attentionVE,
      connectivity,
      leakyReluSlope: config.leakyReluSlope,
    };

    return { embeddings: E_new, attention: attentionVE, cache };
  }

  /**
   * Backward pass: compute gradients for W_source, W_target, a_attention
   *
   * @param dE_new - Gradient from next layer [numCaps][headDim]
   * @param cache - Forward pass cache
   * @param params - Phase parameters (needed for chain rule)
   * @returns Gradients for all parameters and inputs
   */
  backward(
    dE_new: number[][],
    cache: VEForwardCache,
    params: PhaseParameters,
  ): VEGradients {
    const { H, E, H_proj, concatPreAct, aggregated, attention, connectivity, leakyReluSlope } = cache;
    const numTools = H.length;
    const numCaps = E.length;
    const headDim = H_proj[0]?.length ?? 0;
    const embDim = H[0]?.length ?? 0;

    // Initialize gradients
    const dW_source: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
    const dW_target: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
    const da_attention: number[] = Array(2 * headDim).fill(0);
    const dH: number[][] = Array.from({ length: numTools }, () => Array(embDim).fill(0));
    const dE: number[][] = Array.from({ length: numCaps }, () => Array(embDim).fill(0));

    // Intermediate gradients
    const dH_proj: number[][] = Array.from({ length: numTools }, () => Array(headDim).fill(0));
    const dE_proj: number[][] = Array.from({ length: numCaps }, () => Array(headDim).fill(0));

    // Step 1: Through ELU activation
    // dAggregated[c][d] = dE_new[c][d] * ELU'(aggregated[c][d])
    const dAggregated: number[][] = [];
    for (let c = 0; c < numCaps; c++) {
      const dAgg = dE_new[c].map((grad, d) => {
        const x = aggregated[c][d];
        // ELU'(x) = 1 if x >= 0, else exp(x)
        const eluDeriv = x >= 0 ? 1 : Math.exp(x);
        return grad * eluDeriv;
      });
      dAggregated.push(dAgg);
    }

    // Step 2: Through aggregation
    // aggregated[c] = Σ_t attention[t][c] * H_proj[t]
    // → dAttention[t][c] = dAggregated[c] · H_proj[t]
    // → dH_proj[t] += attention[t][c] * dAggregated[c]
    const dAttention: number[][] = Array.from({ length: numTools }, () => Array(numCaps).fill(0));

    for (let c = 0; c < numCaps; c++) {
      for (let t = 0; t < numTools; t++) {
        if (attention[t][c] > 0) {
          // dAttention[t][c] = dot(dAggregated[c], H_proj[t])
          dAttention[t][c] = math.dot(dAggregated[c], H_proj[t]);

          // dH_proj[t] += attention[t][c] * dAggregated[c]
          for (let d = 0; d < headDim; d++) {
            dH_proj[t][d] += attention[t][c] * dAggregated[c][d];
          }
        }
      }
    }

    // Step 3: Through softmax (per capability)
    // softmax jacobian: dScore[t] = attention[t] * (dAttention[t] - Σ attention * dAttention)
    const dScore: number[][] = Array.from({ length: numTools }, () => Array(numCaps).fill(0));

    for (let c = 0; c < numCaps; c++) {
      // Find tools in this capability
      const toolsInCap: number[] = [];
      for (let t = 0; t < numTools; t++) {
        if (connectivity[t][c] === 1) {
          toolsInCap.push(t);
        }
      }

      if (toolsInCap.length === 0) continue;

      // Compute sum: Σ attention[t][c] * dAttention[t][c]
      let sumAttnDAttn = 0;
      for (const t of toolsInCap) {
        sumAttnDAttn += attention[t][c] * dAttention[t][c];
      }

      // dScore[t][c] = attention[t][c] * (dAttention[t][c] - sumAttnDAttn)
      for (const t of toolsInCap) {
        dScore[t][c] = attention[t][c] * (dAttention[t][c] - sumAttnDAttn);
      }
    }

    // Step 4 & 5: Through attention computation and LeakyReLU
    // score[t][c] = a · LeakyReLU(concat)
    // → dActivated = dScore * a (element-wise contribution)
    // → dConcat = dActivated * LeakyReLU'
    // → da_attention += activated * dScore

    for (let t = 0; t < numTools; t++) {
      for (let c = 0; c < numCaps; c++) {
        if (connectivity[t][c] !== 1) continue;

        const concat = concatPreAct.get(`${t}:${c}`);
        if (!concat) continue;

        const score_grad = dScore[t][c];

        // Compute activated values (for da_attention)
        const activated = concat.map((x) => math.leakyRelu(x, leakyReluSlope));

        // da_attention += activated * dScore
        for (let i = 0; i < activated.length; i++) {
          da_attention[i] += activated[i] * score_grad;
        }

        // dActivated = dScore * a_attention
        // dConcat = dActivated * LeakyReLU'(concat)
        const dConcat = concat.map((x, i) => {
          const leakyDeriv = x > 0 ? 1 : leakyReluSlope;
          return score_grad * params.a_attention[i] * leakyDeriv;
        });

        // Split dConcat into dH_proj_attn and dE_proj
        // dConcat = [dH_proj_attn, dE_proj_attn]
        for (let d = 0; d < headDim; d++) {
          dH_proj[t][d] += dConcat[d];
          dE_proj[c][d] += dConcat[headDim + d];
        }
      }
    }

    // Step 6: Through projection matrices (BLAS-accelerated matrix multiplications)
    // H_proj = H @ W_source.T → dW_source += dH_proj.T @ H, dH += dH_proj @ W_source
    // E_proj = E @ W_target.T → dW_target += dE_proj.T @ E, dE += dE_proj @ W_target

    // dW_source = dH_proj.T @ H (BLAS: matmulTranspose computes A @ B^T, so we use transpose of dH_proj)
    // Using batch outer product: dW_source[i][j] = Σ_t dH_proj[t][i] * H[t][j]
    const dW_source_contrib = math.matmulTranspose(math.transpose(dH_proj), H);
    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < embDim; j++) {
        dW_source[i][j] += dW_source_contrib[i]?.[j] ?? 0;
      }
    }

    // dW_target = dE_proj.T @ E
    const dW_target_contrib = math.matmulTranspose(math.transpose(dE_proj), E);
    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < embDim; j++) {
        dW_target[i][j] += dW_target_contrib[i]?.[j] ?? 0;
      }
    }

    // dH = dH_proj @ W_source (BLAS-accelerated)
    const dH_contrib = math.matmul(dH_proj, params.W_source);
    for (let t = 0; t < numTools; t++) {
      for (let j = 0; j < embDim; j++) {
        dH[t][j] += dH_contrib[t]?.[j] ?? 0;
      }
    }

    // dE = dE_proj @ W_target (BLAS-accelerated)
    const dE_contrib = math.matmul(dE_proj, params.W_target);
    for (let c = 0; c < numCaps; c++) {
      for (let j = 0; j < embDim; j++) {
        dE[c][j] += dE_contrib[c]?.[j] ?? 0;
      }
    }

    return { dW_source, dW_target, da_attention, dH, dE };
  }
}
