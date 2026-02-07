/**
 * Hyperedge → Vertex Message Passing Phase
 *
 * Phase 2 of SHGAT message passing: Capabilities (hyperedges) send messages
 * back to the tools (vertices) they contain.
 *
 * Algorithm:
 *   1. Project capability embeddings: E' = E · W_e^T
 *   2. Project tool embeddings: H' = H · W_v^T
 *   3. Compute attention scores: score(c, t) = a^T · LeakyReLU([E'_c || H'_t])
 *      (masked by incidence matrix: only compute for capabilities containing tool)
 *   4. Normalize per tool: α_t = softmax({score(c, t) | t ∈ c})
 *   5. Aggregate: H^new_t = ELU(Σ_c α_ct · E'_c)
 *
 * @module graphrag/algorithms/shgat/message-passing/edge-to-vertex-phase
 */

import * as math from "../utils/math.ts";
import type { MessagePassingPhase, PhaseParameters, PhaseResult } from "./phase-interface.ts";

/**
 * Cache for backward pass
 */
export interface EVForwardCache {
  /** Original capability embeddings [numCaps][embDim] */
  E: number[][];
  /** Original tool embeddings [numTools][embDim] */
  H: number[][];
  /** Projected capability embeddings [numCaps][headDim] */
  E_proj: number[][];
  /** Projected tool embeddings [numTools][headDim] */
  H_proj: number[][];
  /** Pre-activation concatenated vectors for each (c,t) pair */
  concatPreAct: Map<string, number[]>;
  /** Aggregated values before ELU [numTools][headDim] */
  aggregated: number[][];
  /** Attention weights [numCaps][numTools] */
  attention: number[][];
  /** Connectivity matrix [numTools][numCaps] (original orientation) */
  connectivity: number[][];
  /** LeakyReLU slope */
  leakyReluSlope: number;
}

/**
 * Gradients from backward pass
 */
export interface EVGradients {
  /** Gradient for W_source [headDim][embDim] */
  dW_source: number[][];
  /** Gradient for W_target [headDim][embDim] */
  dW_target: number[][];
  /** Gradient for a_attention [2*headDim] */
  da_attention: number[];
  /** Gradient for input E [numCaps][embDim] */
  dE: number[][];
  /** Gradient for input H [numTools][embDim] */
  dH: number[][];
}

/**
 * Extended result with cache
 */
export interface EVPhaseResultWithCache extends PhaseResult {
  cache: EVForwardCache;
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
    connectivity: number[][],
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
    connectivity: number[][],
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): EVPhaseResultWithCache {
    const numCaps = E.length;
    const numTools = H.length;

    // Project embeddings
    const E_proj = math.matmulTranspose(E, params.W_source);
    const H_proj = math.matmulTranspose(H, params.W_target);

    // Cache for backward: pre-activation concat values
    const concatPreAct = new Map<string, number[]>();

    // Compute attention scores (masked by incidence matrix)
    const attentionScores: number[][] = Array.from(
      { length: numCaps },
      () => Array(numTools).fill(-Infinity),
    );

    for (let c = 0; c < numCaps; c++) {
      for (let t = 0; t < numTools; t++) {
        // Note: connectivity is still [t][c] from vertex-to-edge perspective
        if (connectivity[t][c] === 1) {
          // Concatenate projected embeddings
          const concat = [...E_proj[c], ...H_proj[t]];
          concatPreAct.set(`${c}:${t}`, concat); // Cache pre-activation
          const activated = concat.map((x) => math.leakyRelu(x, config.leakyReluSlope));
          attentionScores[c][t] = math.dot(params.a_attention, activated);
        }
      }
    }

    // Softmax per tool (column-wise in transposed view)
    const attentionEV: number[][] = Array.from({ length: numCaps }, () => Array(numTools).fill(0));

    for (let t = 0; t < numTools; t++) {
      // Find all capabilities containing this tool
      const capsForTool: number[] = [];
      for (let c = 0; c < numCaps; c++) {
        if (connectivity[t][c] === 1) {
          capsForTool.push(c);
        }
      }

      if (capsForTool.length === 0) continue;

      // Normalize attention weights for this tool
      const scores = capsForTool.map((c) => attentionScores[c][t]);
      const softmaxed = math.softmax(scores);

      for (let i = 0; i < capsForTool.length; i++) {
        attentionEV[capsForTool[i]][t] = softmaxed[i];
      }
    }

    // Aggregate: H_new = σ(B^T · E_proj)
    const H_new: number[][] = [];
    const aggregated: number[][] = [];
    const hiddenDim = E_proj[0]?.length ?? 0;

    for (let t = 0; t < numTools; t++) {
      const agg = Array(hiddenDim).fill(0);

      // Weighted sum of capability embeddings
      for (let c = 0; c < numCaps; c++) {
        if (attentionEV[c][t] > 0) {
          for (let d = 0; d < hiddenDim; d++) {
            agg[d] += attentionEV[c][t] * E_proj[c][d];
          }
        }
      }

      aggregated.push(agg);
      // Apply ELU activation
      H_new.push(agg.map((x) => math.elu(x)));
    }

    const cache: EVForwardCache = {
      E,
      H,
      E_proj,
      H_proj,
      concatPreAct,
      aggregated,
      attention: attentionEV,
      connectivity,
      leakyReluSlope: config.leakyReluSlope,
    };

    return { embeddings: H_new, attention: attentionEV, cache };
  }

  /**
   * Backward pass: compute gradients for W_source, W_target, a_attention
   *
   * @param dH_new - Gradient from next layer [numTools][headDim]
   * @param cache - Forward pass cache
   * @param params - Phase parameters (needed for chain rule)
   * @returns Gradients for all parameters and inputs
   */
  backward(
    dH_new: number[][],
    cache: EVForwardCache,
    params: PhaseParameters,
  ): EVGradients {
    const { E, H, E_proj, concatPreAct, aggregated, attention, connectivity, leakyReluSlope } = cache;
    const numCaps = E.length;
    const numTools = H.length;
    const headDim = E_proj[0]?.length ?? 0;
    const embDim = E[0]?.length ?? 0;

    // Initialize gradients
    const dW_source: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
    const dW_target: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
    const da_attention: number[] = Array(2 * headDim).fill(0);
    const dE: number[][] = Array.from({ length: numCaps }, () => Array(embDim).fill(0));
    const dH: number[][] = Array.from({ length: numTools }, () => Array(embDim).fill(0));

    // Intermediate gradients
    const dE_proj: number[][] = Array.from({ length: numCaps }, () => Array(headDim).fill(0));
    const dH_proj: number[][] = Array.from({ length: numTools }, () => Array(headDim).fill(0));

    // Step 1: Through ELU activation
    // dAggregated[t][d] = dH_new[t][d] * ELU'(aggregated[t][d])
    const dAggregated: number[][] = [];
    for (let t = 0; t < numTools; t++) {
      const dAgg = dH_new[t].map((grad, d) => {
        const x = aggregated[t][d];
        // ELU'(x) = 1 if x >= 0, else exp(x)
        const eluDeriv = x >= 0 ? 1 : Math.exp(x);
        return grad * eluDeriv;
      });
      dAggregated.push(dAgg);
    }

    // Step 2: Through aggregation
    // aggregated[t] = Σ_c attention[c][t] * E_proj[c]
    // → dAttention[c][t] = dAggregated[t] · E_proj[c]
    // → dE_proj[c] += attention[c][t] * dAggregated[t]
    const dAttention: number[][] = Array.from({ length: numCaps }, () => Array(numTools).fill(0));

    for (let t = 0; t < numTools; t++) {
      for (let c = 0; c < numCaps; c++) {
        if (attention[c][t] > 0) {
          // dAttention[c][t] = dot(dAggregated[t], E_proj[c])
          dAttention[c][t] = math.dot(dAggregated[t], E_proj[c]);

          // dE_proj[c] += attention[c][t] * dAggregated[t]
          for (let d = 0; d < headDim; d++) {
            dE_proj[c][d] += attention[c][t] * dAggregated[t][d];
          }
        }
      }
    }

    // Step 3: Through softmax (per tool)
    // softmax jacobian: dScore[c] = attention[c] * (dAttention[c] - Σ attention * dAttention)
    const dScore: number[][] = Array.from({ length: numCaps }, () => Array(numTools).fill(0));

    for (let t = 0; t < numTools; t++) {
      // Find capabilities containing this tool
      const capsForTool: number[] = [];
      for (let c = 0; c < numCaps; c++) {
        if (connectivity[t][c] === 1) {
          capsForTool.push(c);
        }
      }

      if (capsForTool.length === 0) continue;

      // Compute sum: Σ attention[c][t] * dAttention[c][t]
      let sumAttnDAttn = 0;
      for (const c of capsForTool) {
        sumAttnDAttn += attention[c][t] * dAttention[c][t];
      }

      // dScore[c][t] = attention[c][t] * (dAttention[c][t] - sumAttnDAttn)
      for (const c of capsForTool) {
        dScore[c][t] = attention[c][t] * (dAttention[c][t] - sumAttnDAttn);
      }
    }

    // Step 4 & 5: Through attention computation and LeakyReLU
    // score[c][t] = a · LeakyReLU(concat)
    // → dActivated = dScore * a (element-wise contribution)
    // → dConcat = dActivated * LeakyReLU'
    // → da_attention += activated * dScore

    for (let c = 0; c < numCaps; c++) {
      for (let t = 0; t < numTools; t++) {
        if (connectivity[t][c] !== 1) continue;

        const concat = concatPreAct.get(`${c}:${t}`);
        if (!concat) continue;

        const score_grad = dScore[c][t];

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

        // Split dConcat into dE_proj_attn and dH_proj
        // dConcat = [dE_proj_attn, dH_proj_attn]
        for (let d = 0; d < headDim; d++) {
          dE_proj[c][d] += dConcat[d];
          dH_proj[t][d] += dConcat[headDim + d];
        }
      }
    }

    // Step 6: Through projection matrices (BLAS-accelerated matrix multiplications)
    // E_proj = E @ W_source.T → dW_source += dE_proj.T @ E, dE += dE_proj @ W_source
    // H_proj = H @ W_target.T → dW_target += dH_proj.T @ H, dH += dH_proj @ W_target

    // dW_source = dE_proj.T @ E (BLAS-accelerated)
    const dW_source_contrib = math.matmulTranspose(math.transpose(dE_proj), E);
    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < embDim; j++) {
        dW_source[i][j] += dW_source_contrib[i]?.[j] ?? 0;
      }
    }

    // dW_target = dH_proj.T @ H (BLAS-accelerated)
    const dW_target_contrib = math.matmulTranspose(math.transpose(dH_proj), H);
    for (let i = 0; i < headDim; i++) {
      for (let j = 0; j < embDim; j++) {
        dW_target[i][j] += dW_target_contrib[i]?.[j] ?? 0;
      }
    }

    // dE = dE_proj @ W_source (BLAS-accelerated)
    const dE_contrib = math.matmul(dE_proj, params.W_source);
    for (let c = 0; c < numCaps; c++) {
      for (let j = 0; j < embDim; j++) {
        dE[c][j] += dE_contrib[c]?.[j] ?? 0;
      }
    }

    // dH = dH_proj @ W_target (BLAS-accelerated)
    const dH_contrib = math.matmul(dH_proj, params.W_target);
    for (let t = 0; t < numTools; t++) {
      for (let j = 0; j < embDim; j++) {
        dH[t][j] += dH_contrib[t]?.[j] ?? 0;
      }
    }

    return { dW_source, dW_target, da_attention, dE, dH };
  }
}
