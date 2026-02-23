/**
 * Shared Forward & Backward for GAT Message Passing Phases
 *
 * All three phases (V→E, E→V, E→E) follow the identical algorithm:
 *   1. Project source and target embeddings
 *   2. Compute attention scores: a^T · LeakyReLU([sourceProj || targetProj])
 *   3. Softmax per target node
 *   4. Aggregate: output[t] = ELU(Σ_s α[s,t] · sourceProj[s])
 *
 * The only difference is which connectivity direction each phase iterates:
 *   V→E / E→E: conn.targetToSources
 *   E→V:       conn.sourceToTargets  (keys are still "aggregation targets")
 *
 * This module implements both forward and backward once, eliminating ~900 lines
 * of duplicated logic across the three phase files.
 *
 * @module graphrag/algorithms/shgat/message-passing/phase-shared
 */

import * as math from "../utils/math.ts";
import type {
  PhaseForwardCache,
  PhaseGradients,
  PhaseParameters,
  PhaseResult,
  PhaseResultWithCache,
} from "./phase-interface.ts";
import { edgeKey } from "./phase-interface.ts";

/**
 * Generic forward pass for all GAT message passing phases.
 *
 * @param source - Source node embeddings [numSource][embDim]
 * @param target - Target node embeddings [numTarget][embDim]
 * @param neighborMap - Map: targetIdx → [sourceIdx, ...] (iteration order)
 * @param numTargets - Total number of target nodes
 * @param params - Phase parameters (W_source, W_target, a_attention)
 * @param config - { leakyReluSlope }
 * @returns Phase result with embeddings, dense attention, and cache
 */
export function phaseForward(
  source: number[][],
  target: number[][],
  neighborMap: Map<number, number[]>,
  numTargets: number,
  params: PhaseParameters,
  config: { leakyReluSlope: number },
): PhaseResultWithCache {
  const numSource = source.length;

  // 1. Project embeddings (Float32 for cache RAM)
  const sourceProj = math.matmulTransposeF32(source, params.W_source);
  const targetProj = math.matmulTransposeF32(target, params.W_target);
  const headDim = sourceProj[0]?.length ?? 0;

  // 2. Compute attention scores (only for existing edges)
  const attentionScores = new Map<number, number>();

  for (const [t, sources] of neighborMap) {
    for (const s of sources) {
      const key = edgeKey(s, t, numTargets);
      let score = 0;
      for (let d = 0; d < headDim; d++) {
        score += params.a_attention[d] * math.leakyRelu(sourceProj[s][d], config.leakyReluSlope);
      }
      for (let d = 0; d < headDim; d++) {
        score += params.a_attention[headDim + d] * math.leakyRelu(targetProj[t][d], config.leakyReluSlope);
      }
      attentionScores.set(key, score);
    }
  }

  // 3. Softmax per target
  const attentionMap = new Map<number, number>();

  for (const [t, sources] of neighborMap) {
    if (sources.length === 0) continue;

    const scores = sources.map((s) => attentionScores.get(edgeKey(s, t, numTargets))!);
    const softmaxed = math.softmax(scores);

    for (let i = 0; i < sources.length; i++) {
      attentionMap.set(edgeKey(sources[i], t, numTargets), softmaxed[i]);
    }
  }

  // 4. Aggregate: output[t] = ELU(Σ_s attention[s,t] * sourceProj[s])
  const outputEmbs: number[][] = [];
  const aggregated: Float32Array[] = [];

  for (let t = 0; t < numTargets; t++) {
    const agg = new Float32Array(headDim);
    const sources = neighborMap.get(t);
    if (sources) {
      for (const s of sources) {
        const alpha = attentionMap.get(edgeKey(s, t, numTargets)) ?? 0;
        if (alpha > 0) {
          for (let d = 0; d < headDim; d++) {
            agg[d] += alpha * sourceProj[s][d];
          }
        }
      }
    }
    aggregated.push(agg);
    outputEmbs.push(Array.from(agg, (x) => math.elu(x)));
  }

  // 5. Dense attention matrix [numSource][numTarget]
  const attentionDense: number[][] = Array.from(
    { length: numSource },
    () => Array(numTargets).fill(0),
  );
  for (const [key, val] of attentionMap) {
    const s = Math.floor(key / numTargets);
    const t = key % numTargets;
    attentionDense[s][t] = val;
  }

  const cache: PhaseForwardCache = {
    source,
    target,
    sourceProj,
    targetProj,
    aggregated,
    attention: attentionMap,
    neighborMap,
    numTargets,
    leakyReluSlope: config.leakyReluSlope,
  };

  return { embeddings: outputEmbs, attention: attentionDense, cache };
}

/**
 * Generic forward pass without cache (inference only).
 */
export function phaseForwardInference(
  source: number[][],
  target: number[][],
  neighborMap: Map<number, number[]>,
  numTargets: number,
  params: PhaseParameters,
  config: { leakyReluSlope: number },
): PhaseResult {
  const { embeddings, attention } = phaseForward(source, target, neighborMap, numTargets, params, config);
  return { embeddings, attention };
}

/**
 * Generic backward pass for all GAT message passing phases.
 *
 * Reverses the 5-step forward in exact order:
 *   1. Through ELU activation
 *   2. Through aggregation (sparse)
 *   3. Through softmax (per target)
 *   4-5. Through attention + LeakyReLU
 *   6. Through projection matrices (BLAS-accelerated)
 *
 * @param dOutput - Gradient on output embeddings [numTarget][headDim]
 * @param cache - Forward pass cache
 * @param params - Phase parameters (needed for chain rule)
 * @returns Gradients for all parameters and both input embedding sets
 */
export function phaseBackward(
  dOutput: number[][],
  cache: PhaseForwardCache,
  params: PhaseParameters,
): PhaseGradients {
  const {
    source, target, sourceProj, targetProj,
    aggregated, attention, neighborMap, numTargets, leakyReluSlope,
  } = cache;
  const numSource = source.length;
  const numTarget = target.length;
  const headDim = sourceProj[0]?.length ?? 0;
  const embDim = source[0]?.length ?? 0;

  // Initialize gradients
  const dW_source: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
  const dW_target: number[][] = Array.from({ length: headDim }, () => Array(embDim).fill(0));
  const da_attention: number[] = Array(2 * headDim).fill(0);
  const dSource: number[][] = Array.from({ length: numSource }, () => Array(embDim).fill(0));
  const dTarget: number[][] = Array.from({ length: numTarget }, () => Array(embDim).fill(0));

  // Intermediate gradients
  const dSourceProj: number[][] = Array.from({ length: numSource }, () => Array(headDim).fill(0));
  const dTargetProj: number[][] = Array.from({ length: numTarget }, () => Array(headDim).fill(0));

  // Step 1: Through ELU activation
  const dAggregated: number[][] = [];
  for (let t = 0; t < numTarget; t++) {
    const dAgg = dOutput[t].map((grad, d) => {
      const x = aggregated[t][d];
      const eluDeriv = x >= 0 ? 1 : Math.exp(x);
      return grad * eluDeriv;
    });
    dAggregated.push(dAgg);
  }

  // Step 2: Through aggregation (sparse)
  const dAttention = new Map<number, number>();

  for (const [t, sources] of neighborMap) {
    for (const s of sources) {
      const key = edgeKey(s, t, numTargets);
      const alpha = attention.get(key) ?? 0;
      if (alpha > 0) {
        dAttention.set(key, math.dot(dAggregated[t], sourceProj[s]));

        for (let d = 0; d < headDim; d++) {
          dSourceProj[s][d] += alpha * dAggregated[t][d];
        }
      }
    }
  }

  // Step 3: Through softmax (per target)
  const dScore = new Map<number, number>();

  for (const [t, sources] of neighborMap) {
    if (sources.length === 0) continue;

    let sumAttnDAttn = 0;
    for (const s of sources) {
      const key = edgeKey(s, t, numTargets);
      sumAttnDAttn += (attention.get(key) ?? 0) * (dAttention.get(key) ?? 0);
    }

    for (const s of sources) {
      const key = edgeKey(s, t, numTargets);
      const alpha = attention.get(key) ?? 0;
      dScore.set(key, alpha * ((dAttention.get(key) ?? 0) - sumAttnDAttn));
    }
  }

  // Step 4-5: Through attention computation and LeakyReLU
  for (const [t, sources] of neighborMap) {
    for (const s of sources) {
      const key = edgeKey(s, t, numTargets);
      const score_grad = dScore.get(key) ?? 0;

      for (let d = 0; d < headDim; d++) {
        const x = sourceProj[s][d];
        da_attention[d] += math.leakyRelu(x, leakyReluSlope) * score_grad;
        const leakyDeriv = x > 0 ? 1 : leakyReluSlope;
        dSourceProj[s][d] += score_grad * params.a_attention[d] * leakyDeriv;
      }
      for (let d = 0; d < headDim; d++) {
        const x = targetProj[t][d];
        da_attention[headDim + d] += math.leakyRelu(x, leakyReluSlope) * score_grad;
        const leakyDeriv = x > 0 ? 1 : leakyReluSlope;
        dTargetProj[t][d] += score_grad * params.a_attention[headDim + d] * leakyDeriv;
      }
    }
  }

  // Step 6: Through projection matrices (BLAS-accelerated)
  const dW_source_contrib = math.matmul(math.transpose(dSourceProj), source);
  for (let i = 0; i < headDim; i++) {
    for (let j = 0; j < embDim; j++) {
      dW_source[i][j] += dW_source_contrib[i]?.[j] ?? 0;
    }
  }

  const dW_target_contrib = math.matmul(math.transpose(dTargetProj), target);
  for (let i = 0; i < headDim; i++) {
    for (let j = 0; j < embDim; j++) {
      dW_target[i][j] += dW_target_contrib[i]?.[j] ?? 0;
    }
  }

  const dSource_contrib = math.matmul(dSourceProj, params.W_source);
  for (let i = 0; i < numSource; i++) {
    for (let j = 0; j < embDim; j++) {
      dSource[i][j] += dSource_contrib[i]?.[j] ?? 0;
    }
  }

  const dTarget_contrib = math.matmul(dTargetProj, params.W_target);
  for (let i = 0; i < numTarget; i++) {
    for (let j = 0; j < embDim; j++) {
      dTarget[i][j] += dTarget_contrib[i]?.[j] ?? 0;
    }
  }

  return { dW_source, dW_target, da_attention, dSource, dTarget };
}
