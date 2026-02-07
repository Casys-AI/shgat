/**
 * Vertex → Vertex Message Passing Phase (V→V)
 *
 * Pre-phase for SHGAT: Tools send messages to other tools based on
 * co-occurrence patterns from scraped n8n workflows.
 *
 * This phase enriches tool embeddings with structural information
 * from real-world workflow usage patterns BEFORE the V→E phase.
 *
 * Algorithm (simplified, no projection - keeps 1024d):
 *   1. Compute attention scores: score(i,j) = H_i · H_j (cosine similarity)
 *      (masked by co-occurrence matrix: only compute for co-occurring tools)
 *   2. Weight by co-occurrence frequency: score'(i,j) = score(i,j) * A_cooc[i][j]
 *   3. Normalize per tool: α_i = softmax({score'(i,j) | j co-occurs with i})
 *   4. Aggregate: H'_i = H_i + β · Σ_j α_ij · H_j  (residual connection)
 *
 * @module graphrag/algorithms/shgat/message-passing/vertex-to-vertex-phase
 */

import * as math from "../utils/math.ts";

/**
 * Co-occurrence matrix entry
 * Sparse representation for efficiency
 */
export interface CooccurrenceEntry {
  /** Source tool index */
  from: number;
  /** Target tool index */
  to: number;
  /** Co-occurrence weight (frequency-based, normalized) */
  weight: number;
}

/**
 * V→V phase configuration
 */
export interface VertexToVertexConfig {
  /** Residual connection weight (0 = no enrichment, 1 = full) */
  residualWeight: number;
  /** Use attention-weighted aggregation vs simple weighted sum */
  useAttention: boolean;
  /** Temperature for attention softmax (lower = sharper) */
  temperature: number;
}

/**
 * Default V→V configuration
 */
export const DEFAULT_V2V_CONFIG: VertexToVertexConfig = {
  residualWeight: 0.3, // Conservative: 30% co-occurrence, 70% original
  useAttention: true,
  temperature: 1.0,
};

/**
 * V→V phase result
 */
export interface VertexToVertexResult {
  /** Enriched embeddings [numTools][embeddingDim] */
  embeddings: number[][];
  /** Attention weights (sparse) for debugging */
  attentionWeights: CooccurrenceEntry[];
}

/**
 * Learnable parameters for V→V phase
 *
 * Lightweight: just 2 scalars instead of full projection matrices
 */
export interface V2VParams {
  /** Residual weight (sigmoid-transformed for [0, 1] range) */
  residualLogit: number;
  /** Temperature logit (exp-transformed for positive range) */
  temperatureLogit: number;
}

/**
 * Default V→V learnable parameters
 *
 * Initialized to match DEFAULT_V2V_CONFIG:
 * - residualLogit = logit(0.3) ≈ -0.847
 * - temperatureLogit = log(1.0) = 0.0
 */
export const DEFAULT_V2V_PARAMS: V2VParams = {
  residualLogit: Math.log(0.3 / 0.7), // sigmoid^-1(0.3) ≈ -0.847
  temperatureLogit: 0.0, // exp(0) = 1.0
};

/**
 * Cache for V→V backward pass
 */
export interface V2VForwardCache {
  /** Original embeddings [numTools][embDim] */
  H: number[][];
  /** Aggregated neighbor embeddings [numTools][embDim] */
  aggregated: number[][];
  /** Attention weights per tool: tool_idx → [neighbor weights] */
  attentionPerTool: Map<number, number[]>;
  /** Neighbors per tool: tool_idx → [neighbor indices] */
  neighborsPerTool: Map<number, number[]>;
  /** Pre-softmax scores per tool: tool_idx → [scores] */
  scoresPerTool: Map<number, number[]>;
  /** Pre-normalization enriched embeddings (before L2 norm) */
  enrichedPreNorm: number[][];
  /** L2 norms of enriched embeddings */
  enrichedNorms: number[];
  /** Effective residual weight (after sigmoid) */
  residualWeight: number;
  /** Effective temperature (after exp) */
  temperature: number;
}

/**
 * Gradients for V→V learnable parameters
 */
export interface V2VGradients {
  /** Gradient for residualLogit */
  dResidualLogit: number;
  /** Gradient for temperatureLogit */
  dTemperatureLogit: number;
  /** Gradient for input H [numTools][embDim] */
  dH: number[][];
}

/**
 * Extended result with cache
 */
export interface V2VPhaseResultWithCache extends VertexToVertexResult {
  cache: V2VForwardCache;
}

/**
 * Vertex → Vertex message passing implementation
 *
 * Enriches tool embeddings with co-occurrence information from
 * scraped workflow patterns. Operates on full 1024d embeddings
 * without projection.
 */
export class VertexToVertexPhase {
  private config: VertexToVertexConfig;

  constructor(config: Partial<VertexToVertexConfig> = {}) {
    this.config = { ...DEFAULT_V2V_CONFIG, ...config };
  }

  getName(): string {
    return "Vertex→Vertex";
  }

  /**
   * Execute V→V message passing
   *
   * @param H - Tool embeddings [numTools][embeddingDim] (1024d)
   * @param cooccurrence - Sparse co-occurrence matrix entries
   * @param toolIds - Tool ID to index mapping for debugging
   * @returns Enriched embeddings and attention weights
   */
  forward(
    H: number[][],
    cooccurrence: CooccurrenceEntry[],
    _toolIds?: string[],
  ): VertexToVertexResult {
    const numTools = H.length;
    if (numTools === 0) {
      return { embeddings: [], attentionWeights: [] };
    }

    const embeddingDim = H[0].length;

    // Build adjacency list from sparse co-occurrence
    const neighbors: Map<number, { idx: number; weight: number }[]> = new Map();
    for (const entry of cooccurrence) {
      if (entry.from >= numTools || entry.to >= numTools) continue;

      if (!neighbors.has(entry.from)) {
        neighbors.set(entry.from, []);
      }
      neighbors.get(entry.from)!.push({ idx: entry.to, weight: entry.weight });
    }

    // Compute enriched embeddings
    const H_enriched: number[][] = [];
    const attentionWeights: CooccurrenceEntry[] = [];

    for (let i = 0; i < numTools; i++) {
      const neighborList = neighbors.get(i);

      if (!neighborList || neighborList.length === 0) {
        // No co-occurring tools, keep original embedding
        H_enriched.push([...H[i]]);
        continue;
      }

      let aggregated: number[];

      if (this.config.useAttention) {
        // Attention-weighted aggregation
        const scores: number[] = [];

        for (const neighbor of neighborList) {
          // Cosine similarity * co-occurrence weight
          const sim = math.cosineSimilarity(H[i], H[neighbor.idx]);
          scores.push((sim * neighbor.weight) / this.config.temperature);
        }

        // Softmax normalization
        const attention = math.softmax(scores);

        // Weighted sum of neighbor embeddings
        aggregated = Array(embeddingDim).fill(0);
        for (let n = 0; n < neighborList.length; n++) {
          const neighbor = neighborList[n];
          for (let d = 0; d < embeddingDim; d++) {
            aggregated[d] += attention[n] * H[neighbor.idx][d];
          }

          // Store attention weights for debugging
          attentionWeights.push({
            from: i,
            to: neighbor.idx,
            weight: attention[n],
          });
        }
      } else {
        // Simple weighted sum (normalized by total weight)
        aggregated = Array(embeddingDim).fill(0);
        let totalWeight = 0;

        for (const neighbor of neighborList) {
          totalWeight += neighbor.weight;
          for (let d = 0; d < embeddingDim; d++) {
            aggregated[d] += neighbor.weight * H[neighbor.idx][d];
          }
        }

        if (totalWeight > 0) {
          for (let d = 0; d < embeddingDim; d++) {
            aggregated[d] /= totalWeight;
          }
        }
      }

      // Residual connection: H' = H + β * aggregated
      const enriched = Array(embeddingDim);
      for (let d = 0; d < embeddingDim; d++) {
        enriched[d] = H[i][d] + this.config.residualWeight * aggregated[d];
      }

      // Optional: L2 normalize to keep embeddings on unit sphere
      const norm = Math.sqrt(enriched.reduce((sum, x) => sum + x * x, 0));
      if (norm > 0) {
        for (let d = 0; d < embeddingDim; d++) {
          enriched[d] /= norm;
        }
      }

      H_enriched.push(enriched);
    }

    return {
      embeddings: H_enriched,
      attentionWeights,
    };
  }

  /**
   * Get configuration
   */
  getConfig(): VertexToVertexConfig {
    return { ...this.config };
  }

  /**
   * Forward pass with learnable parameters and cache for backward
   *
   * Uses V2VParams instead of config for residualWeight and temperature.
   * This allows these parameters to be trained via gradient descent.
   *
   * @param H - Tool embeddings [numTools][embDim]
   * @param cooccurrence - Sparse co-occurrence entries
   * @param params - Learnable parameters (residualLogit, temperatureLogit)
   * @returns Enriched embeddings and cache for backward
   */
  forwardWithCache(
    H: number[][],
    cooccurrence: CooccurrenceEntry[],
    params: V2VParams,
  ): V2VPhaseResultWithCache {
    const numTools = H.length;
    if (numTools === 0) {
      return {
        embeddings: [],
        attentionWeights: [],
        cache: {
          H: [],
          aggregated: [],
          attentionPerTool: new Map(),
          neighborsPerTool: new Map(),
          scoresPerTool: new Map(),
          enrichedPreNorm: [],
          enrichedNorms: [],
          residualWeight: 0,
          temperature: 1,
        },
      };
    }

    const embeddingDim = H[0].length;

    // Transform learnable parameters
    const residualWeight = math.sigmoid(params.residualLogit);
    const temperature = Math.exp(params.temperatureLogit);

    // Build adjacency list
    const neighbors: Map<number, { idx: number; weight: number }[]> = new Map();
    for (const entry of cooccurrence) {
      if (entry.from >= numTools || entry.to >= numTools) continue;
      if (!neighbors.has(entry.from)) {
        neighbors.set(entry.from, []);
      }
      neighbors.get(entry.from)!.push({ idx: entry.to, weight: entry.weight });
    }

    // Cache structures
    const aggregatedAll: number[][] = [];
    const attentionPerTool = new Map<number, number[]>();
    const neighborsPerTool = new Map<number, number[]>();
    const scoresPerTool = new Map<number, number[]>();
    const enrichedPreNorm: number[][] = [];
    const enrichedNorms: number[] = [];
    const H_enriched: number[][] = [];
    const attentionWeights: CooccurrenceEntry[] = [];

    for (let i = 0; i < numTools; i++) {
      const neighborList = neighbors.get(i);

      if (!neighborList || neighborList.length === 0) {
        // No neighbors: keep original, cache zeros
        H_enriched.push([...H[i]]);
        aggregatedAll.push(Array(embeddingDim).fill(0));
        enrichedPreNorm.push([...H[i]]);
        enrichedNorms.push(1.0);
        continue;
      }

      // Store neighbor indices
      neighborsPerTool.set(i, neighborList.map((n) => n.idx));

      // Compute attention scores
      const scores: number[] = [];
      for (const neighbor of neighborList) {
        const sim = math.cosineSimilarity(H[i], H[neighbor.idx]);
        scores.push((sim * neighbor.weight) / temperature);
      }
      scoresPerTool.set(i, scores);

      // Softmax
      const attention = math.softmax(scores);
      attentionPerTool.set(i, attention);

      // Weighted sum of neighbor embeddings
      const aggregated = Array(embeddingDim).fill(0);
      for (let n = 0; n < neighborList.length; n++) {
        const neighbor = neighborList[n];
        for (let d = 0; d < embeddingDim; d++) {
          aggregated[d] += attention[n] * H[neighbor.idx][d];
        }
        attentionWeights.push({
          from: i,
          to: neighbor.idx,
          weight: attention[n],
        });
      }
      aggregatedAll.push(aggregated);

      // Residual connection: H' = H + β * aggregated
      const preNorm = Array(embeddingDim);
      for (let d = 0; d < embeddingDim; d++) {
        preNorm[d] = H[i][d] + residualWeight * aggregated[d];
      }
      enrichedPreNorm.push(preNorm);

      // L2 normalize
      const norm = Math.sqrt(preNorm.reduce((sum, x) => sum + x * x, 0));
      enrichedNorms.push(norm);

      const enriched = Array(embeddingDim);
      if (norm > 0) {
        for (let d = 0; d < embeddingDim; d++) {
          enriched[d] = preNorm[d] / norm;
        }
      } else {
        for (let d = 0; d < embeddingDim; d++) {
          enriched[d] = preNorm[d];
        }
      }
      H_enriched.push(enriched);
    }

    const cache: V2VForwardCache = {
      H: H.map((row) => [...row]),
      aggregated: aggregatedAll,
      attentionPerTool,
      neighborsPerTool,
      scoresPerTool,
      enrichedPreNorm,
      enrichedNorms,
      residualWeight,
      temperature,
    };

    return { embeddings: H_enriched, attentionWeights, cache };
  }

  /**
   * Backward pass: compute gradients for learnable parameters
   *
   * @param dH_enriched - Gradient from downstream [numTools][embDim]
   * @param cache - Forward pass cache
   * @param params - Learnable parameters (for chain rule)
   * @returns Gradients for residualLogit, temperatureLogit, and input H
   */
  backward(
    dH_enriched: number[][],
    cache: V2VForwardCache,
    _params: V2VParams,
  ): V2VGradients {
    const { H, aggregated, attentionPerTool, neighborsPerTool, scoresPerTool, enrichedPreNorm, enrichedNorms, residualWeight, temperature } = cache;
    const numTools = H.length;
    const embeddingDim = H[0]?.length ?? 0;

    // Initialize gradients
    let dResidualLogit = 0;
    let dTemperatureLogit = 0;
    const dH: number[][] = Array.from({ length: numTools }, () => Array(embeddingDim).fill(0));

    for (let i = 0; i < numTools; i++) {
      const neighborIndices = neighborsPerTool.get(i);

      if (!neighborIndices || neighborIndices.length === 0) {
        // No neighbors: gradient passes through directly
        for (let d = 0; d < embeddingDim; d++) {
          dH[i][d] += dH_enriched[i][d];
        }
        continue;
      }

      // Step 1: Through L2 normalization
      // y = x / ||x||
      // dy/dx = (I - y * y^T) / ||x||
      const norm = enrichedNorms[i];
      const preNorm = enrichedPreNorm[i];
      const dPreNorm = Array(embeddingDim).fill(0);

      if (norm > 0) {
        // Compute y = preNorm / norm (the normalized vector)
        const y = preNorm.map((v) => v / norm);

        // Compute dot(dH_enriched[i], y)
        let dotDY = 0;
        for (let d = 0; d < embeddingDim; d++) {
          dotDY += dH_enriched[i][d] * y[d];
        }

        // dPreNorm = (dH_enriched - y * dot(dH_enriched, y)) / norm
        for (let d = 0; d < embeddingDim; d++) {
          dPreNorm[d] = (dH_enriched[i][d] - y[d] * dotDY) / norm;
        }
      } else {
        // If norm is 0, just pass through
        for (let d = 0; d < embeddingDim; d++) {
          dPreNorm[d] = dH_enriched[i][d];
        }
      }

      // Step 2: Through residual connection
      // preNorm = H[i] + β * aggregated[i]
      // dH[i] += dPreNorm
      // dAggregated = β * dPreNorm
      // dβ += dot(dPreNorm, aggregated[i])
      for (let d = 0; d < embeddingDim; d++) {
        dH[i][d] += dPreNorm[d];
      }

      const agg = aggregated[i];
      const dAggregated = Array(embeddingDim).fill(0);
      for (let d = 0; d < embeddingDim; d++) {
        dAggregated[d] = residualWeight * dPreNorm[d];
      }

      // dResidualWeight += dot(dPreNorm, aggregated)
      let dResidualWeight = 0;
      for (let d = 0; d < embeddingDim; d++) {
        dResidualWeight += dPreNorm[d] * agg[d];
      }

      // Chain rule: residualWeight = sigmoid(residualLogit)
      // dResidualLogit += dResidualWeight * sigmoid'(residualLogit)
      // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = residualWeight * (1 - residualWeight)
      dResidualLogit += dResidualWeight * residualWeight * (1 - residualWeight);

      // Step 3: Through weighted aggregation
      // aggregated = Σ attention[n] * H[neighbor]
      // dAttention[n] = dot(dAggregated, H[neighbor])
      // dH[neighbor] += attention[n] * dAggregated
      const attention = attentionPerTool.get(i)!;
      const dAttention = Array(neighborIndices.length).fill(0);

      for (let n = 0; n < neighborIndices.length; n++) {
        const neighborIdx = neighborIndices[n];
        dAttention[n] = math.dot(dAggregated, H[neighborIdx]);

        for (let d = 0; d < embeddingDim; d++) {
          dH[neighborIdx][d] += attention[n] * dAggregated[d];
        }
      }

      // Step 4: Through softmax
      // softmax Jacobian: dScore[n] = attention[n] * (dAttention[n] - Σ attention * dAttention)
      let sumAttnDAttn = 0;
      for (let n = 0; n < neighborIndices.length; n++) {
        sumAttnDAttn += attention[n] * dAttention[n];
      }

      const dScore = Array(neighborIndices.length).fill(0);
      for (let n = 0; n < neighborIndices.length; n++) {
        dScore[n] = attention[n] * (dAttention[n] - sumAttnDAttn);
      }

      // Step 5: Through score computation
      // score[n] = (cosineSim * coocWeight) / temperature
      // dTemperature += dScore[n] * (-score[n] / temperature)
      const scores = scoresPerTool.get(i)!;
      for (let n = 0; n < neighborIndices.length; n++) {
        // dTemperature contribution: d(x/T)/dT = -x/T^2 = -score/T
        const dTemp = dScore[n] * (-scores[n] / temperature);

        // Chain rule: temperature = exp(temperatureLogit)
        // dTemperatureLogit += dTemp * exp(temperatureLogit) = dTemp * temperature
        dTemperatureLogit += dTemp * temperature;

        // Note: For dH through cosine similarity, the gradient is complex:
        // score = sim * w / T where sim = H[i] · H[neighbor] / (||H[i]|| * ||H[neighbor]||)
        // For now, we skip this gradient path as:
        // 1. Embeddings are typically frozen (from pre-trained model)
        // 2. The attention gradient already provides learning signal
        // 3. Co-occurrence weights aren't stored in cache for reconstruction
        // Future work could add cosine gradient for end-to-end embedding fine-tuning
      }
    }

    return { dResidualLogit, dTemperatureLogit, dH };
  }
}

/**
 * Build co-occurrence matrix from prior patterns
 *
 * Converts PriorPattern[] to sparse CooccurrenceEntry[] for V→V phase
 *
 * @param patterns - Prior patterns from n8n scraping
 * @param toolIndex - Map from tool ID to index
 * @returns Sparse co-occurrence entries
 */
export function buildCooccurrenceMatrix(
  patterns: { from: string; to: string; weight: number; frequency: number }[],
  toolIndex: Map<string, number>,
): CooccurrenceEntry[] {
  const entries: CooccurrenceEntry[] = [];
  const seen = new Set<string>(); // Deduplicate edges

  for (const pattern of patterns) {
    const fromIdx = toolIndex.get(pattern.from);
    const toIdx = toolIndex.get(pattern.to);

    if (fromIdx === undefined || toIdx === undefined) continue;
    if (fromIdx === toIdx) continue; // Skip self-loops

    // Convert weight to similarity (lower weight = higher co-occurrence)
    // PriorPattern.weight is inverse frequency, so invert it
    const coocWeight = 1.0 / (1.0 + pattern.weight);

    // Add forward edge if not already seen
    const keyFwd = `${fromIdx}:${toIdx}`;
    if (!seen.has(keyFwd)) {
      entries.push({ from: fromIdx, to: toIdx, weight: coocWeight });
      seen.add(keyFwd);
    }

    // Add backward edge if not already seen (co-occurrence is symmetric)
    const keyBwd = `${toIdx}:${fromIdx}`;
    if (!seen.has(keyBwd)) {
      entries.push({ from: toIdx, to: fromIdx, weight: coocWeight });
      seen.add(keyBwd);
    }
  }

  return entries;
}
