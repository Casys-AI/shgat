/**
 * SHGAT Multi-Level Scorer
 *
 * Scoring API for n-SuperHyperGraph with multi-level message passing.
 * Uses propagated embeddings from all hierarchy levels.
 *
 * Key differences from V1Scorer:
 * - Uses `forwardMultiLevel()` instead of `forward()`
 * - Iterates over hierarchy levels
 * - Adds `hierarchyLevel` to AttentionResult
 * - Supports `targetLevel` filter parameter
 *
 * @module graphrag/algorithms/shgat/scoring/multi-level-scorer
 * @since v1 refactor
 * @see 06-scoring-api.md
 */

import * as math from "../utils/math.ts";
import type {
  AttentionResult,
  CapabilityNode,
  FeatureWeights,
  FusionWeights,
  HypergraphFeatures,
  SHGATConfig,
} from "../core/types.ts";
import { DEFAULT_HYPERGRAPH_FEATURES } from "../core/types.ts";

/**
 * Multi-level forward pass result
 */
export interface MultiLevelForwardResult {
  /** Tool embeddings [numTools][embeddingDim] */
  H: number[][];
  /** Capability embeddings by level: level → [numCapsAtLevel][embDim] */
  E: Map<number, number[][]>;
}

/**
 * Dependencies required by MultiLevelScorer
 */
export interface MultiLevelScorerDependencies {
  /** SHGAT configuration */
  config: SHGATConfig;

  /** Capability nodes map */
  capabilityNodes: Map<string, CapabilityNode>;

  /** Hierarchy levels: level → Set of capability IDs at that level */
  hierarchyLevels: Map<number, Set<string>>;

  /** Feature weights (learnable) */
  featureWeights: FeatureWeights;

  /** Fusion weights (learnable) */
  fusionWeights: FusionWeights;

  /** Multi-level forward pass function */
  forwardMultiLevel: () => MultiLevelForwardResult;

  /** Project intent to propagated embedding space */
  projectIntent: (intent: number[]) => number[];
}

/**
 * Multi-Level Scorer for n-SuperHyperGraph
 *
 * Uses multi-level message passing for propagated embeddings and
 * scores capabilities across all hierarchy levels.
 */
export class MultiLevelScorer {
  constructor(private deps: MultiLevelScorerDependencies) {}

  /**
   * Compute normalized fusion weights
   */
  private computeFusionWeights(): { semantic: number; structure: number; temporal: number } {
    const { config, fusionWeights } = this.deps;

    // If fixed weights are provided in config, use them directly
    if (config.headFusionWeights) {
      const [s, st, t] = config.headFusionWeights;
      return { semantic: s, structure: st, temporal: t };
    }

    // Otherwise, use learnable weights with softmax normalization
    const raw = [fusionWeights.semantic, fusionWeights.structure, fusionWeights.temporal];
    const softmaxed = math.softmax(raw);
    return {
      semantic: softmaxed[0],
      structure: softmaxed[1],
      temporal: softmaxed[2],
    };
  }

  /**
   * Score all capabilities using multi-level forward pass
   *
   * @param intentEmbedding User intent embedding (1024-dim BGE-M3)
   * @param targetLevel Optional level filter (0, 1, 2, ...). If undefined, scores all levels.
   * @returns Array of capability scores sorted descending, with hierarchyLevel field
   */
  scoreAllCapabilities(
    intentEmbedding: number[],
    targetLevel?: number,
  ): AttentionResult[] {
    const {
      capabilityNodes,
      hierarchyLevels,
      featureWeights,
      forwardMultiLevel,
      projectIntent,
      config,
    } = this.deps;

    // Run multi-level forward pass to get propagated embeddings
    const { E } = forwardMultiLevel();

    const results: AttentionResult[] = [];

    // Compute normalized fusion weights from learnable params
    const groupWeights = this.computeFusionWeights();

    // Project intent to propagated space for semantic comparison
    const intentProjected = projectIntent(intentEmbedding);

    // Active heads for ablation (default: all 3)
    const activeHeads = config.activeHeads ?? [0, 1, 2];

    // Determine which levels to score
    const levelsToScore = targetLevel !== undefined
      ? [targetLevel]
      : Array.from(hierarchyLevels.keys()).sort((a, b) => a - b);

    for (const level of levelsToScore) {
      const capsAtLevel = hierarchyLevels.get(level);
      if (!capsAtLevel || capsAtLevel.size === 0) continue;

      const E_level = E.get(level);
      if (!E_level) continue;

      // Create ordered array of cap IDs for index lookup
      const capsArray = Array.from(capsAtLevel);

      for (let idx = 0; idx < capsArray.length; idx++) {
        const capId = capsArray[idx];
        const cap = capabilityNodes.get(capId);
        if (!cap) continue;

        // Use PROPAGATED embedding from multi-level message passing
        const capPropagatedEmb = E_level[idx];
        if (!capPropagatedEmb) continue;

        const intentSim = math.cosineSimilarity(intentProjected, capPropagatedEmb);

        // Reliability multiplier
        const reliability = cap.successRate;
        const reliabilityMult = reliability < 0.5 ? 0.5 : reliability > 0.9 ? 1.2 : 1.0;

        // Get hypergraph features
        const features: HypergraphFeatures = cap.hypergraphFeatures || DEFAULT_HYPERGRAPH_FEATURES;

        // === 3-HEAD ARCHITECTURE (all weights learnable) ===
        // Head 0: Semantic - intent similarity scaled by learned weight
        const semanticScore = intentSim * featureWeights.semantic;

        // Head 1: Structure - graph topology features scaled by learned weight
        const structureScore = (features.hypergraphPageRank + (features.adamicAdar ?? 0)) *
          featureWeights.structure;

        // Head 2: Temporal - usage patterns scaled by learned weight
        const temporalScore = (features.recency + (features.heatDiffusion ?? 0)) *
          featureWeights.temporal;

        // Store all head scores
        const allHeadScores = [semanticScore, structureScore, temporalScore];

        // === ABLATION-AWARE FUSION ===
        const activeWeights = [
          activeHeads.includes(0) ? groupWeights.semantic : 0,
          activeHeads.includes(1) ? groupWeights.structure : 0,
          activeHeads.includes(2) ? groupWeights.temporal : 0,
        ];
        const totalActiveWeight = activeWeights.reduce((a, b) => a + b, 0) || 1;

        // Weighted combination of active heads
        const baseScore = (activeWeights[0] * semanticScore +
          activeWeights[1] * structureScore +
          activeWeights[2] * temporalScore) /
          totalActiveWeight;

        // Final score with reliability (raw logit, no sigmoid - softmax applied at discover level)
        const rawScore = baseScore * reliabilityMult;
        const score = Number.isFinite(rawScore) ? rawScore : 0;

        // Compute normalized head weights for interpretability
        const headWeights = [
          activeWeights[0] / totalActiveWeight,
          activeWeights[1] / totalActiveWeight,
          activeWeights[2] / totalActiveWeight,
        ];

        results.push({
          capabilityId: capId,
          score,
          headWeights,
          headScores: allHeadScores,
          recursiveContribution: 0, // TODO: compute from attention weights
          featureContributions: {
            semantic: semanticScore,
            structure: structureScore,
            temporal: temporalScore,
            reliability: reliabilityMult,
          },
          hierarchyLevel: level, // NEW: hierarchy level
        });
      }
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }

  /**
   * Score only leaf capabilities (level 0)
   *
   * Convenience method for scoring only the most specific capabilities.
   *
   * @param intentEmbedding User intent embedding
   * @returns Leaf capability scores sorted descending
   */
  scoreLeafCapabilities(intentEmbedding: number[]): AttentionResult[] {
    return this.scoreAllCapabilities(intentEmbedding, 0);
  }

  /**
   * Score only meta-capabilities (level 1+)
   *
   * Convenience method for scoring higher-level capabilities.
   *
   * @param intentEmbedding User intent embedding
   * @param level The meta level to score (default: 1)
   * @returns Meta-capability scores sorted descending
   */
  scoreMetaCapabilities(intentEmbedding: number[], level: number = 1): AttentionResult[] {
    return this.scoreAllCapabilities(intentEmbedding, level);
  }

  /**
   * Get top capabilities at each hierarchy level
   *
   * Useful for hierarchical exploration: show best at each level.
   *
   * @param intentEmbedding User intent embedding
   * @param topK Number of results per level
   * @returns Map of level → top-K capabilities at that level
   */
  getTopByLevel(
    intentEmbedding: number[],
    topK: number = 5,
  ): Map<number, AttentionResult[]> {
    const allResults = this.scoreAllCapabilities(intentEmbedding);
    const byLevel = new Map<number, AttentionResult[]>();

    for (const result of allResults) {
      const level = result.hierarchyLevel ?? 0;
      let levelResults = byLevel.get(level);
      if (!levelResults) {
        levelResults = [];
        byLevel.set(level, levelResults);
      }
      if (levelResults.length < topK) {
        levelResults.push(result);
      }
    }

    return byLevel;
  }
}
