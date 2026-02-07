/**
 * SHGAT v1 Scorer - Legacy 3-Head Architecture
 *
 * 3-Head Architecture (simplified from original 6):
 * - Head 0: Semantic (intent similarity with propagated embeddings)
 * - Head 1: Structure (PageRank + AdamicAdar)
 * - Head 2: Temporal (recency + heatDiffusion)
 *
 * All weights are learnable:
 * - fusionWeights: how to combine heads (via softmax)
 * - featureWeights: scale factor for each head's features
 *
 * @module graphrag/algorithms/shgat/scoring/v1-scorer
 */

import * as math from "../utils/math.ts";
import type {
  AttentionResult,
  CapabilityNode,
  FeatureWeights,
  FusionWeights,
  HypergraphFeatures,
  SHGATConfig,
  ToolGraphFeatures,
  ToolNode,
} from "../core/types.ts";
import { DEFAULT_HYPERGRAPH_FEATURES } from "../core/types.ts";

/**
 * Forward pass result containing propagated embeddings
 */
export interface ForwardResult {
  /** Propagated tool embeddings [numTools][embeddingDim] */
  H: number[][];
  /** Propagated capability embeddings [numCapabilities][embeddingDim] */
  E: number[][];
}

/**
 * Dependencies required by V1Scorer
 *
 * These are injected from the parent SHGAT class to avoid circular dependencies.
 */
export interface V1ScorerDependencies {
  /** SHGAT configuration */
  config: SHGATConfig;
  /** Tool nodes map */
  toolNodes: Map<string, ToolNode>;
  /** Capability nodes map */
  capabilityNodes: Map<string, CapabilityNode>;
  /** Tool ID → index mapping */
  toolIndex: Map<string, number>;
  /** Capability ID → index mapping */
  capabilityIndex: Map<string, number>;
  /** Feature weights (learnable) */
  featureWeights: FeatureWeights;
  /** Fusion weights (learnable) */
  fusionWeights: FusionWeights;
  /** Forward pass function (message passing) */
  forward: () => ForwardResult;
  /** Project intent to propagated embedding space */
  projectIntent: (intent: number[]) => number[];
  /** Get tool attention for a capability (for interpretability) */
  getCapabilityToolAttention: (capIdx: number) => number[];
}

/**
 * V1 Scorer - Legacy 3-head architecture
 *
 * Uses message passing for propagated embeddings and 3 specialized heads
 * for semantic, structure, and temporal scoring.
 */
export class V1Scorer {
  constructor(private deps: V1ScorerDependencies) {}

  /**
   * Compute normalized fusion weights
   *
   * If headFusionWeights is provided in config, use those (for ablation).
   * Otherwise, use softmax of learnable parameters.
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
   * Score all capabilities given intent embedding
   *
   * Uses 3-head architecture with message passing for propagated embeddings.
   * Context is handled by DR-DSP pathfinding, not here.
   *
   * @param intentEmbedding User intent embedding (1024-dim BGE-M3)
   * @param _contextToolEmbeddings DEPRECATED - kept for API compat, ignored
   * @param _contextCapabilityIds DEPRECATED - kept for API compat, ignored
   * @returns Array of capability scores sorted descending
   */
  scoreAllCapabilities(
    intentEmbedding: number[],
    _contextToolEmbeddings?: number[][],
    _contextCapabilityIds?: string[],
  ): AttentionResult[] {
    const {
      capabilityNodes,
      capabilityIndex,
      featureWeights,
      forward,
      projectIntent,
      getCapabilityToolAttention,
      config,
    } = this.deps;

    // Run forward pass to get propagated embeddings via V→E→V message passing
    const { E } = forward();

    const results: AttentionResult[] = [];

    // Compute normalized fusion weights from learnable params
    const groupWeights = this.computeFusionWeights();

    // Project intent to propagated space for semantic comparison
    const intentProjected = projectIntent(intentEmbedding);

    // Active heads for ablation (default: all 3)
    const activeHeads = config.activeHeads ?? [0, 1, 2];

    for (const [capId, cap] of capabilityNodes) {
      const cIdx = capabilityIndex.get(capId)!;

      // Use PROPAGATED embedding from message passing for semantic similarity
      const capPropagatedEmb = E[cIdx];
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
      // Only include active heads
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

      // Get tool attention for interpretability
      const toolAttention = getCapabilityToolAttention(cIdx);

      results.push({
        capabilityId: capId,
        score,
        headWeights,
        headScores: allHeadScores,
        recursiveContribution: 0,
        featureContributions: {
          semantic: semanticScore,
          structure: structureScore,
          temporal: temporalScore,
          reliability: reliabilityMult,
        },
        toolAttention,
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }

  /**
   * Score all tools given intent embedding
   *
   * 3-Head architecture for tools:
   * - Head 0 (Semantic): Intent similarity with propagated embeddings
   * - Head 1 (Structure): PageRank + AdamicAdar
   * - Head 2 (Temporal): Recency + HeatDiffusion
   *
   * @param intentEmbedding The intent embedding (1024-dim BGE-M3)
   * @returns Array of tool scores sorted by score descending
   */
  scoreAllTools(
    intentEmbedding: number[],
  ): Array<{ toolId: string; score: number; headWeights?: number[] }> {
    const { toolNodes, toolIndex, featureWeights, forward, projectIntent, config } = this.deps;

    // Run forward pass to get propagated embeddings via V→E→V message passing
    const { H } = forward();

    const results: Array<{ toolId: string; score: number; headWeights?: number[] }> = [];

    // Compute normalized fusion weights from learnable params
    const groupWeights = this.computeFusionWeights();

    // Project intent to propagated space for semantic comparison
    const intentProjected = projectIntent(intentEmbedding);

    // Active heads for ablation (default: all 3)
    const activeHeads = config.activeHeads ?? [0, 1, 2];

    for (const [toolId, tool] of toolNodes) {
      const tIdx = toolIndex.get(toolId)!;

      // Use PROPAGATED embedding from message passing
      const toolPropagatedEmb = H[tIdx];
      const intentSim = math.cosineSimilarity(intentProjected, toolPropagatedEmb);

      // Get tool features (may be undefined for tools without features)
      const features: ToolGraphFeatures | undefined = tool.toolFeatures;

      if (!features) {
        // Fallback: pure semantic similarity if no features
        results.push({
          toolId,
          score: Math.max(0, Math.min(intentSim, 0.95)),
        });
        continue;
      }

      // === 3-HEAD ARCHITECTURE (all weights learnable) ===
      // Head 0: Semantic - intent similarity scaled by learned weight
      const semanticScore = intentSim * featureWeights.semantic;

      // Head 1: Structure - graph topology features scaled by learned weight
      const structureScore = (features.pageRank + features.adamicAdar) * featureWeights.structure;

      // Head 2: Temporal - usage patterns scaled by learned weight
      const temporalScore = (features.recency + features.heatDiffusion) * featureWeights.temporal;

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

      // Raw logit score (no sigmoid - softmax applied at discover level)
      const rawScore = baseScore;
      const score = Number.isFinite(rawScore) ? rawScore : 0;

      // Compute normalized head weights for interpretability
      const headWeights = [
        activeWeights[0] / totalActiveWeight,
        activeWeights[1] / totalActiveWeight,
        activeWeights[2] / totalActiveWeight,
      ];

      results.push({
        toolId,
        score,
        headWeights,
      });
    }

    results.sort((a, b) => b.score - a.score);
    return results;
  }
}
