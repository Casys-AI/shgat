/**
 * SHGAT Scoring Module
 *
 * Unified interface for SHGAT scoring architectures:
 * - v1: K-head attention with multi-level message passing
 * - Multi-level: n-SuperHyperGraph with hierarchical message passing
 *
 * @module graphrag/algorithms/shgat/scoring
 */

export { V1Scorer } from "./v1-scorer.ts";
export {
  type MultiLevelForwardResult,
  MultiLevelScorer,
  type MultiLevelScorerDependencies,
} from "./multi-level-scorer.ts";

// K-head scoring functions (extracted from shgat.ts)
export {
  batchComputeKForAllHeads,
  batchComputeScores,
  computeHeadScoreV1,
  computeMultiHeadScoresWithPrecomputedQ,
  precomputeQForAllHeads,
  predictPathSuccess,
  projectIntent,
  scoreAllCapabilities as scoreAllCapabilitiesKHead,
  scoreAllTools as scoreAllToolsKHead,
  // Unified Node API
  scoreNodes,
  type NodeScore,
} from "./khead-scorer.ts";

// NOTE: Tensor-native scoring is now integrated directly into scoreNodes()

// Re-export types from shgat-types for convenience
export type {
  AttentionResult,
  CapabilityNode,
  FeatureWeights,
  FusionWeights,
  HypergraphFeatures,
  SHGATConfig,
  ToolGraphFeatures,
  ToolNode,
  TraceFeatures,
  TraceStats,
} from "../core/types.ts";

export {
  DEFAULT_FEATURE_WEIGHTS,
  DEFAULT_FUSION_WEIGHTS,
  DEFAULT_HYPERGRAPH_FEATURES,
  DEFAULT_SHGAT_CONFIG,
  DEFAULT_TOOL_GRAPH_FEATURES,
  DEFAULT_TRACE_STATS,
  NUM_TRACE_STATS,
} from "../core/types.ts";

/**
 * Common interface for all SHGAT scorers
 *
 * Allows swapping between v1, v2, v3 implementations while maintaining
 * consistent scoring API.
 */
export interface ScorerInterface {
  /**
   * Score all capabilities given intent embedding
   *
   * @param intentEmbedding User intent embedding (1024-dim BGE-M3)
   * @returns Array of capability scores sorted descending
   */
  scoreAllCapabilities(intentEmbedding: number[]): AttentionResult[];

  /**
   * Score all tools given intent embedding
   *
   * @param intentEmbedding User intent embedding (1024-dim BGE-M3)
   * @returns Array of tool scores sorted descending
   */
  scoreAllTools(
    intentEmbedding: number[],
  ): Array<{ toolId: string; score: number; headScores?: number[] }>;
}

/**
 * Scorer version enum
 */
export enum ScorerVersion {
  /** K-head attention with multi-level message passing */
  V1 = "v1",
}

// Import AttentionResult for interface definition
import type { AttentionResult } from "../core/types.ts";
