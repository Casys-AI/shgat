/**
 * SHGAT Scoring Helpers
 *
 * Extracted helper functions for scoring operations.
 *
 * @module shgat/core/scoring-helpers
 */

import type { AttentionResult, ForwardCache, SHGATConfig } from "./types.ts";
import type { GraphBuilder } from "../graph/mod.ts";

// ==========================================================================
// Context Interface
// ==========================================================================

/**
 * Context required for scoring helpers
 */
export interface ScoringContext {
  config: SHGATConfig;
  graphBuilder: GraphBuilder;
  lastCache: ForwardCache | null;
  scoreAllCapabilities(intentEmbedding: number[]): AttentionResult[];
  scoreAllTools(intentEmbedding: number[]): Array<{ toolId: string; score: number; headScores: number[] }>;
}

// ==========================================================================
// Capability Tool Attention
// ==========================================================================

/**
 * Get attention weights from tools to a specific capability
 *
 * Returns the average attention weight across all heads from each tool
 * to the specified capability index.
 */
export function getCapabilityToolAttention(
  ctx: ScoringContext,
  capIdx: number,
): number[] {
  if (!ctx.lastCache || ctx.lastCache.attentionVE.length === 0) return [];

  const lastLayerVE = ctx.lastCache.attentionVE[ctx.config.numLayers - 1];
  const toolCount = ctx.graphBuilder.getToolCount();

  return Array.from({ length: toolCount }, (_, t) => {
    let avg = 0;
    for (let h = 0; h < ctx.config.numHeads; h++) {
      avg += lastLayerVE[h][t][capIdx];
    }
    return avg / ctx.config.numHeads;
  });
}

// ==========================================================================
// Path Success Prediction
// ==========================================================================

/**
 * Predict success probability for a path of tools/capabilities
 *
 * Uses position-weighted scoring where later items in the path
 * contribute more to the final score.
 *
 * @param ctx - Scoring context
 * @param intentEmbedding - User intent embedding
 * @param path - Array of tool or capability IDs
 * @returns Weighted average score in [0, 1]
 */
export function predictPathSuccess(
  ctx: ScoringContext,
  intentEmbedding: number[],
  path: string[],
): number {
  const capabilityNodes = ctx.graphBuilder.getCapabilityNodes();
  const toolNodes = ctx.graphBuilder.getToolNodes();

  if (capabilityNodes.size === 0 && toolNodes.size === 0) return 0.5;
  if (!path || path.length === 0) return 0.5;

  const toolScoresMap = new Map<string, number>();
  const capScoresMap = new Map<string, number>();

  // Only score relevant node types
  if (path.some((id) => toolNodes.has(id))) {
    for (const r of ctx.scoreAllTools(intentEmbedding)) {
      toolScoresMap.set(r.toolId, r.score);
    }
  }
  if (path.some((id) => capabilityNodes.has(id))) {
    for (const r of ctx.scoreAllCapabilities(intentEmbedding)) {
      capScoresMap.set(r.capabilityId, r.score);
    }
  }

  // Position-weighted average
  let weightedSum = 0;
  let weightTotal = 0;
  for (let i = 0; i < path.length; i++) {
    const weight = 1 + i * 0.5;
    const score = toolScoresMap.get(path[i]) ?? capScoresMap.get(path[i]) ?? 0.5;
    weightedSum += score * weight;
    weightTotal += weight;
  }

  return weightedSum / weightTotal;
}

// ==========================================================================
// Single Capability Attention
// ==========================================================================

/**
 * Compute attention for a single capability
 *
 * This is a convenience method that scores all capabilities and
 * returns the result for the specified one.
 *
 * @param ctx - Scoring context
 * @param intentEmbedding - User intent embedding
 * @param capabilityId - ID of the capability to score
 * @returns AttentionResult for the capability, or default if not found
 */
export function computeAttentionForCapability(
  ctx: ScoringContext,
  intentEmbedding: number[],
  capabilityId: string,
): AttentionResult {
  const results = ctx.scoreAllCapabilities(intentEmbedding);
  return (
    results.find((r) => r.capabilityId === capabilityId) || {
      capabilityId,
      score: 0,
      headWeights: new Array(ctx.config.numHeads).fill(0),
      headScores: new Array(ctx.config.numHeads).fill(0),
      recursiveContribution: 0,
    }
  );
}
