/**
 * SHGAT Stats Helper
 *
 * Extracted helper for computing SHGAT statistics.
 *
 * @module shgat/core/stats
 */

import type { SHGATConfig } from "./types.ts";
import type { GraphBuilder } from "../graph/mod.ts";
import { countParameters } from "../initialization/index.ts";

// ==========================================================================
// Stats Interface
// ==========================================================================

/**
 * SHGAT statistics object
 */
export interface SHGATStats {
  numHeads: number;
  hiddenDim: number;
  numLayers: number;
  paramCount: number;
  v2ParamCount: number;
  registeredCapabilities: number;
  registeredTools: number;
  incidenceNonZeros: number;
  fusionWeights: { semantic: number; structure: number; temporal: number };
  mlpHiddenDim: number;
  maxContextLength: number;
}

// ==========================================================================
// Stats Context
// ==========================================================================

/**
 * Context required for computing stats
 */
export interface StatsContext {
  config: SHGATConfig;
  graphBuilder: GraphBuilder;
  getFusionWeights(): { semantic: number; structure: number; temporal: number };
}

// ==========================================================================
// Stats Computation
// ==========================================================================

/**
 * Compute SHGAT statistics
 *
 * Returns comprehensive stats about the model configuration,
 * parameter counts, and registered nodes.
 */
export function computeStats(ctx: StatsContext): SHGATStats {
  const { v1ParamCount, v2ParamCount } = countParameters(ctx.config);
  const incidenceStats = ctx.graphBuilder.getIncidenceStats();

  return {
    numHeads: ctx.config.numHeads,
    hiddenDim: ctx.config.hiddenDim,
    numLayers: ctx.config.numLayers,
    paramCount: v1ParamCount,
    v2ParamCount,
    registeredCapabilities: incidenceStats.numCapabilities,
    registeredTools: incidenceStats.numTools,
    incidenceNonZeros: incidenceStats.nonZeros,
    fusionWeights: ctx.getFusionWeights(),
    mlpHiddenDim: ctx.config.mlpHiddenDim,
    maxContextLength: ctx.config.maxContextLength,
  };
}
