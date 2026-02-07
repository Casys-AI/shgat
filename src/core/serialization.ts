/**
 * SHGAT Serialization Helpers
 *
 * Extracted export/import logic for SHGAT parameters.
 *
 * @module shgat/core/serialization
 */

import type { SHGATConfig, LevelParams } from "./types.ts";
import type { SHGATParams } from "../initialization/index.ts";
import type { V2VParams } from "../message-passing/index.ts";
import {
  exportParams as exportParamsBase,
  importParams as importParamsBase,
} from "../initialization/index.ts";

// ==========================================================================
// Serialization Context
// ==========================================================================

/**
 * Context required for serialization operations
 */
export interface SerializationContext {
  config: SHGATConfig;
  params: SHGATParams;
  levelParams: Map<number, LevelParams>;
  v2vParams: V2VParams;
}

// ==========================================================================
// Export
// ==========================================================================

/**
 * Export all SHGAT parameters to a serializable object
 *
 * PreserveDim mode: skip layerParams (deprecated V1, ~512MB unused)
 * Only levelParams + headParams are used by trainBatchV1KHead()
 *
 * ADR-055: Also exports levelParams for multi-level message passing
 * ADR-057: Also exports V2V trainable params
 */
export function exportSHGATParams(ctx: SerializationContext): Record<string, unknown> {
  const base = exportParamsBase(ctx.config, ctx.params);

  // PreserveDim mode: skip layerParams (deprecated V1)
  if (ctx.config.preserveDim) {
    delete (base as Record<string, unknown>).layerParams;
  }

  // ADR-055: Export levelParams for multi-level message passing
  const levelParamsObj: Record<string, LevelParams> = {};
  for (const [level, params] of ctx.levelParams) {
    levelParamsObj[level.toString()] = params;
  }

  // ADR-057: Export V2V trainable params
  const result: Record<string, unknown> = {
    ...base,
    levelParams: levelParamsObj,
    v2vParams: { ...ctx.v2vParams },
  };

  // Projection head params are already included in base (from SHGATParams.projectionHead)
  return result;
}

// ==========================================================================
// Import
// ==========================================================================

/**
 * Import result containing updated config and params
 */
export interface ImportResult {
  config: SHGATConfig | null;
  params: SHGATParams;
  levelParams: Map<number, LevelParams>;
  v2vParams: V2VParams;
}

/**
 * Import SHGAT parameters from a serialized object
 *
 * ADR-055: Imports levelParams for multi-level message passing
 * ADR-057: Imports V2V trainable params
 */
export function importSHGATParams(
  serialized: Record<string, unknown>,
  currentParams: SHGATParams,
  currentV2VParams: V2VParams,
): ImportResult {
  const baseResult = importParamsBase(serialized, currentParams);

  // ADR-055: Import levelParams for multi-level message passing
  const levelParams = new Map<number, LevelParams>();
  if (serialized.levelParams && typeof serialized.levelParams === "object") {
    const levelParamsObj = serialized.levelParams as Record<string, LevelParams>;
    for (const [levelStr, lp] of Object.entries(levelParamsObj)) {
      levelParams.set(parseInt(levelStr), lp);
    }
  }

  // ADR-057: Import V2V trainable params
  const v2vParams = { ...currentV2VParams };
  if (serialized.v2vParams && typeof serialized.v2vParams === "object") {
    const v2v = serialized.v2vParams as V2VParams;
    if (typeof v2v.residualLogit === "number") {
      v2vParams.residualLogit = v2v.residualLogit;
    }
    if (typeof v2v.temperatureLogit === "number") {
      v2vParams.temperatureLogit = v2v.temperatureLogit;
    }
  }

  return {
    config: baseResult.config ?? null,
    params: baseResult.params,
    levelParams,
    v2vParams,
  };
}
