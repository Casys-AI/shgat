/**
 * SHGAT Forward Pass Helpers
 *
 * Extracted helper functions for building incidence matrices
 * and converting between data formats during forward pass.
 *
 * @module shgat/core/forward-helpers
 */

import type { ForwardCache, LevelParams, SHGATConfig } from "./types.ts";
import type { GraphBuilder, HierarchyResult, MultiLevelIncidence } from "../graph/mod.ts";
import type { MultiLevelOrchestrator } from "../message-passing/index.ts";

// ==========================================================================
// Context Interface
// ==========================================================================

/**
 * Context required for forward pass helpers
 */
export interface ForwardContext {
  config: SHGATConfig;
  graphBuilder: GraphBuilder;
  hierarchy: HierarchyResult | null;
  multiLevelIncidence: MultiLevelIncidence | null;
}

/**
 * Extended context for full forward pass
 */
export interface ForwardPassContext extends ForwardContext {
  levelParams: Map<number, LevelParams>;
  orchestrator: MultiLevelOrchestrator;
  /** Learnable per-level residual logits (sigmoid → α) */
  residualLogits?: number[];
}

/**
 * Cache for backward pass of per-level residuals
 */
export interface ResidualCache {
  /** Original embeddings before message passing */
  E_original: number[][];
  H_original: number[][];
  /** Propagated embeddings after message passing (before residual) */
  E_propagated: number[][];
  H_propagated: number[][];
  /** Hierarchy level for each embedding */
  E_levels: number[];
  H_levels: number[];
  /** Computed alphas for each level (sigmoid of logits) */
  alphas: number[];
  /** Original logits (for gradient computation) */
  logits: number[];
}

// ==========================================================================
// Incidence Matrix Builders
// ==========================================================================

/**
 * Build tool → capability incidence matrix from MultiLevelIncidence
 *
 * Returns matrix A[tool_idx][cap_idx] = 1 if tool is directly in capability
 * Only for level-0 mappings (no transitive closure)
 */
export function buildToolToCapMatrix(ctx: ForwardContext): number[][] {
  const numTools = ctx.graphBuilder.getToolCount();
  const capsAtLevel0 = ctx.hierarchy?.hierarchyLevels.get(0) ?? new Set<string>();
  const numCapsLevel0 = capsAtLevel0.size;

  if (numTools === 0 || numCapsLevel0 === 0) return [];

  // Build cap index for level 0
  const capIndex = new Map<string, number>();
  let idx = 0;
  for (const capId of capsAtLevel0) {
    capIndex.set(capId, idx++);
  }

  const matrix: number[][] = Array.from({ length: numTools }, () => Array(numCapsLevel0).fill(0));

  for (const [toolId, caps] of ctx.multiLevelIncidence!.toolToCapIncidence) {
    const tIdx = ctx.graphBuilder.getToolIndex(toolId);
    if (tIdx === undefined) continue;

    for (const capId of caps) {
      const cIdx = capIndex.get(capId);
      if (cIdx !== undefined) {
        matrix[tIdx][cIdx] = 1;
      }
    }
  }

  return matrix;
}

/**
 * Build capability → capability incidence matrices for each level
 *
 * For level k: matrix A[child_idx][parent_idx] = 1 if child is directly in parent
 */
export function buildCapToCapMatrices(ctx: ForwardContext): Map<number, number[][]> {
  const matrices = new Map<number, number[][]>();
  if (!ctx.hierarchy || !ctx.multiLevelIncidence) return matrices;

  for (let level = 1; level <= ctx.hierarchy.maxHierarchyLevel; level++) {
    const childLevel = level - 1;
    const capsAtChildLevel = ctx.hierarchy.hierarchyLevels.get(childLevel) ?? new Set<string>();
    const capsAtParentLevel = ctx.hierarchy.hierarchyLevels.get(level) ?? new Set<string>();

    if (capsAtChildLevel.size === 0 || capsAtParentLevel.size === 0) continue;

    // Build indices
    const childIndex = new Map<string, number>();
    let idx = 0;
    for (const capId of capsAtChildLevel) childIndex.set(capId, idx++);

    const parentIndex = new Map<string, number>();
    idx = 0;
    for (const capId of capsAtParentLevel) parentIndex.set(capId, idx++);

    // Build matrix [numChildren][numParents]
    const matrix: number[][] = Array.from(
      { length: capsAtChildLevel.size },
      () => Array(capsAtParentLevel.size).fill(0),
    );

    const levelMap = ctx.multiLevelIncidence.capToCapIncidence.get(level);
    if (levelMap) {
      for (const [childId, parents] of levelMap) {
        const cIdx = childIndex.get(childId);
        if (cIdx === undefined) continue;

        for (const parentId of parents) {
          const pIdx = parentIndex.get(parentId);
          if (pIdx !== undefined) {
            matrix[cIdx][pIdx] = 1;
          }
        }
      }
    }

    matrices.set(level, matrix);
  }

  return matrices;
}

// ==========================================================================
// Embedding Flattening
// ==========================================================================

/**
 * Flatten multi-level embeddings to match graphBuilder capability order
 */
export function flattenEmbeddingsByCapabilityOrder(
  ctx: ForwardContext,
  E_levels: Map<number, number[][]>,
): number[][] {
  const capabilityNodes = ctx.graphBuilder.getCapabilityNodes();
  const result: number[][] = [];

  // Create a map from capId to embedding
  const embeddingMap = new Map<string, number[]>();

  for (let level = 0; level <= (ctx.hierarchy?.maxHierarchyLevel ?? 0); level++) {
    const capsAtLevel = ctx.hierarchy?.hierarchyLevels.get(level) ?? new Set<string>();
    const embeddings = E_levels.get(level) ?? [];

    let idx = 0;
    for (const capId of capsAtLevel) {
      if (idx < embeddings.length) {
        embeddingMap.set(capId, embeddings[idx]);
      }
      idx++;
    }
  }

  // Return in graphBuilder order
  for (const [capId] of capabilityNodes) {
    const emb = embeddingMap.get(capId);
    if (emb) {
      result.push(emb);
    } else {
      // Fallback: use zero embedding with correct dimension
      const dim = E_levels.get(0)?.[0]?.length ?? ctx.config.hiddenDim;
      result.push(new Array(dim).fill(0));
    }
  }

  return result;
}

// ==========================================================================
// Index Mapping
// ==========================================================================

/**
 * Build mapping from flat capability index to (level, withinLevelIndex)
 *
 * Used by training to route dCapEmbedding gradients to correct level.
 */
export function buildCapIndexToLevelMap(
  ctx: ForwardContext,
): Map<number, { level: number; withinLevelIdx: number }> {
  const mapping = new Map<number, { level: number; withinLevelIdx: number }>();
  const capabilityNodes = ctx.graphBuilder.getCapabilityNodes();

  if (!ctx.hierarchy) return mapping;

  // Build capId → (level, withinLevelIdx)
  const capIdToLevel = new Map<string, { level: number; withinLevelIdx: number }>();
  for (let level = 0; level <= ctx.hierarchy.maxHierarchyLevel; level++) {
    const capsAtLevel = ctx.hierarchy.hierarchyLevels.get(level) ?? new Set<string>();
    let withinLevelIdx = 0;
    for (const capId of capsAtLevel) {
      capIdToLevel.set(capId, { level, withinLevelIdx });
      withinLevelIdx++;
    }
  }

  // Map flat index to (level, withinLevelIdx)
  let flatIdx = 0;
  for (const [capId] of capabilityNodes) {
    const levelInfo = capIdToLevel.get(capId);
    if (levelInfo) {
      mapping.set(flatIdx, levelInfo);
    }
    flatIdx++;
  }

  return mapping;
}

// ==========================================================================
// Attention Format Conversion
// ==========================================================================

/**
 * Convert multi-level attention format to layer-based format for training
 *
 * Multi-level: Map<level, [head][source][target]>
 * Layer-based: [layer][head][source][target]
 *
 * Training backward pass expects attention matrices per layer.
 * We replicate the multi-level attention to fill all layers.
 */
export function convertAttentionToLayerFormat(
  config: SHGATConfig,
  multiLevelAttention: Map<number, number[][][]>,
  numSources: number,
  numTargets: number,
): number[][][][] {
  const numLayers = config.numLayers;
  const numHeads = config.numHeads;
  const result: number[][][][] = [];

  // Initialize empty attention matrices for each layer
  for (let l = 0; l < numLayers; l++) {
    const layerAttention: number[][][] = [];
    for (let h = 0; h < numHeads; h++) {
      const headMatrix: number[][] = Array.from(
        { length: numSources },
        () => Array(numTargets).fill(0),
      );
      layerAttention.push(headMatrix);
    }
    result.push(layerAttention);
  }

  // Fill with multi-level attention data
  // Strategy: distribute levels across layers
  const levels = Array.from(multiLevelAttention.keys()).sort((a, b) => a - b);
  if (levels.length === 0) return result;

  for (let l = 0; l < numLayers; l++) {
    // Map layer index to level (round-robin if more layers than levels)
    const levelIdx = Math.min(l, levels.length - 1);
    const level = levels[levelIdx];
    const levelAttention = multiLevelAttention.get(level);

    if (!levelAttention) continue;

    // Copy attention weights
    for (let h = 0; h < Math.min(numHeads, levelAttention.length); h++) {
      const srcMatrix = levelAttention[h];
      if (!srcMatrix) continue;

      for (let s = 0; s < Math.min(numSources, srcMatrix.length); s++) {
        const srcRow = srcMatrix[s];
        if (!srcRow) continue;

        for (let t = 0; t < Math.min(numTargets, srcRow.length); t++) {
          result[l][h][s][t] = srcRow[t];
        }
      }
    }
  }

  return result;
}

// ==========================================================================
// Forward Pass Core
// ==========================================================================

/**
 * Execute multi-level message passing (core logic)
 *
 * n-SHG Architecture:
 * - Upward: V → E^0 → E^1 → ... → E^L_max
 * - Downward: E^L_max → ... → E^1 → E^0 → V
 */
export function forwardCore(
  ctx: ForwardPassContext,
): { H: number[][]; E: number[][]; cache: ForwardCache; residualCache: ResidualCache | null } {
  const H_init = ctx.graphBuilder.getToolEmbeddings();
  const capabilityNodes = ctx.graphBuilder.getCapabilityNodes();

  // Handle empty graph
  if (capabilityNodes.size === 0 || !ctx.hierarchy || !ctx.multiLevelIncidence) {
    return {
      H: H_init,
      E: [],
      cache: { H: [H_init], E: [[]], attentionVE: [], attentionEV: [] },
      residualCache: null,
    };
  }

  // Build E_levels_init: initial embeddings grouped by level
  const E_levels_init = new Map<number, number[][]>();
  for (let level = 0; level <= ctx.hierarchy.maxHierarchyLevel; level++) {
    const capsAtLevel = ctx.hierarchy.hierarchyLevels.get(level) ?? new Set<string>();
    const embeddings: number[][] = [];
    for (const capId of capsAtLevel) {
      const cap = capabilityNodes.get(capId);
      if (cap) embeddings.push([...cap.embedding]);
    }
    if (embeddings.length > 0) {
      E_levels_init.set(level, embeddings);
    }
  }

  // Build incidence matrices
  const toolToCapMatrix = buildToolToCapMatrix(ctx);
  const capToCapMatrices = buildCapToCapMatrices(ctx);

  // Execute multi-level forward pass
  const { result, cache: multiCache } = ctx.orchestrator.forwardMultiLevel(
    H_init,
    E_levels_init,
    toolToCapMatrix,
    capToCapMatrices,
    ctx.levelParams,
    {
      numHeads: ctx.config.numHeads,
      numLayers: ctx.config.numLayers,
      dropout: ctx.config.dropout,
      leakyReluSlope: ctx.config.leakyReluSlope,
      downwardResidual: ctx.config.downwardResidual,
    },
  );

  // Flatten E for backward compatibility
  let E_flat = flattenEmbeddingsByCapabilityOrder(ctx, result.E);
  let H_final = result.H;

  // Store propagated embeddings BEFORE residual for backward pass
  const E_propagated = E_flat.map(row => [...row]);
  const H_propagated = H_final.map(row => [...row]);

  // Cache for backward pass of learnable residuals
  let residualCache: ResidualCache | null = null;

  // PreserveDim: add residual connection to ORIGINAL embeddings
  if (ctx.config.preserveDim) {
    const E_original = ctx.graphBuilder.getCapabilityEmbeddings();
    const H_original = ctx.graphBuilder.getToolEmbeddings();
    const defaultResidual = ctx.config.preserveDimResidual ?? 0.3;

    // Check for learnable per-level residuals (from params.residualLogits)
    const maxLevel = ctx.hierarchy?.maxHierarchyLevel ?? 0;
    if (ctx.residualLogits && ctx.residualLogits.length > 0 && maxLevel > 0) {
      // Convert logits to alphas: α = sigmoid(logit)
      // Only use levels up to maxLevel in the hierarchy
      const logits = ctx.residualLogits.slice(0, maxLevel + 1);
      const perLevelResiduals = logits.map(logit =>
        1 / (1 + Math.exp(-Math.max(-500, Math.min(500, logit))))
      );

      const E_levels = ctx.graphBuilder.getCapabilityLevels();
      const H_levels = ctx.graphBuilder.getToolLevels();

      // Store cache for backward pass
      residualCache = {
        E_original,
        H_original,
        E_propagated,
        H_propagated,
        E_levels,
        H_levels,
        alphas: perLevelResiduals,
        logits,
      };

      E_flat = applyResidualConnectionPerLevel(E_flat, E_original, E_levels, perLevelResiduals, defaultResidual);
      H_final = applyResidualConnectionPerLevel(H_final, H_original, H_levels, perLevelResiduals, defaultResidual);
    } else if (ctx.config.preserveDimResiduals && ctx.config.preserveDimResiduals.length > 0) {
      // Fixed per-level residuals from config (non-learnable)
      const E_levels = ctx.graphBuilder.getCapabilityLevels();
      const H_levels = ctx.graphBuilder.getToolLevels();

      E_flat = applyResidualConnectionPerLevel(E_flat, E_original, E_levels, ctx.config.preserveDimResiduals, defaultResidual);
      H_final = applyResidualConnectionPerLevel(H_final, H_original, H_levels, ctx.config.preserveDimResiduals, defaultResidual);
    } else {
      // Single residual value for all nodes
      E_flat = applyResidualConnection(E_flat, E_original, defaultResidual);
      H_final = applyResidualConnection(H_final, H_original, defaultResidual);
    }
  }

  // Build cache for training backward compatibility
  const cache = buildForwardCache(
    ctx,
    H_init,
    H_final,
    E_flat,
    multiCache.attentionUpward,
    multiCache.attentionDownward,
  );

  return { H: H_final, E: E_flat, cache, residualCache };
}

/**
 * Backward pass for learnable per-level residual logits
 *
 * Computes gradients for residualLogits based on the loss gradient.
 * Formula: d(loss)/d(logit) = d(loss)/d(α) * d(α)/d(logit)
 * Where d(α)/d(logit) = sigmoid'(logit) = α * (1 - α)
 *
 * @param dE - Gradient of loss w.r.t. capability embeddings
 * @param dH - Gradient of loss w.r.t. tool embeddings
 * @param cache - ResidualCache from forward pass
 * @param lr - Learning rate
 * @returns Updated logits (mutates in place and returns)
 */
export function backwardResidualLogits(
  dE: number[][] | null,
  dH: number[][] | null,
  cache: ResidualCache,
  residualLogits: number[],
  lr: number,
): void {
  const dLogits = new Array(cache.logits.length).fill(0);

  // Accumulate gradients from capability embeddings
  if (dE) {
    for (let i = 0; i < dE.length && i < cache.E_propagated.length; i++) {
      const level = cache.E_levels[i] ?? 0;
      if (level >= cache.alphas.length) continue;

      const alpha = cache.alphas[level];
      const orig = cache.E_original[i];
      const prop = cache.E_propagated[i];
      const dOut = dE[i];

      if (!orig || !prop || !dOut) continue;

      // d(loss)/d(alpha) = sum over dims of dOut * (original - propagated)
      let dAlpha = 0;
      for (let d = 0; d < dOut.length; d++) {
        dAlpha += dOut[d] * ((orig[d] ?? 0) - (prop[d] ?? 0));
      }

      // Chain rule: d(alpha)/d(logit) = alpha * (1 - alpha)
      dLogits[level] += dAlpha * alpha * (1 - alpha);
    }
  }

  // Accumulate gradients from tool embeddings
  if (dH) {
    for (let i = 0; i < dH.length && i < cache.H_propagated.length; i++) {
      const level = cache.H_levels[i] ?? 0;
      if (level >= cache.alphas.length) continue;

      const alpha = cache.alphas[level];
      const orig = cache.H_original[i];
      const prop = cache.H_propagated[i];
      const dOut = dH[i];

      if (!orig || !prop || !dOut) continue;

      let dAlpha = 0;
      for (let d = 0; d < dOut.length; d++) {
        dAlpha += dOut[d] * ((orig[d] ?? 0) - (prop[d] ?? 0));
      }

      dLogits[level] += dAlpha * alpha * (1 - alpha);
    }
  }

  // Update logits with gradient descent
  for (let l = 0; l < residualLogits.length && l < dLogits.length; l++) {
    // Clip gradient to prevent explosion
    const clippedGrad = Math.max(-1.0, Math.min(1.0, dLogits[l]));
    residualLogits[l] -= lr * clippedGrad;
  }
}

/**
 * Apply residual connection with normalization
 */
export function applyResidualConnection(
  propagated: number[][],
  original: number[][],
  residual: number,
): number[][] {
  return propagated.map((vec, idx) => {
    const orig = original[idx];
    if (!orig || orig.length !== vec.length) return vec;

    const mixed = vec.map((v, i) => (1 - residual) * v + residual * orig[i]);
    const norm = Math.sqrt(mixed.reduce((s, x) => s + x * x, 0));
    return norm > 0 ? mixed.map((x) => x / norm) : mixed;
  });
}

/**
 * Apply per-level residual connection with normalization
 * Each node uses a residual value based on its hierarchy level.
 *
 * @param propagated - Propagated embeddings from message passing
 * @param original - Original (input) embeddings
 * @param levels - Hierarchy level for each embedding (same order as propagated/original)
 * @param perLevelResiduals - Residual values indexed by level [L0, L1, L2, ...]
 * @param defaultResidual - Fallback residual for levels not in perLevelResiduals
 */
export function applyResidualConnectionPerLevel(
  propagated: number[][],
  original: number[][],
  levels: number[],
  perLevelResiduals: number[],
  defaultResidual: number,
): number[][] {
  return propagated.map((vec, idx) => {
    const orig = original[idx];
    if (!orig || orig.length !== vec.length) return vec;

    // Get residual for this node's level
    const level = levels[idx] ?? 0;
    const residual = perLevelResiduals[level] ?? defaultResidual;

    const mixed = vec.map((v, i) => (1 - residual) * v + residual * orig[i]);
    const norm = Math.sqrt(mixed.reduce((s, x) => s + x * x, 0));
    return norm > 0 ? mixed.map((x) => x / norm) : mixed;
  });
}

/**
 * Build ForwardCache from multi-level results
 */
function buildForwardCache(
  ctx: ForwardPassContext,
  H_init: number[][],
  H_final: number[][],
  E_flat: number[][],
  attentionUpward: Map<number, number[][][]>,
  attentionDownward: Map<number, number[][][]>,
): ForwardCache {
  const E_init = ctx.graphBuilder.getCapabilityEmbeddings();
  const numLayers = ctx.config.numLayers;

  // Interpolate intermediate layers
  const H_layers: number[][][] = [H_init];
  const E_layers: number[][][] = [E_init];

  for (let l = 1; l < numLayers; l++) {
    const alpha = l / numLayers;
    H_layers.push(
      H_init.map((row, i) =>
        row.map((v, j) => v * (1 - alpha) + (H_final[i]?.[j] ?? v) * alpha)
      ),
    );
    E_layers.push(
      E_init.map((row, i) =>
        row.map((v, j) => v * (1 - alpha) + (E_flat[i]?.[j] ?? v) * alpha)
      ),
    );
  }
  H_layers.push(H_final);
  E_layers.push(E_flat);

  // Convert attention format
  const attentionVE = convertAttentionToLayerFormat(
    ctx.config,
    attentionUpward,
    ctx.graphBuilder.getToolCount(),
    ctx.graphBuilder.getCapabilityCount(),
  );
  const attentionEV = convertAttentionToLayerFormat(
    ctx.config,
    attentionDownward,
    ctx.graphBuilder.getCapabilityCount(),
    ctx.graphBuilder.getToolCount(),
  );

  return { H: H_layers, E: E_layers, attentionVE, attentionEV };
}
