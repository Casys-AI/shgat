/**
 * SHGAT Factory Functions
 *
 * Factory for creating SHGAT inference instances.
 * For training, use SHGATBuilder from builder.ts or AutogradTrainer directly.
 *
 * @module shgat-tf/core/factory
 */

import { SHGAT } from "./shgat.ts";
import type { Node, SHGATConfig } from "./types.ts";
import { buildGraph } from "./types.ts";
import { getAdaptiveHeadsByGraphSize } from "../initialization/index.ts";

// ============================================================================
// Factory Functions (Unified Node API)
// ============================================================================

/**
 * Create SHGAT from unified nodes
 *
 * This is the new recommended API that uses the unified Node type.
 *
 * @param nodes Array of nodes (leaves have children: [], composites have children: [...])
 * @param config Optional SHGAT configuration
 * @returns SHGAT instance with all nodes registered
 *
 * @example
 * ```typescript
 * const nodes: Node[] = [
 *   { id: 'tool-1', embedding: [...], children: [], level: 0 },
 *   { id: 'tool-2', embedding: [...], children: [], level: 0 },
 *   { id: 'cap-1', embedding: [...], children: ['tool-1', 'tool-2'], level: 0 },
 * ];
 * const shgat = createSHGAT(nodes);
 * ```
 */
export function createSHGAT(
  nodes: Node[],
  config?: Partial<SHGATConfig>,
): SHGAT {
  // Build graph to compute levels
  const graph = buildGraph(nodes);

  // Count leaves and composites
  let leafCount = 0;
  let compositeCount = 0;
  let maxLevel = 0;
  for (const node of graph.values()) {
    if (node.children.length === 0) {
      leafCount++;
    } else {
      compositeCount++;
    }
    if (node.level > maxLevel) {
      maxLevel = node.level;
    }
  }

  // Get embedding dimension from first node
  const embeddingDim = nodes[0]?.embedding.length || 1024;
  const preserveDim = config?.preserveDim ?? true;

  // Adaptive K based on graph size (ADR-053)
  const adaptiveConfig = getAdaptiveHeadsByGraphSize(
    leafCount,
    compositeCount,
    maxLevel,
    preserveDim,
    embeddingDim,
  );

  // Merge configs
  const mergedConfig: Partial<SHGATConfig> = {
    numHeads: adaptiveConfig.numHeads,
    hiddenDim: adaptiveConfig.hiddenDim,
    headDim: adaptiveConfig.headDim,
    ...config,
  };

  const shgat = new SHGAT(mergedConfig);

  // Register all nodes
  for (const node of graph.values()) {
    shgat.registerNode(node);
  }

  // Finalize: rebuild indices once after all nodes are registered
  shgat.finalizeNodes();

  return shgat;
}

