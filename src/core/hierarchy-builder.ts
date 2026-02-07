/**
 * SHGAT Hierarchy Builder
 *
 * Extracted helper for building multi-level hierarchy structures.
 *
 * @module shgat/core/hierarchy-builder
 */

import { getLogger } from "./logger.ts";
import type { LevelParams, SHGATConfig } from "./types.ts";
import {
  buildMultiLevelIncidence,
  computeHierarchyLevels,
  type GraphBuilder,
  type HierarchyResult,
  type MultiLevelIncidence,
} from "../graph/mod.ts";
import { initializeLevelParameters } from "../initialization/index.ts";

const log = getLogger();

// ==========================================================================
// Result Interface
// ==========================================================================

/**
 * Result of hierarchy rebuild
 */
export interface HierarchyBuildResult {
  hierarchy: HierarchyResult;
  multiLevelIncidence: MultiLevelIncidence;
  levelParams: Map<number, LevelParams>;
}

// ==========================================================================
// Empty Hierarchy
// ==========================================================================

/**
 * Create empty hierarchy structures for graphs with no capabilities
 */
export function createEmptyHierarchy(): HierarchyBuildResult {
  return {
    hierarchy: {
      hierarchyLevels: new Map(),
      maxHierarchyLevel: 0,
      capabilities: new Map(),
    },
    multiLevelIncidence: {
      toolToCapIncidence: new Map(),
      capToCapIncidence: new Map(),
      parentToChildIncidence: new Map(),
      capToToolIncidence: new Map(),
    },
    levelParams: new Map(),
  };
}

// ==========================================================================
// Hierarchy Builder
// ==========================================================================

/**
 * Rebuild multi-level hierarchy and incidence structures
 *
 * This function:
 * 1. Uses pre-computed levels from unified Node API (preferred)
 *    OR computes hierarchy levels from capability parents/children (legacy)
 * 2. Updates hierarchyLevel on each capability node
 * 3. Builds multi-level incidence structure
 * 4. Initializes level parameters if needed
 *
 * @param config - SHGAT configuration
 * @param graphBuilder - Graph builder with registered nodes
 * @param existingLevelParams - Existing level params to preserve if possible
 * @returns HierarchyBuildResult with all structures
 */
export function rebuildHierarchy(
  config: SHGATConfig,
  graphBuilder: GraphBuilder,
  existingLevelParams: Map<number, LevelParams>,
): HierarchyBuildResult {
  const capabilityNodes = graphBuilder.getCapabilityNodes();

  // Handle empty graph
  if (capabilityNodes.size === 0) {
    return createEmptyHierarchy();
  }

  // Try to use unified Node API levels first (new path)
  const unifiedNodes = graphBuilder.getNodes();
  const useUnifiedLevels = unifiedNodes.size > 0;

  let hierarchy: HierarchyResult;

  if (useUnifiedLevels) {
    // Use pre-computed levels from unified Node API
    const hierarchyLevels = graphBuilder.getNodeIdsByLevel();
    const maxHierarchyLevel = graphBuilder.getMaxLevel();

    // Include ALL nodes in hierarchy (tools at level 0, capabilities at level 1+)
    // This enables training on both tools and capabilities
    const allNodeLevels = new Map<number, Set<string>>();
    for (const [level, nodeIds] of hierarchyLevels) {
      allNodeLevels.set(level, new Set(nodeIds));
    }

    hierarchy = {
      hierarchyLevels: allNodeLevels,
      maxHierarchyLevel,
      capabilities: capabilityNodes,
    };

    // Update hierarchyLevel on each capability from unified nodes
    for (const [capId, cap] of capabilityNodes) {
      const node = unifiedNodes.get(capId);
      if (node) {
        cap.hierarchyLevel = node.level;
      }
    }
  } else {
    // Legacy path: compute hierarchy from capability containment
    hierarchy = computeHierarchyLevels(capabilityNodes);

    // Update hierarchyLevel on each capability
    for (const [level, capIds] of hierarchy.hierarchyLevels) {
      for (const capId of capIds) {
        const cap = capabilityNodes.get(capId);
        if (cap) cap.hierarchyLevel = level;
      }
    }
  }

  // 3. Build multi-level incidence structure
  const multiLevelIncidence = buildMultiLevelIncidence(capabilityNodes, hierarchy);

  // 4. Initialize level parameters if needed
  let levelParams = existingLevelParams;
  if (levelParams.size === 0 || levelParams.size <= hierarchy.maxHierarchyLevel) {
    levelParams = initializeLevelParameters(config, hierarchy.maxHierarchyLevel);
  }

  log.debug("[SHGAT] Rebuilt hierarchy", {
    maxLevel: hierarchy.maxHierarchyLevel,
    levels: Array.from(hierarchy.hierarchyLevels.keys()),
  });

  return {
    hierarchy,
    multiLevelIncidence,
    levelParams,
  };
}
