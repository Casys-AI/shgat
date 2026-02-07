/**
 * SHGAT Multi-Level Incidence Structure
 *
 * Builds incidence mappings for n-SuperHyperGraph message passing.
 *
 * Structure:
 * - I₀: Tools → Level-0 Capabilities (direct membership only)
 * - I_k: Level-(k-1) Caps → Level-k Caps (k ≥ 1, direct membership only)
 *
 * CRITICAL: NO TRANSITIVE CLOSURE. Each mapping captures direct membership only.
 *
 * @module graphrag/algorithms/shgat/graph/incidence
 * @see 03-incidence-structure.md
 */

import type { CapabilityNode } from "../core/types.ts";
import { getDirectCapabilities, getDirectTools } from "../core/types.ts";
import type { HierarchyResult } from "./hierarchy.ts";

/**
 * Multi-level incidence structure for n-SuperHyperGraph
 */
export interface MultiLevelIncidence {
  /**
   * I₀: Tools → Level-0 Capabilities
   *
   * Maps tool ID → set of level-0 capability IDs that directly contain it.
   * toolToCapIncidence.get("tool1") = Set{"cap-a", "cap-b"}
   */
  toolToCapIncidence: Map<string, Set<string>>;

  /**
   * I_k: Level-(k-1) Caps → Level-k Caps (forward mapping)
   *
   * For each level k ≥ 1, maps child capability → parent capabilities.
   * capToCapIncidence.get(1).get("cap-a") = Set{"meta-ab"} // cap-a is in meta-ab
   */
  capToCapIncidence: Map<number, Map<string, Set<string>>>;

  /**
   * Reverse mapping: Parent → Children at each level
   *
   * For each level k ≥ 1, maps parent capability → child capabilities.
   * parentToChildIncidence.get(1).get("meta-ab") = Set{"cap-a", "cap-b"}
   */
  parentToChildIncidence: Map<number, Map<string, Set<string>>>;

  /**
   * Reverse mapping: Capability → Tools (for level-0 caps only)
   *
   * Maps level-0 capability ID → set of tool IDs it directly contains.
   * capToToolIncidence.get("cap-a") = Set{"tool1", "tool2"}
   */
  capToToolIncidence: Map<string, Set<string>>;
}

/**
 * Build multi-level incidence structure from capabilities and hierarchy
 *
 * Algorithm:
 * 1. Build tool mappings for ALL capabilities (tools can be at any level!)
 * 2. For each level k ≥ 1, build I_k by iterating level-k caps and their cap members
 * 3. Build reverse mappings for downward pass
 *
 * NOTE: A capability at level k can contain BOTH tools AND child capabilities.
 * Example: cap-mixed with members=[t1, cap-a] is level 1 but has direct tool t1.
 *
 * Time complexity: O(C × M_avg) where C = capabilities, M_avg = avg members per cap
 *
 * @param capabilities Map of capability ID → CapabilityNode (with hierarchyLevel set)
 * @param hierarchy Hierarchy result from computeHierarchyLevels()
 * @returns MultiLevelIncidence structure
 */
export function buildMultiLevelIncidence(
  capabilities: Map<string, CapabilityNode>,
  hierarchy: HierarchyResult,
): MultiLevelIncidence {
  const { hierarchyLevels, maxHierarchyLevel } = hierarchy;

  // Initialize structures
  const toolToCapIncidence = new Map<string, Set<string>>();
  const capToCapIncidence = new Map<number, Map<string, Set<string>>>();
  const parentToChildIncidence = new Map<number, Map<string, Set<string>>>();
  const capToToolIncidence = new Map<string, Set<string>>();

  // ==========================================================================
  // Step 1: Build tool mappings for ALL capabilities (not just level 0!)
  //
  // A capability at ANY level can contain tools directly (mixed members).
  // Example: cap-mixed at level 1 has members=[t1, cap-a]
  // ==========================================================================
  for (const [capId, cap] of capabilities) {
    const toolIds = getDirectTools(cap);

    if (toolIds.length > 0) {
      // Build capToToolIncidence (reverse for downward pass)
      capToToolIncidence.set(capId, new Set(toolIds));

      // Build toolToCapIncidence (forward for upward pass)
      for (const toolId of toolIds) {
        let caps = toolToCapIncidence.get(toolId);
        if (!caps) {
          caps = new Set();
          toolToCapIncidence.set(toolId, caps);
        }
        caps.add(capId);
      }
    }
  }

  // ==========================================================================
  // Step 2: Build I_k for k ≥ 1 (Cap → Parent Cap)
  // ==========================================================================
  for (let level = 1; level <= maxHierarchyLevel; level++) {
    const capsAtLevel = hierarchyLevels.get(level) ?? new Set<string>();

    const childToParent = new Map<string, Set<string>>();
    const parentToChild = new Map<string, Set<string>>();

    for (const parentId of capsAtLevel) {
      const parent = capabilities.get(parentId);
      if (!parent) continue;

      // Get direct child capabilities (not tools)
      const childCapIds = getDirectCapabilities(parent);

      // Initialize parent → children set
      parentToChild.set(parentId, new Set(childCapIds));

      // Build child → parent mappings
      for (const childId of childCapIds) {
        let parents = childToParent.get(childId);
        if (!parents) {
          parents = new Set();
          childToParent.set(childId, parents);
        }
        parents.add(parentId);
      }
    }

    capToCapIncidence.set(level, childToParent);
    parentToChildIncidence.set(level, parentToChild);
  }

  return {
    toolToCapIncidence,
    capToCapIncidence,
    parentToChildIncidence,
    capToToolIncidence,
  };
}

/**
 * Get capabilities that contain a specific tool (level-0 caps only)
 *
 * @param incidence The incidence structure
 * @param toolId Tool ID
 * @returns Set of capability IDs that directly contain this tool
 */
export function getCapsContainingTool(
  incidence: MultiLevelIncidence,
  toolId: string,
): Set<string> {
  return incidence.toolToCapIncidence.get(toolId) ?? new Set();
}

/**
 * Get tools directly in a capability (level-0 caps only)
 *
 * @param incidence The incidence structure
 * @param capId Capability ID
 * @returns Set of tool IDs directly in this capability
 */
export function getToolsInCap(
  incidence: MultiLevelIncidence,
  capId: string,
): Set<string> {
  return incidence.capToToolIncidence.get(capId) ?? new Set();
}

/**
 * Get parent capabilities at level k that contain a child capability
 *
 * @param incidence The incidence structure
 * @param childCapId Child capability ID
 * @param parentLevel Level of parent capabilities (k)
 * @returns Set of parent capability IDs at level k
 */
export function getParentCaps(
  incidence: MultiLevelIncidence,
  childCapId: string,
  parentLevel: number,
): Set<string> {
  const levelMap = incidence.capToCapIncidence.get(parentLevel);
  if (!levelMap) return new Set();
  return levelMap.get(childCapId) ?? new Set();
}

/**
 * Get child capabilities of a parent at level k
 *
 * @param incidence The incidence structure
 * @param parentCapId Parent capability ID
 * @param parentLevel Level of parent capability (k)
 * @returns Set of child capability IDs (at level k-1)
 */
export function getChildCaps(
  incidence: MultiLevelIncidence,
  parentCapId: string,
  parentLevel: number,
): Set<string> {
  const levelMap = incidence.parentToChildIncidence.get(parentLevel);
  if (!levelMap) return new Set();
  return levelMap.get(parentCapId) ?? new Set();
}

/**
 * Get statistics about the incidence structure
 */
export function getIncidenceStats(incidence: MultiLevelIncidence): {
  numToolMappings: number;
  numLevel0Caps: number;
  numLevels: number;
  totalCapToCapEdges: number;
} {
  let totalCapToCapEdges = 0;
  for (const levelMap of incidence.capToCapIncidence.values()) {
    for (const parents of levelMap.values()) {
      totalCapToCapEdges += parents.size;
    }
  }

  return {
    numToolMappings: incidence.toolToCapIncidence.size,
    numLevel0Caps: incidence.capToToolIncidence.size,
    numLevels: incidence.capToCapIncidence.size + 1, // +1 for level 0
    totalCapToCapEdges,
  };
}
