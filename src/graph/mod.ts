/**
 * SHGAT Graph Module
 *
 * Graph construction and management for SHGAT hypergraphs.
 *
 * @module graphrag/algorithms/shgat/graph
 */

export {
  generateDefaultToolEmbedding,
  type GraphBuildData,
  GraphBuilder,
} from "./graph-builder.ts";

// Hierarchy computation (n-SuperHyperGraph)
export {
  computeHierarchyLevels,
  computeNodeHierarchyLevels,
  getCapabilitiesAtLevel,
  getNodesAtLevel,
  getSortedLevels,
  HierarchyCycleError,
  type HierarchyResult,
  type NodeHierarchyResult,
  validateAcyclic,
  validateNodeAcyclic,
} from "./hierarchy.ts";

// Multi-level incidence structure (n-SuperHyperGraph)
export {
  buildMultiLevelIncidence,
  getCapsContainingTool,
  getChildCaps,
  getIncidenceStats,
  getParentCaps,
  getToolsInCap,
  type MultiLevelIncidence,
} from "./incidence.ts";
