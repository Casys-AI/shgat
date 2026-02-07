/**
 * SHGAT Graph Builder Module
 *
 * Handles graph construction and management for SHGAT:
 * - Unified node registration
 * - Tool node registration (vertices) - legacy
 * - Capability node registration (hyperedges) - legacy
 * - Incidence matrix construction with transitive closure
 * - Index management
 *
 * @module graphrag/algorithms/shgat/graph/graph-builder
 */

import type {
  CapabilityNode,
  HypergraphFeatures,
  Node,
  ToolGraphFeatures,
  ToolNode,
} from "../core/types.ts";
import {
  computeAllLevels,
  createMembersFromLegacy,
  DEFAULT_HYPERGRAPH_FEATURES,
  DEFAULT_TOOL_GRAPH_FEATURES,
} from "../core/types.ts";

/**
 * Input data for building a hypergraph
 */
export interface GraphBuildData {
  tools: Array<{ id: string; embedding: number[] }>;
  capabilities: Array<{
    id: string;
    embedding: number[];
    toolsUsed: string[];
    successRate: number;
    parents?: string[];
    children?: string[];
  }>;
}

/**
 * Manages the hypergraph structure for SHGAT
 *
 * Handles:
 * - Unified nodes (new API)
 * - Tool nodes (vertices in hypergraph) - legacy
 * - Capability nodes (hyperedges) - legacy
 * - Incidence matrix with transitive closure for hierarchical capabilities
 */
export class GraphBuilder {
  // Unified node storage (new API)
  private nodes: Map<string, Node> = new Map();

  // Legacy node storage (maintained for backward compatibility)
  private toolNodes: Map<string, ToolNode> = new Map();
  private capabilityNodes: Map<string, CapabilityNode> = new Map();
  private toolIndex: Map<string, number> = new Map();
  private capabilityIndex: Map<string, number> = new Map();
  private incidenceMatrix: number[][] = [];

  // =========================================================================
  // Unified Node API (new)
  // =========================================================================

  /**
   * Register a unified node
   *
   * Also populates legacy Maps for backward compatibility with scoring functions.
   * Note: Call finalizeNodes() after registering all nodes to rebuild indices.
   *
   * @param node Node to register
   */
  registerNode(node: Node): void {
    this.nodes.set(node.id, node);

    // Populate legacy Maps for backward compatibility with scoreAllTools/scoreAllCapabilities
    if (node.children.length === 0) {
      // Leaf node → ToolNode
      const toolNode: ToolNode = {
        id: node.id,
        embedding: node.embedding,
        toolFeatures: DEFAULT_TOOL_GRAPH_FEATURES,
      };
      this.toolNodes.set(node.id, toolNode);
    } else {
      // Composite node → CapabilityNode
      const capNode: CapabilityNode = {
        id: node.id,
        embedding: node.embedding,
        members: createMembersFromLegacy(node.children, []),
        hierarchyLevel: node.level,
        successRate: 1.0,
        toolsUsed: node.children,
        hypergraphFeatures: DEFAULT_HYPERGRAPH_FEATURES,
      };
      this.capabilityNodes.set(node.id, capNode);
    }
  }

  /**
   * Finalize node registration - call after registering all nodes
   * Recomputes levels and rebuilds indices once.
   */
  finalizeNodes(): void {
    computeAllLevels(this.nodes);

    // Sync computed levels to legacy capabilityNodes Map
    for (const node of this.nodes.values()) {
      const capNode = this.capabilityNodes.get(node.id);
      if (capNode) {
        capNode.hierarchyLevel = node.level;
      }
    }

    this.rebuildIndices();
  }

  /**
   * Get a unified node by ID
   */
  getNode(id: string): Node | undefined {
    return this.nodes.get(id);
  }

  /**
   * Get all unified nodes
   */
  getNodes(): Map<string, Node> {
    return this.nodes;
  }

  /**
   * Get nodes at a specific hierarchy level
   */
  getNodesByLevel(level: number): Node[] {
    return Array.from(this.nodes.values()).filter((n) => n.level === level);
  }

  /**
   * Get maximum hierarchy level from unified nodes
   */
  getMaxLevel(): number {
    let maxLevel = 0;
    for (const node of this.nodes.values()) {
      if (node.level > maxLevel) maxLevel = node.level;
    }
    return maxLevel;
  }

  /**
   * Get node IDs grouped by level (for hierarchy building)
   */
  getNodeIdsByLevel(): Map<number, Set<string>> {
    const byLevel = new Map<number, Set<string>>();
    for (const node of this.nodes.values()) {
      let levelSet = byLevel.get(node.level);
      if (!levelSet) {
        levelSet = new Set();
        byLevel.set(node.level, levelSet);
      }
      levelSet.add(node.id);
    }
    return byLevel;
  }

  /**
   * Get all descendant IDs of a node (transitive)
   */
  getDescendants(id: string, visited: Set<string> = new Set<string>()): string[] {
    if (visited.has(id)) return [];
    visited.add(id);

    const node = this.nodes.get(id);
    if (!node) return [];

    const descendants: string[] = [];
    for (const childId of node.children) {
      descendants.push(childId);
      descendants.push(...this.getDescendants(childId, visited));
    }
    return descendants;
  }

  /**
   * Check if a unified node exists
   */
  hasNode(id: string): boolean {
    return this.nodes.has(id);
  }

  /**
   * Get the number of unified nodes
   */
  getNodeCount(): number {
    return this.nodes.size;
  }

  /**
   * Get all unified node IDs
   */
  getNodeIds(): string[] {
    return Array.from(this.nodes.keys());
  }

  /**
   * Clear all unified nodes
   */
  clearNodes(): void {
    this.nodes.clear();
  }

  // =========================================================================
  // Legacy Node Registration (maintained for backward compatibility)
  // =========================================================================

  /**
   * Register a tool (vertex)
   * @deprecated Use registerNode() with children: [] instead
   */
  registerTool(node: ToolNode): void {
    this.toolNodes.set(node.id, node);
    // Also add to unified nodes for scoreNodes compatibility
    const unifiedNode: Node = {
      id: node.id,
      embedding: node.embedding,
      children: [],
      level: 0,
    };
    this.nodes.set(node.id, unifiedNode);
    this.rebuildIndices();
  }

  /**
   * Register a capability (hyperedge)
   * @deprecated Use registerNode() with children: [...] instead
   */
  registerCapability(node: CapabilityNode): void {
    this.capabilityNodes.set(node.id, node);
    // Also add to unified nodes for scoreNodes compatibility
    const unifiedNode: Node = {
      id: node.id,
      embedding: node.embedding,
      children: node.toolsUsed ?? [],
      level: node.hierarchyLevel,
    };
    this.nodes.set(node.id, unifiedNode);
    this.rebuildIndices();
  }

  /**
   * Check if a tool node exists
   */
  hasToolNode(toolId: string): boolean {
    return this.toolNodes.has(toolId);
  }

  /**
   * Check if a capability node exists
   */
  hasCapabilityNode(capabilityId: string): boolean {
    return this.capabilityNodes.has(capabilityId);
  }

  /**
   * Get the number of registered tools
   */
  getToolCount(): number {
    return this.toolNodes.size;
  }

  /**
   * Get the number of registered capabilities
   */
  getCapabilityCount(): number {
    return this.capabilityNodes.size;
  }

  /**
   * Get all registered tool IDs
   */
  getToolIds(): string[] {
    return Array.from(this.toolNodes.keys());
  }

  /**
   * Get all registered capability IDs
   */
  getCapabilityIds(): string[] {
    return Array.from(this.capabilityNodes.keys());
  }

  /**
   * Get a tool node by ID
   */
  getToolNode(toolId: string): ToolNode | undefined {
    return this.toolNodes.get(toolId);
  }

  /**
   * Get a capability node by ID
   */
  getCapabilityNode(capabilityId: string): CapabilityNode | undefined {
    return this.capabilityNodes.get(capabilityId);
  }

  /**
   * Get all tool nodes
   */
  getToolNodes(): Map<string, ToolNode> {
    return this.toolNodes;
  }

  /**
   * Get all capability nodes
   */
  getCapabilityNodes(): Map<string, CapabilityNode> {
    return this.capabilityNodes;
  }

  /**
   * Get tool index (for incidence matrix)
   */
  getToolIndex(toolId: string): number | undefined {
    return this.toolIndex.get(toolId);
  }

  /**
   * Get full tool index map (for V→V co-occurrence)
   */
  getToolIndexMap(): Map<string, number> {
    return new Map(this.toolIndex);
  }

  /**
   * Get capability index (for incidence matrix)
   */
  getCapabilityIndex(capabilityId: string): number | undefined {
    return this.capabilityIndex.get(capabilityId);
  }

  /**
   * Get the incidence matrix
   */
  getIncidenceMatrix(): number[][] {
    return this.incidenceMatrix;
  }

  // =========================================================================
  // Graph Construction
  // =========================================================================

  /**
   * Build hypergraph from tools and capabilities
   */
  buildFromData(data: GraphBuildData): void {
    this.toolNodes.clear();
    this.capabilityNodes.clear();

    for (const tool of data.tools) {
      this.toolNodes.set(tool.id, {
        id: tool.id,
        embedding: tool.embedding,
      });
    }

    for (const cap of data.capabilities) {
      this.capabilityNodes.set(cap.id, {
        id: cap.id,
        embedding: cap.embedding,
        members: createMembersFromLegacy(cap.toolsUsed, cap.children),
        hierarchyLevel: 0, // Will be recomputed by computeHierarchyLevels()
        toolsUsed: cap.toolsUsed,
        successRate: cap.successRate,
        parents: cap.parents,
        children: cap.children,
      });
    }

    this.rebuildIndices();
  }

  /**
   * Clear all nodes and rebuild from scratch
   */
  clear(): void {
    this.toolNodes.clear();
    this.capabilityNodes.clear();
    this.toolIndex.clear();
    this.capabilityIndex.clear();
    this.incidenceMatrix = [];
  }

  // =========================================================================
  // Feature Updates
  // =========================================================================

  /**
   * Update hypergraph features for a capability
   */
  updateHypergraphFeatures(capabilityId: string, features: Partial<HypergraphFeatures>): void {
    const node = this.capabilityNodes.get(capabilityId);
    if (node) {
      node.hypergraphFeatures = {
        ...(node.hypergraphFeatures || DEFAULT_HYPERGRAPH_FEATURES),
        ...features,
      };
    }
  }

  /**
   * Update graph features for a tool
   */
  updateToolFeatures(toolId: string, features: Partial<ToolGraphFeatures>): void {
    const node = this.toolNodes.get(toolId);
    if (node) {
      node.toolFeatures = {
        ...(node.toolFeatures || DEFAULT_TOOL_GRAPH_FEATURES),
        ...features,
      };
    }
  }

  /**
   * Batch update hypergraph features for capabilities
   */
  batchUpdateCapabilityFeatures(updates: Map<string, Partial<HypergraphFeatures>>): void {
    for (const [capId, features] of updates) {
      this.updateHypergraphFeatures(capId, features);
    }
  }

  /**
   * Batch update graph features for tools
   */
  batchUpdateToolFeatures(updates: Map<string, Partial<ToolGraphFeatures>>): void {
    for (const [toolId, features] of updates) {
      this.updateToolFeatures(toolId, features);
    }
  }

  // =========================================================================
  // Embedding Extraction
  // =========================================================================

  /**
   * Get all tool embeddings in index order
   */
  getToolEmbeddings(): number[][] {
    const embeddings: number[][] = [];
    for (const [_, tool] of this.toolNodes) {
      embeddings.push([...tool.embedding]);
    }
    return embeddings;
  }

  /**
   * Get all capability embeddings in index order
   */
  getCapabilityEmbeddings(): number[][] {
    const embeddings: number[][] = [];
    for (const [_, cap] of this.capabilityNodes) {
      embeddings.push([...cap.embedding]);
    }
    return embeddings;
  }

  /**
   * Get capability hierarchy levels in index order (same order as getCapabilityEmbeddings)
   * Used for per-level residual connections.
   */
  getCapabilityLevels(): number[] {
    const levels: number[] = [];
    for (const [_, cap] of this.capabilityNodes) {
      levels.push(cap.hierarchyLevel ?? 0);
    }
    return levels;
  }

  /**
   * Get tool hierarchy levels in index order (same order as getToolEmbeddings)
   * Tools are typically level 0 (leaves).
   */
  getToolLevels(): number[] {
    const levels: number[] = [];
    for (const [id, _] of this.toolNodes) {
      const node = this.nodes.get(id);
      levels.push(node?.level ?? 0);
    }
    return levels;
  }

  // =========================================================================
  // Index and Incidence Matrix Management
  // =========================================================================

  /**
   * Recursively collect all tools from a capability and its children (transitive closure)
   *
   * This enables hierarchical capabilities (meta-meta-capabilities → meta-capabilities → capabilities)
   * to inherit all tools from their descendants in the incidence matrix.
   *
   * Example:
   *   release-cycle (meta-meta) contains [deploy-full, rollback-plan]
   *   deploy-full (meta) contains [build, test]
   *   build (capability) has tools [compiler, linker]
   *   test (capability) has tools [pytest]
   *
   *   collectTransitiveTools("release-cycle") returns [compiler, linker, pytest, ...]
   *
   * @param capId - The capability ID to collect tools from
   * @param visited - Set of already visited capability IDs (cycle detection)
   * @returns Set of all tool IDs transitively reachable from this capability
   *
   * @deprecated Use multi-level message passing with buildMultiLevelIncidence() instead.
   * This method flattens the n-SuperHyperGraph structure which loses hierarchical information.
   * For proper multi-level message passing, use the incidence structures from graph/incidence.ts
   * that preserve the direct membership relationships at each level.
   *
   * @see 03-incidence-structure.md for the new multi-level approach
   */
  private collectTransitiveTools(capId: string, visited: Set<string> = new Set()): Set<string> {
    // Cycle detection - prevent infinite recursion
    if (visited.has(capId)) {
      return new Set();
    }
    visited.add(capId);

    const cap = this.capabilityNodes.get(capId);
    if (!cap) {
      return new Set();
    }

    // Start with direct tools (use members for new format, fallback to toolsUsed)
    const directTools = cap.members
      ? cap.members.filter((m) => m.type === "tool").map((m) => m.id)
      : cap.toolsUsed ?? [];
    const tools = new Set<string>(directTools);

    // Recursively collect from children (contained capabilities)
    const childCapIds = cap.members
      ? cap.members.filter((m) => m.type === "capability").map((m) => m.id)
      : cap.children ?? [];
    for (const childId of childCapIds) {
      const childTools = this.collectTransitiveTools(childId, visited);
      for (const tool of childTools) {
        tools.add(tool);
      }
    }

    return tools;
  }

  /**
   * Rebuild indices and incidence matrix
   */
  private rebuildIndices(): void {
    this.toolIndex.clear();
    this.capabilityIndex.clear();

    let tIdx = 0;
    for (const tId of this.toolNodes.keys()) {
      this.toolIndex.set(tId, tIdx++);
    }

    let cIdx = 0;
    for (const cId of this.capabilityNodes.keys()) {
      this.capabilityIndex.set(cId, cIdx++);
    }

    // Build incidence matrix A[tool][capability] with transitive closure
    // Meta-capabilities inherit all tools from their child capabilities
    // This enables infinite hierarchical nesting (meta-meta-meta... → meta → capability)
    const numTools = this.toolNodes.size;
    const numCaps = this.capabilityNodes.size;

    this.incidenceMatrix = Array.from({ length: numTools }, () => Array(numCaps).fill(0));

    for (const [capId] of this.capabilityNodes) {
      const cIdx = this.capabilityIndex.get(capId)!;
      // Use transitive collection to get all tools from this capability
      // and all its descendants (children, grandchildren, etc.)
      const transitiveTools = this.collectTransitiveTools(capId);
      for (const toolId of transitiveTools) {
        const tIdx = this.toolIndex.get(toolId);
        if (tIdx !== undefined) {
          this.incidenceMatrix[tIdx][cIdx] = 1;
        }
      }
    }
  }

  // =========================================================================
  // Statistics
  // =========================================================================

  /**
   * Get incidence matrix statistics
   */
  getIncidenceStats(): { numTools: number; numCapabilities: number; nonZeros: number } {
    let nonZeros = 0;
    for (const row of this.incidenceMatrix) {
      nonZeros += row.filter((x) => x > 0).length;
    }

    return {
      numTools: this.toolNodes.size,
      numCapabilities: this.capabilityNodes.size,
      nonZeros,
    };
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Generate a deterministic default embedding for a tool based on its ID
 */
export function generateDefaultToolEmbedding(toolId: string, dim: number): number[] {
  const embedding: number[] = [];

  // Use hash-like seed from tool ID for deterministic pseudo-random values
  let seed = 0;
  for (let i = 0; i < toolId.length; i++) {
    seed = ((seed << 5) - seed + toolId.charCodeAt(i)) | 0;
  }
  for (let i = 0; i < dim; i++) {
    // Deterministic pseudo-random based on seed and index
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    embedding.push((seed / 0x7fffffff - 0.5) * 0.1);
  }
  return embedding;
}
