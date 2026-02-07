/**
 * SHGAT Types and Configuration
 *
 * Type definitions and default configurations for SHGAT v2.
 * Extracted from shgat.ts for maintainability.
 *
 * @module graphrag/algorithms/shgat-types
 */

// ============================================================================
// Trace Features (v2)
// ============================================================================

/**
 * Trace-derived statistics for multi-head attention (v2)
 *
 * These statistics are extracted from execution_trace and episodic_events tables.
 * All features are fed to ALL heads - each head learns different patterns.
 */
export interface TraceStats {
  // === Success patterns ===
  /** Success rate of this tool overall (0-1) */
  historicalSuccessRate: number;
  /** Success rate when used after context tools (0-1) */
  contextualSuccessRate: number;
  /** Success rate for similar intents (0-1) */
  intentSimilarSuccessRate: number;

  // === Co-occurrence patterns ===
  /** How often this tool follows context tools (0-1) */
  cooccurrenceWithContext: number;
  /** Typical position in workflows (0=start, 1=end) */
  sequencePosition: number;

  // === Temporal patterns ===
  /** Exponential decay since last use (0-1, 1=very recent) */
  recencyScore: number;
  /** Normalized usage count (0-1) */
  usageFrequency: number;
  /** Normalized average duration (0-1) */
  avgExecutionTime: number;

  // === Error patterns ===
  /** Success rate after errors in context (0-1) */
  errorRecoveryRate: number;

  // === Path patterns ===
  /** Average steps to reach successful outcome */
  avgPathLengthToSuccess: number;
  /** Variance in path lengths */
  pathVariance: number;

  // === Error type patterns ===
  /** Success rate per error type (one-hot style: TIMEOUT, PERMISSION, NOT_FOUND, VALIDATION, NETWORK, UNKNOWN) */
  errorTypeAffinity: number[];
}

/**
 * Default trace stats for cold start
 */
export const DEFAULT_TRACE_STATS: TraceStats = {
  historicalSuccessRate: 0.5,
  contextualSuccessRate: 0.5,
  intentSimilarSuccessRate: 0.5,
  cooccurrenceWithContext: 0,
  sequencePosition: 0.5,
  recencyScore: 0.5,
  usageFrequency: 0,
  avgExecutionTime: 0.5,
  errorRecoveryRate: 0.5,
  avgPathLengthToSuccess: 3,
  pathVariance: 0,
  errorTypeAffinity: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], // TIMEOUT, PERMISSION, NOT_FOUND, VALIDATION, NETWORK, UNKNOWN
};

/**
 * Number of scalar features in TraceStats (derived from DEFAULT_TRACE_STATS)
 * = 11 scalar fields + 6 errorTypeAffinity values = 17
 */
export const NUM_TRACE_STATS: number = Object.keys(DEFAULT_TRACE_STATS).length - 1 +
  DEFAULT_TRACE_STATS.errorTypeAffinity.length;
// -1 because errorTypeAffinity is counted as array length, not as 1 key

/**
 * Rich features derived from execution traces (v2)
 *
 * All features fed to ALL heads (heads learn different patterns).
 * This replaces the 3 specialized heads (semantic/structure/temporal).
 */
export interface TraceFeatures {
  // === Core Embeddings ===
  /** User intent embedding (BGE-M3, 1024D) */
  intentEmbedding: number[];
  /** Tool/capability being scored (BGE-M3, 1024D) */
  candidateEmbedding: number[];

  // === Context Embeddings ===
  /** Recent tools in current session (max 5) */
  contextEmbeddings: number[][];
  /** Mean pooling of context embeddings */
  contextAggregated: number[];

  // === Trace-Derived Statistics ===
  traceStats: TraceStats;
}

/**
 * Create default TraceFeatures for cold start
 */
export function createDefaultTraceFeatures(
  intentEmbedding: number[],
  candidateEmbedding: number[],
): TraceFeatures {
  return {
    intentEmbedding,
    candidateEmbedding,
    contextEmbeddings: [],
    contextAggregated: new Array(intentEmbedding.length).fill(0),
    traceStats: { ...DEFAULT_TRACE_STATS },
  };
}

// ============================================================================
// Legacy Types (kept for backward compatibility)
// ============================================================================

/**
 * @deprecated Use TraceStats instead. Kept for API compatibility.
 */
export interface FusionWeights {
  semantic: number;
  structure: number;
  temporal: number;
}

/**
 * @deprecated Use DEFAULT_TRACE_STATS instead.
 */
export const DEFAULT_FUSION_WEIGHTS: FusionWeights = {
  semantic: 1.0,
  structure: 0.5,
  temporal: 0.5,
};

/**
 * @deprecated Use TraceStats instead.
 */
export interface FeatureWeights {
  semantic: number;
  structure: number;
  temporal: number;
}

/**
 * @deprecated Use DEFAULT_TRACE_STATS instead.
 */
export const DEFAULT_FEATURE_WEIGHTS: FeatureWeights = {
  semantic: 0.5,
  structure: 0.1,
  temporal: 0.1,
};

// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration for SHGAT v2
 *
 * Key changes from v1:
 * - numHeads: Now 4-16 (adaptive based on trace volume), not fixed 3
 * - headDim: hiddenDim / numHeads (for parallel attention)
 * - mlpHiddenDim: Fusion MLP hidden size
 * - maxContextLength: Max recent tools for context
 * - maxBufferSize: PER buffer cap
 * - minTracesForTraining: Cold start threshold
 */
export interface SHGATConfig {
  // === Architecture ===
  /** Number of attention heads (4-16, adaptive based on trace volume) */
  numHeads: number;
  /** Hidden dimension for projections (scales with numHeads) */
  hiddenDim: number;
  /** Dimension per head (hiddenDim / numHeads) */
  headDim: number;
  /** Embedding dimension (should match BGE-M3: 1024) */
  embeddingDim: number;
  /** Number of message passing layers */
  numLayers: number;
  /** Fusion MLP hidden dimension */
  mlpHiddenDim: number;

  // === Training ===
  /** Learning rate for training */
  learningRate: number;
  /** Batch size for training */
  batchSize: number;
  /** Max recent tools in context */
  maxContextLength: number;

  // === Buffer Management ===
  /** PER buffer cap */
  maxBufferSize: number;
  /** Cold start threshold - min traces before training */
  minTracesForTraining: number;

  // === Regularization ===
  /** Dropout rate (0 = no dropout) */
  dropout: number;
  /** L2 regularization weight */
  l2Lambda: number;
  /** LeakyReLU negative slope */
  leakyReluSlope: number;
  /** Decay factor for recursive depth */
  depthDecay: number;

  // === Dimension Preservation ===
  /**
   * Keep embedding dimension (1024) throughout message passing.
   * Fixes discriminability loss from 1024→64 compression.
   * @default false
   */
  preserveDim?: boolean;
  /**
   * Residual weight: final = (1-r)*propagated + r*original
   * @default 0.3
   */
  preserveDimResidual?: number;

  /**
   * Per-level residual weights for adaptive blending.
   * Index corresponds to node level (0=leaves, 1=intermediate, 2=root, etc.)
   * If provided, overrides preserveDimResidual for nodes at each level.
   * Falls back to preserveDimResidual for levels not specified.
   * @example [0.9, 0.3, 0.5] // L0: 90% original, L1: 30%, L2: 50%
   */
  preserveDimResiduals?: number[];

  // === Multi-Location Residuals ===
  /**
   * Residual weight for V2V (vertex-to-vertex) message passing phase.
   * output = (1-r)*enriched + r*original
   * @default 0 (no residual in V2V)
   */
  v2vResidual?: number;

  /**
   * Residual weight for downward message passing phase.
   * output = (1-r)*propagated + r*original
   * @default 0 (no residual in downward)
   */
  downwardResidual?: number;

  // === Gradient Scaling ===
  /**
   * Learning rate multiplier for message passing parameters.
   * Compensates for vanishing gradients through attention layers.
   * MP gradients are typically ~100x smaller than K-head gradients.
   * @default 1 (same learning rate as K-head)
   * @recommended 50-100 (to match K-head gradient scale)
   */
  mpLearningRateScale?: number;

  // === Projection Head (contrastive discrimination) ===
  /**
   * Enable learned projection head for fine-grained tool discrimination.
   * Maps enriched embeddings to a compact contrastive space where
   * semantically similar but functionally distinct tools are separated.
   * @default false (pure K-head scoring)
   */
  useProjectionHead?: boolean;

  /** Projection head hidden dimension (bottleneck) @default 256 */
  projectionHiddenDim?: number;

  /** Projection head output dimension (contrastive space) @default 256 */
  projectionOutputDim?: number;

  /** Blend weight: final = (1-α)*khead + α*projection. @default 0.5 */
  projectionBlendAlpha?: number;

  /** Temperature for projection scoring dot product. @default 0.07 */
  projectionTemperature?: number;

  // === Legacy (kept for backward compatibility) ===
  /** @deprecated Which heads are active - all heads active in v2 */
  activeHeads?: number[];
  /** @deprecated Fixed fusion weights - learned in v2 */
  headFusionWeights?: number[];
}

/**
 * Default configuration for SHGAT v2
 *
 * Uses 16 heads by default for optimal performance.
 * 16 heads × 64 dim = 1024 exactly matches BGE-M3 embedding dimension.
 * Benchmarks show 16 heads consistently outperforms 4 heads (+12% train, +5% test).
 */
export const DEFAULT_SHGAT_CONFIG: SHGATConfig = {
  // Architecture: 16 heads is optimal for BGE-M3 embeddings
  // hiddenDim/headDim are for SCORING (K-head attention)
  // Message passing with preserveDim uses embeddingDim/numHeads separately
  numHeads: 16, // Fixed at 16 for optimal performance
  hiddenDim: 1024, // = numHeads * 64 for scoring (16 * 64 = 1024)
  headDim: 64, // Fixed at 64 for scoring K-head
  embeddingDim: 1024,
  numLayers: 2,
  mlpHiddenDim: 32,

  // ADR-055: Keep d=1024 throughout message passing for discriminability
  // (initializeLevelParametersPreserveDim handles this separately)
  preserveDim: true,
  preserveDimResidual: 0.3, // 30% original + 70% propagated

  // Multi-location residuals (default: disabled)
  v2vResidual: 0,       // No residual in V2V phase
  downwardResidual: 0,  // No residual in downward phase

  // Gradient scaling for MP (compensate vanishing gradients)
  mpLearningRateScale: 1, // Default 1 for backward compatibility, set to 50-100 to enable MP learning

  // Training
  learningRate: 0.05,  // Increased 5x for InfoNCE with fixed τ=0.07 (CLIP-style)
  batchSize: 32,
  maxContextLength: 5,

  // Buffer management
  maxBufferSize: 50_000,
  minTracesForTraining: 100,

  // Regularization
  dropout: 0.1,
  l2Lambda: 0.0001,
  leakyReluSlope: 0.2,
  depthDecay: 0.8,
};

/**
 * @deprecated Use getAdaptiveHeadsByGraphSize() from initialization/parameters.ts instead.
 * That function is now called automatically in createSHGATFromCapabilities().
 *
 * This function was based on trace count, but graph size is available at init time
 * while trace count requires async DB query and changes over time.
 */
export function getAdaptiveConfig(_traceCount: number): Partial<SHGATConfig> {
  // Deprecated - kept for backward compatibility with tests
  // Use getAdaptiveHeadsByGraphSize() instead
  return { numHeads: 16, hiddenDim: 1024, headDim: 64, mlpHiddenDim: 32 };
}

// ============================================================================
// Training Types
// ============================================================================

/**
 * Number of negatives used per training example (sampled from curriculum tier)
 */
export const NUM_NEGATIVES = 8;

/**
 * Training example from episodic events
 *
 * For contrastive training:
 * - candidateId = positive (the capability that was executed)
 * - negativeCapIds = negatives (random other capabilities)
 */
export interface TrainingExample {
  /** Intent embedding (1024-dim) */
  intentEmbedding: number[];
  /** Context tool IDs that were active */
  contextTools: string[];
  /** Candidate capability ID (positive - the one that was executed) */
  candidateId: string;
  /** Outcome: 1 = success, 0 = failure (legacy, kept for compatibility) */
  outcome: number;
  /** Negative capability IDs for contrastive learning (optional) */
  negativeCapIds?: string[];
  /**
   * ALL negatives sorted by similarity (descending: hard → easy)
   * Excludes only the anchor capability itself.
   * Used for curriculum learning with dynamic tiers:
   * - accuracy < 0.35: sample from last third (easy negatives)
   * - accuracy > 0.55: sample from first third (hard negatives)
   * - else: sample from middle third (medium negatives)
   */
  allNegativesSorted?: string[];
}

// ============================================================================
// Graph Feature Types
// ============================================================================

/**
 * Hypergraph features for SHGAT 3-head attention (CAPABILITIES)
 *
 * These features are used by the 3-head architecture:
 * - Head 0 (semantic): uses embedding + featureWeights.semantic
 * - Head 1 (structure): hypergraphPageRank + adamicAdar × featureWeights.structure
 * - Head 2 (temporal): recency + heatDiffusion × featureWeights.temporal
 *
 * NOTE: For capabilities (hyperedges), these use HYPERGRAPH algorithms.
 * For tools, use ToolGraphFeatures instead (simple graph algorithms).
 */
export interface HypergraphFeatures {
  /** Spectral cluster ID on the hypergraph (0-based) */
  spectralCluster: number;
  /** Hypergraph PageRank score (0-1) */
  hypergraphPageRank: number;
  /** Co-occurrence frequency from episodic traces (0-1) */
  cooccurrence: number;
  /** Recency score - how recently used (0-1, 1 = very recent) */
  recency: number;
  /** Adamic-Adar similarity with neighboring capabilities (0-1) */
  adamicAdar?: number;
  /** Heat diffusion score (0-1) */
  heatDiffusion?: number;
}

/**
 * Default hypergraph features (cold start)
 */
export const DEFAULT_HYPERGRAPH_FEATURES: HypergraphFeatures = {
  spectralCluster: 0,
  hypergraphPageRank: 0.01,
  cooccurrence: 0,
  recency: 0,
  adamicAdar: 0,
  heatDiffusion: 0,
};

/**
 * Tool graph features for SHGAT 3-head attention (TOOLS)
 *
 * These features use SIMPLE GRAPH algorithms (not hypergraph):
 * - Head 1 (structure): pageRank + adamicAdar × featureWeights.structure
 * - Head 2 (temporal): cooccurrence + recency × featureWeights.temporal
 *
 * This is separate from HypergraphFeatures because tools exist in a
 * simple directed graph (Graphology), not the superhypergraph.
 */
export interface ToolGraphFeatures {
  /** Regular PageRank score from Graphology (0-1) */
  pageRank: number;
  /** Louvain community ID (0-based integer) */
  louvainCommunity: number;
  /** Adamic-Adar similarity with neighboring tools (0-1) */
  adamicAdar: number;
  /** Co-occurrence frequency from execution_trace (0-1) */
  cooccurrence: number;
  /** Recency score - exponential decay since last use (0-1, 1 = very recent) */
  recency: number;
  /** Heat diffusion score from graph topology (0-1) */
  heatDiffusion: number;
}

/**
 * Default tool graph features (cold start)
 */
export const DEFAULT_TOOL_GRAPH_FEATURES: ToolGraphFeatures = {
  pageRank: 0.01,
  louvainCommunity: 0,
  adamicAdar: 0,
  cooccurrence: 0,
  recency: 0,
  heatDiffusion: 0,
};

// ============================================================================
// Unified Node Type
// ============================================================================

/**
 * Unified node type for n-SuperHyperGraph
 *
 * Replaces separate ToolNode and CapabilityNode types with a single unified type.
 * The hierarchy is implicit from structure:
 * - children.length === 0 → leaf node (level 0)
 * - children.length > 0 → composite node (level = 1 + max child level)
 *
 * @since Unified Node refactor
 */
export interface Node {
  /** Unique identifier */
  id: string;
  /** Embedding vector (e.g., BGE-M3 1024-dim) */
  embedding: number[];
  /** Child node IDs. Empty array = leaf node (level 0) */
  children: string[];
  /** Hierarchy level. Computed at graph construction time. */
  level: number;
}

/**
 * Build a graph from an array of nodes, computing levels via DFS
 *
 * @param nodes Array of nodes (level field will be computed)
 * @returns Map of node ID to Node with computed levels
 */
export function buildGraph(nodes: Node[]): Map<string, Node> {
  const graph = new Map(nodes.map((n) => [n.id, { ...n, level: 0 }]));
  computeAllLevels(graph);
  return graph;
}

/**
 * Compute levels for all nodes in a graph using DFS with memoization
 *
 * @param nodes Map of node ID to Node (mutates level field)
 */
export function computeAllLevels(nodes: Map<string, Node>): void {
  const cache = new Map<string, number>();
  for (const id of nodes.keys()) {
    computeLevel(id, nodes, cache);
  }
}

/**
 * Compute level for a single node recursively
 *
 * @param id Node ID to compute level for
 * @param nodes Map of all nodes
 * @param cache Memoization cache
 * @returns Computed level
 */
function computeLevel(
  id: string,
  nodes: Map<string, Node>,
  cache: Map<string, number>,
): number {
  if (cache.has(id)) return cache.get(id)!;
  const node = nodes.get(id);
  if (!node || node.children.length === 0) {
    cache.set(id, 0);
    if (node) node.level = 0;
    return 0;
  }
  const maxChildLevel = Math.max(
    ...node.children.map((c) => computeLevel(c, nodes, cache)),
  );
  const level = 1 + maxChildLevel;
  cache.set(id, level);
  node.level = level;
  return level;
}

// ============================================================================
// Batched Operations for Node (BLAS-optimized)
// ============================================================================

/**
 * Result of batched embeddings lookup
 */
export interface BatchedEmbeddings {
  /** Embeddings matrix [N x dim] - row i is embedding for ids[i] */
  matrix: number[][];
  /** Node IDs in order */
  ids: string[];
  /** Index lookup: id → row index */
  indexMap: Map<string, number>;
}

/**
 * Get embeddings as a matrix for batched BLAS operations
 *
 * Instead of N separate lookups, returns a [N x dim] matrix.
 * Use for K-head scoring: K_all = E @ W_k^T (single matmul vs N matVecs)
 *
 * @param nodes Map of nodes
 * @param ids Optional subset of IDs (default: all nodes)
 * @returns BatchedEmbeddings with matrix and index maps
 *
 * @example
 * ```typescript
 * const { matrix, indexMap } = batchGetEmbeddings(graph);
 * // matrix is [numNodes x embDim]
 * // Use with matmulTranspose for batched K computation
 * const K_all = matmulTranspose(matrix, W_k); // [numNodes x scoringDim]
 * ```
 */
export function batchGetEmbeddings(
  nodes: Map<string, Node>,
  ids?: string[],
): BatchedEmbeddings {
  const nodeIds = ids ?? Array.from(nodes.keys());
  const matrix: number[][] = new Array(nodeIds.length);
  const indexMap = new Map<string, number>();

  for (let i = 0; i < nodeIds.length; i++) {
    const id = nodeIds[i];
    const node = nodes.get(id);
    matrix[i] = node?.embedding ?? [];
    indexMap.set(id, i);
  }

  return { matrix, ids: nodeIds, indexMap };
}

/**
 * Get embeddings for nodes at a specific level as a matrix
 *
 * @param nodes Map of nodes
 * @param level Hierarchy level to filter
 * @returns BatchedEmbeddings for nodes at that level only
 */
export function batchGetEmbeddingsByLevel(
  nodes: Map<string, Node>,
  level: number,
): BatchedEmbeddings {
  const ids = Array.from(nodes.values())
    .filter((n) => n.level === level)
    .map((n) => n.id);
  return batchGetEmbeddings(nodes, ids);
}

/**
 * Build parent-child incidence matrix in one pass
 *
 * Returns matrix A where A[child_idx][parent_idx] = 1 if child is in parent.
 * Used for batched message passing: E_parent = A^T @ E_child
 *
 * @param nodes Map of nodes
 * @param childLevel Level of child nodes
 * @param parentLevel Level of parent nodes (should be childLevel + 1)
 * @returns Incidence matrix and index maps
 *
 * @example
 * ```typescript
 * const { matrix, childIndex, parentIndex } = buildIncidenceMatrix(graph, 0, 1);
 * // Upward aggregation: E_level1 = matrix^T @ E_level0
 * const E_agg = matmulTranspose(transpose(matrix), E_level0);
 * ```
 */
export function buildIncidenceMatrix(
  nodes: Map<string, Node>,
  childLevel: number,
  parentLevel: number,
): {
  matrix: number[][];
  childIndex: Map<string, number>;
  parentIndex: Map<string, number>;
} {
  // Get nodes at each level
  const childNodes = Array.from(nodes.values()).filter((n) => n.level === childLevel);
  const parentNodes = Array.from(nodes.values()).filter((n) => n.level === parentLevel);

  // Build index maps
  const childIndex = new Map<string, number>();
  const parentIndex = new Map<string, number>();

  for (let i = 0; i < childNodes.length; i++) {
    childIndex.set(childNodes[i].id, i);
  }
  for (let i = 0; i < parentNodes.length; i++) {
    parentIndex.set(parentNodes[i].id, i);
  }

  // Build matrix [numChildren x numParents]
  const matrix: number[][] = Array.from(
    { length: childNodes.length },
    () => new Array(parentNodes.length).fill(0),
  );

  // Fill matrix: for each parent, mark its children
  for (let p = 0; p < parentNodes.length; p++) {
    const parent = parentNodes[p];
    for (const childId of parent.children) {
      const c = childIndex.get(childId);
      if (c !== undefined) {
        matrix[c][p] = 1;
      }
    }
  }

  return { matrix, childIndex, parentIndex };
}

/**
 * Build all incidence matrices for multi-level hierarchy
 *
 * Returns matrices for each level transition (0→1, 1→2, etc.)
 *
 * @param nodes Map of nodes
 * @returns Map of parentLevel → incidence info
 */
export function buildAllIncidenceMatrices(
  nodes: Map<string, Node>,
): Map<number, {
  matrix: number[][];
  childIndex: Map<string, number>;
  parentIndex: Map<string, number>;
}> {
  const result = new Map<number, {
    matrix: number[][];
    childIndex: Map<string, number>;
    parentIndex: Map<string, number>;
  }>();

  // Find max level
  let maxLevel = 0;
  for (const node of nodes.values()) {
    if (node.level > maxLevel) maxLevel = node.level;
  }

  // Build matrix for each level transition
  for (let level = 1; level <= maxLevel; level++) {
    result.set(level, buildIncidenceMatrix(nodes, level - 1, level));
  }

  return result;
}

/**
 * Batch lookup nodes by IDs
 *
 * @param nodes Map of all nodes
 * @param ids IDs to lookup
 * @returns Array of nodes (undefined for missing IDs)
 */
export function batchGetNodes(
  nodes: Map<string, Node>,
  ids: string[],
): (Node | undefined)[] {
  return ids.map((id) => nodes.get(id));
}

/**
 * Group nodes by level for batched processing
 *
 * @param nodes Map of nodes
 * @returns Map of level → array of nodes at that level
 */
export function groupNodesByLevel(
  nodes: Map<string, Node>,
): Map<number, Node[]> {
  const groups = new Map<number, Node[]>();

  for (const node of nodes.values()) {
    let group = groups.get(node.level);
    if (!group) {
      group = [];
      groups.set(node.level, group);
    }
    group.push(node);
  }

  return groups;
}

// ============================================================================
// Legacy Node Types (kept for backward compatibility during migration)
// ============================================================================

/**
 * Member of a capability (tool OR capability)
 *
 * @deprecated Use Node.children instead
 */
export type Member =
  | { type: "tool"; id: string }
  | { type: "capability"; id: string };

/**
 * Tool node (vertex in hypergraph)
 *
 * @deprecated Use Node with children: [] instead
 */
export interface ToolNode {
  id: string;
  /** Embedding (from tool description) */
  embedding: number[];
  /** Tool graph features (simple graph algorithms) */
  toolFeatures?: ToolGraphFeatures;
}

/**
 * Capability node (hyperedge in n-SuperHyperGraph)
 *
 * @deprecated Use Node with children: [...] instead
 */
export interface CapabilityNode {
  id: string;
  /** Embedding (from description or aggregated tools) */
  embedding: number[];

  /**
   * Members: tools (V₀) OR capabilities (P^k, k < level)
   * @deprecated Use Node.children instead
   */
  members: Member[];

  /**
   * Hierarchy level (computed via topological sort)
   * @deprecated Use Node.level instead
   */
  hierarchyLevel: number;

  /** Success rate from history (reliability) */
  successRate: number;

  // === Legacy fields (kept for backward compatibility, will be removed) ===

  /**
   * @deprecated Use members.filter(m => m.type === 'tool') instead
   * Tools in this capability (vertex IDs)
   */
  toolsUsed?: string[];

  /**
   * @deprecated Use members.filter(m => m.type === 'capability') instead
   * Child capabilities (via contains)
   */
  children?: string[];

  /**
   * @deprecated Compute via reverse incidence mapping
   * Parent capabilities (via contains)
   */
  parents?: string[];

  /** Hypergraph features for multi-head attention */
  hypergraphFeatures?: HypergraphFeatures;
}

/**
 * Attention result for a capability
 */
export interface AttentionResult {
  capabilityId: string;
  /** Final attention score (0-1) */
  score: number;
  /** Per-head attention weights */
  headWeights: number[];
  /** Per-head raw scores before fusion */
  headScores: number[];
  /** Contribution from recursive parents */
  recursiveContribution: number;
  /** Feature contributions for interpretability */
  featureContributions?: {
    semantic: number;
    structure: number;
    temporal: number;
    reliability: number;
  };
  /** Attention over tools (for interpretability) */
  toolAttention?: number[];

  /**
   * Hierarchy level of this capability (n-SuperHyperGraph)
   *
   * - Level 0: Leaf capabilities (contain only tools)
   * - Level 1: Meta-capabilities (contain level-0 caps)
   * - Level k: Meta^k capabilities (contain level-(k-1) caps)
   *
   * @since v1 refactor
   */
  hierarchyLevel?: number;
}

/**
 * Cached activations for backpropagation
 */
export interface ForwardCache {
  /** Vertex (tool) embeddings at each layer */
  H: number[][][];
  /** Hyperedge (capability) embeddings at each layer */
  E: number[][][];
  /** Attention weights vertex→edge [layer][head][vertex][edge] */
  attentionVE: number[][][][];
  /** Attention weights edge→vertex [layer][head][edge][vertex] */
  attentionEV: number[][][][];
}

// ============================================================================
// Multi-Level Message Passing Types (n-SuperHyperGraph v1 refactor)
// ============================================================================

/**
 * Multi-level embeddings structure for n-SuperHyperGraph
 *
 * After forward pass, contains:
 * - H: Final tool embeddings (V, level -1)
 * - E: Capability embeddings per hierarchy level (E^0, E^1, ..., E^L_max)
 * - Attention weights for interpretability
 *
 * @since v1 refactor
 * @see 04-message-passing.md
 */
export interface MultiLevelEmbeddings {
  /** Tool embeddings (level -1) [numTools][embeddingDim] */
  H: number[][];

  /** Capability embeddings by level (E^0, E^1, ..., E^L_max) */
  E: Map<number, number[][]>;

  /** Attention weights for upward pass: level → [head][child][parent] */
  attentionUpward: Map<number, number[][][]>;

  /** Attention weights for downward pass: level → [head][parent][child] */
  attentionDownward: Map<number, number[][][]>;
}

/**
 * Learnable parameters per hierarchy level
 *
 * Each level has projection matrices and attention vectors for:
 * - Upward pass: child → parent aggregation
 * - Downward pass: parent → child propagation
 *
 * @since v1 refactor
 * @see 05-parameters.md
 */
export interface LevelParams {
  /**
   * Child projection matrices per head [numHeads][headDim][embeddingDim]
   *
   * Projects child embeddings (tools or lower-level caps) for attention.
   */
  W_child: number[][][];

  /**
   * Parent projection matrices per head [numHeads][headDim][embeddingDim]
   *
   * Projects parent embeddings (higher-level caps) for attention.
   */
  W_parent: number[][][];

  /**
   * Attention vectors for upward pass per head [numHeads][2*headDim]
   *
   * Computes attention: a^T · LeakyReLU([W_child · e_i || W_parent · e_j])
   */
  a_upward: number[][];

  /**
   * Attention vectors for downward pass per head [numHeads][2*headDim]
   *
   * Computes attention: b^T · LeakyReLU([W_parent · e_i || W_child · e_j])
   */
  a_downward: number[][];
}

/**
 * Extended forward cache for multi-level message passing
 *
 * Extends ForwardCache with per-level intermediate embeddings for backprop.
 *
 * @since v1 refactor
 */
export interface MultiLevelForwardCache {
  /** Initial tool embeddings [numTools][embDim] */
  H_init: number[][];

  /** Final tool embeddings after downward pass [numTools][embDim] */
  H_final: number[][];

  /** Capability embeddings per level: level → [numCapsAtLevel][embDim] */
  E_init: Map<number, number[][]>;

  /** Final capability embeddings per level after forward pass */
  E_final: Map<number, number[][]>;

  /** Intermediate upward embeddings: level → [numCapsAtLevel][embDim] */
  intermediateUpward: Map<number, number[][]>;

  /** Intermediate downward embeddings: level → [numCapsAtLevel][embDim] */
  intermediateDownward: Map<number, number[][]>;

  /** Attention weights upward: level → [head][child][parent] */
  attentionUpward: Map<number, number[][][]>;

  /** Attention weights downward: level → [head][parent][child] */
  attentionDownward: Map<number, number[][][]>;
}

// ============================================================================
// n-SuperHyperGraph Helper Functions
// ============================================================================

/** Type alias for tool member */
export type ToolMember = Extract<Member, { type: "tool" }>;

/** Type alias for capability member */
export type CapabilityMember = Extract<Member, { type: "capability" }>;

/**
 * Get direct tools from a capability (no transitive closure)
 *
 * @param cap Capability node
 * @returns Array of tool IDs
 */
export function getDirectTools(cap: CapabilityNode): string[] {
  return cap.members
    .filter((m): m is ToolMember => m.type === "tool")
    .map((m) => m.id);
}

/**
 * Get direct child capabilities (no transitive closure)
 *
 * @param cap Capability node
 * @returns Array of capability IDs
 */
export function getDirectCapabilities(cap: CapabilityNode): string[] {
  return cap.members
    .filter((m): m is CapabilityMember => m.type === "capability")
    .map((m) => m.id);
}

/**
 * Create members array from legacy toolsUsed + children
 *
 * @param toolsUsed Tool IDs (default: empty array)
 * @param children Child capability IDs (default: empty array)
 * @returns Unified members array
 */
export function createMembersFromLegacy(
  toolsUsed: string[] = [],
  children: string[] = [],
): Member[] {
  return [
    ...toolsUsed.map((id) => ({ type: "tool" as const, id })),
    ...children.map((id) => ({ type: "capability" as const, id })),
  ];
}

/**
 * Legacy capability node format (before n-SuperHyperGraph refactor)
 */
export interface LegacyCapabilityNode {
  id: string;
  embedding: number[];
  toolsUsed: string[];
  children?: string[];
  parents?: string[];
  successRate: number;
  hypergraphFeatures?: HypergraphFeatures;
}

/**
 * Migrate legacy CapabilityNode to new format with members
 *
 * Converts old format (toolsUsed + children) to new unified members array.
 *
 * @param legacy Legacy capability node
 * @returns Capability node with members field (hierarchyLevel = 0, needs recomputation)
 */
export function migrateCapabilityNode(legacy: LegacyCapabilityNode): CapabilityNode {
  return {
    id: legacy.id,
    embedding: legacy.embedding,
    members: createMembersFromLegacy(legacy.toolsUsed, legacy.children),
    hierarchyLevel: 0, // Will be recomputed by computeHierarchyLevels()
    successRate: legacy.successRate,
    toolsUsed: legacy.toolsUsed, // Keep for backward compat
    children: legacy.children,
    parents: legacy.parents,
    hypergraphFeatures: legacy.hypergraphFeatures,
  };
}
