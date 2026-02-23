/**
 * Message Passing Phase Interface
 *
 * Defines the contract for pluggable message passing phases in SHGAT.
 * Enables multi-level message passing for n-SuperHyperGraph hierarchies.
 *
 * @module graphrag/algorithms/shgat/message-passing/phase-interface
 */

/**
 * Sparse connectivity — replaces dense number[][] incidence matrices.
 *
 * Memory: O(edges) instead of O(sources × targets).
 * For 35K edges vs 13.4M dense entries = ~380x less memory.
 */
export interface SparseConnectivity {
  /** For each source index, list of connected target indices */
  sourceToTargets: Map<number, number[]>;
  /** For each target index, list of connected source indices */
  targetToSources: Map<number, number[]>;
  numSources: number;
  numTargets: number;
}

/**
 * Build SparseConnectivity from a dense incidence matrix.
 * Used for backward compatibility with legacy consumers.
 *
 * @param dense - Dense matrix [numSources][numTargets] where 1 = connected
 * @returns Sparse adjacency list representation
 */
export function denseToSparse(dense: number[][]): SparseConnectivity {
  const numSources = dense.length;
  const numTargets = dense[0]?.length ?? 0;
  const sourceToTargets = new Map<number, number[]>();
  const targetToSources = new Map<number, number[]>();

  for (let s = 0; s < numSources; s++) {
    for (let t = 0; t < numTargets; t++) {
      if (dense[s][t] === 1) {
        if (!sourceToTargets.has(s)) sourceToTargets.set(s, []);
        sourceToTargets.get(s)!.push(t);
        if (!targetToSources.has(t)) targetToSources.set(t, []);
        targetToSources.get(t)!.push(s);
      }
    }
  }

  return { sourceToTargets, targetToSources, numSources, numTargets };
}

/**
 * Transpose a SparseConnectivity (swap source/target roles).
 */
export function transposeSparse(conn: SparseConnectivity): SparseConnectivity {
  return {
    sourceToTargets: conn.targetToSources,
    targetToSources: conn.sourceToTargets,
    numSources: conn.numTargets,
    numTargets: conn.numSources,
  };
}

/**
 * Parameters for a single message passing phase
 *
 * Each head has its own set of parameters for projection and attention.
 */
export interface PhaseParameters {
  /** Projection matrix for source nodes [headDim][embeddingDim] */
  W_source: number[][];
  /** Projection matrix for target nodes [headDim][embeddingDim] */
  W_target: number[][];
  /** Attention vector for computing scores [2 * headDim] */
  a_attention: number[];
}

/**
 * Result of a message passing phase
 */
export interface PhaseResult {
  /** Updated embeddings for target nodes [numTargets][headDim] */
  embeddings: number[][];
  /** Attention weights [numSources][numTargets] */
  attention: number[][];
}

/**
 * Message passing phase interface
 *
 * Implementations:
 * - VertexToEdgePhase: V → E^0 (L0 nodes → base L1 nodes)
 * - EdgeToEdgePhase: E^k → E^(k+1) (level-k nodes → level-(k+1) nodes)
 * - EdgeToVertexPhase: E → V (L1+ nodes → L0 nodes, backward pass)
 *
 * All phases follow the same pattern:
 * 1. Project source and target embeddings
 * 2. Compute attention scores (masked by connectivity)
 * 3. Apply softmax normalization
 * 4. Aggregate messages with attention weights
 * 5. Apply activation function
 */
export interface MessagePassingPhase {
  /**
   * Execute message passing from source to target nodes
   *
   * @param sourceEmbeddings - Embeddings of source nodes [numSources][embeddingDim]
   * @param targetEmbeddings - Embeddings of target nodes [numTargets][embeddingDim]
   * @param connectivity - Adjacency/incidence matrix [numSources][numTargets]
   *                       connectivity[s][t] = 1 if source s connects to target t
   * @param params - Phase-specific parameters (W matrices, attention vector)
   * @param config - SHGAT configuration (for activation params)
   * @returns Updated embeddings and attention weights
   */
  forward(
    sourceEmbeddings: number[][],
    targetEmbeddings: number[][],
    connectivity: SparseConnectivity,
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): PhaseResult;

  /**
   * Get human-readable name for this phase
   * Used for logging and debugging
   */
  getName(): string;
}

// ============================================================================
// Unified Phase Cache & Gradients (all phases share the same structure)
// ============================================================================

/** Compute edge key from source/target indices (avoids string GC) */
export function edgeKey(sourceIdx: number, targetIdx: number, numTargets: number): number {
  return sourceIdx * numTargets + targetIdx;
}

/**
 * Unified forward cache for all GAT message passing phases (V→E, E→V, E→E).
 *
 * All phases share the same source→target attention-aggregation structure.
 * Phase-specific names map to source/target:
 *   V→E: source=H(L0),   target=E(L1+)
 *   E→V: source=E(L1+),  target=H(L0)
 *   E→E: source=E_k,     target=E_{k+1}
 */
export interface PhaseForwardCache {
  /** Original source node embeddings [numSource][embDim] */
  source: number[][];
  /** Original target node embeddings [numTarget][embDim] */
  target: number[][];
  /** Projected source embeddings [numSource][headDim] — Float32 for RAM */
  sourceProj: Float32Array[];
  /** Projected target embeddings [numTarget][headDim] — Float32 for RAM */
  targetProj: Float32Array[];
  /** Aggregated values before ELU [numTarget][headDim] — Float32 for RAM */
  aggregated: Float32Array[];
  /** Sparse attention weights (edgeKey → weight) */
  attention: Map<number, number>;
  /** Neighbor map: targetIdx → [sourceIdx, ...] for iteration */
  neighborMap: Map<number, number[]>;
  /** Number of target nodes (for edgeKey computation) */
  numTargets: number;
  /** LeakyReLU slope */
  leakyReluSlope: number;
}

/**
 * Unified gradients from backward pass.
 */
export interface PhaseGradients {
  /** Gradient for W_source [headDim][embDim] */
  dW_source: number[][];
  /** Gradient for W_target [headDim][embDim] */
  dW_target: number[][];
  /** Gradient for a_attention [2*headDim] */
  da_attention: number[];
  /** Gradient for source embeddings [numSource][embDim] */
  dSource: number[][];
  /** Gradient for target embeddings [numTarget][embDim] */
  dTarget: number[][];
}

/**
 * Phase result with forward cache for backward pass.
 */
export interface PhaseResultWithCache extends PhaseResult {
  cache: PhaseForwardCache;
}

/**
 * Multi-level message passing orchestrator
 *
 * Coordinates message passing across multiple hierarchy levels
 * for n-SuperHyperGraph structures.
 */
export interface MultiLevelOrchestrator {
  /**
   * Execute multi-level forward pass
   *
   * @param toolEmbeddings - Initial L0 node embeddings [numL0][embeddingDim]
   * @param capabilityEmbeddings - Initial higher-level node embeddings per level
   *                                [[numL1_0][embeddingDim], [numL1_1][embeddingDim], ...]
   * @param incidenceMatrices - Connectivity matrices per level
   *                            [I_0, I_1, ...] where I_k: V or E^(k-1) → E^k
   * @param layerParams - Parameters for all phases
   * @param config - SHGAT configuration
   * @returns Final embeddings for L0 nodes and all higher-level nodes
   */
  forward(
    toolEmbeddings: number[][],
    capabilityEmbeddings: number[][][],
    incidenceMatrices: number[][][],
    layerParams: any, // Will be typed properly in implementation
    config: any,
  ): {
    toolEmbeddings: number[][];
    capabilityEmbeddings: number[][][];
    attentionWeights: any; // Cache for backprop
  };
}
