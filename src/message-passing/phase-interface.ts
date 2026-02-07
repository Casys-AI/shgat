/**
 * Message Passing Phase Interface
 *
 * Defines the contract for pluggable message passing phases in SHGAT.
 * Enables multi-level message passing for n-SuperHyperGraph hierarchies.
 *
 * @module graphrag/algorithms/shgat/message-passing/phase-interface
 */

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
 * - VertexToEdgePhase: V → E^0 (tools → base capabilities)
 * - EdgeToEdgePhase: E^k → E^(k+1) (capability level k → level k+1)
 * - EdgeToVertexPhase: E → V (capabilities → tools, backward pass)
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
    connectivity: number[][],
    params: PhaseParameters,
    config: { leakyReluSlope: number },
  ): PhaseResult;

  /**
   * Get human-readable name for this phase
   * Used for logging and debugging
   */
  getName(): string;
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
   * @param toolEmbeddings - Initial tool embeddings [numTools][embeddingDim]
   * @param capabilityEmbeddings - Initial capability embeddings per level
   *                                [[numCaps_0][embeddingDim], [numCaps_1][embeddingDim], ...]
   * @param incidenceMatrices - Connectivity matrices per level
   *                            [I_0, I_1, ...] where I_k: V or E^(k-1) → E^k
   * @param layerParams - Parameters for all phases
   * @param config - SHGAT configuration
   * @returns Final embeddings for tools and all capability levels
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
