/**
 * SHGAT (SuperHyperGraph Attention Networks)
 *
 * Implementation based on "SuperHyperGraph Attention Networks" research paper.
 * Key architecture:
 * - Multi-level message passing: V→E→...→V across hierarchy levels
 * - Incidence matrix A where A[v][e] = 1 if vertex v is in hyperedge e
 * - K-head attention (K=4-16, adaptive) with InfoNCE contrastive loss
 *
 * This file is the main orchestrator that delegates to specialized modules:
 * - graph/: Node registration and incidence matrix
 * - initialization/: Parameter initialization
 * - message-passing/: Multi-level message passing
 * - scoring/: K-head attention scoring
 * - training/: K-head training with PER and curriculum learning
 *
 * @module shgat
 */

import { getLogger } from "./logger.ts";

// Module imports
import {
  GraphBuilder,
  type HierarchyResult,
  type MultiLevelIncidence,
} from "../graph/mod.ts";
import {
  createTensorScoringParamsSync,
  disposeTensorScoringParams,
  initializeParameters,
  type SHGATParams,
  type TensorScoringParams,
} from "../initialization/index.ts";
import { tf } from "../tf/backend.ts";
import {
  DEFAULT_V2V_PARAMS,
  MultiLevelOrchestrator,
  type CooccurrenceEntry,
  type V2VParams,
} from "../message-passing/index.ts";
import {
  tensorForwardPass,
  createTensorLevelParams,
  disposeTensorLevelParams,
  type TensorLevelParams,
  type TensorForwardResult,
} from "../message-passing/tensor-forward.ts";

// K-head scoring functions (extracted)
import {
  scoreAllCapabilities as scoreAllCapabilitiesFn,
  scoreAllTools as scoreAllToolsFn,
  scoreNodesTensorDirect as scoreNodesTensorDirectFn,
  type NodeScore,
} from "../attention/khead-scorer.ts";

// NOTE: Training moved to AutogradTrainer in training/autograd-trainer.ts
// Use AutogradTrainer for training with TF.js autograd

// Forward pass helpers (extracted)
import {
  forwardCore,
  buildToolToCapMatrix,
  buildCapToCapMatrices,
  type ForwardPassContext,
} from "./forward-helpers.ts";

// Serialization helpers (extracted)
import {
  exportSHGATParams,
  importSHGATParams,
  type SerializationContext,
} from "./serialization.ts";

// Scoring helpers (extracted)
import {
  getCapabilityToolAttention as getCapToolAttentionFn,
  predictPathSuccess as predictPathSuccessFn,
  computeAttentionForCapability,
  type ScoringContext,
} from "./scoring-helpers.ts";

// Hierarchy builder (extracted)
import { rebuildHierarchy as rebuildHierarchyFn } from "./hierarchy-builder.ts";

// Stats helper (extracted)
import { computeStats, type SHGATStats } from "./stats.ts";

// Re-export all types from ./shgat/types.ts for backward compatibility
export {
  type AttentionResult,
  buildGraph,
  type CapabilityNode,
  computeAllLevels,
  createDefaultTraceFeatures,
  DEFAULT_FEATURE_WEIGHTS,
  DEFAULT_FUSION_WEIGHTS,
  DEFAULT_HYPERGRAPH_FEATURES,
  DEFAULT_SHGAT_CONFIG,
  DEFAULT_TOOL_GRAPH_FEATURES,
  DEFAULT_TRACE_STATS,
  type FeatureWeights,
  type ForwardCache,
  type FusionWeights,
  getAdaptiveConfig,
  type HypergraphFeatures,
  type Node,
  NUM_TRACE_STATS,
  type SHGATConfig,
  type ToolGraphFeatures,
  type ToolNode,
  type TraceFeatures,
  type TraceStats,
  type TrainingExample,
} from "./types.ts";

// Export seeded RNG for reproducibility
export { seedRng } from "../initialization/parameters.ts";

// Export helper for generating tool embeddings
export { generateDefaultToolEmbedding } from "../graph/mod.ts";

import {
  type AttentionResult,
  type CapabilityNode,
  createMembersFromLegacy,
  DEFAULT_SHGAT_CONFIG,
  type ForwardCache,
  type FusionWeights,
  type HypergraphFeatures,
  type LevelParams,
  type Node,
  type SHGATConfig,
  type ToolGraphFeatures,
  type ToolNode,
  type TrainingExample,
} from "./types.ts";

// Auto-initialize BLAS acceleration on module load
import { initBlasAcceleration } from "../utils/math.ts";
await initBlasAcceleration();

const log = getLogger();

// ============================================================================
// SHGAT Implementation
// ============================================================================

/**
 * SuperHyperGraph Attention Networks
 *
 * Implements proper two-phase message passing:
 * 1. Vertex → Hyperedge: Aggregate tool features to capabilities
 * 2. Hyperedge → Vertex: Propagate capability features back to tools
 */
export class SHGAT {
  private config: SHGATConfig;
  private graphBuilder: GraphBuilder;
  private params: SHGATParams;
  private orchestrator: MultiLevelOrchestrator;
  private trainingMode = false;
  private lastCache: ForwardCache | null = null;

  // Multi-level n-SuperHyperGraph structures
  private hierarchy: HierarchyResult | null = null;
  private multiLevelIncidence: MultiLevelIncidence | null = null;
  private levelParams: Map<number, LevelParams> = new Map();
  private hierarchyDirty = true; // Flag to rebuild hierarchy when graph changes

  // V→V trainable parameters (co-occurrence enrichment)
  private v2vParams: V2VParams = { ...DEFAULT_V2V_PARAMS };

  // GPU-accelerated tensor parameters for scoring
  // Initialized lazily on first scoreNodes() call
  private tensorParams: TensorScoringParams | null = null;

  // Tensor-based level parameters for message passing
  // Initialized lazily on first tensor forward pass
  private tensorLevelParams: Map<number, TensorLevelParams> | null = null;

  // Cached tensor embeddings (converted once from arrays)
  private tensorEmbeddingsCache: {
    H: tf.Tensor2D;
    E: Map<number, tf.Tensor2D>;
    toolIds: string[];
    capIdsByLevel: Map<number, string[]>;
    toolToCapMatrix: tf.Tensor2D;
    capToCapMatrices: Map<number, tf.Tensor2D>;
  } | null = null;

  constructor(config: Partial<SHGATConfig> = {}) {
    this.config = { ...DEFAULT_SHGAT_CONFIG, ...config };

    // Note: preserveDim affects levelParams (message passing keeps 1024-dim)
    // hiddenDim = numHeads * 16 for K-head scoring (adaptive: 64, 128, etc.)
    // Each head gets 16 dims for consistent expressiveness

    this.graphBuilder = new GraphBuilder();
    // Pass v2vResidual to the V2V phase config
    const v2vConfig = this.config.v2vResidual !== undefined && this.config.v2vResidual > 0
      ? { residualWeight: this.config.v2vResidual }
      : undefined;
    this.orchestrator = new MultiLevelOrchestrator(this.trainingMode, v2vConfig);
    this.params = initializeParameters(this.config);
  }

  // ==========================================================================
  // Graph Management (delegated to GraphBuilder)
  // ==========================================================================

  /**
   * Register a unified node
   *
   * @param node Node to register
   */
  registerNode(node: Node): void {
    this.graphBuilder.registerNode(node);
    this.hierarchyDirty = true;
  }

  /**
   * Finalize node registration - call after registering all nodes
   * Rebuilds indices once for efficiency.
   */
  finalizeNodes(): void {
    this.graphBuilder.finalizeNodes();
    this.hierarchyDirty = true;
  }

  /**
   * Register a tool (vertex)
   * @deprecated Use registerNode() with children: [] instead
   */
  registerTool(node: ToolNode): void {
    this.graphBuilder.registerTool(node);
    this.hierarchyDirty = true;
  }

  /**
   * Register a capability (hyperedge)
   * @deprecated Use registerNode() with children: [...] instead
   */
  registerCapability(node: CapabilityNode): void {
    this.graphBuilder.registerCapability(node);
    this.hierarchyDirty = true;
  }

  /** Set V→V co-occurrence data for tool embedding enrichment */
  setCooccurrenceData(data: CooccurrenceEntry[]): void {
    this.orchestrator.setCooccurrenceData(data);
    log.info(`[SHGAT] V→V co-occurrence enabled with ${data.length} edges`);
  }

  /**
   * Get tool ID to index mapping for co-occurrence loader
   */
  getToolIndexMap(): Map<string, number> {
    return this.graphBuilder.getToolIndexMap();
  }

  /**
   * Rebuild multi-level hierarchy and incidence structures
   *
   * Called lazily before forward() when hierarchyDirty is true.
   */
  private rebuildHierarchy(): void {
    if (!this.hierarchyDirty) return;

    const result = rebuildHierarchyFn(this.config, this.graphBuilder, this.levelParams);
    this.hierarchy = result.hierarchy;
    this.multiLevelIncidence = result.multiLevelIncidence;
    this.levelParams = result.levelParams;
    this.hierarchyDirty = false;
  }

  /** @deprecated Use registerCapability() with members array */
  addCapabilityLegacy(
    id: string,
    embedding: number[],
    toolsUsed: string[],
    children: string[] = [],
    successRate: number = 0.5,
  ): void {
    const members = createMembersFromLegacy(toolsUsed, children);

    this.registerCapability({
      id,
      embedding,
      members,
      hierarchyLevel: 0, // Will be recomputed during rebuild
      toolsUsed, // Keep for backward compat
      children,
      successRate,
    });
  }

  hasToolNode(toolId: string): boolean { return this.graphBuilder.hasToolNode(toolId); }
  hasCapabilityNode(capabilityId: string): boolean { return this.graphBuilder.hasCapabilityNode(capabilityId); }
  getToolCount(): number { return this.graphBuilder.getToolCount(); }
  getCapabilityCount(): number { return this.graphBuilder.getCapabilityCount(); }
  getToolIds(): string[] { return this.graphBuilder.getToolIds(); }
  getCapabilityIds(): string[] { return this.graphBuilder.getCapabilityIds(); }

  buildFromData(
    tools: Array<{ id: string; embedding: number[] }>,
    capabilities: Array<{
      id: string;
      embedding: number[];
      toolsUsed: string[];
      successRate: number;
      parents?: string[];
      children?: string[];
    }>,
  ): void {
    this.graphBuilder.buildFromData({ tools, capabilities });
  }

  updateHypergraphFeatures(capabilityId: string, features: Partial<HypergraphFeatures>): void {
    this.graphBuilder.updateHypergraphFeatures(capabilityId, features);
  }
  updateToolFeatures(toolId: string, features: Partial<ToolGraphFeatures>): void {
    this.graphBuilder.updateToolFeatures(toolId, features);
  }
  batchUpdateFeatures(updates: Map<string, Partial<HypergraphFeatures>>): void {
    this.graphBuilder.batchUpdateCapabilityFeatures(updates);
  }
  batchUpdateToolFeatures(updates: Map<string, Partial<ToolGraphFeatures>>): void {
    this.graphBuilder.batchUpdateToolFeatures(updates);
  }

  // ==========================================================================
  // Multi-Level Message Passing (n-SuperHyperGraph)
  // ==========================================================================

  /** Execute multi-level message passing (V→E→...→V) */
  forward(): { H: number[][]; E: number[][]; cache: ForwardCache } {
    // Return cached result if graph hasn't changed
    if (!this.hierarchyDirty && this.lastCache) {
      const H = this.lastCache.H[this.lastCache.H.length - 1] ?? [];
      const E = this.lastCache.E[this.lastCache.E.length - 1] ?? [];
      return { H, E, cache: this.lastCache };
    }

    // Rebuild hierarchy if needed
    this.rebuildHierarchy();

    // Delegate to extracted core function
    const result = forwardCore(this.getForwardPassContext());
    this.lastCache = result.cache;
    return result;
  }

  /** Get context for forward pass */
  private getForwardPassContext(): ForwardPassContext {
    return {
      config: this.config,
      graphBuilder: this.graphBuilder,
      hierarchy: this.hierarchy,
      multiLevelIncidence: this.multiLevelIncidence,
      levelParams: this.levelParams,
      orchestrator: this.orchestrator,
      residualLogits: this.params.residualLogits,
    };
  }

  // ==========================================================================
  // Scoring (K-head Attention)
  // ==========================================================================

  /**
   * Score all capabilities using K-head attention after message passing
   * @deprecated Use scoreNodes(intentEmbedding, 1) instead for better performance
   */
  scoreAllCapabilities(intentEmbedding: number[], _contextToolIds?: string[]): AttentionResult[] {
    // Use tensor-native forward pass then convert to arrays for legacy API
    const { E: E_map } = this.forwardTensor();
    const cache = this.tensorEmbeddingsCache!;

    // Build E array from all hierarchy levels (matching original behavior)
    const E: number[][] = [];
    for (const [hierLevel] of cache.capIdsByLevel) {
      const E_tensor = E_map.get(hierLevel);
      if (E_tensor) {
        const E_array = E_tensor.arraySync() as number[][];
        E.push(...E_array);
      }
    }

    // Cleanup tensors
    for (const [, tensor] of E_map) {
      tensor.dispose();
    }

    return scoreAllCapabilitiesFn(
      E,
      intentEmbedding,
      this.graphBuilder.getCapabilityNodes(),
      this.params.headParams,
      this.params.W_intent,
      this.config,
      (capIdx) => this.getCapabilityToolAttention(capIdx),
    );
  }

  /**
   * Score all tools using K-head attention
   * @deprecated Use scoreNodes(intentEmbedding, 0) instead for better performance
   */
  scoreAllTools(
    intentEmbedding: number[],
    _contextToolIds?: string[],
  ): Array<{ toolId: string; score: number; headScores: number[] }> {
    // Use tensor-native forward pass then convert to arrays for legacy API
    const { H: H_tensor, E: E_map } = this.forwardTensor();
    const H = H_tensor.arraySync() as number[][];

    // Cleanup tensors
    H_tensor.dispose();
    for (const [, tensor] of E_map) {
      tensor.dispose();
    }

    const toolIds = Array.from(this.graphBuilder.getToolNodes().keys());
    return scoreAllToolsFn(
      H,
      intentEmbedding,
      toolIds,
      this.params.headParams,
      this.params.W_intent,
      this.config,
    );
  }

  // ==========================================================================
  // Unified Node Scoring (new API)
  // ==========================================================================

  /**
   * Ensure tensor scoring parameters are initialized (lazy initialization)
   */
  private ensureTensorParams(): TensorScoringParams {
    if (!this.tensorParams) {
      this.tensorParams = createTensorScoringParamsSync(this.params, tf);
    }
    return this.tensorParams;
  }

  /**
   * Ensure tensor level parameters are initialized (lazy initialization)
   */
  private ensureTensorLevelParams(): Map<number, TensorLevelParams> {
    if (!this.tensorLevelParams) {
      this.tensorLevelParams = new Map();
      let instanceCounter = 0;
      for (const [level, params] of this.levelParams) {
        this.tensorLevelParams.set(level, createTensorLevelParams(params, instanceCounter++));
      }
    }
    return this.tensorLevelParams;
  }

  /**
   * Ensure tensor embeddings cache is initialized
   *
   * Converts array embeddings to tensors once, then reuses for all forward passes.
   */
  private ensureTensorEmbeddings(): NonNullable<typeof this.tensorEmbeddingsCache> {
    if (!this.tensorEmbeddingsCache) {
      // Rebuild hierarchy if not already built
      if (!this.hierarchy) {
        this.rebuildHierarchy();
      }

      // Get array embeddings
      const H_array = this.graphBuilder.getToolEmbeddings();
      const capabilityNodes = this.graphBuilder.getCapabilityNodes();
      const toolIds = this.graphBuilder.getToolIds();

      // Build E_levels (capabilities grouped by hierarchy level) with corresponding IDs
      const E_levels = new Map<number, number[][]>();
      const capIdsByLevel = new Map<number, string[]>();
      if (this.hierarchy) {
        for (let level = 0; level <= this.hierarchy.maxHierarchyLevel; level++) {
          const capsAtLevel = this.hierarchy.hierarchyLevels.get(level) ?? new Set<string>();
          const embeddings: number[][] = [];
          const ids: string[] = [];
          for (const capId of capsAtLevel) {
            const cap = capabilityNodes.get(capId);
            if (cap) {
              embeddings.push([...cap.embedding]);
              ids.push(capId);
            }
          }
          if (embeddings.length > 0) {
            E_levels.set(level, embeddings);
            capIdsByLevel.set(level, ids);
          }
        }
      }

      // Build incidence matrices
      const ctx = {
        config: this.config,
        graphBuilder: this.graphBuilder,
        hierarchy: this.hierarchy,
        multiLevelIncidence: this.multiLevelIncidence,
      };
      const toolToCapMatrix_array = buildToolToCapMatrix(ctx);
      const capToCapMatrices_array = buildCapToCapMatrices(ctx);

      // Convert to tensors
      const H = tf.tensor2d(H_array);
      const E = new Map<number, tf.Tensor2D>();
      for (const [level, embs] of E_levels) {
        E.set(level, tf.tensor2d(embs));
      }
      const toolToCapMatrix = toolToCapMatrix_array.length > 0
        ? tf.tensor2d(toolToCapMatrix_array)
        : tf.zeros([H_array.length, capabilityNodes.size]) as tf.Tensor2D;
      const capToCapMatrices = new Map<number, tf.Tensor2D>();
      for (const [level, matrix] of capToCapMatrices_array) {
        capToCapMatrices.set(level, tf.tensor2d(matrix));
      }

      this.tensorEmbeddingsCache = { H, E, toolIds, capIdsByLevel, toolToCapMatrix, capToCapMatrices };
    }
    return this.tensorEmbeddingsCache;
  }

  /**
   * Tensor-native forward pass
   *
   * All computations stay in tensors. Much faster than array-based forward().
   */
  private forwardTensor(): TensorForwardResult {
    const tensorLevelParams = this.ensureTensorLevelParams();
    const { H, E, toolToCapMatrix, capToCapMatrices } = this.ensureTensorEmbeddings();

    return tensorForwardPass(
      H,
      E,
      toolToCapMatrix,
      capToCapMatrices,
      tensorLevelParams,
      {
        numHeads: this.config.numHeads,
        leakyReluSlope: this.config.leakyReluSlope,
        preserveDim: this.config.preserveDim ?? false,
        preserveDimResidual: this.config.preserveDimResidual,
      },
    );
  }

  /**
   * Score nodes using K-head attention (unified API) - FULLY GPU-ACCELERATED
   *
   * This is the main scoring function for the unified Node API.
   * It replaces the legacy scoreAllCapabilities/scoreAllTools for new code.
   *
   * PERFORMANCE: Uses tensor-native forward pass + tensor-native scoring.
   * All computations stay in tensors except final array conversion.
   * Expected ~10-20x speedup over array-based version.
   *
   * @param intentEmbedding - User intent embedding
   * @param level - Optional level filter. If undefined, scores all nodes.
   * @returns Sorted array of node scores
   */
  scoreNodes(intentEmbedding: number[], level?: number): NodeScore[] {
    // Run tensor-native forward pass
    const { H: H_tensor, E: E_map } = this.forwardTensor();

    // Get cached IDs (same order as embeddings in forward pass)
    const cache = this.tensorEmbeddingsCache!;

    // Build node metadata and collect tensors to concatenate
    const nodeIds: string[] = [];
    const nodeLevels: number[] = [];
    const tensorsToConcat: tf.Tensor2D[] = [];

    // Tools are level 0 (API level, not hierarchy level)
    if (level === undefined || level === 0) {
      for (const toolId of cache.toolIds) {
        nodeIds.push(toolId);
        nodeLevels.push(0);
      }
      tensorsToConcat.push(H_tensor);
    }

    // Capabilities are level 1 (API level) - combine all hierarchy levels
    if (level === undefined || level === 1) {
      for (const [hierLevel, capIds] of cache.capIdsByLevel) {
        const E_tensor = E_map.get(hierLevel);
        if (!E_tensor) continue;
        for (const capId of capIds) {
          nodeIds.push(capId);
          nodeLevels.push(1);
        }
        tensorsToConcat.push(E_tensor);
      }
    }

    if (tensorsToConcat.length === 0 || nodeIds.length === 0) {
      // Cleanup and return empty
      H_tensor.dispose();
      for (const [, tensor] of E_map) {
        tensor.dispose();
      }
      return [];
    }

    // Concatenate all embeddings into single tensor (GPU operation)
    const embeddingsTensor = tensorsToConcat.length === 1
      ? tensorsToConcat[0]
      : tf.concat(tensorsToConcat, 0) as tf.Tensor2D;

    // Score using tensor-native function (no array conversion until the very end)
    const tensorParams = this.ensureTensorParams();
    const results = scoreNodesTensorDirectFn(
      embeddingsTensor,
      nodeIds,
      nodeLevels,
      intentEmbedding,
      tensorParams,
      this.config,
      tensorParams.projectionHead,
    );

    // Clean up forward pass tensors
    if (tensorsToConcat.length > 1) {
      embeddingsTensor.dispose(); // Only dispose if we created a new concat tensor
    }
    H_tensor.dispose();
    for (const [, tensor] of E_map) {
      tensor.dispose();
    }

    return results;
  }

  /**
   * Score only leaf nodes (level 0)
   *
   * Convenience method equivalent to scoreNodes(intent, 0)
   */
  scoreLeaves(intentEmbedding: number[]): NodeScore[] {
    return this.scoreNodes(intentEmbedding, 0);
  }

  /**
   * Score only composite nodes at a given level (default: 1)
   *
   * Convenience method for scoring higher-level nodes
   */
  scoreComposites(intentEmbedding: number[], level: number = 1): NodeScore[] {
    return this.scoreNodes(intentEmbedding, level);
  }

  predictPathSuccess(intentEmbedding: number[], path: string[]): number {
    return predictPathSuccessFn(this.getScoringContext(), intentEmbedding, path);
  }

  computeAttention(
    intentEmbedding: number[],
    _contextToolEmbeddings: number[][],
    capabilityId: string,
    _contextCapabilityIds?: string[],
  ): AttentionResult {
    return computeAttentionForCapability(this.getScoringContext(), intentEmbedding, capabilityId);
  }

  /** Get scoring context for extracted scoring functions */
  private getScoringContext(): ScoringContext {
    return {
      config: this.config,
      graphBuilder: this.graphBuilder,
      lastCache: this.lastCache,
      scoreAllCapabilities: (e) => this.scoreAllCapabilities(e),
      scoreAllTools: (e) => this.scoreAllTools(e),
    };
  }

  private getCapabilityToolAttention(capIdx: number): number[] {
    return getCapToolAttentionFn(this.getScoringContext(), capIdx);
  }

  // ==========================================================================
  // Training
  // ==========================================================================

  /**
   * @deprecated Use AutogradTrainer from training/autograd-trainer.ts instead.
   *
   * The new trainer uses TensorFlow.js automatic differentiation which:
   * - Eliminates 3000+ lines of manual backward passes
   * - Provides GPU acceleration via WebGPU
   * - Is more maintainable and less error-prone
   *
   * @example
   * ```typescript
   * import { AutogradTrainer } from "./training/autograd-trainer.ts";
   *
   * const trainer = new AutogradTrainer(config);
   * trainer.setNodeEmbeddings(embeddings);
   * const metrics = trainer.trainBatch(examples);
   * ```
   */
  trainBatchV1KHeadBatched(
    _examples: TrainingExample[],
    _isWeights?: number[],
    _evaluateOnly = false,
    _temperature = 0.1,
  ) {
    throw new Error(
      "trainBatchV1KHeadBatched is deprecated in shgat-tf. " +
      "Use AutogradTrainer from training/autograd-trainer.ts instead."
    );
  }

  // ==========================================================================
  // Serialization (delegated)
  // ==========================================================================

  exportParams(): Record<string, unknown> {
    return exportSHGATParams(this.getSerializationContext());
  }

  importParams(serialized: Record<string, unknown>): void {
    const result = importSHGATParams(serialized, this.params, this.v2vParams);
    if (result.config) this.config = result.config;
    this.params = result.params;
    if (result.levelParams.size > 0) this.levelParams = result.levelParams;
    this.v2vParams = result.v2vParams;

    // Invalidate tensor params - they'll be recreated on next scoreNodes() call
    if (this.tensorParams) {
      disposeTensorScoringParams(this.tensorParams);
      this.tensorParams = null;
    }
  }

  /** Get serialization context */
  private getSerializationContext(): SerializationContext {
    return {
      config: this.config,
      params: this.params,
      levelParams: this.levelParams,
      v2vParams: this.v2vParams,
    };
  }

  // ==========================================================================
  // Accessors & Utilities
  // ==========================================================================

  /** @deprecated Fusion weights are legacy - K-head scoring uses learned attention */
  getFusionWeights(): FusionWeights { return this.params.fusionWeights; }
  setFusionWeights(weights: Partial<FusionWeights>): void {
    Object.assign(this.params.fusionWeights, weights);
  }
  getLearningRate(): number { return this.config.learningRate; }
  setLearningRate(lr: number): void { this.config.learningRate = lr; }

  /** @deprecated Use getToolIds() */
  getRegisteredToolIds(): string[] { return this.getToolIds(); }

  /** @deprecated Use getCapabilityIds() */
  getRegisteredCapabilityIds(): string[] { return this.getCapabilityIds(); }

  /** Get all tool embeddings for negative sampling */
  getToolEmbeddings(): Map<string, number[]> {
    return new Map(
      Array.from(this.graphBuilder.getToolNodes())
        .filter(([, t]) => t.embedding)
        .map(([id, t]) => [id, t.embedding!]),
    );
  }

  getStats(): SHGATStats {
    return computeStats({
      config: this.config,
      graphBuilder: this.graphBuilder,
      getFusionWeights: () => this.getFusionWeights(),
    });
  }

  /**
   * Dispose GPU resources
   *
   * IMPORTANT: Call this when the SHGAT instance is no longer needed
   * to free GPU memory used by tensor parameters.
   *
   * After calling dispose(), the instance can still be used - tensor
   * parameters will be recreated on the next scoreNodes() call.
   */
  dispose(): void {
    // Dispose scoring params
    if (this.tensorParams) {
      disposeTensorScoringParams(this.tensorParams);
      this.tensorParams = null;
    }

    // Dispose level params
    if (this.tensorLevelParams) {
      for (const [, params] of this.tensorLevelParams) {
        disposeTensorLevelParams(params);
      }
      this.tensorLevelParams = null;
    }

    // Dispose embeddings cache
    if (this.tensorEmbeddingsCache) {
      this.tensorEmbeddingsCache.H.dispose();
      for (const [, tensor] of this.tensorEmbeddingsCache.E) {
        tensor.dispose();
      }
      this.tensorEmbeddingsCache.toolToCapMatrix.dispose();
      for (const [, tensor] of this.tensorEmbeddingsCache.capToCapMatrices) {
        tensor.dispose();
      }
      this.tensorEmbeddingsCache = null;
    }
  }
}

// ============================================================================
// Factory Functions (re-exported from factory.ts)
// ============================================================================

export {
  createSHGAT,
  createSHGATFromCapabilities,
  trainSHGATOnEpisodes,
  trainSHGATOnEpisodesKHead,
  trainSHGATOnExecution,
} from "./factory.ts";
