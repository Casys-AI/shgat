/**
 * SHGAT-TF Builder & Ports
 *
 * Clean API for constructing and using SHGAT-TF.
 * Consolidates the scattered constructor parameters into a single builder.
 *
 * ## Architecture Ports:
 * - `SHGATScorer` — Score nodes for an intent (inference)
 * - `SHGATTrainer` — Train parameters from examples
 * - `SHGATTrainerScorer` — Full training + inference (combined)
 *
 * ## Usage:
 * ```typescript
 * import { SHGATBuilder, type SHGATTrainerScorer } from "@casys/shgat-tf";
 *
 * const shgat: SHGATTrainerScorer = await SHGATBuilder.create()
 *   .nodes(myNodes)                        // Nodes with embeddings
 *   .training({ learningRate: 0.05 })      // Training hyperparams
 *   .build();
 *
 * // Score
 * const scores = shgat.score(intentEmbedding, capabilityIds);
 *
 * // Train
 * const metrics = await shgat.trainBatch(examples);
 *
 * // Dispose when done
 * shgat.dispose();
 * ```
 *
 * @module shgat-tf/core/builder
 */

import { tf } from "../tf/backend.ts";
import type { Node, SHGATConfig, SoftTargetExample, TrainingExample } from "./types.ts";
import { DEFAULT_SHGAT_CONFIG } from "./types.ts";
import {
  AutogradTrainer,
  buildGraphStructure,
  type CapabilityInfo,
  DEFAULT_TRAINER_CONFIG,
  disposeGraphStructure,
  type GraphStructure,
  type KLTrainingMetrics,
  type TrainerConfig,
  type TrainingMetrics,
} from "../training/autograd-trainer.ts";
import { initTensorFlow } from "../tf/backend.ts";
import type { BackendMode } from "../tf/backend.ts";

// ============================================================================
// Ports (Interfaces)
// ============================================================================

/**
 * Inference port — score nodes for a given intent.
 *
 * Implementations may use message passing or direct K-head attention.
 */
export interface SHGATScorer {
  /**
   * Score nodes for an intent embedding.
   *
   * @param intentEmbedding - User intent (e.g. BGE-M3, 1024-dim)
   * @param nodeIds - IDs of nodes to score
   * @returns Scores in same order as nodeIds (higher = better match)
   */
  score(intentEmbedding: number[], nodeIds: string[]): number[];

  /** Whether message passing is enabled (enriches embeddings before scoring) */
  readonly hasMessagePassing: boolean;

  /** Release GPU/tensor resources */
  dispose(): void;
}

/**
 * Training port — train SHGAT parameters from contrastive examples.
 */
export interface SHGATTrainer {
  /**
   * Train on a batch of contrastive examples.
   *
   * Each example has a positive candidate + negative candidates.
   * Uses InfoNCE loss with temperature annealing.
   *
   * @param examples - Training batch
   * @returns Metrics (loss, accuracy, gradient norm)
   */
  trainBatch(examples: TrainingExample[]): Promise<TrainingMetrics>;

  /**
   * Train on a batch of KL divergence examples (soft targets from n8n workflows).
   *
   * Scores ALL tools for each intent, then minimizes KL(target || predicted).
   * K-heads learn to match cosine similarity distributions.
   *
   * @param examples - Soft target examples with sparse probability distributions
   * @param toolIds - All tool IDs in the vocabulary (order matches target indices)
   * @param klTemperature - Temperature for softmax on predicted scores
   * @returns KL training metrics
   */
  trainBatchKL(
    examples: SoftTargetExample[],
    toolIds: string[],
    klTemperature: number,
    klWeight?: number,
  ): Promise<KLTrainingMetrics>;

  /**
   * Update InfoNCE temperature (for cosine annealing during training).
   *
   * @param temperature - New temperature (e.g. 0.10 → 0.06 over epochs)
   */
  setTemperature(temperature: number): void;

  /**
   * Update learning rate (for warmup + cosine decay scheduling).
   *
   * @param lr - New learning rate
   */
  setLearningRate(lr: number): void;

  /**
   * Configure subgraph neighbor sampling K parameter for mini-batch MP.
   * @param K - Max neighbors per node (default 8, recommended 16)
   */
  setSubgraphK(K: number): void;

  /**
   * Pre-compute enriched embeddings via message passing OUTSIDE the gradient tape.
   * Call once per epoch before trainBatch/trainBatchKL.
   * Reduces autograd tape memory from ~3GB to ~50MB.
   */
  precomputeEnrichedEmbeddings(): void;

  /**
   * Export trained parameters as plain JS arrays for serialization.
   * Returns a JSON-serializable object.
   */
  exportParams(): Record<string, unknown>;

  /** Release GPU/tensor resources */
  dispose(): void;
}

/**
 * Combined training + inference port.
 *
 * This is the main interface for most use cases.
 */
export interface SHGATTrainerScorer extends SHGATScorer, SHGATTrainer {}

// ============================================================================
// Builder Configuration
// ============================================================================

/**
 * Node data for the builder.
 *
 * Minimal shape: `{ id, embedding, children }`.
 * The builder computes levels automatically.
 */
export type NodeInput = Pick<Node, "id" | "embedding" | "children">;

/**
 * Training options for the builder.
 */
export interface TrainingOptions {
  /** Learning rate (default: 0.001) */
  learningRate?: number;
  /** Batch size (default: 32) */
  batchSize?: number;
  /** InfoNCE temperature (default: 0.07) */
  temperature?: number;
  /** Max gradient norm (default: 1.0) */
  gradientClip?: number;
  /** L2 regularization weight (default: 0.0001) */
  l2Lambda?: number;
}

/**
 * Architecture options for the builder.
 *
 * Most users should NOT set these — the defaults are tuned for BGE-M3 (1024-dim).
 * Only customize if using different embeddings.
 */
export interface ArchitectureOptions {
  /** Embedding dimension (default: 1024 for BGE-M3) */
  embeddingDim?: number;
  /** Number of attention heads (default: 16) */
  numHeads?: number;
  /** Dimension per head (default: 64) */
  headDim?: number;
  /** Hidden dimension for projections (default: 1024) */
  hiddenDim?: number;
  /** Preserve embedding dimension through message passing (default: true) */
  preserveDim?: boolean;
  /** Enable projection head for fine-grained discrimination (default: false) */
  useProjectionHead?: boolean;
  /**
   * Residual weight for downward message passing phase.
   * output = (1-r)*propagated + r*original
   * @default 0 (no residual in downward — WARNING: causes collapse with hierarchy)
   */
  downwardResidual?: number;
  /**
   * Global residual weight applied after message passing.
   * final = (1-r)*propagated + r*original
   * @default 0.3
   */
  preserveDimResidual?: number;
  /**
   * Per-level residual weights applied after message passing.
   * Index corresponds to node level (0=leaves, 1=intermediate, 2=root, etc.)
   * Overrides preserveDimResidual for nodes at each specified level.
   * @example [0.95, 0.7, 0.5] // L0: 95% original, L1: 70%, L2: 50%
   */
  preserveDimResiduals?: number[];
  /**
   * Random seed for deterministic parameter initialization.
   * When set, all random matrices (W_k, W_up, W_down, etc.) are initialized
   * with deterministic values, making results reproducible across runs.
   */
  seed?: number;
  /**
   * Learning rate multiplier for MP parameters (W_up, W_down, a_up, a_down).
   * Values < 1 dampen noisy gradients from subgraph sampling.
   * Values > 1 amplify vanishing gradients through attention layers.
   * @default 1 (same learning rate as K-head)
   */
  mpLearningRateScale?: number;
}

// ============================================================================
// Builder Implementation
// ============================================================================

/**
 * Fluent builder for SHGAT-TF.
 *
 * Consolidates all configuration into a single, discoverable API.
 * Call `.build()` to get a ready-to-use `SHGATTrainerScorer`.
 *
 * @example
 * ```typescript
 * // Minimal (inference only)
 * const scorer = await SHGATBuilder.create()
 *   .nodes(myNodes)
 *   .build();
 *
 * // Full training setup
 * const shgat = await SHGATBuilder.create()
 *   .nodes(myNodes)
 *   .training({ learningRate: 0.05, temperature: 0.10 })
 *   .architecture({ numHeads: 16 })
 *   .backend("training")
 *   .build();
 * ```
 */
export class SHGATBuilder {
  private _nodes: NodeInput[] = [];
  private _trainingOpts: TrainingOptions = {};
  private _archOpts: ArchitectureOptions = {};
  private _backendMode: BackendMode | "cpu" | "webgpu" | undefined;

  private constructor() {}

  /** Create a new builder */
  static create(): SHGATBuilder {
    return new SHGATBuilder();
  }

  /**
   * Set the graph nodes.
   *
   * Leaves have `children: []`, composites list their child IDs.
   * Levels are computed automatically from the structure.
   *
   * @param nodes - Array of nodes with embeddings
   */
  nodes(nodes: NodeInput[]): this {
    this._nodes = nodes;
    return this;
  }

  /**
   * Set training hyperparameters.
   *
   * If not called, default training config is used.
   * @param opts - Training options (all optional, merged with defaults)
   */
  training(opts: TrainingOptions): this {
    this._trainingOpts = opts;
    return this;
  }

  /**
   * Set architecture parameters.
   *
   * Most users should NOT call this — defaults are tuned for BGE-M3 (1024-dim).
   *
   * @param opts - Architecture options (all optional, merged with defaults)
   */
  architecture(opts: ArchitectureOptions): this {
    this._archOpts = opts;
    return this;
  }

  /**
   * Set TensorFlow.js backend mode.
   *
   * - `"training"`: WebGPU > CPU (full autograd, never WASM)
   * - `"inference"`: WebGPU > WASM > CPU (speed priority)
   * - `"cpu"` / `"webgpu"`: Force specific backend
   *
   * Default: `"training"` (safe choice — supports both training and inference).
   *
   * @param mode - Backend selection mode
   */
  backend(mode: BackendMode | "cpu" | "webgpu"): this {
    this._backendMode = mode;
    return this;
  }

  /**
   * Build and return a ready-to-use SHGAT instance.
   *
   * This:
   * 1. Initializes the TF.js backend
   * 2. Separates nodes into tools (leaves) and capabilities (composites)
   * 3. Builds the graph structure (incidence matrices)
   * 4. Creates the trainer with all parameters
   * 5. Returns a `SHGATTrainerScorer` ready for scoring and training
   *
   * @throws Error if no nodes are provided
   */
  async build(): Promise<SHGATTrainerScorer> {
    // Validate
    if (this._nodes.length === 0) {
      throw new Error(
        "[SHGATBuilder] No nodes provided. Call .nodes([...]) before .build().",
      );
    }

    // 1. Initialize backend
    const backendMode = this._backendMode ?? "training";
    await initTensorFlow(backendMode as BackendMode);

    // 2. Build config
    const config: SHGATConfig = {
      ...DEFAULT_SHGAT_CONFIG,
      ...(this._archOpts.embeddingDim !== undefined &&
        { embeddingDim: this._archOpts.embeddingDim }),
      ...(this._archOpts.numHeads !== undefined && { numHeads: this._archOpts.numHeads }),
      ...(this._archOpts.headDim !== undefined && { headDim: this._archOpts.headDim }),
      ...(this._archOpts.hiddenDim !== undefined && { hiddenDim: this._archOpts.hiddenDim }),
      ...(this._archOpts.preserveDim !== undefined && { preserveDim: this._archOpts.preserveDim }),
      ...(this._archOpts.useProjectionHead !== undefined &&
        { useProjectionHead: this._archOpts.useProjectionHead }),
      ...(this._archOpts.downwardResidual !== undefined &&
        { downwardResidual: this._archOpts.downwardResidual }),
      ...(this._archOpts.preserveDimResidual !== undefined &&
        { preserveDimResidual: this._archOpts.preserveDimResidual }),
      ...(this._archOpts.preserveDimResiduals !== undefined &&
        { preserveDimResiduals: this._archOpts.preserveDimResiduals }),
      ...(this._archOpts.mpLearningRateScale !== undefined &&
        { mpLearningRateScale: this._archOpts.mpLearningRateScale }),
    };

    const trainerConfig: Partial<TrainerConfig> = {
      ...DEFAULT_TRAINER_CONFIG,
      ...(this._trainingOpts.learningRate !== undefined &&
        { learningRate: this._trainingOpts.learningRate }),
      ...(this._trainingOpts.batchSize !== undefined &&
        { batchSize: this._trainingOpts.batchSize }),
      ...(this._trainingOpts.temperature !== undefined &&
        { temperature: this._trainingOpts.temperature }),
      ...(this._trainingOpts.gradientClip !== undefined &&
        { gradientClip: this._trainingOpts.gradientClip }),
      ...(this._trainingOpts.l2Lambda !== undefined && { l2Lambda: this._trainingOpts.l2Lambda }),
    };

    // 3. Separate tools (leaves) and capabilities (composites)
    const toolIds: string[] = [];
    const capInfos: CapabilityInfo[] = [];
    const embeddings = new Map<string, number[]>();

    // Detect embedding dim from first node
    const detectedDim = this._nodes[0]?.embedding.length ?? config.embeddingDim;
    if (detectedDim !== config.embeddingDim) {
      config.embeddingDim = detectedDim;
      // Adjust hiddenDim if it was not explicitly set and matches old embeddingDim
      if (this._archOpts.hiddenDim === undefined) {
        config.hiddenDim = detectedDim;
      }
    }

    const nodeIdSet = new Set(this._nodes.map((n) => n.id));

    // Pass 1: Identify leaf nodes (no valid children) to distinguish
    // tool-children from capability-children in the hierarchy.
    const leafIds = new Set<string>();
    for (const node of this._nodes) {
      const validChildren = node.children.filter((id) => nodeIdSet.has(id));
      if (validChildren.length === 0) {
        leafIds.add(node.id);
      }
    }

    // Pass 2: Build toolIds and capInfos with separate toolsUsed vs children.
    // - toolsUsed: direct tool (leaf) children → creates edges in toolToCapMatrix
    // - children: sub-capability children → creates edges in capToCapMatrices
    // This enables multi-level hierarchy (L0→L1→L2→...→root) for message passing.
    for (const node of this._nodes) {
      embeddings.set(node.id, node.embedding);

      const validChildren = node.children.filter((id) => nodeIdSet.has(id));

      if (validChildren.length === 0) {
        // Leaf (tool)
        toolIds.push(node.id);
      } else {
        // Composite (capability) — separate leaf children from cap children
        const childTools = validChildren.filter((id) => leafIds.has(id));
        const childCaps = validChildren.filter((id) => !leafIds.has(id));

        capInfos.push({
          id: node.id,
          toolsUsed: childTools,
          children: childCaps.length > 0 ? childCaps : undefined,
        });
      }
    }

    // 4. Build graph structure
    const graph = buildGraphStructure(capInfos, toolIds);

    // 5. Create trainer
    const trainer = new AutogradTrainer(config, trainerConfig, graph.maxLevel, this._archOpts.seed);
    trainer.setNodeEmbeddings(embeddings);
    trainer.setGraph(graph);

    // 6. Wrap in port implementation
    return new BuiltSHGAT(trainer, graph);
  }
}

// ============================================================================
// Internal Implementation
// ============================================================================

/**
 * Concrete implementation of SHGATTrainerScorer backed by AutogradTrainer.
 *
 * @internal
 */
class BuiltSHGAT implements SHGATTrainerScorer {
  private trainer: AutogradTrainer;
  private graph: GraphStructure;
  private disposed = false;

  constructor(trainer: AutogradTrainer, graph: GraphStructure) {
    this.trainer = trainer;
    this.graph = graph;
  }

  get hasMessagePassing(): boolean {
    return this.trainer.hasMessagePassing();
  }

  score(intentEmbedding: number[], nodeIds: string[]): number[] {
    if (this.disposed) {
      throw new Error("[SHGAT] Instance has been disposed. Create a new one via SHGATBuilder.");
    }
    return this.trainer.score(intentEmbedding, nodeIds);
  }

  async trainBatch(examples: TrainingExample[]): Promise<TrainingMetrics> {
    if (this.disposed) {
      throw new Error("[SHGAT] Instance has been disposed. Create a new one via SHGATBuilder.");
    }
    return this.trainer.trainBatch(examples);
  }

  async trainBatchKL(
    examples: SoftTargetExample[],
    toolIds: string[],
    klTemperature: number,
    klWeight: number = 1.0,
  ): Promise<KLTrainingMetrics> {
    if (this.disposed) {
      throw new Error("[SHGAT] Instance has been disposed. Create a new one via SHGATBuilder.");
    }
    return this.trainer.trainBatchKL(examples, toolIds, klTemperature, klWeight);
  }

  setTemperature(temperature: number): void {
    this.trainer.setTemperature(temperature);
  }

  setLearningRate(lr: number): void {
    this.trainer.setLearningRate(lr);
  }

  setSubgraphK(K: number): void {
    this.trainer.setSubgraphK(K);
  }

  precomputeEnrichedEmbeddings(): void {
    if (this.disposed) {
      throw new Error("[SHGAT] Instance has been disposed.");
    }
    this.trainer.precomputeEnrichedEmbeddings();
  }

  exportParams(): Record<string, unknown> {
    if (this.disposed) {
      throw new Error("[SHGAT] Instance has been disposed.");
    }
    const params = this.trainer.getParams();
    const toArr = (v: tf.Variable) => v.arraySync();

    const result: Record<string, unknown> = {
      W_k: params.W_k.map(toArr),
      W_intent: toArr(params.W_intent),
      W_up: Object.fromEntries(
        [...params.W_up.entries()].map(([l, ws]) => [l, ws.map(toArr)]),
      ),
      W_down: Object.fromEntries(
        [...params.W_down.entries()].map(([l, ws]) => [l, ws.map(toArr)]),
      ),
      a_up: Object.fromEntries(
        [...params.a_up.entries()].map(([l, ws]) => [l, ws.map(toArr)]),
      ),
      a_down: Object.fromEntries(
        [...params.a_down.entries()].map(([l, ws]) => [l, ws.map(toArr)]),
      ),
    };
    if (params.W_q) result.W_q = params.W_q.map(toArr);
    if (params.residualWeights) result.residualWeights = toArr(params.residualWeights);
    if (params.projectionHead) {
      result.projectionHead = this.trainer.exportProjectionHeadToArray();
    }
    return result;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    disposeGraphStructure(this.graph);
    this.trainer.dispose();
  }
}
