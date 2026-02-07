/**
 * SHGAT Layers Trainer
 *
 * Training with tf.layers.* and model.trainOnBatch() for proper gradient tracking.
 * Uses custom FFI kernel for UnsortedSegmentSum to enable gather gradients on WASM.
 *
 * Pattern from lib/gru/src/transition/gru-model.ts
 *
 * @module shgat-tf/training/layers-trainer
 */

import * as tf from "npm:@tensorflow/tfjs@4.22.0";
import "npm:@tensorflow/tfjs-backend-wasm@4.22.0";
import { registerUnsortedSegmentSumKernel } from "../tf/kernels/unsorted-segment-sum.ts";
import type { SHGATConfig, TrainingExample } from "../core/types.ts";

// ============================================================================
// Types
// ============================================================================

/**
 * Training configuration
 */
export interface LayersTrainerConfig {
  learningRate: number;
  temperature: number;
  dropout: number;
}

/**
 * Training metrics
 */
export interface LayersTrainingMetrics {
  loss: number;
  accuracy: number;
}

/**
 * Default config
 */
export const DEFAULT_LAYERS_TRAINER_CONFIG: LayersTrainerConfig = {
  learningRate: 0.001,
  temperature: 0.07,
  dropout: 0.1,
};

// ============================================================================
// Custom Score Layer
// ============================================================================

/**
 * Custom layer for K-head attention scoring
 */
class ScoreLayer extends tf.layers.Layer {
  private numHeads: number;
  private headDim: number;

  constructor(config: { numHeads: number; headDim: number }) {
    super({});
    this.numHeads = config.numHeads;
    this.headDim = config.headDim;
  }

  override computeOutputShape(inputShape: tf.Shape[]): tf.Shape {
    // Input: [queryProj, keyProj]
    // keyProj: [batchSize, numCandidates, headDim * numHeads]
    // Output: [batchSize, numCandidates]
    const keyShape = inputShape[1] as number[];
    return [keyShape[0], keyShape[1]];
  }

  override call(
    inputs: tf.Tensor | tf.Tensor[],
    _kwargs?: Record<string, unknown>
  ): tf.Tensor {
    return tf.tidy(() => {
      const [query, keys] = inputs as [tf.Tensor2D, tf.Tensor3D];

      // query: [batchSize, headDim * numHeads]
      // keys: [batchSize, numCandidates, headDim * numHeads]

      const batchSize = query.shape[0];
      const numCandidates = keys.shape[1];

      // Reshape query: [batchSize, numHeads, headDim]
      const queryReshaped = query.reshape([batchSize, this.numHeads, this.headDim]);

      // Reshape keys: [batchSize, numCandidates, numHeads, headDim]
      const keysReshaped = keys.reshape([
        batchSize,
        numCandidates,
        this.numHeads,
        this.headDim,
      ]);

      // Expand query: [batchSize, numHeads, 1, headDim]
      const queryExpanded = queryReshaped.expandDims(2);

      // Transpose keys: [batchSize, numHeads, numCandidates, headDim]
      const keysTransposed = keysReshaped.transpose([0, 2, 1, 3]);

      // Batched matmul: [batchSize, numHeads, 1, headDim] @ [batchSize, numHeads, headDim, numCandidates]
      const keysForMatmul = keysTransposed.transpose([0, 1, 3, 2]);
      const scores = tf.matMul(queryExpanded, keysForMatmul);

      // Squeeze and average heads: [batchSize, numHeads, numCandidates] -> [batchSize, numCandidates]
      const scoresSqueezeD = scores.squeeze([2]);
      const meanScores = tf.mean(scoresSqueezeD, 1);

      return meanScores;
    });
  }

  static className = "ScoreLayer";

  override getConfig(): tf.serialization.ConfigDict {
    return {
      ...super.getConfig(),
      numHeads: this.numHeads,
      headDim: this.headDim,
    };
  }
}

tf.serialization.registerClass(ScoreLayer);

// ============================================================================
// SHGAT Layers Trainer
// ============================================================================

/**
 * SHGAT Trainer using tf.layers.* and model.trainOnBatch()
 *
 * Architecture:
 * - Intent projection: intent @ W_intent -> [hiddenDim]
 * - Node projection: nodes @ W_k -> [numNodes, headDim] per head
 * - K-head scoring: mean(heads(intentProj @ nodeProj.T))
 * - InfoNCE loss with temperature
 */
export class LayersTrainer {
  readonly config: SHGATConfig;
  readonly trainerConfig: LayersTrainerConfig;

  // Keras model
  private model: tf.LayersModel | null = null;

  // Node embeddings
  private nodeEmbeddings: Map<string, number[]> = new Map();

  // Compiled state
  private compiled = false;

  constructor(
    config: SHGATConfig,
    trainerConfig: Partial<LayersTrainerConfig> = {}
  ) {
    this.config = config;
    this.trainerConfig = { ...DEFAULT_LAYERS_TRAINER_CONFIG, ...trainerConfig };
  }

  /**
   * Build the Keras model
   */
  private buildModel(): void {
    const { embeddingDim, hiddenDim, numHeads, headDim } = this.config;
    const { dropout } = this.trainerConfig;

    // Input: intent embedding [batchSize, embDim]
    const intentInput = tf.input({
      shape: [embeddingDim],
      name: "intent_input",
    });

    // Input: candidate node embeddings [batchSize, numCandidates, embDim]
    const nodesInput = tf.input({
      shape: [null, embeddingDim], // Variable number of candidates
      name: "nodes_input",
    });

    // Intent projection: [batchSize, hiddenDim]
    const intentProj = tf.layers.dense({
      units: hiddenDim,
      activation: "relu",
      name: "intent_proj",
    }).apply(intentInput) as tf.SymbolicTensor;

    // Dropout
    const intentDropout = tf.layers.dropout({
      rate: dropout,
    }).apply(intentProj) as tf.SymbolicTensor;

    // K-head scoring - project intent to headDim * numHeads
    const queryProj = tf.layers.dense({
      units: headDim * numHeads,
      name: "query_proj",
    }).apply(intentDropout) as tf.SymbolicTensor;

    // Project nodes: [batchSize, numCandidates, headDim * numHeads]
    const keyProj = tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: headDim * numHeads,
        name: "key_dense",
      }),
      name: "key_proj",
    }).apply(nodesInput) as tf.SymbolicTensor;

    // Compute attention scores using custom layer
    const scores = new ScoreLayer({ numHeads, headDim }).apply([
      queryProj,
      keyProj,
    ]) as tf.SymbolicTensor;

    // Create model for inference (outputs scores)
    this.model = tf.model({
      inputs: [intentInput, nodesInput],
      outputs: scores,
      name: "SHGATScorer",
    });
  }

  /**
   * Compile the model for training
   */
  private compileModel(): void {
    if (!this.model) {
      this.buildModel();
    }

    // Custom loss function for InfoNCE
    const temperature = this.trainerConfig.temperature;

    this.model!.compile({
      optimizer: tf.train.adam(this.trainerConfig.learningRate),
      loss: (_yTrue: tf.Tensor, yPred: tf.Tensor) => {
        return tf.tidy(() => {
          // yPred: scores [batchSize, numCandidates]
          // Label is always index 0 (positive is first)
          const batchSize = yPred.shape[0] || 1;
          const numCandidates = yPred.shape[1] || 1;
          const logits = tf.div(yPred, temperature);
          const labels = tf.oneHot(tf.zeros([batchSize], "int32"), numCandidates);
          return tf.losses.softmaxCrossEntropy(labels, logits);
        });
      },
    });

    this.compiled = true;
  }

  /**
   * Set node embeddings
   */
  setNodeEmbeddings(embeddings: Map<string, number[]>): void {
    this.nodeEmbeddings = embeddings;
  }

  /**
   * Prepare batch data
   */
  private prepareBatch(examples: TrainingExample[]): {
    intentTensor: tf.Tensor2D;
    nodesTensor: tf.Tensor3D;
    labelsTensor: tf.Tensor2D;
  } {
    const embDim = this.config.embeddingDim;

    // Find max candidates across batch
    let maxCandidates = 0;
    for (const ex of examples) {
      const numCands = 1 + (ex.negativeCapIds?.length || 0);
      maxCandidates = Math.max(maxCandidates, numCands);
    }

    // Prepare arrays
    const intents: number[][] = [];
    const nodes: number[][][] = [];
    const labels: number[][] = [];

    for (const ex of examples) {
      // Intent
      intents.push(ex.intentEmbedding);

      // Candidates: [positive, ...negatives]
      const candidateIds = [ex.candidateId, ...(ex.negativeCapIds || [])];
      const candidateEmbs: number[][] = [];

      for (const id of candidateIds) {
        const emb = this.nodeEmbeddings.get(id);
        candidateEmbs.push(emb || new Array(embDim).fill(0));
      }

      // Pad to maxCandidates
      while (candidateEmbs.length < maxCandidates) {
        candidateEmbs.push(new Array(embDim).fill(0));
      }

      nodes.push(candidateEmbs);

      // Label: one-hot with positive at index 0
      const label = new Array(maxCandidates).fill(0);
      label[0] = 1;
      labels.push(label);
    }

    return {
      intentTensor: tf.tensor2d(intents),
      nodesTensor: tf.tensor3d(nodes),
      labelsTensor: tf.tensor2d(labels),
    };
  }

  /**
   * Train on a batch of examples
   */
  async trainBatch(examples: TrainingExample[]): Promise<LayersTrainingMetrics> {
    if (!this.compiled) {
      this.compileModel();
    }

    const { intentTensor, nodesTensor, labelsTensor } = this.prepareBatch(examples);

    // Train on batch
    const result = await this.model!.trainOnBatch(
      [intentTensor, nodesTensor],
      labelsTensor
    );

    // Get loss
    const loss = typeof result === "number" ? result : result[0];

    // Compute accuracy
    const predictions = this.model!.predict([intentTensor, nodesTensor]) as tf.Tensor;
    const predArgmax = predictions.argMax(-1).arraySync() as number[];

    let correct = 0;
    for (const pred of predArgmax) {
      if (pred === 0) correct++; // Positive is always at index 0
    }
    const accuracy = correct / examples.length;

    // Cleanup
    intentTensor.dispose();
    nodesTensor.dispose();
    labelsTensor.dispose();
    predictions.dispose();

    return { loss, accuracy };
  }

  /**
   * Score nodes for an intent
   */
  score(intentEmb: number[], nodeIds: string[]): number[] {
    if (!this.model) {
      this.buildModel();
    }

    return tf.tidy(() => {
      // Prepare input
      const intentTensor = tf.tensor2d([intentEmb]);
      const nodeEmbs = nodeIds.map(
        (id) => this.nodeEmbeddings.get(id) || new Array(this.config.embeddingDim).fill(0)
      );
      const nodesTensor = tf.tensor3d([nodeEmbs]);

      // Forward pass
      const scores = this.model!.predict([intentTensor, nodesTensor]) as tf.Tensor;

      return scores.squeeze().arraySync() as number[];
    });
  }

  /**
   * Model summary
   */
  summary(): void {
    if (!this.model) {
      this.buildModel();
    }
    this.model!.summary();
  }

  /**
   * Dispose model
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.compiled = false;
  }
}

// ============================================================================
// Initialization
// ============================================================================

let initialized = false;

/**
 * Initialize TensorFlow with WASM backend and custom FFI kernel
 */
export async function initLayersTrainer(): Promise<string> {
  if (initialized) return tf.getBackend();

  // Set WASM backend
  await tf.setBackend("wasm");
  await tf.ready();

  // Register custom kernel for UnsortedSegmentSum
  registerUnsortedSegmentSumKernel();

  initialized = true;
  const backend = tf.getBackend();
  console.error(`[LayersTrainer] Initialized with ${backend} backend`);

  return backend;
}
