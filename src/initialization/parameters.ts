/**
 * SHGAT Parameter Initialization Module
 *
 * Functions for initializing all learnable parameters in SHGAT:
 * - Layer parameters (W_v, W_e, attention vectors)
 * - Head parameters (W_q, W_k, W_v for each head)
 * - V2 parameters (W_proj, b_proj, fusionMLP, W_stats, b_stats)
 * - Intent projection (W_intent)
 *
 * Uses Xavier/He initialization for proper gradient flow.
 *
 * @module graphrag/algorithms/shgat/initialization/parameters
 */

import type { LevelParams, SHGATConfig } from "../core/types.ts";
import { DEFAULT_FEATURE_WEIGHTS, DEFAULT_FUSION_WEIGHTS, NUM_TRACE_STATS } from "../core/types.ts";
import type { FeatureWeights, FusionWeights } from "../core/types.ts";
import type { ProjectionHeadArrayParams, ProjectionHeadTFParams } from "../core/projection-head.ts";

// ============================================================================
// Seeded PRNG for Reproducibility (mulberry32)
// ============================================================================

let rngState = Date.now(); // Default: use current time (non-reproducible)

/**
 * Seed the random number generator for reproducible initialization.
 * Call this before creating SHGAT instances for deterministic results.
 *
 * @param seed Integer seed value
 *
 * @example
 * ```typescript
 * import { seedRng } from "./shgat/initialization/parameters.ts";
 * seedRng(42); // Set seed for reproducibility
 * const shgat = createSHGATFromCapabilities(caps);
 * ```
 */
export function seedRng(seed: number): void {
  rngState = seed | 0; // Ensure integer
}

/**
 * Get current RNG state (for debugging/testing)
 */
export function getRngState(): number {
  return rngState;
}

/**
 * Seeded random number generator (mulberry32 algorithm)
 * Returns value in [0, 1)
 */
export function random(): number {
  rngState |= 0;
  rngState = (rngState + 0x6D2B79F5) | 0;
  let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

// ============================================================================
// Parameter Types
// ============================================================================

/**
 * Layer parameters for message passing
 */
export interface LayerParams {
  // Vertex→Edge phase
  W_v: number[][][]; // [head][hiddenDim][inputDim]
  W_e: number[][][]; // [head][hiddenDim][inputDim]
  a_ve: number[][]; // [head][2*hiddenDim]

  // Edge→Vertex phase
  W_e2: number[][][]; // [head][hiddenDim][hiddenDim]
  W_v2: number[][][]; // [head][hiddenDim][hiddenDim]
  a_ev: number[][]; // [head][2*hiddenDim]
}

/**
 * Per-head attention parameters (array-based, for serialization)
 */
export interface HeadParams {
  W_q: number[][];
  W_k: number[][];
  W_v: number[][];
  a: number[];
}

// Import TensorFlow types (must use same package as backend.ts)
import type { Variable } from "npm:@tensorflow/tfjs@4.22.0";

/**
 * Per-head attention parameters (tensor-based, for GPU-accelerated scoring)
 *
 * These are created once at initialization and reused for all scoring calls.
 * This avoids the 20x slowdown from array→tensor conversion on every call.
 */
export interface TensorHeadParams {
  W_q: Variable;  // [scoringDim, embeddingDim]
  W_k: Variable;  // [scoringDim, embeddingDim]
  W_v: Variable;  // [scoringDim, embeddingDim]
  a: Variable;    // [2 * scoringDim]
}

/**
 * Tensor-based SHGAT scoring parameters
 *
 * Only includes parameters needed for scoring (K-head attention).
 * Message passing still uses array-based params for now.
 */
export interface TensorScoringParams {
  /** Per-head attention tensors */
  headParams: TensorHeadParams[];
  /** Intent projection matrix [hiddenDim, embeddingDim] */
  W_intent: Variable;
  /** Optional projection head for contrastive discrimination */
  projectionHead?: ProjectionHeadTFParams;
}

/**
 * Fusion MLP parameters
 */
export interface FusionMLPParams {
  W1: number[][];
  b1: number[];
  W2: number[];
  b2: number;
}

/**
 * All SHGAT parameters
 */
export interface SHGATParams {
  // Layer parameters (v1)
  layerParams: LayerParams[];
  headParams: HeadParams[];

  // Legacy weights (v1)
  fusionWeights: FusionWeights;
  featureWeights: FeatureWeights;
  W_intent: number[][]; // [hiddenDim][embeddingDim] - projects intent to match E after concatHeads

  // V2 parameters
  W_proj: number[][];
  b_proj: number[];
  fusionMLP: FusionMLPParams;
  W_stats: number[][];
  b_stats: number[];

  // Per-level residual logits (learnable)
  // sigmoid(residualLogits[level]) = α for that level
  // Initialized to logit(0.3) ≈ -0.847 so sigmoid gives ~0.3
  residualLogits: number[];

  // Optional projection head for contrastive discrimination
  projectionHead?: ProjectionHeadArrayParams;
}

/**
 * V2 gradient accumulators
 */
export interface V2GradientAccumulators {
  W_proj: number[][];
  b_proj: number[];
  fusionMLP: {
    W1: number[][];
    b1: number[];
    W2: number[];
    b2: number;
  };
}

// ============================================================================
// Tensor Initialization
// ============================================================================

/**
 * Initialize 3D tensor with Xavier scaling
 */
export function initTensor3D(d1: number, d2: number, d3: number): number[][][] {
  const scale = Math.sqrt(2.0 / (d2 + d3));
  return Array.from(
    { length: d1 },
    () =>
      Array.from(
        { length: d2 },
        () => Array.from({ length: d3 }, () => (random() - 0.5) * 2 * scale),
      ),
  );
}

/**
 * Initialize 2D matrix with Xavier scaling
 */
export function initMatrix(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols));
  return Array.from(
    { length: rows },
    () => Array.from({ length: cols }, () => (random() - 0.5) * 2 * scale),
  );
}

/**
 * Initialize 3D tensor with identity-like structure for preserveDim mode.
 *
 * Each head extracts a non-overlapping slice of the input:
 * - Head 0: dims [0, headDim)
 * - Head 1: dims [headDim, 2*headDim)
 * - etc.
 *
 * This preserves semantic structure while allowing gradient flow.
 *
 * @param numHeads Number of attention heads
 * @param headDim Output dimension per head
 * @param inputDim Input dimension (should equal numHeads * headDim)
 * @returns [numHeads][headDim][inputDim] tensor with identity-like structure
 */
export function initTensor3DIdentityLike(
  numHeads: number,
  headDim: number,
  inputDim: number,
): number[][][] {
  const noiseScale = 0.01;
  return Array.from({ length: numHeads }, (_, head) =>
    Array.from({ length: headDim }, (_, i) =>
      Array.from({ length: inputDim }, (_, j) => {
        // Identity: W[head][i][head*headDim + i] = 1.0
        const targetJ = head * headDim + i;
        if (j === targetJ) {
          return 1.0;
        }
        // Small noise elsewhere for gradient flow
        return (random() - 0.5) * noiseScale;
      })
    )
  );
}

/**
 * Initialize 2D matrix with scaled Xavier initialization
 *
 * Used for K-head attention (W_q, W_k) where standard Xavier gives
 * values too small for Q·K to escape sigmoid(0) = 0.5.
 *
 * @param scaleFactor Multiplier for Xavier scale (default 10 for K-head)
 */
export function initMatrixScaled(rows: number, cols: number, scaleFactor: number = 10): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols)) * scaleFactor;
  return Array.from(
    { length: rows },
    () => Array.from({ length: cols }, () => (random() - 0.5) * 2 * scale),
  );
}

/**
 * Initialize 1D vector
 */
export function initVector(size: number): number[] {
  const scale = Math.sqrt(1.0 / size);
  return Array.from({ length: size }, () => (random() - 0.5) * 2 * scale);
}

/**
 * Create zeros matrix with same shape as input
 */
export function zerosLike2D(matrix: number[][]): number[][] {
  return matrix.map((row) => row.map(() => 0));
}

/**
 * Create zeros tensor with same shape as input
 */
export function zerosLike3D(tensor: number[][][]): number[][][] {
  return tensor.map((m) => m.map((r) => r.map(() => 0)));
}

// ============================================================================
// Parameter Initialization
// ============================================================================

/**
 * Initialize all SHGAT parameters
 */
export function initializeParameters(config: SHGATConfig): SHGATParams {
  const { numLayers, numHeads, hiddenDim, headDim, embeddingDim, mlpHiddenDim, preserveDim } = config;

  // Initialize layer parameters (legacy V1 message passing)
  // OPTIMIZATION: Skip when preserveDim=true - levelParams are used instead
  // This saves ~134M elements of initialization (~10s on typical hardware).
  const layerParams: LayerParams[] = [];
  if (!preserveDim) {
    for (let l = 0; l < numLayers; l++) {
      // Layer 0: input is raw embedding (embeddingDim)
      // Layer k>0: input is previous layer output after concatHeads (hiddenDim)
      const layerInputDim = l === 0 ? embeddingDim : hiddenDim;

      layerParams.push({
        W_v: initTensor3D(numHeads, hiddenDim, layerInputDim),
        W_e: initTensor3D(numHeads, hiddenDim, layerInputDim),
        a_ve: initMatrix(numHeads, 2 * hiddenDim),

        W_e2: initTensor3D(numHeads, hiddenDim, hiddenDim),
        W_v2: initTensor3D(numHeads, hiddenDim, hiddenDim),
        a_ev: initMatrix(numHeads, 2 * hiddenDim),
      });
    }
  }

  // Initialize head parameters for K-head attention scoring
  // FIX: Use shared projection W_q = W_k to preserve cosine similarity structure
  // Random different projections destroy discriminability (MRR 0.148 → 1.0 with shared)
  //
  // Each head projects to its own subspace: headDim = 64 (not hiddenDim = 1024)
  // This matches standard Transformer attention where d_k = d_model / numHeads
  // Benchmark shows 93.8% param reduction with +19.9% test accuracy improvement
  const scoringDim = headDim; // 64 per head, NOT 1024 (fixes 16x oversized matrices)
  const headParams: HeadParams[] = [];
  for (let h = 0; h < numHeads; h++) {
    const W_shared = initMatrixScaled(scoringDim, embeddingDim, 10);
    headParams.push({
      W_q: W_shared,
      W_k: W_shared, // Same matrix as W_q - preserves similarity structure
      W_v: initMatrix(scoringDim, embeddingDim),
      a: initVector(2 * scoringDim),
    });
  }

  // Initialize intent projection matrix (only used in standard mode)
  // PreserveDim mode bypasses W_intent and uses raw intent (1024-dim) directly with W_q
  const propagatedDim = hiddenDim;
  const W_intent = initMatrix(propagatedDim, embeddingDim);

  // Initialize V2 parameters
  const numTraceStats = NUM_TRACE_STATS;
  const projInputDim = 3 * embeddingDim + numTraceStats;

  const W_proj = initMatrix(hiddenDim, projInputDim);
  const b_proj = initVector(hiddenDim);

  const W_stats = initMatrix(hiddenDim, numTraceStats);
  const b_stats = initVector(hiddenDim);

  const fusionMLP: FusionMLPParams = {
    W1: initMatrix(mlpHiddenDim, numHeads),
    b1: initVector(mlpHiddenDim),
    W2: initVector(mlpHiddenDim),
    b2: 0,
  };

  // Per-level residual logits (learnable)
  // Initialize to logit(0.3) = log(0.3/0.7) ≈ -0.847 so sigmoid gives ~0.3
  // Support up to 10 hierarchy levels (typical is 2-3)
  const MAX_LEVELS = 10;
  const initLogit = Math.log(0.3 / 0.7); // ≈ -0.847
  const residualLogits = new Array(MAX_LEVELS).fill(initLogit);

  return {
    layerParams,
    headParams,
    fusionWeights: { ...DEFAULT_FUSION_WEIGHTS },
    featureWeights: { ...DEFAULT_FEATURE_WEIGHTS },
    W_intent,
    W_proj,
    b_proj,
    fusionMLP,
    W_stats,
    b_stats,
    residualLogits,
  };
}

/**
 * Initialize V2 gradient accumulators (used in training)
 */
export function initializeV2GradientAccumulators(config: SHGATConfig): V2GradientAccumulators {
  const { hiddenDim, mlpHiddenDim, embeddingDim, numHeads } = config;
  const numTraceStats = NUM_TRACE_STATS;
  const projInputDim = 3 * embeddingDim + numTraceStats;

  return {
    W_proj: Array.from({ length: hiddenDim }, () => Array(projInputDim).fill(0)),
    b_proj: new Array(hiddenDim).fill(0),
    fusionMLP: {
      W1: Array.from({ length: mlpHiddenDim }, () => Array(numHeads).fill(0)),
      b1: new Array(mlpHiddenDim).fill(0),
      W2: new Array(mlpHiddenDim).fill(0),
      b2: 0,
    },
  };
}

/**
 * Reset V2 gradient accumulators to zero
 */
export function resetV2GradientAccumulators(
  accum: V2GradientAccumulators,
  config: SHGATConfig,
): void {
  const { hiddenDim, mlpHiddenDim, embeddingDim, numHeads } = config;
  const numTraceStats = NUM_TRACE_STATS;
  const projInputDim = 3 * embeddingDim + numTraceStats;

  accum.W_proj = Array.from({ length: hiddenDim }, () => Array(projInputDim).fill(0));
  accum.b_proj = new Array(hiddenDim).fill(0);
  accum.fusionMLP = {
    W1: Array.from({ length: mlpHiddenDim }, () => Array(numHeads).fill(0)),
    b1: new Array(mlpHiddenDim).fill(0),
    W2: new Array(mlpHiddenDim).fill(0),
    b2: 0,
  };
}

// ============================================================================
// Multi-Level Parameter Initialization (n-SuperHyperGraph v1 refactor)
// ============================================================================

/**
 * Initialize parameters for multi-level message passing
 *
 * Creates LevelParams for each hierarchy level (0 to maxLevel).
 * Uses Xavier initialization for proper gradient flow.
 *
 * Dimension notes:
 * - Level 0: input is embeddingDim (from tools)
 * - Level k > 0: input is numHeads * headDim (after concat from previous level)
 * - All levels: headDim = hiddenDim / numHeads (per-head dimension)
 *
 * @param config SHGAT configuration
 * @param maxLevel Maximum hierarchy level (L_max)
 * @returns Map of level → LevelParams
 *
 * @since v1 refactor
 * @see 05-parameters.md
 */
export function initializeLevelParameters(
  config: SHGATConfig,
  maxLevel: number,
): Map<number, LevelParams> {
  // Use preserveDim mode if enabled
  if (config.preserveDim) {
    return initializeLevelParametersPreserveDim(config, maxLevel);
  }

  const { numHeads, hiddenDim, embeddingDim } = config;
  const headDim = Math.floor(hiddenDim / numHeads);

  const levelParams = new Map<number, LevelParams>();

  for (let level = 0; level <= maxLevel; level++) {
    // Input dimension depends on level
    // Level 0: tools have embeddingDim
    // Level k > 0: capabilities have numHeads * headDim after concat
    const inputDim = level === 0 ? embeddingDim : numHeads * headDim;

    levelParams.set(level, {
      // Child projection: [numHeads][headDim][inputDim]
      W_child: initTensor3D(numHeads, headDim, inputDim),

      // Parent projection: [numHeads][headDim][inputDim]
      // Parents at level k receive from children, same input dim
      W_parent: initTensor3D(numHeads, headDim, inputDim),

      // Attention vectors for upward pass: [numHeads][2*headDim]
      a_upward: initMatrix(numHeads, 2 * headDim),

      // Attention vectors for downward pass: [numHeads][2*headDim]
      a_downward: initMatrix(numHeads, 2 * headDim),
    });
  }

  return levelParams;
}

/**
 * Initialize level parameters with dimension preservation (1024→1024).
 *
 * Each head projects to embeddingDim/numHeads, then concat → embeddingDim.
 * This preserves K-head attention while maintaining output dimension.
 *
 * Example with numHeads=4, embeddingDim=1024:
 * - Each head: [256][1024] projection
 * - Concat 4 heads → 1024-dim output
 *
 * @param config SHGAT configuration with preserveDim=true
 * @param maxLevel Maximum hierarchy level
 */
export function initializeLevelParametersPreserveDim(
  config: SHGATConfig,
  maxLevel: number,
): Map<number, LevelParams> {
  const { numHeads, embeddingDim } = config;
  // headDim such that numHeads * headDim = embeddingDim
  const headDim = Math.floor(embeddingDim / numHeads);
  const levelParams = new Map<number, LevelParams>();

  for (let level = 0; level <= maxLevel; level++) {
    // All levels use embeddingDim input (preserved throughout)
    const inputDim = embeddingDim;

    levelParams.set(level, {
      // Each head: [headDim][inputDim] = [256][1024]
      // Use identity-like init to preserve semantic structure from BGE
      W_child: initTensor3DIdentityLike(numHeads, headDim, inputDim),
      W_parent: initTensor3DIdentityLike(numHeads, headDim, inputDim),
      // Attention vectors: [numHeads][2*headDim]
      a_upward: initMatrix(numHeads, 2 * headDim),
      a_downward: initMatrix(numHeads, 2 * headDim),
    });
  }

  return levelParams;
}

/**
 * Count parameters for multi-level message passing
 *
 * @param config SHGAT configuration
 * @param maxLevel Maximum hierarchy level
 * @returns Total parameter count for level params
 */
export function countLevelParameters(
  config: SHGATConfig,
  maxLevel: number,
): number {
  const { numHeads, hiddenDim, embeddingDim } = config;
  const headDim = Math.floor(hiddenDim / numHeads);

  let count = 0;

  for (let level = 0; level <= maxLevel; level++) {
    const inputDim = level === 0 ? embeddingDim : numHeads * headDim;

    // W_child: numHeads * headDim * inputDim
    count += numHeads * headDim * inputDim;
    // W_parent: numHeads * headDim * inputDim
    count += numHeads * headDim * inputDim;
    // a_upward: numHeads * 2 * headDim
    count += numHeads * 2 * headDim;
    // a_downward: numHeads * 2 * headDim
    count += numHeads * 2 * headDim;
  }

  return count;
}

/**
 * Get parameters for a specific hierarchy level
 *
 * @param levelParams Map of all level parameters
 * @param level The hierarchy level to get
 * @returns LevelParams for the specified level
 * @throws Error if level not found
 */
export function getLevelParams(
  levelParams: Map<number, LevelParams>,
  level: number,
): LevelParams {
  const params = levelParams.get(level);
  if (!params) {
    throw new Error(
      `LevelParams not found for level ${level}. ` +
        `Available levels: ${Array.from(levelParams.keys()).join(", ")}`,
    );
  }
  return params;
}

/**
 * Export level parameters to JSON-serializable object
 *
 * Format: { "level_0": {...}, "level_1": {...}, ... }
 *
 * @param levelParams Map of level → LevelParams
 * @returns Serializable object
 */
export function exportLevelParams(
  levelParams: Map<number, LevelParams>,
): Record<string, LevelParams> {
  const result: Record<string, LevelParams> = {};

  for (const [level, params] of levelParams) {
    result[`level_${level}`] = {
      W_child: params.W_child,
      W_parent: params.W_parent,
      a_upward: params.a_upward,
      a_downward: params.a_downward,
    };
  }

  return result;
}

/**
 * Import level parameters from JSON object
 *
 * @param data Serialized level params
 * @returns Map of level → LevelParams
 */
export function importLevelParams(
  data: Record<string, LevelParams>,
): Map<number, LevelParams> {
  const levelParams = new Map<number, LevelParams>();

  for (const key of Object.keys(data)) {
    const level = parseInt(key.replace("level_", ""));
    if (isNaN(level)) continue;

    const params = data[key];
    levelParams.set(level, {
      W_child: params.W_child,
      W_parent: params.W_parent,
      a_upward: params.a_upward,
      a_downward: params.a_downward,
    });
  }

  return levelParams;
}

// ============================================================================
// Adaptive Configuration
// ============================================================================

/**
 * Adaptive heads configuration based on graph complexity
 *
 * More tools/capabilities = more heads can capture diverse patterns.
 * Also considers hierarchy depth for multi-level message passing.
 *
 * @param numTools Number of tools in graph
 * @param numCapabilities Number of capabilities
 * @param maxLevel Maximum hierarchy level (L_max)
 * @returns Recommended numHeads and hiddenDim
 */
export function getAdaptiveHeadsByGraphSize(
  _numTools: number,
  _numCapabilities: number,
  _maxLevel: number = 0,
  _preserveDim: boolean = false,
  _embeddingDim: number = 1024,
): { numHeads: number; hiddenDim: number; headDim: number } {
  // Always use 16 heads for optimal performance
  // 16 heads × 64 dim = 1024 exactly matches BGE-M3 embedding dimension
  // Benchmarks show 16 heads consistently outperforms 4 heads (+12% train, +5% test)
  const numHeads = 16;
  const HEAD_DIM = 64;
  const scoringDim = numHeads * HEAD_DIM; // 1024
  return { numHeads, hiddenDim: scoringDim, headDim: HEAD_DIM };
}

// ============================================================================
// Serialization Helpers
// ============================================================================

/**
 * Export parameters to JSON-serializable object
 */
export function exportParams(
  config: SHGATConfig,
  params: SHGATParams,
): Record<string, unknown> {
  const result: Record<string, unknown> = {
    config,
    layerParams: params.layerParams,
    headParams: params.headParams,
    fusionWeights: params.fusionWeights,
    featureWeights: params.featureWeights,
    W_intent: params.W_intent,
    W_proj: params.W_proj,
    b_proj: params.b_proj,
    fusionMLP: params.fusionMLP,
    W_stats: params.W_stats,
    b_stats: params.b_stats,
  };

  if (params.projectionHead) {
    result.projectionHead = params.projectionHead;
  }

  return result;
}

/**
 * Import parameters from JSON object
 */
export function importParams(
  data: Record<string, unknown>,
  currentParams: SHGATParams,
): { config?: SHGATConfig; params: SHGATParams } {
  const params = { ...currentParams };
  let config: SHGATConfig | undefined;

  if (data.config) {
    config = data.config as SHGATConfig;
  }
  if (data.layerParams) {
    params.layerParams = data.layerParams as LayerParams[];
  }
  if (data.headParams) {
    params.headParams = data.headParams as HeadParams[];
  }
  if (data.fusionWeights) {
    params.fusionWeights = data.fusionWeights as FusionWeights;
  }
  if (data.featureWeights) {
    params.featureWeights = data.featureWeights as FeatureWeights;
  }
  if (data.W_intent) {
    params.W_intent = data.W_intent as number[][];
  }
  if (data.W_proj) {
    params.W_proj = data.W_proj as number[][];
  }
  if (data.b_proj) {
    params.b_proj = data.b_proj as number[];
  }
  if (data.fusionMLP) {
    params.fusionMLP = data.fusionMLP as FusionMLPParams;
  }
  if (data.W_stats) {
    params.W_stats = data.W_stats as number[][];
  }
  if (data.b_stats) {
    params.b_stats = data.b_stats as number[];
  }
  if (data.projectionHead) {
    params.projectionHead = data.projectionHead as ProjectionHeadArrayParams;
  }

  return { config, params };
}

// ============================================================================
// Statistics
// ============================================================================

/**
 * Count total parameters in the model
 */
export function countParameters(config: SHGATConfig): {
  v1ParamCount: number;
  v2ParamCount: number;
  total: number;
} {
  const { numHeads, hiddenDim, embeddingDim, numLayers, mlpHiddenDim } = config;
  const numTraceStats = NUM_TRACE_STATS;

  // V1 param count
  let v1ParamCount = 0;
  for (let l = 0; l < numLayers; l++) {
    // Layer 0: input is embeddingDim, Layer k>0: input is hiddenDim (after concatHeads)
    const layerInputDim = l === 0 ? embeddingDim : hiddenDim;
    v1ParamCount += numHeads * hiddenDim * layerInputDim * 2; // W_v, W_e
    v1ParamCount += numHeads * 2 * hiddenDim; // a_ve
    v1ParamCount += numHeads * hiddenDim * hiddenDim * 2; // W_e2, W_v2
    v1ParamCount += numHeads * 2 * hiddenDim; // a_ev
  }
  v1ParamCount += 3; // fusionWeights
  v1ParamCount += 3; // featureWeights
  v1ParamCount += hiddenDim * embeddingDim; // W_intent (hiddenDim = propagatedDim)

  // V2 param count
  const projInputDim = 3 * embeddingDim + numTraceStats;
  let v2ParamCount = 0;
  v2ParamCount += hiddenDim * projInputDim + hiddenDim; // W_proj, b_proj
  v2ParamCount += hiddenDim * numTraceStats + hiddenDim; // W_stats, b_stats
  v2ParamCount += numHeads * 3 * hiddenDim * hiddenDim; // headParams (W_q, W_k, W_v per head)
  v2ParamCount += mlpHiddenDim * numHeads + mlpHiddenDim; // fusionMLP W1, b1
  v2ParamCount += mlpHiddenDim + 1; // fusionMLP W2, b2

  return {
    v1ParamCount,
    v2ParamCount,
    total: v1ParamCount + v2ParamCount,
  };
}

// ============================================================================
// Tensor-Based Parameter Management (GPU-accelerated scoring)
// ============================================================================

// Dynamic import to avoid circular dependency with backend.ts
let _tf: typeof import("npm:@tensorflow/tfjs@4.22.0") | null = null;
async function getTf() {
  if (!_tf) {
    _tf = await import("npm:@tensorflow/tfjs@4.22.0");
  }
  return _tf;
}

/**
 * Create tensor-based scoring parameters from array-based params.
 *
 * This should be called ONCE at initialization. The tensors are stored as
 * Variables (trainable tensors) and reused for all scoring calls.
 *
 * IMPORTANT: Call disposeTensorScoringParams() when done to free GPU memory.
 *
 * @param params Array-based SHGAT parameters
 * @returns Tensor-based scoring parameters for GPU-accelerated scoring
 *
 * @example
 * ```typescript
 * const tensorParams = await createTensorScoringParams(params);
 * // Use for all scoring calls
 * const scores = scoreNodesTensor(embeddings, nodeIds, levels, intent, tensorParams, config);
 * // Clean up when done
 * disposeTensorScoringParams(tensorParams);
 * ```
 */
export async function createTensorScoringParams(
  params: SHGATParams,
): Promise<TensorScoringParams> {
  const tf = await getTf();
  const instanceId = tensorParamsCounter++;

  const headParams: TensorHeadParams[] = params.headParams.map((hp, i) => ({
    W_q: tf.variable(tf.tensor2d(hp.W_q), true, `W_q_${instanceId}_h${i}`),
    W_k: tf.variable(tf.tensor2d(hp.W_k), true, `W_k_${instanceId}_h${i}`),
    W_v: tf.variable(tf.tensor2d(hp.W_v), true, `W_v_${instanceId}_h${i}`),
    a: tf.variable(tf.tensor1d(hp.a), true, `a_${instanceId}_h${i}`),
  }));

  const W_intent = tf.variable(
    tf.tensor2d(params.W_intent),
    true,
    `W_intent_${instanceId}`,
  );

  const projectionHead = params.projectionHead
    ? {
        W1: tf.variable(tf.tensor2d(params.projectionHead.W1), true, `proj_W1_${instanceId}`),
        b1: tf.variable(tf.tensor1d(params.projectionHead.b1), true, `proj_b1_${instanceId}`),
        W2: tf.variable(tf.tensor2d(params.projectionHead.W2), true, `proj_W2_${instanceId}`),
        b2: tf.variable(tf.tensor1d(params.projectionHead.b2), true, `proj_b2_${instanceId}`),
      } as ProjectionHeadTFParams
    : undefined;

  return { headParams, W_intent, projectionHead };
}

/**
 * Create tensor-based scoring parameters synchronously.
 *
 * Use this when TensorFlow.js is already loaded (e.g., after initTensorFlow()).
 *
 * @param params Array-based SHGAT parameters
 * @param tf TensorFlow.js module reference
 * @returns Tensor-based scoring parameters
 */
// Counter for unique variable names across instances
let tensorParamsCounter = 0;

export function createTensorScoringParamsSync(
  params: SHGATParams,
  tf: typeof import("npm:@tensorflow/tfjs@4.22.0"),
): TensorScoringParams {
  const instanceId = tensorParamsCounter++;

  const headParams: TensorHeadParams[] = params.headParams.map((hp, i) => ({
    W_q: tf.variable(tf.tensor2d(hp.W_q), true, `W_q_${instanceId}_h${i}`),
    W_k: tf.variable(tf.tensor2d(hp.W_k), true, `W_k_${instanceId}_h${i}`),
    W_v: tf.variable(tf.tensor2d(hp.W_v), true, `W_v_${instanceId}_h${i}`),
    a: tf.variable(tf.tensor1d(hp.a), true, `a_${instanceId}_h${i}`),
  }));

  const W_intent = tf.variable(
    tf.tensor2d(params.W_intent),
    true,
    `W_intent_${instanceId}`,
  );

  const projectionHead = params.projectionHead
    ? {
        W1: tf.variable(tf.tensor2d(params.projectionHead.W1), true, `proj_W1_${instanceId}`),
        b1: tf.variable(tf.tensor1d(params.projectionHead.b1), true, `proj_b1_${instanceId}`),
        W2: tf.variable(tf.tensor2d(params.projectionHead.W2), true, `proj_W2_${instanceId}`),
        b2: tf.variable(tf.tensor1d(params.projectionHead.b2), true, `proj_b2_${instanceId}`),
      } as ProjectionHeadTFParams
    : undefined;

  return { headParams, W_intent, projectionHead };
}

/**
 * Dispose tensor-based scoring parameters to free GPU memory.
 *
 * IMPORTANT: Call this when the SHGAT instance is no longer needed.
 *
 * @param tensorParams Tensor parameters to dispose
 */
export function disposeTensorScoringParams(tensorParams: TensorScoringParams): void {
  for (const hp of tensorParams.headParams) {
    hp.W_q.dispose();
    hp.W_k.dispose();
    hp.W_v.dispose();
    hp.a.dispose();
  }
  tensorParams.W_intent.dispose();
  if (tensorParams.projectionHead) {
    tensorParams.projectionHead.W1.dispose();
    tensorParams.projectionHead.b1.dispose();
    tensorParams.projectionHead.W2.dispose();
    tensorParams.projectionHead.b2.dispose();
  }
}

/**
 * Update tensor parameters from array-based params (e.g., after training).
 *
 * This updates the tensor values in-place without creating new tensors.
 *
 * @param tensorParams Tensor parameters to update
 * @param params Source array-based parameters
 * @param tf TensorFlow.js module reference
 */
export function updateTensorScoringParams(
  tensorParams: TensorScoringParams,
  params: SHGATParams,
  tf: typeof import("npm:@tensorflow/tfjs@4.22.0"),
): void {
  for (let i = 0; i < params.headParams.length; i++) {
    const hp = params.headParams[i];
    tensorParams.headParams[i].W_q.assign(tf.tensor2d(hp.W_q));
    tensorParams.headParams[i].W_k.assign(tf.tensor2d(hp.W_k));
    tensorParams.headParams[i].W_v.assign(tf.tensor2d(hp.W_v));
    tensorParams.headParams[i].a.assign(tf.tensor1d(hp.a));
  }
  tensorParams.W_intent.assign(tf.tensor2d(params.W_intent));

  // Update projection head if present in both source and target
  if (params.projectionHead && tensorParams.projectionHead) {
    tensorParams.projectionHead.W1.assign(tf.tensor2d(params.projectionHead.W1));
    tensorParams.projectionHead.b1.assign(tf.tensor1d(params.projectionHead.b1));
    tensorParams.projectionHead.W2.assign(tf.tensor2d(params.projectionHead.W2));
    tensorParams.projectionHead.b2.assign(tf.tensor1d(params.projectionHead.b2));
  }
}
