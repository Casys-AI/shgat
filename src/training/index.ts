/**
 * SHGAT-TF Training Module
 *
 * Training with TensorFlow.js automatic differentiation.
 * Replaces 3000+ lines of manual backward passes with ~300 lines of autograd.
 *
 * Includes sparse message passing (2026-01-28) for:
 * - WASM backend compatibility (no UnsortedSegmentSum kernel needed)
 * - ~10x faster training on large graphs
 * - Full gradient flow through W_up, W_down, a_up, a_down
 *
 * @module shgat-tf/training
 */

// Autograd trainer (NEW - replaces v1-trainer, multi-level-trainer, etc.)
export {
  AutogradTrainer,
  trainStep,
  forwardScoring,
  kHeadScoring,
  infoNCELoss,
  batchContrastiveLoss,
  initTFParams,
  DEFAULT_TRAINER_CONFIG,
  // Message passing (2026-01-28)
  messagePassingForward,
  buildGraphStructure,
  disposeGraphStructure,
} from "./autograd-trainer.ts";

export type {
  TFParams,
  TrainerConfig,
  TrainingMetrics,
  // Message passing types (2026-01-28)
  GraphStructure,
  CapabilityInfo,
  MessagePassingContext,
} from "./autograd-trainer.ts";

// Sparse message passing (2026-01-28)
export {
  buildSparseConnectivity,
  sparseMPForward,
  sparseMPBackward,
  applySparseMPGradients,
} from "./sparse-mp.ts";

export type {
  SparseConnectivity,
  SparseMPForwardCache,
  SparseMPGradients,
  SparseMPForwardResult,
} from "./sparse-mp.ts";

// PER buffer (kept - no gradients, just replay logic)
export {
  PERBuffer,
  annealBeta,
  annealTemperature,
  type PERConfig,
} from "./per-buffer.ts";
