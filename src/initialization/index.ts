/**
 * SHGAT Initialization Module
 *
 * Parameter initialization and management for SHGAT networks.
 *
 * @module graphrag/algorithms/shgat/initialization
 */

export {
  countLevelParameters,
  // Statistics
  countParameters,
  // Tensor parameter management
  createTensorScoringParams,
  createTensorScoringParamsSync,
  disposeTensorScoringParams,
  // Level params serialization (v1 refactor)
  exportLevelParams,
  // Legacy serialization
  exportParams,
  type FusionMLPParams,
  // Adaptive configuration (v1 refactor)
  getAdaptiveHeadsByGraphSize,
  getLevelParams,
  // RNG seeding for reproducibility
  getRngState,
  random,
  type HeadParams,
  importLevelParams,
  importParams,
  // Multi-level parameter initialization (v1 refactor)
  initializeLevelParameters,
  // Parameter initialization
  initializeParameters,
  initializeV2GradientAccumulators,
  initMatrix,
  // Tensor initialization
  initTensor3D,
  initVector,
  // Types
  type LayerParams,
  resetV2GradientAccumulators,
  seedRng,
  type SHGATParams,
  // Tensor parameter types
  type TensorHeadParams,
  type TensorScoringParams,
  updateTensorScoringParams,
  type V2GradientAccumulators,
  zerosLike2D,
  zerosLike3D,
} from "./parameters.ts";
