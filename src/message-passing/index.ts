/**
 * Message Passing Module Index
 *
 * Exports all message passing phases and orchestrator for SHGAT.
 * Supports both legacy 2-level (V→E→V) and multi-level n-SuperHyperGraph.
 *
 * @module graphrag/algorithms/shgat/message-passing
 */

export type {
  MessagePassingPhase,
  PhaseForwardCache,
  PhaseGradients,
  PhaseParameters,
  PhaseResult,
  PhaseResultWithCache,
  SparseConnectivity,
} from "./phase-interface.ts";
export { denseToSparse, edgeKey, transposeSparse } from "./phase-interface.ts";
export { phaseBackward, phaseForward } from "./phase-shared.ts";
export { VertexToEdgePhase } from "./vertex-to-edge-phase.ts";
export { EdgeToVertexPhase } from "./edge-to-vertex-phase.ts";
export { EdgeToEdgePhase } from "./edge-to-edge-phase.ts";
export {
  type ForwardCache,
  type LayerParameters,
  type LevelParamsGradients,
  type MultiLevelBackwardCache,
  type MultiLevelGradients,
  MultiLevelOrchestrator,
  type OrchestratorConfig,
} from "./multi-level-orchestrator.ts";

// V→V co-occurrence phase
export {
  buildCooccurrenceFromWorkflows,
  buildCooccurrenceMatrix,
  DEFAULT_V2V_PARAMS,
  v2vEnrich,
  VertexToVertexPhase,
} from "./vertex-to-vertex-phase.ts";
export type {
  CooccurrenceEntry,
  V2VForwardCache,
  V2VGradients,
  V2VParams,
  V2VPhaseResultWithCache,
  VertexToVertexConfig,
} from "./vertex-to-vertex-phase.ts";

// Re-export multi-level types from main types for convenience
export type { LevelParams, MultiLevelEmbeddings, MultiLevelForwardCache } from "../core/types.ts";

// NOTE: Tensor-native message passing was experimental and has been removed.
// The standard phases now use optimized TF.js operations internally.
