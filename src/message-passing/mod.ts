/**
 * SHGAT Message Passing Module
 *
 * Multi-level message passing for n-SuperHyperGraph structures.
 *
 * Phases:
 * - V→V: L0 node co-occurrence (from scraped workflow patterns)
 * - V→E: L0 nodes → L1 nodes (vertex-to-edge)
 * - E→E: L1+ nodes → higher-level nodes (edge-to-edge)
 * - E→V: L1 nodes → L0 nodes (edge-to-vertex)
 *
 * @module graphrag/algorithms/shgat/message-passing
 */

// Phase interface
export type {
  MessagePassingPhase,
  MultiLevelOrchestrator as MultiLevelOrchestratorInterface,
  PhaseParameters,
  PhaseResult,
  SparseConnectivity,
} from "./phase-interface.ts";

export { denseToSparse, transposeSparse } from "./phase-interface.ts";

// V→V phase (co-occurrence enrichment)
export {
  buildCooccurrenceFromWorkflows,
  buildCooccurrenceMatrix,
  DEFAULT_V2V_CONFIG,
  v2vEnrich,
  VertexToVertexPhase,
} from "./vertex-to-vertex-phase.ts";
export type {
  CooccurrenceEntry,
  VertexToVertexConfig,
  VertexToVertexResult,
} from "./vertex-to-vertex-phase.ts";

// V→E phase
export { VertexToEdgePhase } from "./vertex-to-edge-phase.ts";

// E→V phase
export { EdgeToVertexPhase } from "./edge-to-vertex-phase.ts";

// E→E phase
export { EdgeToEdgePhase } from "./edge-to-edge-phase.ts";

// Orchestrator
export { MultiLevelOrchestrator } from "./multi-level-orchestrator.ts";
export type {
  ForwardCache,
  LayerParameters,
  OrchestratorConfig,
} from "./multi-level-orchestrator.ts";
