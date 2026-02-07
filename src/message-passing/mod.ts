/**
 * SHGAT Message Passing Module
 *
 * Multi-level message passing for n-SuperHyperGraph structures.
 *
 * Phases:
 * - V→V: Tool-to-tool co-occurrence (from scraped patterns)
 * - V→E: Tools → Capabilities
 * - E→E: Capabilities → Higher-level capabilities
 * - E→V: Capabilities → Tools (backward pass)
 *
 * @module graphrag/algorithms/shgat/message-passing
 */

// Phase interface
export type {
  MessagePassingPhase,
  MultiLevelOrchestrator as MultiLevelOrchestratorInterface,
  PhaseParameters,
  PhaseResult,
} from "./phase-interface.ts";

// V→V phase (co-occurrence enrichment)
export {
  buildCooccurrenceMatrix,
  DEFAULT_V2V_CONFIG,
  VertexToVertexPhase,
} from "./vertex-to-vertex-phase.ts";
export type {
  CooccurrenceEntry,
  VertexToVertexConfig,
  VertexToVertexResult,
} from "./vertex-to-vertex-phase.ts";

// Co-occurrence loader
export {
  getToolEmbeddings,
  loadCooccurrenceData,
  mergeEmbeddings,
} from "./cooccurrence-loader.ts";
export type { CooccurrenceData, LoaderOptions } from "./cooccurrence-loader.ts";

// V→E phase
export { VertexToEdgePhase } from "./vertex-to-edge-phase.ts";

// E→V phase
export { EdgeToVertexPhase } from "./edge-to-vertex-phase.ts";

// E→E phase
export { EdgeToEdgePhase } from "./edge-to-edge-phase.ts";

// Orchestrator
export { MultiLevelOrchestrator } from "./multi-level-orchestrator.ts";
export type { ForwardCache, LayerParameters, OrchestratorConfig } from "./multi-level-orchestrator.ts";
