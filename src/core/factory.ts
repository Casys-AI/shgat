/**
 * SHGAT Factory Functions
 *
 * Factory functions for creating SHGAT instances.
 * Training functions moved to AutogradTrainer.
 *
 * @module shgat-tf/core/factory
 */

import { getLogger } from "./logger.ts";
import { SHGAT } from "./shgat.ts";
import type { HypergraphFeatures, Node, SHGATConfig, TrainingExample } from "./types.ts";
import { buildGraph, createMembersFromLegacy } from "./types.ts";
import { getAdaptiveHeadsByGraphSize } from "../initialization/index.ts";
import { generateDefaultToolEmbedding } from "../graph/mod.ts";

const log = getLogger();

// ============================================================================
// Factory Functions (Unified Node API)
// ============================================================================

/**
 * Create SHGAT from unified nodes
 *
 * This is the new recommended API that uses the unified Node type.
 *
 * @param nodes Array of nodes (leaves have children: [], composites have children: [...])
 * @param config Optional SHGAT configuration
 * @returns SHGAT instance with all nodes registered
 *
 * @example
 * ```typescript
 * const nodes: Node[] = [
 *   { id: 'tool-1', embedding: [...], children: [], level: 0 },
 *   { id: 'tool-2', embedding: [...], children: [], level: 0 },
 *   { id: 'cap-1', embedding: [...], children: ['tool-1', 'tool-2'], level: 0 },
 * ];
 * const shgat = createSHGAT(nodes);
 * ```
 */
export function createSHGAT(
  nodes: Node[],
  config?: Partial<SHGATConfig>,
): SHGAT {
  // Build graph to compute levels
  const graph = buildGraph(nodes);

  // Count leaves and composites
  let leafCount = 0;
  let compositeCount = 0;
  let maxLevel = 0;
  for (const node of graph.values()) {
    if (node.children.length === 0) {
      leafCount++;
    } else {
      compositeCount++;
    }
    if (node.level > maxLevel) {
      maxLevel = node.level;
    }
  }

  // Get embedding dimension from first node
  const embeddingDim = nodes[0]?.embedding.length || 1024;
  const preserveDim = config?.preserveDim ?? true;

  // Adaptive K based on graph size (ADR-053)
  const adaptiveConfig = getAdaptiveHeadsByGraphSize(
    leafCount,
    compositeCount,
    maxLevel,
    preserveDim,
    embeddingDim,
  );

  // Merge configs
  const mergedConfig: Partial<SHGATConfig> = {
    numHeads: adaptiveConfig.numHeads,
    hiddenDim: adaptiveConfig.hiddenDim,
    headDim: adaptiveConfig.headDim,
    ...config,
  };

  const shgat = new SHGAT(mergedConfig);

  // Register all nodes
  for (const node of graph.values()) {
    shgat.registerNode(node);
  }

  // Finalize: rebuild indices once after all nodes are registered
  shgat.finalizeNodes();

  return shgat;
}

// ============================================================================
// Legacy Factory Functions
// ============================================================================

/**
 * Create SHGAT from capability records
 *
 * @deprecated Use createSHGAT(nodes) instead
 *
 * @param capabilities Array of capability records
 * @param configOrToolEmbeddings Either config or tool embeddings map
 * @param config Config (if second param is tool embeddings)
 */
export function createSHGATFromCapabilities(
  capabilities: Array<
    {
      id: string;
      embedding: number[];
      toolsUsed: string[];
      successRate: number;
      parents?: string[];
      children?: string[];
      hypergraphFeatures?: HypergraphFeatures;
    }
  >,
  configOrToolEmbeddings?: Partial<SHGATConfig> | Map<string, number[]>,
  config?: Partial<SHGATConfig>,
): SHGAT {
  let toolEmbeddings: Map<string, number[]> | undefined;
  let actualConfig: Partial<SHGATConfig> | undefined;

  if (configOrToolEmbeddings instanceof Map) {
    toolEmbeddings = configOrToolEmbeddings;
    actualConfig = config;
  } else {
    actualConfig = configOrToolEmbeddings;
  }

  // Collect all unique tools
  const allTools = new Set<string>();
  for (const cap of capabilities) for (const toolId of cap.toolsUsed) allTools.add(toolId);

  // Compute max hierarchy level from children relationships
  const hasChildren = capabilities.some((c) => c.children && c.children.length > 0);
  const maxLevel = hasChildren ? 1 : 0;

  // Get embeddingDim and preserveDim from config
  const embeddingDim = capabilities[0]?.embedding.length || 1024;
  const preserveDim = actualConfig?.preserveDim ?? true;

  // Adaptive K based on graph size (ADR-053)
  const adaptiveConfig = getAdaptiveHeadsByGraphSize(
    allTools.size,
    capabilities.length,
    maxLevel,
    preserveDim,
    embeddingDim,
  );

  // Merge: user config overrides adaptive, adaptive overrides defaults
  const mergedConfig: Partial<SHGATConfig> = {
    numHeads: adaptiveConfig.numHeads,
    hiddenDim: adaptiveConfig.hiddenDim,
    headDim: adaptiveConfig.headDim,
    ...actualConfig,
  };

  // Validate config consistency
  const finalHiddenDim = mergedConfig.hiddenDim ?? adaptiveConfig.hiddenDim;
  const finalNumHeads = mergedConfig.numHeads ?? adaptiveConfig.numHeads;
  const expectedHiddenDim = finalNumHeads * 64;
  if (finalHiddenDim !== expectedHiddenDim) {
    log.warn(
      `[SHGAT] hiddenDim should be numHeads * 64 = ${expectedHiddenDim}, got ${finalHiddenDim}. ` +
        `Each head needs 64 dims for full expressiveness.`,
    );
  }

  const shgat = new SHGAT(mergedConfig);

  for (const toolId of allTools) {
    shgat.registerTool({
      id: toolId,
      embedding: toolEmbeddings?.get(toolId) || generateDefaultToolEmbedding(toolId, embeddingDim),
    });
  }

  for (const cap of capabilities) {
    shgat.registerCapability({
      id: cap.id,
      embedding: cap.embedding,
      members: createMembersFromLegacy(cap.toolsUsed, cap.children),
      hierarchyLevel: 0,
      toolsUsed: cap.toolsUsed,
      successRate: cap.successRate,
      parents: cap.parents,
      children: cap.children,
    });
    if (cap.hypergraphFeatures) shgat.updateHypergraphFeatures(cap.id, cap.hypergraphFeatures);
  }

  // Finalize: rebuild indices once after all nodes are registered
  shgat.finalizeNodes();

  return shgat;
}

// ============================================================================
// Deprecated Training Functions
// ============================================================================

/**
 * @deprecated Use AutogradTrainer from training/autograd-trainer.ts instead
 */
export async function trainSHGATOnEpisodes(
  _shgat: SHGAT,
  _episodes: TrainingExample[],
  _getEmbedding: (id: string) => number[] | null,
  _options?: {
    epochs?: number;
    batchSize?: number;
    onEpoch?: (epoch: number, loss: number, accuracy: number) => void;
  },
): Promise<{ finalLoss: number; finalAccuracy: number }> {
  throw new Error(
    "trainSHGATOnEpisodes is deprecated in shgat-tf. " +
    "Use AutogradTrainer from training/autograd-trainer.ts instead."
  );
}

/**
 * @deprecated Use AutogradTrainer from training/autograd-trainer.ts instead
 */
export async function trainSHGATOnEpisodesKHead(
  _shgat: SHGAT,
  _episodes: TrainingExample[],
  _getEmbedding: (id: string) => number[] | null,
  _options?: {
    epochs?: number;
    learningRate?: number;
    batchSize?: number;
    onEpoch?: (epoch: number, loss: number, accuracy: number) => void;
  },
): Promise<{ finalLoss: number; finalAccuracy: number }> {
  throw new Error(
    "trainSHGATOnEpisodesKHead is deprecated in shgat-tf. " +
    "Use AutogradTrainer from training/autograd-trainer.ts instead."
  );
}

/**
 * @deprecated Use AutogradTrainer from training/autograd-trainer.ts instead
 */
export async function trainSHGATOnExecution(
  _shgat: SHGAT,
  _execution: {
    intentEmbedding: number[];
    targetCapId: string;
    outcome: number;
  },
): Promise<{ loss: number; accuracy: number; gradNorm: number }> {
  throw new Error(
    "trainSHGATOnExecution is deprecated in shgat-tf. " +
    "Use AutogradTrainer from training/autograd-trainer.ts instead."
  );
}
