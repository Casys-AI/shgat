/**
 * Multi-Level Message Passing Orchestrator
 *
 * Coordinates message passing across multiple hierarchy levels for
 * n-SuperHyperGraph structures.
 *
 * Implements both:
 * - Legacy 2-level: V → E → V
 * - Multi-level: V → E^0 → E^1 → ... → E^n → ... → E^0 → V
 *
 * @module graphrag/algorithms/shgat/message-passing/multi-level-orchestrator
 */

import * as math from "../utils/math.ts";
import type { PhaseParameters } from "./phase-interface.ts";
import type { LevelParams, MultiLevelEmbeddings, MultiLevelForwardCache } from "../core/types.ts";
import { VertexToEdgePhase, type VEForwardCache } from "./vertex-to-edge-phase.ts";
import { EdgeToVertexPhase, type EVForwardCache } from "./edge-to-vertex-phase.ts";
import { EdgeToEdgePhase, type EEForwardCache } from "./edge-to-edge-phase.ts";
import {
  VertexToVertexPhase,
  type CooccurrenceEntry,
  type V2VForwardCache,
  type V2VGradients,
  type V2VParams,
  type VertexToVertexConfig,
} from "./vertex-to-vertex-phase.ts";

/**
 * Extended cache for backward pass with per-phase caches
 */
export interface MultiLevelBackwardCache extends MultiLevelForwardCache {
  /** V→E phase caches per level per head: level → head → cache */
  veCaches: Map<number, VEForwardCache[]>;
  /** E→E upward phase caches per level per head: level → head → cache */
  eeUpwardCaches: Map<number, EEForwardCache[]>;
  /** E→E downward phase caches per level per head: level → head → cache */
  eeDownwardCaches: Map<number, EEForwardCache[]>;
  /** E→V phase caches per head: head → cache */
  evCaches: EVForwardCache[];
  /** V→V phase cache (optional, only if V2V enabled) */
  v2vCache?: V2VForwardCache;
  /** Tool-to-cap connectivity matrix */
  toolToCapMatrix: number[][];
  /** Cap-to-cap connectivity matrices: level → matrix */
  capToCapMatrices: Map<number, number[][]>;
  /** Max hierarchy level */
  maxLevel: number;
  /** Config used */
  config: OrchestratorConfig;
}

/**
 * Accumulated gradients for all level parameters
 */
export interface MultiLevelGradients {
  /** Gradients per level: level → LevelParamsGradients */
  levelGrads: Map<number, LevelParamsGradients>;
  /** Gradient for input H [numTools][embDim] */
  dH: number[][];
  /** Gradient for input E per level: level → [numCaps][embDim] */
  dE: Map<number, number[][]>;
  /** V→V gradients (optional, only if V2V enabled) */
  v2vGrads?: V2VGradients;
}

/**
 * Gradients for a single level's parameters
 */
export interface LevelParamsGradients {
  /** Per-head gradients for W_child */
  dW_child: number[][][];
  /** Per-head gradients for W_parent */
  dW_parent: number[][][];
  /** Per-head gradients for a_upward */
  da_upward: number[][];
  /** Per-head gradients for a_downward */
  da_downward: number[][];
}

/**
 * Layer parameters for all phases
 *
 * Each layer has parameters for all heads.
 */
export interface LayerParameters {
  /** Vertex projection matrices per head [numHeads][headDim][embeddingDim] */
  W_v: number[][][];
  /** Edge projection matrices per head [numHeads][headDim][embeddingDim] */
  W_e: number[][][];
  /** Attention vectors V→E per head [numHeads][2*headDim] */
  a_ve: number[][];
  /** Edge projection matrices (phase 2) per head [numHeads][headDim][embeddingDim] */
  W_e2: number[][][];
  /** Vertex projection matrices (phase 2) per head [numHeads][headDim][embeddingDim] */
  W_v2: number[][][];
  /** Attention vectors E→V per head [numHeads][2*headDim] */
  a_ev: number[][];
}

/**
 * Forward pass cache for backpropagation
 */
export interface ForwardCache {
  /** Tool embeddings per layer [numLayers+1][numTools][dim] */
  H: number[][][];
  /** Capability embeddings per layer [numLayers+1][numCaps][dim] */
  E: number[][][];
  /** Attention weights V→E [layer][head][numTools][numCaps] */
  attentionVE: number[][][][];
  /** Attention weights E→V [layer][head][numCaps][numTools] */
  attentionEV: number[][][][];
}

/**
 * Configuration for orchestrator
 */
export interface OrchestratorConfig {
  numHeads: number;
  numLayers: number;
  dropout: number;
  leakyReluSlope: number;
  /**
   * Residual weight for downward message passing.
   * output = (1-α)*propagated + α*original
   * @default 0 (pure propagation, no residual)
   */
  downwardResidual?: number;
}

/**
 * Multi-level message passing orchestrator
 *
 * Handles both current 2-level (V→E→V) and future multi-level (V→E^0→E^1→...→V)
 * message passing architectures.
 *
 * Optionally includes V→V pre-phase for co-occurrence enrichment from scraped patterns.
 */
export class MultiLevelOrchestrator {
  private readonly vertexToEdgePhase: VertexToEdgePhase;
  private readonly edgeToVertexPhase: EdgeToVertexPhase;
  private vertexToVertexPhase: VertexToVertexPhase | null;
  private readonly trainingMode: boolean;
  private cooccurrenceData: CooccurrenceEntry[] | null = null;

  constructor(
    trainingMode: boolean = false,
    v2vConfig?: Partial<VertexToVertexConfig>,
  ) {
    this.vertexToEdgePhase = new VertexToEdgePhase();
    this.edgeToVertexPhase = new EdgeToVertexPhase();
    this.vertexToVertexPhase = v2vConfig ? new VertexToVertexPhase(v2vConfig) : null;
    this.trainingMode = trainingMode;
  }

  /**
   * Set co-occurrence data for V→V enrichment
   *
   * @param data - Sparse co-occurrence entries from scraped patterns
   */
  setCooccurrenceData(data: CooccurrenceEntry[]): void {
    this.cooccurrenceData = data;
    // Initialize V→V phase if not already done
    if (!this.vertexToVertexPhase) {
      this.vertexToVertexPhase = new VertexToVertexPhase();
    }
  }

  /**
   * Apply V→V enrichment if configured
   *
   * @param H - Tool embeddings [numTools][embeddingDim]
   * @returns Enriched embeddings (or original if no co-occurrence data)
   */
  private applyV2VEnrichment(H: number[][]): number[][] {
    if (!this.vertexToVertexPhase || !this.cooccurrenceData || this.cooccurrenceData.length === 0) {
      return H;
    }

    const { embeddings } = this.vertexToVertexPhase.forward(H, this.cooccurrenceData);
    return embeddings;
  }

  /**
   * Execute forward pass through all layers
   *
   * Current implementation: 2-level (V→E→V)
   *
   * For each layer l:
   *   For each head k:
   *     1. V → E: Tools aggregate to capabilities
   *     2. E → V: Capabilities aggregate back to tools
   *   Concatenate all heads
   *   Apply dropout (if training)
   *
   * @param H_init - Initial tool embeddings [numTools][embeddingDim]
   * @param E_init - Initial capability embeddings [numCaps][embeddingDim]
   * @param incidenceMatrix - Connectivity [numTools][numCaps]
   * @param layerParams - Parameters for all layers
   * @param config - Configuration (numHeads, dropout, etc.)
   * @returns Final embeddings and cache for backprop
   */
  forward(
    H_init: number[][],
    E_init: number[][],
    incidenceMatrix: number[][],
    layerParams: LayerParameters[],
    config: OrchestratorConfig,
  ): { H: number[][]; E: number[][]; cache: ForwardCache } {
    const cache: ForwardCache = {
      H: [],
      E: [],
      attentionVE: [],
      attentionEV: [],
    };

    // Apply V→V co-occurrence enrichment (if configured)
    let H = this.applyV2VEnrichment(H_init);
    let E = E_init;

    cache.H.push(H);
    cache.E.push(E);

    // Process each layer
    for (let l = 0; l < config.numLayers; l++) {
      const params = layerParams[l];
      const layerAttentionVE: number[][][] = [];
      const layerAttentionEV: number[][][] = [];

      const headsH: number[][][] = [];
      const headsE: number[][][] = [];

      // Process each head in parallel
      for (let head = 0; head < config.numHeads; head++) {
        // Phase 1: Vertex → Hyperedge
        const veParams: PhaseParameters = {
          W_source: params.W_v[head],
          W_target: params.W_e[head],
          a_attention: params.a_ve[head],
        };

        const { embeddings: E_new, attention: attentionVE } = this.vertexToEdgePhase.forward(
          H,
          E,
          incidenceMatrix,
          veParams,
          { leakyReluSlope: config.leakyReluSlope },
        );

        layerAttentionVE.push(attentionVE);

        // Phase 2: Hyperedge → Vertex
        const evParams: PhaseParameters = {
          W_source: params.W_e2[head],
          W_target: params.W_v2[head],
          a_attention: params.a_ev[head],
        };

        const { embeddings: H_new, attention: attentionEV } = this.edgeToVertexPhase.forward(
          E_new,
          H,
          incidenceMatrix,
          evParams,
          { leakyReluSlope: config.leakyReluSlope },
        );

        layerAttentionEV.push(attentionEV);

        headsH.push(H_new);
        headsE.push(E_new);
      }

      // Concatenate heads
      H = math.concatHeads(headsH);
      E = math.concatHeads(headsE);

      // Apply dropout during training
      if (this.trainingMode && config.dropout > 0) {
        H = math.applyDropout(H, config.dropout);
        E = math.applyDropout(E, config.dropout);
      }

      cache.H.push(H);
      cache.E.push(E);
      cache.attentionVE.push(layerAttentionVE);
      cache.attentionEV.push(layerAttentionEV);
    }

    return { H, E, cache };
  }

  /**
   * Multi-level forward pass: V → E^0 → E^1 → ... → E^L_max → ... → E^0 → V
   *
   * Implements n-SuperHyperGraph message passing with:
   * 1. Upward aggregation: Tools → Level-0 Caps → ... → Level-L_max Caps
   * 2. Downward propagation: Level-L_max → ... → Level-0 → Tools
   *
   * @param H_init - Initial tool embeddings [numTools][embDim]
   * @param E_levels_init - Initial embeddings per level: level → [numCapsAtLevel][embDim]
   * @param toolToCapMatrix - I₀: Tool-to-level-0 connectivity [numTools][numCaps0]
   * @param capToCapMatrices - I_k: Level-(k-1) to level-k connectivity, keyed by parent level
   * @param levelParams - Parameters per hierarchy level
   * @param config - Configuration (numHeads, dropout, etc.)
   * @returns MultiLevelEmbeddings with final embeddings and attention weights
   */
  forwardMultiLevel(
    H_init: number[][],
    E_levels_init: Map<number, number[][]>,
    toolToCapMatrix: number[][],
    capToCapMatrices: Map<number, number[][]>,
    levelParams: Map<number, LevelParams>,
    config: OrchestratorConfig,
  ): { result: MultiLevelEmbeddings; cache: MultiLevelForwardCache } {
    // Validate inputs
    if (E_levels_init.size === 0) {
      throw new Error("forwardMultiLevel requires at least one level of capability embeddings");
    }

    const maxLevel = Math.max(...Array.from(E_levels_init.keys()));

    // Pre-create EdgeToEdgePhase instances to avoid repeated allocation
    const edgeToEdgePhases = new Map<string, EdgeToEdgePhase>();
    for (let level = 1; level <= maxLevel; level++) {
      edgeToEdgePhases.set(`up-${level}`, new EdgeToEdgePhase(level - 1, level));
      edgeToEdgePhases.set(`down-${level}`, new EdgeToEdgePhase(level, level - 1));
    }

    // Initialize result structures
    const E = new Map<number, number[][]>();
    const attentionUpward = new Map<number, number[][][]>();
    const attentionDownward = new Map<number, number[][][]>();

    // Initialize cache
    const cache: MultiLevelForwardCache = {
      H_init: H_init.map((row) => [...row]),
      H_final: [],
      E_init: new Map(),
      E_final: new Map(),
      intermediateUpward: new Map(),
      intermediateDownward: new Map(),
      attentionUpward: new Map(),
      attentionDownward: new Map(),
    };

    // Copy initial embeddings to cache
    for (const [level, embs] of E_levels_init) {
      cache.E_init.set(level, embs.map((row) => [...row]));
      E.set(level, embs.map((row) => [...row]));
    }

    // Apply V→V co-occurrence enrichment (if configured)
    const H_enriched = this.applyV2VEnrichment(H_init);

    // Track current tool embeddings
    let H = H_enriched.map((row) => [...row]);

    // ========================================================================
    // UPWARD PASS: V → E^0 → E^1 → ... → E^L_max
    // ========================================================================

    for (let level = 0; level <= maxLevel; level++) {
      const params = levelParams.get(level);
      if (!params) {
        throw new Error(`Missing LevelParams for level ${level}`);
      }

      const capsAtLevel = E.get(level);
      if (!capsAtLevel || capsAtLevel.length === 0) continue;

      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];

      for (let head = 0; head < config.numHeads; head++) {
        if (level === 0) {
          // Phase: Tools (V) → Level-0 Capabilities (E^0)
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const { embeddings, attention } = this.vertexToEdgePhase.forward(
            H,
            capsAtLevel,
            toolToCapMatrix,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(embeddings);
          levelAttention.push(attention);
        } else {
          // Phase: Level-(k-1) → Level-k Capabilities
          const E_prev = E.get(level - 1);
          if (!E_prev) continue;

          const connectivity = capToCapMatrices.get(level);
          if (!connectivity) continue;

          const phase = edgeToEdgePhases.get(`up-${level}`)!;
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const { embeddings, attention } = phase.forward(
            E_prev,
            capsAtLevel,
            connectivity,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(embeddings);
          levelAttention.push(attention);
        }
      }

      // Concatenate heads
      if (headsE.length > 0) {
        const E_new = math.concatHeads(headsE);
        E.set(level, E_new);
        cache.intermediateUpward.set(level, E_new.map((row) => [...row]));
      }

      attentionUpward.set(level, levelAttention);
      cache.attentionUpward.set(level, levelAttention);
    }

    // ========================================================================
    // DOWNWARD PASS: E^L_max → ... → E^1 → E^0 → V
    // ========================================================================

    // Downward: Higher levels → Lower levels
    for (let level = maxLevel - 1; level >= 0; level--) {
      const params = levelParams.get(level);
      if (!params) continue;

      const capsAtLevel = E.get(level);
      const capsAtParentLevel = E.get(level + 1);

      if (!capsAtLevel || capsAtLevel.length === 0) continue;
      if (!capsAtParentLevel || capsAtParentLevel.length === 0) continue;

      // Save pre-downward embeddings for residual connection
      const capsAtLevelPreDownward = capsAtLevel.map((row) => [...row]);

      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];

      // Get reverse connectivity (parent → child)
      const forwardConnectivity = capToCapMatrices.get(level + 1);
      if (!forwardConnectivity) continue;

      // Transpose the connectivity matrix for downward pass
      const reverseConnectivity = this.transposeMatrix(forwardConnectivity);

      const phase = edgeToEdgePhases.get(`down-${level + 1}`)!;

      for (let head = 0; head < config.numHeads; head++) {
        // Downward phase: Level-(k+1) → Level-k
        const phaseParams: PhaseParameters = {
          W_source: params.W_parent[head], // Parents are source in downward
          W_target: params.W_child[head], // Children are target in downward
          a_attention: params.a_downward[head],
        };

        const { embeddings: propagated, attention } = phase.forward(
          capsAtParentLevel,
          capsAtLevel,
          reverseConnectivity,
          phaseParams,
          { leakyReluSlope: config.leakyReluSlope },
        );

        headsE.push(propagated);
        levelAttention.push(attention);
      }

      // Concatenate heads first
      if (headsE.length > 0) {
        const E_concat = math.concatHeads(headsE);

        // Apply residual connection with configurable weight
        // E_new = (1-α)*propagated + α*original, where α = downwardResidual
        const alpha = config.downwardResidual ?? 0;
        const E_new = capsAtLevelPreDownward.map((row, i) =>
          row.map((val, j) => {
            const propagated = E_concat[i]?.[j] ?? 0;
            return (1 - alpha) * propagated + alpha * val;
          })
        );

        E.set(level, E_new);
        cache.intermediateDownward.set(level, E_new.map((row) => [...row]));
      }

      attentionDownward.set(level, levelAttention);
      cache.attentionDownward.set(level, levelAttention);
    }

    // Final phase: Level-0 → Tools (downward)
    const E_level0 = E.get(0);
    if (E_level0 && E_level0.length > 0) {
      const params = levelParams.get(0);
      if (params) {
        // Save pre-downward tool embeddings for residual connection
        const H_preDownward = H.map((row) => [...row]);

        // EdgeToVertexPhase expects [tool][cap] format (same as upward pass)
        // It internally handles the reverse aggregation direction

        const headsH: number[][][] = [];
        const levelAttention: number[][][] = [];

        for (let head = 0; head < config.numHeads; head++) {
          const phaseParams: PhaseParameters = {
            W_source: params.W_parent[head],
            W_target: params.W_child[head],
            a_attention: params.a_downward[head],
          };

          const { embeddings: propagated, attention } = this.edgeToVertexPhase.forward(
            E_level0,
            H,
            toolToCapMatrix,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsH.push(propagated);
          levelAttention.push(attention);
        }

        // Concatenate heads first
        if (headsH.length > 0) {
          const H_concat = math.concatHeads(headsH);

          // Apply residual connection with configurable weight
          // H = (1-α)*propagated + α*original, where α = downwardResidual
          const alpha = config.downwardResidual ?? 0;
          H = H_preDownward.map((row, i) =>
            row.map((val, j) => {
              const propagated = H_concat[i]?.[j] ?? 0;
              return (1 - alpha) * propagated + alpha * val;
            })
          );
        }

        attentionDownward.set(-1, levelAttention);
        cache.attentionDownward.set(-1, levelAttention);
      }
    }

    // Apply dropout during training
    if (this.trainingMode && config.dropout > 0) {
      H = math.applyDropout(H, config.dropout);
      for (const [level, embs] of E) {
        E.set(level, math.applyDropout(embs, config.dropout));
      }
    }

    // Store final embeddings in cache
    cache.H_final = H.map((row) => [...row]);
    for (const [level, embs] of E) {
      cache.E_final.set(level, embs.map((row) => [...row]));
    }

    const result: MultiLevelEmbeddings = {
      H,
      E,
      attentionUpward,
      attentionDownward,
    };

    return { result, cache };
  }

  /**
   * Transpose a matrix
   * @param matrix Input matrix [rows][cols]
   * @returns Transposed matrix [cols][rows]
   */
  private transposeMatrix(matrix: number[][]): number[][] {
    if (matrix.length === 0) return [];
    const rows = matrix.length;
    const cols = matrix[0].length;

    return Array.from(
      { length: cols },
      (_, j) => Array.from({ length: rows }, (_, i) => matrix[i][j]),
    );
  }

  /**
   * Forward pass with extended cache for backward
   *
   * Same as forwardMultiLevel but stores per-phase caches for backward pass.
   *
   * @param v2vParams - Optional V2V learnable parameters. If provided, uses trainable V2V.
   */
  forwardMultiLevelWithCache(
    H_init: number[][],
    E_levels_init: Map<number, number[][]>,
    toolToCapMatrix: number[][],
    capToCapMatrices: Map<number, number[][]>,
    levelParams: Map<number, LevelParams>,
    config: OrchestratorConfig,
    v2vParams?: V2VParams,
  ): { result: MultiLevelEmbeddings; cache: MultiLevelBackwardCache } {
    if (E_levels_init.size === 0) {
      throw new Error("forwardMultiLevelWithCache requires at least one level");
    }

    const maxLevel = Math.max(...Array.from(E_levels_init.keys()));

    // Initialize extended cache
    const veCaches = new Map<number, VEForwardCache[]>();
    const eeUpwardCaches = new Map<number, EEForwardCache[]>();
    const eeDownwardCaches = new Map<number, EEForwardCache[]>();
    const evCaches: EVForwardCache[] = [];

    // Initialize result structures
    const E = new Map<number, number[][]>();
    const attentionUpward = new Map<number, number[][][]>();
    const attentionDownward = new Map<number, number[][][]>();

    // Copy initial embeddings
    for (const [level, embs] of E_levels_init) {
      E.set(level, embs.map((row) => [...row]));
    }

    // Apply V→V enrichment (with cache for backward pass)
    let v2vCache: V2VForwardCache | undefined;
    let H_enriched: number[][];

    if (v2vParams && this.vertexToVertexPhase && this.cooccurrenceData && this.cooccurrenceData.length > 0) {
      // Use trainable V2V with cache
      const v2vResult = this.vertexToVertexPhase.forwardWithCache(H_init, this.cooccurrenceData, v2vParams);
      H_enriched = v2vResult.embeddings;
      v2vCache = v2vResult.cache;
    } else if (v2vParams) {
      // v2vParams provided but V2V not configured - this is a bug
      throw new Error(
        "[MultiLevelOrchestrator] v2vParams provided but V2V phase not configured. " +
        "Either call setCooccurrenceData() or don't pass v2vParams."
      );
    } else {
      // No V2V requested - pass through unchanged
      H_enriched = H_init.map(row => [...row]);
    }

    let H = H_enriched.map((row) => [...row]);

    // Pre-create EdgeToEdgePhase instances
    const edgeToEdgePhases = new Map<string, EdgeToEdgePhase>();
    for (let level = 1; level <= maxLevel; level++) {
      edgeToEdgePhases.set(`up-${level}`, new EdgeToEdgePhase(level - 1, level));
      edgeToEdgePhases.set(`down-${level}`, new EdgeToEdgePhase(level, level - 1));
    }

    // ========================================================================
    // UPWARD PASS with caching
    // ========================================================================
    for (let level = 0; level <= maxLevel; level++) {
      const params = levelParams.get(level);
      if (!params) continue;

      const capsAtLevel = E.get(level);
      if (!capsAtLevel || capsAtLevel.length === 0) continue;

      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];
      const levelVECaches: VEForwardCache[] = [];
      const levelEECaches: EEForwardCache[] = [];

      for (let head = 0; head < config.numHeads; head++) {
        if (level === 0) {
          // V→E with cache
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const { embeddings, attention, cache: veCache } = this.vertexToEdgePhase.forwardWithCache(
            H,
            capsAtLevel,
            toolToCapMatrix,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(embeddings);
          levelAttention.push(attention);
          levelVECaches.push(veCache);
        } else {
          // E^(k-1)→E^k with cache
          const E_prev = E.get(level - 1);
          const connectivity = capToCapMatrices.get(level);
          if (!E_prev || !connectivity) continue;

          const phase = edgeToEdgePhases.get(`up-${level}`)!;
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const { embeddings, attention, cache: eeCache } = phase.forwardWithCache(
            E_prev,
            capsAtLevel,
            connectivity,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(embeddings);
          levelAttention.push(attention);
          levelEECaches.push(eeCache);
        }
      }

      // Store caches
      if (level === 0) {
        veCaches.set(level, levelVECaches);
      } else {
        eeUpwardCaches.set(level, levelEECaches);
      }

      // Concatenate heads
      if (headsE.length > 0) {
        E.set(level, math.concatHeads(headsE));
      }
      attentionUpward.set(level, levelAttention);
    }

    // ========================================================================
    // DOWNWARD PASS with caching
    // ========================================================================
    for (let level = maxLevel - 1; level >= 0; level--) {
      const params = levelParams.get(level);
      if (!params) continue;

      const capsAtLevel = E.get(level);
      const capsAtParentLevel = E.get(level + 1);
      if (!capsAtLevel || !capsAtParentLevel) continue;

      const capsAtLevelPreDownward = capsAtLevel.map((row) => [...row]);
      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];
      const levelEECaches: EEForwardCache[] = [];

      const forwardConnectivity = capToCapMatrices.get(level + 1);
      if (!forwardConnectivity) continue;
      const reverseConnectivity = this.transposeMatrix(forwardConnectivity);

      const phase = edgeToEdgePhases.get(`down-${level + 1}`)!;

      for (let head = 0; head < config.numHeads; head++) {
        const phaseParams: PhaseParameters = {
          W_source: params.W_parent[head],
          W_target: params.W_child[head],
          a_attention: params.a_downward[head],
        };

        const { embeddings, attention, cache: eeCache } = phase.forwardWithCache(
          capsAtParentLevel,
          capsAtLevel,
          reverseConnectivity,
          phaseParams,
          { leakyReluSlope: config.leakyReluSlope },
        );

        headsE.push(embeddings);
        levelAttention.push(attention);
        levelEECaches.push(eeCache);
      }

      eeDownwardCaches.set(level, levelEECaches);

      if (headsE.length > 0) {
        const E_concat = math.concatHeads(headsE);
        // Residual connection
        const E_new = capsAtLevelPreDownward.map((row, i) =>
          row.map((val, j) => val + (E_concat[i]?.[j] ?? 0))
        );
        E.set(level, E_new);
      }
      attentionDownward.set(level, levelAttention);
    }

    // Final E^0→V phase with caching
    const E_level0 = E.get(0);
    if (E_level0 && E_level0.length > 0) {
      const params = levelParams.get(0);
      if (params) {
        const H_preDownward = H.map((row) => [...row]);
        const headsH: number[][][] = [];
        const levelAttention: number[][][] = [];

        for (let head = 0; head < config.numHeads; head++) {
          const phaseParams: PhaseParameters = {
            W_source: params.W_parent[head],
            W_target: params.W_child[head],
            a_attention: params.a_downward[head],
          };

          const { embeddings, attention, cache: evCache } = this.edgeToVertexPhase.forwardWithCache(
            E_level0,
            H,
            toolToCapMatrix,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsH.push(embeddings);
          levelAttention.push(attention);
          evCaches.push(evCache);
        }

        if (headsH.length > 0) {
          const H_concat = math.concatHeads(headsH);
          H = H_preDownward.map((row, i) => row.map((val, j) => val + (H_concat[i]?.[j] ?? 0)));
        }
        attentionDownward.set(-1, levelAttention);
      }
    }

    // Apply dropout during training
    if (this.trainingMode && config.dropout > 0) {
      H = math.applyDropout(H, config.dropout);
      for (const [level, embs] of E) {
        E.set(level, math.applyDropout(embs, config.dropout));
      }
    }

    const cache: MultiLevelBackwardCache = {
      H_init: H_init.map((row) => [...row]),
      H_final: H.map((row) => [...row]),
      E_init: new Map(Array.from(E_levels_init.entries()).map(([k, v]) => [k, v.map(r => [...r])])),
      E_final: new Map(Array.from(E.entries()).map(([k, v]) => [k, v.map(r => [...r])])),
      intermediateUpward: new Map(),
      intermediateDownward: new Map(),
      attentionUpward,
      attentionDownward,
      veCaches,
      eeUpwardCaches,
      eeDownwardCaches,
      evCaches,
      v2vCache,
      toolToCapMatrix,
      capToCapMatrices,
      maxLevel,
      config,
    };

    const result: MultiLevelEmbeddings = { H, E, attentionUpward, attentionDownward };
    return { result, cache };
  }

  /**
   * Backward pass through multi-level message passing
   *
   * Reverses the forward pass order:
   * 1. Backward through E^0→V (gives dE^0)
   * 2. Backward through downward passes E^(k+1)→E^k (gives dE^k+1, dE^k)
   * 3. Backward through upward passes E^k→E^(k+1) and V→E^0 (gives gradients)
   * 4. Backward through V→V (if trainable V2V enabled)
   *
   * @param dE_final - Gradient on final capability embeddings (from K-head scoring)
   * @param dH_final - Gradient on final tool embeddings (optional, usually zero)
   * @param cache - Forward pass cache
   * @param levelParams - Level parameters for gradient computation
   * @param v2vParams - Optional V2V learnable parameters (for backward)
   * @returns Accumulated gradients for all level parameters
   */
  backwardMultiLevel(
    dE_final: Map<number, number[][]>,
    dH_final: number[][] | null,
    cache: MultiLevelBackwardCache,
    levelParams: Map<number, LevelParams>,
    v2vParams?: V2VParams,
  ): MultiLevelGradients {
    const { maxLevel, config } = cache;
    const numHeads = config.numHeads;

    // Initialize gradients
    const levelGrads = new Map<number, LevelParamsGradients>();
    for (let level = 0; level <= maxLevel; level++) {
      const params = levelParams.get(level);
      if (!params) continue;

      const headDim = params.W_child[0]?.length ?? 0;
      const embDim = params.W_child[0]?.[0]?.length ?? 0;

      levelGrads.set(level, {
        dW_child: Array.from({ length: numHeads }, () =>
          Array.from({ length: headDim }, () => Array(embDim).fill(0))),
        dW_parent: Array.from({ length: numHeads }, () =>
          Array.from({ length: headDim }, () => Array(embDim).fill(0))),
        da_upward: Array.from({ length: numHeads }, () => Array(2 * headDim).fill(0)),
        da_downward: Array.from({ length: numHeads }, () => Array(2 * headDim).fill(0)),
      });
    }

    // Current gradients flowing backward
    const dE = new Map<number, number[][]>();
    for (const [level, grad] of dE_final) {
      dE.set(level, grad.map(r => [...r]));
    }

    let dH: number[][] = dH_final
      ? dH_final.map(r => [...r])
      : Array.from({ length: cache.H_init.length }, () =>
          Array(cache.H_init[0]?.length ?? 0).fill(0));

    // ========================================================================
    // BACKWARD through E^0→V (final downward phase)
    // ========================================================================
    if (cache.evCaches.length > 0) {
      const params = levelParams.get(0);
      if (params) {
        const grads = levelGrads.get(0)!;

        // Split dH into per-head gradients (reverse of concat)
        const headDim = cache.evCaches[0]?.E_proj[0]?.length ?? 0;
        const dH_perHead: number[][][] = [];
        for (let head = 0; head < numHeads; head++) {
          dH_perHead.push(dH.map(row =>
            row.slice(head * headDim, (head + 1) * headDim)
          ));
        }

        for (let head = 0; head < numHeads; head++) {
          const evCache = cache.evCaches[head];
          const phaseParams: PhaseParameters = {
            W_source: params.W_parent[head],
            W_target: params.W_child[head],
            a_attention: params.a_downward[head],
          };

          const evGrads = this.edgeToVertexPhase.backward(dH_perHead[head], evCache, phaseParams);

          // Accumulate gradients
          this.accumulateMatrix(grads.dW_parent[head], evGrads.dW_source);
          this.accumulateMatrix(grads.dW_child[head], evGrads.dW_target);
          this.accumulateVector(grads.da_downward[head], evGrads.da_attention);

          // Accumulate dE for level 0
          const dE0 = dE.get(0) ?? evGrads.dE.map(r => Array(r.length).fill(0));
          for (let i = 0; i < evGrads.dE.length; i++) {
            for (let j = 0; j < evGrads.dE[i].length; j++) {
              dE0[i][j] += evGrads.dE[i][j];
            }
          }
          dE.set(0, dE0);
        }
      }
    }

    // ========================================================================
    // BACKWARD through downward passes E^(k+1)→E^k
    // ========================================================================
    for (let level = 0; level < maxLevel; level++) {
      const eeCaches = cache.eeDownwardCaches.get(level);
      if (!eeCaches || eeCaches.length === 0) continue;

      const params = levelParams.get(level);
      if (!params) continue;

      const grads = levelGrads.get(level)!;
      const dE_level = dE.get(level);
      if (!dE_level) continue;

      // Split into per-head
      const headDim = eeCaches[0]?.E_k_proj[0]?.length ?? 0;
      const dE_perHead: number[][][] = [];
      for (let head = 0; head < numHeads; head++) {
        dE_perHead.push(dE_level.map(row =>
          row.slice(head * headDim, (head + 1) * headDim)
        ));
      }

      for (let head = 0; head < numHeads; head++) {
        const eeCache = eeCaches[head];
        const phaseParams: PhaseParameters = {
          W_source: params.W_parent[head],
          W_target: params.W_child[head],
          a_attention: params.a_downward[head],
        };

        // Note: In downward, E_k is child (target), E_kPlus1 is parent (source)
        const eeGrads = this.createEdgeToEdgePhaseForBackward(level + 1, level)
          .backward(dE_perHead[head], eeCache, phaseParams);

        this.accumulateMatrix(grads.dW_parent[head], eeGrads.dW_source);
        this.accumulateMatrix(grads.dW_child[head], eeGrads.dW_target);
        this.accumulateVector(grads.da_downward[head], eeGrads.da_attention);

        // Accumulate dE for parent level (k+1)
        const dE_parent = dE.get(level + 1) ?? eeGrads.dE_k.map(r => Array(r.length).fill(0));
        for (let i = 0; i < eeGrads.dE_k.length; i++) {
          for (let j = 0; j < eeGrads.dE_k[i].length; j++) {
            dE_parent[i][j] += eeGrads.dE_k[i][j];
          }
        }
        dE.set(level + 1, dE_parent);
      }
    }

    // ========================================================================
    // BACKWARD through upward passes
    // ========================================================================
    for (let level = maxLevel; level >= 0; level--) {
      if (level === 0) {
        // V→E backward
        const veCaches = cache.veCaches.get(0);
        if (!veCaches || veCaches.length === 0) continue;

        const params = levelParams.get(0);
        if (!params) continue;

        const grads = levelGrads.get(0)!;
        const dE0 = dE.get(0);
        if (!dE0) continue;

        const headDim = veCaches[0]?.H_proj[0]?.length ?? 0;
        const dE_perHead: number[][][] = [];
        for (let head = 0; head < numHeads; head++) {
          dE_perHead.push(dE0.map(row =>
            row.slice(head * headDim, (head + 1) * headDim)
          ));
        }

        for (let head = 0; head < numHeads; head++) {
          const veCache = veCaches[head];
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const veGrads = this.vertexToEdgePhase.backward(dE_perHead[head], veCache, phaseParams);

          this.accumulateMatrix(grads.dW_child[head], veGrads.dW_source);
          this.accumulateMatrix(grads.dW_parent[head], veGrads.dW_target);
          this.accumulateVector(grads.da_upward[head], veGrads.da_attention);

          // Accumulate dH
          for (let i = 0; i < veGrads.dH.length; i++) {
            for (let j = 0; j < veGrads.dH[i].length; j++) {
              dH[i][j] += veGrads.dH[i][j];
            }
          }
        }
      } else {
        // E^(k-1)→E^k backward
        const eeCaches = cache.eeUpwardCaches.get(level);
        if (!eeCaches || eeCaches.length === 0) continue;

        const params = levelParams.get(level);
        if (!params) continue;

        const grads = levelGrads.get(level)!;
        const dE_level = dE.get(level);
        if (!dE_level) continue;

        const headDim = eeCaches[0]?.E_k_proj[0]?.length ?? 0;
        const dE_perHead: number[][][] = [];
        for (let head = 0; head < numHeads; head++) {
          dE_perHead.push(dE_level.map(row =>
            row.slice(head * headDim, (head + 1) * headDim)
          ));
        }

        for (let head = 0; head < numHeads; head++) {
          const eeCache = eeCaches[head];
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const eeGrads = this.createEdgeToEdgePhaseForBackward(level - 1, level)
            .backward(dE_perHead[head], eeCache, phaseParams);

          this.accumulateMatrix(grads.dW_child[head], eeGrads.dW_source);
          this.accumulateMatrix(grads.dW_parent[head], eeGrads.dW_target);
          this.accumulateVector(grads.da_upward[head], eeGrads.da_attention);

          // Accumulate dE for child level (k-1)
          const dE_child = dE.get(level - 1) ?? eeGrads.dE_k.map(r => Array(r.length).fill(0));
          for (let i = 0; i < eeGrads.dE_k.length; i++) {
            for (let j = 0; j < eeGrads.dE_k[i].length; j++) {
              dE_child[i][j] += eeGrads.dE_k[i][j];
            }
          }
          dE.set(level - 1, dE_child);
        }
      }
    }

    // ========================================================================
    // BACKWARD through V→V (if trainable V2V enabled)
    // ========================================================================
    let v2vGrads: V2VGradients | undefined;

    if (v2vParams && cache.v2vCache && this.vertexToVertexPhase) {
      // V2V was applied first in forward, so backward through it last
      // dH contains gradients from all downstream phases
      v2vGrads = this.vertexToVertexPhase.backward(dH, cache.v2vCache, v2vParams);

      // Update dH to include gradients flowing to original H_init
      // (v2vGrads.dH contains the gradient for the original input)
      dH = v2vGrads.dH;
    }

    return { levelGrads, dH, dE, v2vGrads };
  }

  /**
   * Helper to create EdgeToEdge phase for backward
   */
  private createEdgeToEdgePhaseForBackward(levelK: number, levelKPlus1: number): EdgeToEdgePhase {
    return new EdgeToEdgePhase(levelK, levelKPlus1);
  }

  /**
   * Accumulate matrix gradients
   */
  private accumulateMatrix(target: number[][], source: number[][]): void {
    for (let i = 0; i < source.length; i++) {
      for (let j = 0; j < source[i].length; j++) {
        target[i][j] += source[i][j];
      }
    }
  }

  /**
   * Accumulate vector gradients
   */
  private accumulateVector(target: number[], source: number[]): void {
    for (let i = 0; i < source.length; i++) {
      target[i] += source[i];
    }
  }
}
