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
import type { PhaseForwardCache, PhaseParameters, SparseConnectivity } from "./phase-interface.ts";
import { denseToSparse, transposeSparse } from "./phase-interface.ts";
import type { LevelParams, MultiLevelEmbeddings, MultiLevelForwardCache } from "../core/types.ts";
import { VertexToEdgePhase } from "./vertex-to-edge-phase.ts";
import { EdgeToVertexPhase } from "./edge-to-vertex-phase.ts";
import { EdgeToEdgePhase } from "./edge-to-edge-phase.ts";
import {
  VertexToVertexPhase,
  type CooccurrenceEntry,
  type V2VForwardCache,
  type V2VGradients,
  type V2VParams,
  type VertexToVertexConfig,
} from "./vertex-to-vertex-phase.ts";
import type {
  LevelIntermediates,
  ExtendedMultiLevelForwardCache,
} from "../training/multi-level-trainer.ts";

/**
 * Extended cache for backward pass with per-phase caches
 */
export interface MultiLevelBackwardCache extends MultiLevelForwardCache {
  /** V→E phase caches per level per head: level → head → cache */
  veCaches: Map<number, PhaseForwardCache[]>;
  /** E→E upward phase caches per level per head: level → head → cache */
  eeUpwardCaches: Map<number, PhaseForwardCache[]>;
  /** E→E downward phase caches per level per head: level → head → cache */
  eeDownwardCaches: Map<number, PhaseForwardCache[]>;
  /** E→V phase caches per head: head → cache */
  evCaches: PhaseForwardCache[];
  /** V→V phase cache (optional, only if V2V enabled) */
  v2vCache?: V2VForwardCache;
  /** L0-to-L1 sparse connectivity */
  l0ToL1Conn: SparseConnectivity;
  /** Inter-level sparse connectivity: level → sparse */
  interLevelConns: Map<number, SparseConnectivity>;
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
  /** Gradient for input H [numL0][embDim] */
  dH: number[][];
  /** Gradient for input E per level: level → [numL1][embDim] */
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
  /** L0 node projection matrices per head [numHeads][headDim][embeddingDim] */
  W_v: number[][][];
  /** L1+ node projection matrices per head [numHeads][headDim][embeddingDim] */
  W_e: number[][][];
  /** Attention vectors V→E per head [numHeads][2*headDim] */
  a_ve: number[][];
  /** L1+ node projection matrices (phase 2) per head [numHeads][headDim][embeddingDim] */
  W_e2: number[][][];
  /** L0 node projection matrices (phase 2) per head [numHeads][headDim][embeddingDim] */
  W_v2: number[][][];
  /** Attention vectors E→V per head [numHeads][2*headDim] */
  a_ev: number[][];
}

/**
 * Forward pass cache for backpropagation
 */
export interface ForwardCache {
  /** L0 node embeddings per layer [numLayers+1][numL0][dim] */
  H: number[][][];
  /** L1+ node embeddings per layer [numLayers+1][numL1][dim] */
  E: number[][][];
  /** Attention weights V→E [layer][head][numL0][numL1] */
  attentionVE: number[][][][];
  /** Attention weights E→V [layer][head][numL1][numL0] */
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
   * @param H - L0 node embeddings [numL0][embeddingDim]
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
   *     1. V → E: L0 nodes aggregate to L1+ nodes
   *     2. E → V: L1+ nodes aggregate back to L0 nodes
   *   Concatenate all heads
   *   Apply dropout (if training)
   *
   * @param H_init - Initial L0 node embeddings [numL0][embeddingDim]
   * @param E_init - Initial L1+ node embeddings [numL1][embeddingDim]
   * @param incidenceMatrix - Connectivity [numL0][numL1] (dense, auto-converted to sparse)
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
    // Convert dense → sparse once (legacy backward compat)
    const conn = denseToSparse(incidenceMatrix);

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
        // Phase 1: L0 → L1+ (upward aggregation)
        const veParams: PhaseParameters = {
          W_source: params.W_v[head],
          W_target: params.W_e[head],
          a_attention: params.a_ve[head],
        };

        const { embeddings: E_new, attention: attentionVE } = this.vertexToEdgePhase.forward(
          H,
          E,
          conn,
          veParams,
          { leakyReluSlope: config.leakyReluSlope },
        );

        layerAttentionVE.push(attentionVE);

        // Phase 2: L1+ → L0 (downward propagation)
        const evParams: PhaseParameters = {
          W_source: params.W_e2[head],
          W_target: params.W_v2[head],
          a_attention: params.a_ev[head],
        };

        const { embeddings: H_new, attention: attentionEV } = this.edgeToVertexPhase.forward(
          E_new,
          H,
          conn,
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
   * 1. Upward aggregation: L0 nodes → L1 nodes → ... → L_max nodes
   * 2. Downward propagation: L_max → ... → L1 → L0 nodes
   *
   * @param H_init - Initial L0 node embeddings [numL0][embDim]
   * @param E_levels_init - Initial embeddings per level: level → [numNodesAtLevel][embDim]
   * @param l0ToL1Matrix - I₀: L0-to-L1 connectivity [numL0][numL1] (dense, auto-converted)
   * @param interLevelMatrices - I_k: Level-(k-1) to level-k connectivity, keyed by parent level (dense)
   * @param levelParams - Parameters per hierarchy level
   * @param config - Configuration (numHeads, dropout, etc.)
   * @returns MultiLevelEmbeddings with final embeddings and attention weights
   */
  forwardMultiLevel(
    H_init: number[][],
    E_levels_init: Map<number, number[][]>,
    l0ToL1Matrix: number[][],
    interLevelMatrices: Map<number, number[][]>,
    levelParams: Map<number, LevelParams>,
    config: OrchestratorConfig,
  ): { result: MultiLevelEmbeddings; cache: MultiLevelForwardCache } {
    // Convert dense → sparse once (legacy backward compat for this method)
    const l0ToL1Conn = denseToSparse(l0ToL1Matrix);
    const interLevelConns = new Map<number, SparseConnectivity>();
    for (const [level, matrix] of interLevelMatrices) {
      interLevelConns.set(level, denseToSparse(matrix));
    }
    // Validate inputs
    if (E_levels_init.size === 0) {
      throw new Error("forwardMultiLevel requires at least one level of higher-level node embeddings");
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

    // Initialize cache (store references — cache is read-only after forward)
    const cache: MultiLevelForwardCache = {
      H_init,
      H_final: [],
      E_init: new Map(),
      E_final: new Map(),
      intermediateUpward: new Map(),
      intermediateDownward: new Map(),
      attentionUpward: new Map(),
      attentionDownward: new Map(),
    };

    // Copy initial embeddings for mutation safety in upward pass
    for (const [level, embs] of E_levels_init) {
      cache.E_init.set(level, embs);
      E.set(level, embs.map((row) => [...row]));
    }

    // Apply V→V co-occurrence enrichment (if configured)
    let H = this.applyV2VEnrichment(H_init);

    // ========================================================================
    // UPWARD PASS: V → E^0 → E^1 → ... → E^L_max
    // ========================================================================

    // In training mode, collect LevelIntermediates for backward pass
    const intermediateUpwardActivations = this.trainingMode
      ? new Map<number, LevelIntermediates>()
      : undefined;
    const intermediateDownwardActivations = this.trainingMode
      ? new Map<number, LevelIntermediates>()
      : undefined;

    for (let level = 0; level <= maxLevel; level++) {
      const params = levelParams.get(level);
      if (!params) {
        throw new Error(`Missing LevelParams for level ${level}`);
      }

      const edgesAtLevel = E.get(level);
      if (!edgesAtLevel || edgesAtLevel.length === 0) continue;

      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];

      // Per-head intermediates collectors (training mode only)
      const childProjPerHead: Float32Array[][] = [];
      const parentProjPerHead: Float32Array[][] = [];
      const scoresPerHead: number[][][] = [];
      const attentionPerHead: number[][][] = [];

      for (let head = 0; head < config.numHeads; head++) {
        if (level === 0) {
          // Phase: L0 nodes (V) → Level-0 L1+ nodes (E^0)
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          if (this.trainingMode) {
            const { embeddings, attention, cache: veCache } = this.vertexToEdgePhase.forwardWithCache(
              H,
              edgesAtLevel,
              l0ToL1Conn,
              phaseParams,
              { leakyReluSlope: config.leakyReluSlope },
            );

            headsE.push(embeddings);
            levelAttention.push(attention);

            // Collect intermediates: sourceProj=childProj, targetProj=parentProj
            childProjPerHead.push(veCache.sourceProj);
            parentProjPerHead.push(veCache.targetProj);
            attentionPerHead.push(attention); // use dense attention from PhaseResult
            // scores not directly stored in PhaseForwardCache, use empty placeholder
            scoresPerHead.push([]);
          } else {
            const { embeddings, attention } = this.vertexToEdgePhase.forward(
              H,
              edgesAtLevel,
              l0ToL1Conn,
              phaseParams,
              { leakyReluSlope: config.leakyReluSlope },
            );

            headsE.push(embeddings);
            levelAttention.push(attention);
          }
        } else {
          // Phase: Level-(k-1) → Level-k nodes
          const E_prev = E.get(level - 1);
          if (!E_prev) continue;

          const connectivity = interLevelConns.get(level);
          if (!connectivity) continue;

          const phase = edgeToEdgePhases.get(`up-${level}`)!;
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          if (this.trainingMode) {
            const { embeddings, attention, cache: eeCache } = phase.forwardWithCache(
              E_prev,
              edgesAtLevel,
              connectivity,
              phaseParams,
              { leakyReluSlope: config.leakyReluSlope },
            );

            headsE.push(embeddings);
            levelAttention.push(attention);

            // Collect intermediates: sourceProj=childProj, targetProj=parentProj
            childProjPerHead.push(eeCache.sourceProj);
            parentProjPerHead.push(eeCache.targetProj);
            attentionPerHead.push(attention); // use dense attention from PhaseResult
            scoresPerHead.push([]);
          } else {
            const { embeddings, attention } = phase.forward(
              E_prev,
              edgesAtLevel,
              connectivity,
              phaseParams,
              { leakyReluSlope: config.leakyReluSlope },
            );

            headsE.push(embeddings);
            levelAttention.push(attention);
          }
        }
      }

      // Concatenate heads
      if (headsE.length > 0) {
        const E_new = math.concatHeads(headsE);
        E.set(level, E_new);
        cache.intermediateUpward.set(level, E_new);
      }

      // Store LevelIntermediates for training backward pass
      if (intermediateUpwardActivations && childProjPerHead.length > 0) {
        intermediateUpwardActivations.set(level, {
          childProj: childProjPerHead,
          parentProj: parentProjPerHead,
          scores: scoresPerHead,
          attention: attentionPerHead,
        });
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

      const edgesAtLevel = E.get(level);
      const edgesAtParentLevel = E.get(level + 1);

      if (!edgesAtLevel || edgesAtLevel.length === 0) continue;
      if (!edgesAtParentLevel || edgesAtParentLevel.length === 0) continue;

      // Save pre-downward embeddings for residual connection
      const edgesAtLevelPreDownward = edgesAtLevel.map((row) => [...row]);

      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];

      // Per-head intermediates collectors (training mode only)
      const childProjPerHead: Float32Array[][] = [];
      const parentProjPerHead: Float32Array[][] = [];
      const scoresPerHead: number[][][] = [];
      const attentionPerHead: number[][][] = [];

      // Get reverse connectivity (parent → child)
      const forwardConn = interLevelConns.get(level + 1);
      if (!forwardConn) continue;

      // Transpose the connectivity for downward pass
      const reverseConnectivity = transposeSparse(forwardConn);

      const phase = edgeToEdgePhases.get(`down-${level + 1}`)!;

      for (let head = 0; head < config.numHeads; head++) {
        // Downward phase: Level-(k+1) → Level-k
        const phaseParams: PhaseParameters = {
          W_source: params.W_parent[head], // Parents are source in downward
          W_target: params.W_child[head], // Children are target in downward
          a_attention: params.a_downward[head],
        };

        if (this.trainingMode) {
          const { embeddings: propagated, attention, cache: eeCache } = phase.forwardWithCache(
            edgesAtParentLevel,
            edgesAtLevel,
            reverseConnectivity,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(propagated);
          levelAttention.push(attention);

          // In downward: source=parent, target=child
          parentProjPerHead.push(eeCache.sourceProj);
          childProjPerHead.push(eeCache.targetProj);
          attentionPerHead.push(attention); // use dense attention from PhaseResult
          scoresPerHead.push([]);
        } else {
          const { embeddings: propagated, attention } = phase.forward(
            edgesAtParentLevel,
            edgesAtLevel,
            reverseConnectivity,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(propagated);
          levelAttention.push(attention);
        }
      }

      // Concatenate heads first
      if (headsE.length > 0) {
        const E_concat = math.concatHeads(headsE);

        // Apply residual connection with configurable weight
        // E_new = (1-α)*propagated + α*original, where α = downwardResidual
        const alpha = config.downwardResidual ?? 0;
        const E_new = edgesAtLevelPreDownward.map((row, i) =>
          row.map((val, j) => {
            const propagated = E_concat[i]?.[j] ?? 0;
            return (1 - alpha) * propagated + alpha * val;
          })
        );

        E.set(level, E_new);
        cache.intermediateDownward.set(level, E_new);
      }

      // Store LevelIntermediates for training backward pass
      if (intermediateDownwardActivations && childProjPerHead.length > 0) {
        intermediateDownwardActivations.set(level, {
          childProj: childProjPerHead,
          parentProj: parentProjPerHead,
          scores: scoresPerHead,
          attention: attentionPerHead,
        });
      }

      attentionDownward.set(level, levelAttention);
      cache.attentionDownward.set(level, levelAttention);
    }

    // Final phase: Level-0 → L0 nodes (downward)
    const E_level0 = E.get(0);
    if (E_level0 && E_level0.length > 0) {
      const params = levelParams.get(0);
      if (params) {
        // Save pre-downward L0 node embeddings for residual connection
        const H_preDownward = H.map((row) => [...row]);

        // EdgeToVertexPhase expects [L0][L1] format (same as upward pass)
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
            l0ToL1Conn,
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

    // Store references to final embeddings (cache is read-only after forward)
    cache.H_final = H;
    for (const [level, embs] of E) {
      cache.E_final.set(level, embs);
    }

    const result: MultiLevelEmbeddings = {
      H,
      E,
      attentionUpward,
      attentionDownward,
    };

    // In training mode, return an ExtendedMultiLevelForwardCache with LevelIntermediates
    if (this.trainingMode && intermediateUpwardActivations && intermediateDownwardActivations) {
      const extendedCache: ExtendedMultiLevelForwardCache = {
        ...cache,
        intermediateUpwardActivations,
        intermediateDownwardActivations,
      };
      return { result, cache: extendedCache };
    }

    return { result, cache };
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
    l0ToL1Conn: SparseConnectivity,
    interLevelConns: Map<number, SparseConnectivity>,
    levelParams: Map<number, LevelParams>,
    config: OrchestratorConfig,
    v2vParams?: V2VParams,
  ): { result: MultiLevelEmbeddings; cache: MultiLevelBackwardCache } {
    if (E_levels_init.size === 0) {
      throw new Error("forwardMultiLevelWithCache requires at least one level");
    }

    const maxLevel = Math.max(...Array.from(E_levels_init.keys()));

    // Initialize extended cache
    const veCaches = new Map<number, PhaseForwardCache[]>();
    const eeUpwardCaches = new Map<number, PhaseForwardCache[]>();
    const eeDownwardCaches = new Map<number, PhaseForwardCache[]>();
    const evCaches: PhaseForwardCache[] = [];

    // Initialize result structures
    const E = new Map<number, number[][]>();
    const attentionUpward = new Map<number, number[][][]>();
    const attentionDownward = new Map<number, number[][][]>();

    // Use initial embeddings (copy only for upward mutation safety)
    for (const [level, embs] of E_levels_init) {
      E.set(level, embs.map((row) => [...row]));
    }

    // Apply V→V enrichment (with cache for backward pass)
    let v2vCache: V2VForwardCache | undefined;
    let H: number[][];

    if (v2vParams && this.vertexToVertexPhase && this.cooccurrenceData && this.cooccurrenceData.length > 0) {
      // Use trainable V2V with cache (creates new embeddings)
      const v2vResult = this.vertexToVertexPhase.forwardWithCache(H_init, this.cooccurrenceData, v2vParams);
      H = v2vResult.embeddings;
      v2vCache = v2vResult.cache;
    } else if (v2vParams) {
      // v2vParams provided but V2V not configured - this is a bug
      throw new Error(
        "[MultiLevelOrchestrator] v2vParams provided but V2V phase not configured. " +
        "Either call setCooccurrenceData() or don't pass v2vParams."
      );
    } else {
      // No V2V requested - pass through unchanged
      H = H_init;
    }

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

      const edgesAtLevel = E.get(level);
      if (!edgesAtLevel || edgesAtLevel.length === 0) continue;

      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];
      const levelVECaches: PhaseForwardCache[] = [];
      const levelEECaches: PhaseForwardCache[] = [];

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
            edgesAtLevel,
            l0ToL1Conn,
            phaseParams,
            { leakyReluSlope: config.leakyReluSlope },
          );

          headsE.push(embeddings);
          levelAttention.push(attention);
          levelVECaches.push(veCache);
        } else {
          // E^(k-1)→E^k with cache
          const E_prev = E.get(level - 1);
          const connectivity = interLevelConns.get(level);
          if (!E_prev || !connectivity) continue;

          const phase = edgeToEdgePhases.get(`up-${level}`)!;
          const phaseParams: PhaseParameters = {
            W_source: params.W_child[head],
            W_target: params.W_parent[head],
            a_attention: params.a_upward[head],
          };

          const { embeddings, attention, cache: eeCache } = phase.forwardWithCache(
            E_prev,
            edgesAtLevel,
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

      const edgesAtLevel = E.get(level);
      const edgesAtParentLevel = E.get(level + 1);
      if (!edgesAtLevel || !edgesAtParentLevel) continue;

      const edgesAtLevelPreDownward = edgesAtLevel.map((row) => [...row]);
      const headsE: number[][][] = [];
      const levelAttention: number[][][] = [];
      const levelEECaches: PhaseForwardCache[] = [];

      const forwardConn2 = interLevelConns.get(level + 1);
      if (!forwardConn2) continue;
      const reverseConnectivity = transposeSparse(forwardConn2);

      const phase = edgeToEdgePhases.get(`down-${level + 1}`)!;

      for (let head = 0; head < config.numHeads; head++) {
        const phaseParams: PhaseParameters = {
          W_source: params.W_parent[head],
          W_target: params.W_child[head],
          a_attention: params.a_downward[head],
        };

        const { embeddings, attention, cache: eeCache } = phase.forwardWithCache(
          edgesAtParentLevel,
          edgesAtLevel,
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
        const E_new = edgesAtLevelPreDownward.map((row, i) =>
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
            l0ToL1Conn,
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

    // Store references in cache — backward pass only reads from these, never mutates.
    const cache: MultiLevelBackwardCache = {
      H_init,
      H_final: H,
      E_init: E_levels_init,
      E_final: E,
      intermediateUpward: new Map(),
      intermediateDownward: new Map(),
      attentionUpward,
      attentionDownward,
      veCaches,
      eeUpwardCaches,
      eeDownwardCaches,
      evCaches,
      v2vCache,
      l0ToL1Conn,
      interLevelConns,
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
   * @param dE_final - Gradient on final L1+ node embeddings (from K-head scoring)
   * @param dH_final - Gradient on final L0 node embeddings (optional, usually zero)
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
        const headDim = cache.evCaches[0]?.sourceProj[0]?.length ?? 0;
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
      const headDim = eeCaches[0]?.sourceProj[0]?.length ?? 0;
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

        const headDim = veCaches[0]?.sourceProj[0]?.length ?? 0;
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

        const headDim = eeCaches[0]?.sourceProj[0]?.length ?? 0;
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
