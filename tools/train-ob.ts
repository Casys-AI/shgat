#!/usr/bin/env -S DENO_V8_FLAGS=--max-old-space-size=10240 deno run --unstable-ffi --allow-ffi --allow-read --allow-write --allow-env
/**
 * SHGAT-TF OB Training — Manual backward + OpenBLAS FFI
 *
 * Replaces TF.js autograd training (11-15GB RAM) with manual backward
 * passes and OpenBLAS FFI acceleration (target: 1-3GB RAM).
 *
 * Full gradient chain:
 *   InfoNCE → dLogits → K-head backward (dW_q, dW_k, dNodeEmbedding)
 *   → W_intent backward (dW_intent)
 *   → MP backward (dW_child, dW_parent, da_upward, da_downward)
 *
 * Decision log:
 *   - W_q/W_k are SEPARATE for training (not shared like inference init)
 *   - preserveDim=true → embeddingDim=hiddenDim=1024 → no dimension mismatch
 *   - MP backward via orchestrator.backwardMultiLevel() — handles dH (L0 nodes)
 *     through E→V backward AND dE (L1+ nodes) through E→E/V→E backward
 *   - Both InfoNCE and KL paths propagate gradients into MP weights
 *   - Uses forwardMultiLevelWithCache (not forwardMultiLevel) for per-phase caches
 *   - Adam optimizer with per-epoch LR via .lr setter
 *   - MP weights use mpLrScale for smaller updates (noisy subgraph gradients)
 *
 * Data: Parquet files (default) or msgpack.gz (--msgpack) from tools/export-dataset.ts
 *
 * Usage:
 *   cd lib/shgat-tf
 *   deno run --unstable-ffi --allow-ffi --allow-read --allow-write --allow-env \
 *     tools/train-ob.ts --epochs 15 --lr 0.005 --kl --seed 42
 *
 * @module shgat-tf/tools/train-ob
 */

import { dirname, resolve, fromFileUrl } from "https://deno.land/std@0.224.0/path/mod.ts";
import { decode as msgpackDecode } from "@msgpack/msgpack";
import pako from "pako";
import { loadFullDataset } from "./load-parquet.ts";

import { ensureBLAS } from "../src/utils/blas-ffi.ts";
import * as math from "../src/utils/math.ts";
import {
  batchContrastiveForward,
  batchContrastiveBackward,
} from "../src/training/batch-contrastive-loss.ts";
import { AdamOptimizer } from "../src/training/adam-optimizer.ts";
import {
  backpropWIntent,
  initMultiLevelKHeadGradients,
  resetMultiLevelKHeadGradients,
} from "../src/training/multi-level-trainer-khead.ts";
import type { MultiLevelKHeadGradientAccumulators } from "../src/training/multi-level-trainer-khead.ts";
import { MultiLevelOrchestrator } from "../src/message-passing/multi-level-orchestrator.ts";
import type { MultiLevelGradients } from "../src/message-passing/multi-level-orchestrator.ts";
import type { SparseConnectivity } from "../src/message-passing/phase-interface.ts";
import type { LevelParams, SHGATConfig } from "../src/core/types.ts";
import type { HeadParams } from "../src/initialization/parameters.ts";

// ==========================================================================
// BLAS — FAIL-FAST (must succeed or the script is useless)
// ==========================================================================

ensureBLAS();
// Also init BLAS in math.ts module (separate lazy-load state from blas-ffi.ts)
await math.initBlasAcceleration();
console.log("[BLAS] OpenBLAS FFI loaded (blas-ffi + math module).");

// ==========================================================================
// Dataset type (matches export-dataset.ts output)
// ==========================================================================

interface ExportedNode {
  id: string;
  embedding: number[];
  children: string[];
  level: number;
}

interface ProdExample {
  intentEmbedding: number[];
  contextToolIds: string[];
  targetToolId: string;
  isTerminal: number;
  _traceId: string;
}

interface N8nExample {
  intentEmbedding: number[];
  contextToolIds: string[];
  targetToolId: string;
  isTerminal: number;
  softTargetSparse: [number, number][];
}

interface ExportedDataset {
  nodes: ExportedNode[];
  leafIds: string[];
  embeddingDim: number;
  workflowToolLists: string[][];
  prodTrain: ProdExample[];
  prodTest: ProdExample[];
  n8nTrain: N8nExample[];
  n8nEval: N8nExample[];
}

// ==========================================================================
// CLI args
// ==========================================================================

const cliArgs = Deno.args;

function getArg(name: string, def: string): string {
  const idx = cliArgs.indexOf(`--${name}`);
  if (idx === -1) return def;
  const next = cliArgs[idx + 1];
  return next && !next.startsWith("--") ? next : def;
}
function boolArg(name: string, def: boolean): boolean {
  if (cliArgs.includes(`--no-${name}`)) return false;
  if (cliArgs.includes(`--${name}`)) return true;
  return def;
}

if (cliArgs.includes("--help")) {
  console.log(`
SHGAT-TF OB Training — Manual backward + OpenBLAS FFI

Options:
  --epochs <n>         Training epochs (default: 10)
  --batch-size <n>     Batch size (default: 32)
  --lr <n>             Peak learning rate (default: 0.005)
  --lr-warmup <n>      LR warmup epochs (default: 3)
  --temperature <n>    InfoNCE temperature start (default: 0.10)
  --seed <n>           Random seed (default: 42)
  --kl / --no-kl       KL divergence on n8n soft targets (default: ON)
  --kl-warmup <n>      KL warmup epochs (default: 3)
  --kl-weight <n>      KL loss weight at plateau (default: 0.2)
  --mp-lr-scale <n>    LR scale for MP weights (default: 1.0)
  --eval-every <n>     Run full eval every N epochs (default: 2)
  --kl-subsample <n>   Max n8n examples per epoch (default: 2000, 0=all)
  --kl-batch-size <n>  KL batch size (default: 128, larger = fewer backward passes)
  --kl-accum <n>       KL gradient accumulation steps (default: 4, 1=no accum)
  --kl-update-khead    Let KL update W_q/W_k (default: isolated, KL only updates W_intent+MP)
  --msgpack            Use msgpack.gz loader instead of Parquet (default: Parquet)
  --data-path <path>   Path to msgpack.gz dataset (only with --msgpack)
  --help               Show this help
`);
  Deno.exit(0);
}

const EPOCHS = parseInt(getArg("epochs", "10"), 10);
const BATCH_SIZE = parseInt(getArg("batch-size", "32"), 10);
const LEARNING_RATE = parseFloat(getArg("lr", "0.005"));
const LR_WARMUP = parseInt(getArg("lr-warmup", "3"), 10);
const TAU_START = parseFloat(getArg("temperature", "0.10"));
const TAU_END = 0.06;
const SEED = parseInt(getArg("seed", "42"), 10);
const USE_KL = boolArg("kl", true);
const KL_WARMUP = parseInt(getArg("kl-warmup", "3"), 10);
const KL_WEIGHT_PLATEAU = parseFloat(getArg("kl-weight", "0.2"));
const MP_LR_SCALE = parseFloat(getArg("mp-lr-scale", "1.0"));
const EVAL_EVERY = Math.max(1, parseInt(getArg("eval-every", "2"), 10));
const KL_SUBSAMPLE = parseInt(getArg("kl-subsample", "0"), 10);
const KL_BATCH_SIZE = parseInt(getArg("kl-batch-size", "30000"), 10);
const KL_ACCUM_STEPS = Math.max(1, parseInt(getArg("kl-accum", "1"), 10));
const KL_ISOLATE_KHEAD = Deno.args.includes("--kl-isolate-khead"); // default: false (KL updates W_q/W_k)

// ==========================================================================
// Seeded PRNG (mulberry32 — inlined from parameters.ts to avoid TF.js import)
// ==========================================================================

let rngState = SEED | 0;

function random(): number {
  rngState |= 0;
  rngState = (rngState + 0x6D2B79F5) | 0;
  let t = Math.imul(rngState ^ (rngState >>> 15), 1 | rngState);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}

function gaussianRandom(): number {
  const u1 = random() || 1e-10;
  const u2 = random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function shuffleInPlace<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ==========================================================================
// Parameter initialization (inlined from parameters.ts — avoids TF.js import)
// ==========================================================================

function initMatrix(rows: number, cols: number): number[][] {
  const scale = Math.sqrt(2.0 / (rows + cols));
  return Array.from({ length: rows },
    () => Array.from({ length: cols }, () => (random() - 0.5) * 2 * scale));
}

function initOrthogonalMatrix(rows: number, cols: number): number[][] {
  const M: number[][] = Array.from({ length: rows },
    () => Array.from({ length: cols }, () => gaussianRandom()));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < i; j++) {
      let dot = 0;
      for (let k = 0; k < cols; k++) dot += M[i][k] * M[j][k];
      for (let k = 0; k < cols; k++) M[i][k] -= dot * M[j][k];
    }
    let norm = 0;
    for (let k = 0; k < cols; k++) norm += M[i][k] * M[i][k];
    norm = Math.sqrt(norm);
    if (norm > 1e-10) {
      for (let k = 0; k < cols; k++) M[i][k] /= norm;
    }
  }
  const scale = Math.sqrt(cols / rows);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < cols; k++) M[i][k] *= scale;
  }
  return M;
}

function initVector(size: number): number[] {
  const scale = Math.sqrt(1.0 / size);
  return Array.from({ length: size }, () => (random() - 0.5) * 2 * scale);
}

function initTensor3DIdentityLike(numHeads: number, headDim: number, inputDim: number): number[][][] {
  const noiseScale = 0.01;
  return Array.from({ length: numHeads }, (_, head) =>
    Array.from({ length: headDim }, (_, i) =>
      Array.from({ length: inputDim }, (_, j) => {
        const targetJ = head * headDim + i;
        return j === targetJ ? 1.0 : (random() - 0.5) * noiseScale;
      })));
}

// ==========================================================================
// LR / KL / Temperature scheduling (from train-from-bench.ts)
// ==========================================================================

function scheduleLR(epoch: number, totalEpochs: number, lrPeak: number, warmupEpochs: number): number {
  const lrMin = lrPeak * 0.01;
  if (epoch < warmupEpochs) {
    const progress = (epoch + 1) / Math.max(warmupEpochs, 1);
    return lrMin + (lrPeak - lrMin) * progress;
  }
  const decayEpochs = totalEpochs - warmupEpochs;
  const decayProgress = Math.min((epoch - warmupEpochs) / Math.max(decayEpochs - 1, 1), 1.0);
  return lrMin + (lrPeak - lrMin) * 0.5 * (1 + Math.cos(Math.PI * decayProgress));
}

function scheduleKLWeight(epoch: number, warmupEpochs: number, plateau: number): number {
  // Always provide minimum signal (10% of plateau) from epoch 0.
  // Old design: 0 for warmupEpochs then ramp. Problem: MP gradients starved
  // during warmup because KL is the main source of dense dH gradients.
  const minWeight = plateau * 0.1;
  const rampEnd = warmupEpochs * 2;
  if (epoch >= rampEnd) return plateau;
  const progress = epoch / Math.max(rampEnd, 1);
  return minWeight + (plateau - minWeight) * progress;
}

function scheduleTemperature(epoch: number, totalEpochs: number, start: number, end: number): number {
  const progress = epoch / Math.max(totalEpochs - 1, 1);
  return end + (start - end) * 0.5 * (1 + Math.cos(Math.PI * progress));
}

// ==========================================================================
// Gradient norm helpers (logging only — no training logic changes)
// ==========================================================================

/** L2 norm of a 2D matrix (flattened) */
function matrixL2Norm(m: number[][]): number {
  let sum = 0;
  for (let i = 0; i < m.length; i++) {
    const row = m[i];
    for (let j = 0; j < row.length; j++) {
      sum += row[j] * row[j];
    }
  }
  return Math.sqrt(sum);
}

/** L2 norm of a 3D tensor (flattened) */
function tensor3DL2Norm(t: number[][][]): number {
  let sum = 0;
  for (let i = 0; i < t.length; i++) {
    for (let j = 0; j < t[i].length; j++) {
      for (let k = 0; k < t[i][j].length; k++) {
        sum += t[i][j][k] * t[i][j][k];
      }
    }
  }
  return Math.sqrt(sum);
}

/**
 * Compute gradient norms for K-head and W_intent accumulators.
 *
 * |dWq| = average L2 norm across heads, |dWk| = same, |dWi| = L2 of dW_intent.
 * total = sqrt(sum of squares of all gradient elements).
 */
function computeGradNorms(grads: MultiLevelKHeadGradientAccumulators): {
  wq: number; wk: number; wIntent: number; total: number;
} {
  const numHeads = grads.khead.dW_q.length;
  let wqSum = 0, wkSum = 0;
  let totalSqSum = 0;

  for (let h = 0; h < numHeads; h++) {
    const nq = matrixL2Norm(grads.khead.dW_q[h]);
    const nk = matrixL2Norm(grads.khead.dW_k[h]);
    wqSum += nq;
    wkSum += nk;
    totalSqSum += nq * nq + nk * nk;
  }

  const wIntentNorm = matrixL2Norm(grads.dW_intent);
  totalSqSum += wIntentNorm * wIntentNorm;

  // Include MP level gradients in total norm
  for (const [, lg] of grads.levelGradients) {
    const wcNorm = tensor3DL2Norm(lg.dW_child);
    const wpNorm = tensor3DL2Norm(lg.dW_parent);
    const auNorm = matrixL2Norm(lg.da_upward);
    const adNorm = matrixL2Norm(lg.da_downward);
    totalSqSum += wcNorm * wcNorm + wpNorm * wpNorm + auNorm * auNorm + adNorm * adNorm;
  }

  return {
    wq: wqSum / Math.max(numHeads, 1),
    wk: wkSum / Math.max(numHeads, 1),
    wIntent: wIntentNorm,
    total: Math.sqrt(totalSqSum),
  };
}

/** Compute gradient norms from epoch-level MP backward output */
function computeMPGradNorms(mpGrads: MultiLevelGradients): { wChild: number; wParent: number; atten: number } {
  let wChildSq = 0, wParentSq = 0, attenSq = 0;
  for (const [, lg] of mpGrads.levelGrads) {
    for (let h = 0; h < lg.dW_child.length; h++) {
      for (const row of lg.dW_child[h]) for (const v of row) wChildSq += v * v;
      for (const row of lg.dW_parent[h]) for (const v of row) wParentSq += v * v;
      for (const v of lg.da_upward[h]) attenSq += v * v;
      for (const v of lg.da_downward[h]) attenSq += v * v;
    }
  }
  return { wChild: Math.sqrt(wChildSq), wParent: Math.sqrt(wParentSq), atten: Math.sqrt(attenSq) };
}

// ==========================================================================
// Graph structure building (pure JS, no TF.js)
// ==========================================================================

interface PureGraphStructure {
  l0ToL1Conn: SparseConnectivity;
  interLevelConns: Map<number, SparseConnectivity>;
  l0Ids: string[];
  l0IdxMap: Map<string, number>;
  nodeIdsByLevel: Map<number, string[]>;
  maxLevel: number;
  E_levels_init: Map<number, number[][]>;
  H_init: number[][];
  /** l0 tool index → ancestor indices at each orchestrator level.
   *  ancestors[l0Idx] = Map<orchLevel, idx[]>
   *  Recursive: works for any number of hierarchy levels. */
  l0Ancestors: Map<number, number[]>[];
}

/**
 * Build graph structure from dataset nodes.
 *
 * Level mapping:
 *   dataset level 0  = L0 nodes (leaves)
 *   dataset level 1  = orchestrator level 0 (first L1+ parent level)
 *   dataset level k+1 = orchestrator level k
 */
function buildGraphStructure(nodes: ExportedNode[], leafIds: string[]): PureGraphStructure {
  const nodeMap = new Map<string, ExportedNode>();
  for (const n of nodes) nodeMap.set(n.id, n);

  const l0Ids = leafIds;
  const l0IdxMap = new Map<string, number>();
  for (let i = 0; i < l0Ids.length; i++) l0IdxMap.set(l0Ids[i], i);

  // Infer levels from children (bottom-up BFS)
  // Leaves (children=[]) → level 0, else level = max(child levels) + 1
  const levelMap = new Map<string, number>();
  for (const id of leafIds) levelMap.set(id, 0);

  let changed = true;
  while (changed) {
    changed = false;
    for (const n of nodes) {
      if (levelMap.has(n.id)) continue;
      if (!n.children || n.children.length === 0) {
        levelMap.set(n.id, 0);
        changed = true;
        continue;
      }
      const childLevels = n.children.map(c => levelMap.get(c));
      if (childLevels.every(l => l !== undefined)) {
        levelMap.set(n.id, Math.max(...(childLevels as number[])) + 1);
        changed = true;
      }
    }
  }

  // Group non-leaf nodes by orchestrator level (dataset level - 1)
  const nodeIdsByLevel = new Map<number, string[]>();
  let maxLevel = 0;
  for (const n of nodes) {
    const dsLevel = levelMap.get(n.id) ?? 0;
    if (dsLevel === 0) continue; // L0 leaf nodes
    const orchLevel = dsLevel - 1;
    if (!nodeIdsByLevel.has(orchLevel)) nodeIdsByLevel.set(orchLevel, []);
    nodeIdsByLevel.get(orchLevel)!.push(n.id);
    if (orchLevel > maxLevel) maxLevel = orchLevel;
  }

  // Level index maps per level
  const levelIdxMaps = new Map<number, Map<string, number>>();
  for (const [level, ids] of nodeIdsByLevel) {
    const map = new Map<string, number>();
    for (let i = 0; i < ids.length; i++) map.set(ids[i], i);
    levelIdxMaps.set(level, map);
  }

  // Build sparse L0→L1 connectivity (orch level 0)
  const nodesL1 = nodeIdsByLevel.get(0) ?? [];
  const l0ToL1Src = new Map<number, number[]>(); // L0 → L1
  const l0ToL1Tgt = new Map<number, number[]>(); // L1 → L0

  for (let c = 0; c < nodesL1.length; c++) {
    const parentNode = nodeMap.get(nodesL1[c]);
    if (!parentNode) continue;
    for (const childId of parentNode.children) {
      const tIdx = l0IdxMap.get(childId);
      if (tIdx !== undefined) {
        if (!l0ToL1Src.has(tIdx)) l0ToL1Src.set(tIdx, []);
        l0ToL1Src.get(tIdx)!.push(c);
        if (!l0ToL1Tgt.has(c)) l0ToL1Tgt.set(c, []);
        l0ToL1Tgt.get(c)!.push(tIdx);
      }
    }
  }

  const l0ToL1Conn: SparseConnectivity = {
    sourceToTargets: l0ToL1Src,
    targetToSources: l0ToL1Tgt,
    numSources: l0Ids.length,
    numTargets: nodesL1.length,
  };

  // Build sparse inter-level connectivity per parent level
  const interLevelConns = new Map<number, SparseConnectivity>();
  for (let parentLevel = 1; parentLevel <= maxLevel; parentLevel++) {
    const children = nodeIdsByLevel.get(parentLevel - 1) ?? [];
    const parents = nodeIdsByLevel.get(parentLevel) ?? [];
    const childIdx = levelIdxMaps.get(parentLevel - 1) ?? new Map();

    const src = new Map<number, number[]>(); // child → parents
    const tgt = new Map<number, number[]>(); // parent → children

    for (let p = 0; p < parents.length; p++) {
      const parentNode = nodeMap.get(parents[p]);
      if (!parentNode) continue;
      for (const childId of parentNode.children) {
        const cIdx = childIdx.get(childId);
        if (cIdx !== undefined) {
          if (!src.has(cIdx)) src.set(cIdx, []);
          src.get(cIdx)!.push(p);
          if (!tgt.has(p)) tgt.set(p, []);
          tgt.get(p)!.push(cIdx);
        }
      }
    }

    interLevelConns.set(parentLevel, {
      sourceToTargets: src,
      targetToSources: tgt,
      numSources: children.length,
      numTargets: parents.length,
    });
  }

  // Build embedding matrices
  const H_init: number[][] = l0Ids.map(id => [...(nodeMap.get(id)?.embedding ?? [])]);
  const E_levels_init = new Map<number, number[][]>();
  for (const [level, ids] of nodeIdsByLevel) {
    E_levels_init.set(level, ids.map(id => [...(nodeMap.get(id)?.embedding ?? [])]));
  }

  // Build l0Ancestors: for each L0 tool, find ancestor indices at every level.
  // Recursive: walks up the hierarchy from L0 → L1 (orch 0) → L2 (orch 1) → ...
  // Uses the connectivity graph (sourceToTargets = child→parent mappings).
  const l0Ancestors: Map<number, number[]>[] = new Array(l0Ids.length);
  for (let i = 0; i < l0Ids.length; i++) {
    const ancestors = new Map<number, number[]>();
    // Level 0 (orch): L0 tool → L1 caps via l0ToL1Conn.sourceToTargets
    const l1Parents = l0ToL1Conn.sourceToTargets.get(i) ?? [];
    if (l1Parents.length > 0) ancestors.set(0, [...l1Parents]);
    // Higher levels: walk up via interLevelConns
    let currentParents = l1Parents;
    for (let orchLevel = 1; orchLevel <= maxLevel; orchLevel++) {
      const conn = interLevelConns.get(orchLevel);
      if (!conn) break;
      const nextParents = new Set<number>();
      for (const pIdx of currentParents) {
        const grandParents = conn.sourceToTargets.get(pIdx) ?? [];
        for (const gp of grandParents) nextParents.add(gp);
      }
      const nextArr = [...nextParents];
      if (nextArr.length > 0) ancestors.set(orchLevel, nextArr);
      currentParents = nextArr;
    }
    l0Ancestors[i] = ancestors;
  }

  // Debug: verify ancestor coverage
  let withAncestors = 0;
  const toolsWithAncestorIdxs = new Set<number>();
  for (let i = 0; i < l0Ids.length; i++) {
    if (l0Ancestors[i] && l0Ancestors[i].size > 0) {
      withAncestors++;
      toolsWithAncestorIdxs.add(i);
    }
  }
  console.log(`  l0Ancestors: ${withAncestors}/${l0Ids.length} tools have ≥1 ancestor (l0ToL1Src.size=${l0ToL1Src.size})`);

  return { l0ToL1Conn, interLevelConns, l0Ids, l0IdxMap, nodeIdsByLevel, maxLevel, E_levels_init, H_init, l0Ancestors };
}

// ==========================================================================
// MAIN
// ==========================================================================

const scriptDir = dirname(fromFileUrl(import.meta.url));
const GRU_DATA_DIR = resolve(scriptDir, "../../gru/data");

console.log("=== SHGAT-TF OB Training (Manual Backward + OpenBLAS) ===");
console.log(`    Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}, LR: ${LEARNING_RATE} (warmup: ${LR_WARMUP}ep)`);
console.log(`    \u03C4: ${TAU_START}\u2192${TAU_END}`);
console.log(`    KL: ${USE_KL}, KL warmup: ${KL_WARMUP}ep, KL weight: ${KL_WEIGHT_PLATEAU}`);
console.log(`    MP LR scale: ${MP_LR_SCALE}, Seed: ${SEED}`);
console.log(`    KL subsample: ${KL_SUBSAMPLE > 0 ? KL_SUBSAMPLE : 'all'}, KL batch: ${KL_BATCH_SIZE}, KL accum: ${KL_ACCUM_STEPS}, KL isolate K-head: ${KL_ISOLATE_KHEAD}`);
console.log(`    Eval every: ${EVAL_EVERY}\n`);

// ---- Load dataset ----
// Default: Parquet (lower peak memory, lazy per-table loading).
// Fallback: --msgpack flag → monolithic msgpack.gz (1.2GB compressed, ~5GB peak).
const useMsgpack = cliArgs.includes("--msgpack");
let ds: ExportedDataset;
if (useMsgpack) {
  const dataPath = getArg("data-path", resolve(GRU_DATA_DIR, "bench-dataset-export.msgpack.gz"));
  console.log(`[Data] Loading msgpack from ${dataPath}...`);
  // Stage 1: read compressed (1.2GB)
  let compressed: Uint8Array | null = Deno.readFileSync(dataPath);
  // Stage 2: decompress (~3-5GB raw), then drop compressed
  let raw: Uint8Array | null = pako.ungzip(compressed);
  compressed = null; // free 1.2GB
  // Stage 3: decode msgpack → JS objects, then drop raw
  ds = msgpackDecode(raw) as ExportedDataset;
  raw = null; // free 3-5GB
} else {
  console.log(`[Data] Loading Parquet from ${GRU_DATA_DIR}...`);
  ds = await loadFullDataset(GRU_DATA_DIR);
}

console.log(`  Nodes: ${ds.nodes.length} (${ds.leafIds.length} leaves), EmbDim: ${ds.embeddingDim}`);
console.log(`  Prod: ${ds.prodTrain.length} train / ${ds.prodTest.length} test`);
console.log(`  N8n: ${ds.n8nTrain.length} train / ${ds.n8nEval.length} eval`);

// ---- Build graph ----
console.log("\n[Graph] Building sparse connectivity...");
const graph = buildGraphStructure(ds.nodes, ds.leafIds);
const l0IdxMap = graph.l0IdxMap;
console.log(`  L0 (tools):   ${graph.l0Ids.length} leaves`);
{
  let totalEdges = 0;
  for (const [, targets] of graph.l0ToL1Conn.sourceToTargets) totalEdges += targets.length;
  const numL1 = graph.l0ToL1Conn.numTargets;
  console.log(`  L1 (caps):    ${numL1} nodes, ${totalEdges} edges L0→L1 (${(totalEdges / (graph.l0ToL1Conn.numSources * numL1) * 100).toFixed(1)}% fill)`);
  for (const [orchLevel, conn] of graph.interLevelConns) {
    let edges = 0;
    for (const [, targets] of conn.sourceToTargets) edges += targets.length;
    const dsLevel = orchLevel + 1; // orchLevel 1 = dataset level 2
    console.log(`  L${dsLevel} (super):  ${conn.numTargets} nodes, ${edges} edges L${dsLevel - 1}→L${dsLevel}`);
  }
  console.log(`  Max level: ${graph.maxLevel + 1} (${graph.maxLevel + 2} tiers total: tools → caps${graph.maxLevel > 0 ? " → super-caps" : ""})`);
}

// ---- Free heavy data no longer needed ----
// ds.nodes has 8884 entries × 1024D embeddings (~144MB JS overhead).
// Embeddings are now in graph.H_init / E_levels_init, nodes no longer needed.
(ds as { nodes: unknown }).nodes = [];

// Subsample n8n train in-place: KL path only uses KL_SUBSAMPLE per epoch,
// no need to keep all 30K in memory (~500MB). Keep 2× subsample for shuffle variance.
if (KL_SUBSAMPLE > 0 && ds.n8nTrain.length > KL_SUBSAMPLE * 2) {
  const kept = KL_SUBSAMPLE * 2;
  console.log(`  [Mem] Trimming n8nTrain: ${ds.n8nTrain.length} → ${kept} (2× KL subsample)`);
  shuffleInPlace(ds.n8nTrain);
  ds.n8nTrain.length = kept;
}

{
  const rss = (Deno.memoryUsage().rss / 1024 / 1024).toFixed(0);
  console.log(`  [Mem] Post-cleanup RSS: ${rss}MB`);
}

// ---- Config ----
const NUM_HEADS = 16;
const HEAD_DIM = Math.floor(ds.embeddingDim / NUM_HEADS); // 64 for 1024
const config: SHGATConfig = {
  numHeads: NUM_HEADS,
  hiddenDim: ds.embeddingDim,
  headDim: HEAD_DIM,
  embeddingDim: ds.embeddingDim,
  numLayers: 1,
  mlpHiddenDim: 128,
  learningRate: LEARNING_RATE,
  batchSize: BATCH_SIZE,
  maxContextLength: 10,
  maxBufferSize: 5000,
  minTracesForTraining: 10,
  dropout: 0,
  leakyReluSlope: 0.2,
  depthDecay: 0.8,
  preserveDim: true,
  l2Lambda: 0,
};

// ---- Init parameters ----
console.log("\n[Init] Parameters...");

// K-head: W_q and W_k SEPARATE for training (decision: not shared like inference)
const headParams: HeadParams[] = [];
for (let h = 0; h < NUM_HEADS; h++) {
  headParams.push({
    W_q: initOrthogonalMatrix(HEAD_DIM, ds.embeddingDim),
    W_k: initOrthogonalMatrix(HEAD_DIM, ds.embeddingDim),
    W_v: initMatrix(HEAD_DIM, ds.embeddingDim),
    a: initVector(2 * HEAD_DIM),
  });
}

const W_intent: number[][] = initMatrix(ds.embeddingDim, ds.embeddingDim);

// MP level params: identity-like init (preserveDim)
const levelParams = new Map<number, LevelParams>();
for (let level = 0; level <= graph.maxLevel; level++) {
  levelParams.set(level, {
    W_child: initTensor3DIdentityLike(NUM_HEADS, HEAD_DIM, ds.embeddingDim),
    W_parent: initTensor3DIdentityLike(NUM_HEADS, HEAD_DIM, ds.embeddingDim),
    a_upward: initMatrix(NUM_HEADS, 2 * HEAD_DIM),
    a_downward: initMatrix(NUM_HEADS, 2 * HEAD_DIM),
  });
}

let paramCount = 0;
paramCount += NUM_HEADS * HEAD_DIM * ds.embeddingDim * 2; // W_q + W_k
paramCount += ds.embeddingDim * ds.embeddingDim; // W_intent
for (let level = 0; level <= graph.maxLevel; level++) {
  paramCount += NUM_HEADS * HEAD_DIM * ds.embeddingDim * 2; // W_child + W_parent
  paramCount += NUM_HEADS * 2 * HEAD_DIM * 2; // a_upward + a_downward
}
console.log(`  Trainable: ${(paramCount / 1e6).toFixed(2)}M params`);

// ---- Adam ----
const adam = new AdamOptimizer({ lr: LEARNING_RATE, gradientClip: 1.0 });
for (let h = 0; h < NUM_HEADS; h++) {
  adam.register(`W_q_${h}`, [HEAD_DIM, ds.embeddingDim]);
  adam.register(`W_k_${h}`, [HEAD_DIM, ds.embeddingDim]);
}
adam.register("W_intent", [ds.embeddingDim, ds.embeddingDim]);
for (let level = 0; level <= graph.maxLevel; level++) {
  for (let h = 0; h < NUM_HEADS; h++) {
    adam.register(`W_child_L${level}_H${h}`, [HEAD_DIM, ds.embeddingDim]);
    adam.register(`W_parent_L${level}_H${h}`, [HEAD_DIM, ds.embeddingDim]);
  }
  adam.register(`a_up_L${level}`, [NUM_HEADS, 2 * HEAD_DIM]);
  adam.register(`a_down_L${level}`, [NUM_HEADS, 2 * HEAD_DIM]);
}

// ---- Gradient accumulators ----
const grads = initMultiLevelKHeadGradients(levelParams, headParams, config);

// ---- Orchestrator (training mode = true → caches per-phase backward caches) ----
const orchestrator = new MultiLevelOrchestrator(true);

// ==========================================================================
// Pretty logging helpers (ANSI colors + progress bar)
// ==========================================================================

const isTTY = Deno.stdout.isTerminal();
const C = {
  reset: isTTY ? "\x1b[0m" : "",
  bold: isTTY ? "\x1b[1m" : "",
  dim: isTTY ? "\x1b[2m" : "",
  red: isTTY ? "\x1b[31m" : "",
  green: isTTY ? "\x1b[32m" : "",
  yellow: isTTY ? "\x1b[33m" : "",
  blue: isTTY ? "\x1b[34m" : "",
  magenta: isTTY ? "\x1b[35m" : "",
  cyan: isTTY ? "\x1b[36m" : "",
  white: isTTY ? "\x1b[37m" : "",
  bgBlue: isTTY ? "\x1b[44m" : "",
};

function progressBar(current: number, total: number, width = 20): string {
  const ratio = Math.min(current / Math.max(total, 1), 1);
  const filled = Math.round(ratio * width);
  const empty = width - filled;
  const bar = "█".repeat(filled) + "░".repeat(empty);
  const pct = (ratio * 100).toFixed(0).padStart(3);
  return `${C.cyan}${bar}${C.reset} ${pct}%`;
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const min = Math.floor(ms / 60000);
  const sec = Math.floor((ms % 60000) / 1000);
  return `${min}m${sec.toString().padStart(2, "0")}s`;
}

function formatMem(): string {
  const rss = Deno.memoryUsage().rss;
  return rss >= 1024 * 1024 * 1024
    ? `${(rss / 1024 / 1024 / 1024).toFixed(1)}GB`
    : `${(rss / 1024 / 1024).toFixed(0)}MB`;
}

const enc = new TextEncoder();
function writeInPlace(msg: string): void {
  if (isTTY) {
    Deno.stdout.writeSync(enc.encode(`\x1b[2K\r${msg}`));
  } else {
    // Pipe/file: print as regular line (no \r tricks)
    console.log(msg);
  }
}

// ==========================================================================
// Training loop
// ==========================================================================

/** Serialize current params for checkpoint/export */
function serializeParams() {
  return {
    headParams: headParams.map(hp => ({ W_q: hp.W_q, W_k: hp.W_k, W_v: hp.W_v, a: hp.a })),
    W_intent,
    levelParams: Object.fromEntries(
      Array.from(levelParams.entries()).map(([level, lp]) => [level, {
        W_child: lp.W_child, W_parent: lp.W_parent,
        a_upward: lp.a_upward, a_downward: lp.a_downward,
      }])),
    config: { numHeads: NUM_HEADS, headDim: HEAD_DIM, embeddingDim: ds.embeddingDim, preserveDim: true, maxLevel: graph.maxLevel },
  };
}

console.log(`\n${C.bold}${C.bgBlue} SHGAT-TF Training ${C.reset} ${EPOCHS} epochs\n`);
const trainingStartMs = Date.now();
let bestHit1 = 0, bestMRR = 0, bestEpoch = 0;

/** Per-epoch log for report JSON */
interface EpochLogEntry {
  epoch: number;
  lr: number;
  tau: number;
  klWeight: number;
  infoLoss: number;
  hierLoss: number;
  klLoss: number;
  trainAcc: number;
  testHit1: number;
  testHit3: number;
  testHit5: number;
  testMRR: number;
  testNDCG5: number;
  hierR1: number;
  orphR1: number;
  gradNorm: number;
  weightNormIntent: number;
  weightNormWq: number;
  weightNormWk: number;
  mpDeltaHier: number;
  mpDeltaOrph: number;
  silhouette: number;
  scoreMean: number;
  scoreStd: number;
  top1ModeCount: number;  // how many test examples map to the single most predicted tool
  top1UniqueTools: number;  // number of distinct tools in top-1 predictions
  durationMs: number;
}
const epochLog: EpochLogEntry[] = [];

/** Epoch durations for ETA calculation */
const epochDurationsMs: number[] = [];

// Pre-allocate gradient buffer for epoch-level MP backward accumulation.
// dH gradients from ALL batches (InfoNCE + KL) are accumulated here, then
// ONE MP backward pass propagates them through the graph at epoch end.
const _epochDH: number[][] = graph.l0Ids.map(() => new Array<number>(ds.embeddingDim).fill(0));
// dE gradients at L1+ from contrastive capability-level loss.
// Previously always zero; now accumulates gradients from L1+ contrastive batches.
const _epochDE = new Map<number, number[][]>();
for (const [level, ids] of graph.nodeIdsByLevel) {
  _epochDE.set(level, ids.map(() => new Array<number>(ds.embeddingDim).fill(0)));
}

function zeroDH(dh: number[][]): void {
  for (const row of dh) row.fill(0);
}

function zeroDE(de: Map<number, number[][]>): void {
  for (const [, rows] of de) {
    for (const row of rows) row.fill(0);
  }
}

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  const t0 = Date.now();
  const epochLR = scheduleLR(epoch, EPOCHS, LEARNING_RATE, LR_WARMUP);
  const tau = scheduleTemperature(epoch, EPOCHS, TAU_START, TAU_END);
  const klWeight = USE_KL ? scheduleKLWeight(epoch, KL_WARMUP, KL_WEIGHT_PLATEAU) : 0;

  // Update Adam LR for this epoch
  adam.lr = epochLR;

  // ---- MP Forward (once per epoch) ----
  // Uses forwardMultiLevelWithCache so we get MultiLevelBackwardCache with
  // per-phase caches (VE/EE/EV), required for orchestrator.backwardMultiLevel().
  const mpT0 = Date.now();
  const orchConfig = { numHeads: NUM_HEADS, numLayers: 1, dropout: 0, leakyReluSlope: 0.2 };
  let { result: mpResult, cache: mpBackwardCache } = orchestrator.forwardMultiLevelWithCache(
    graph.H_init,
    graph.E_levels_init,
    graph.l0ToL1Conn,
    graph.interLevelConns,
    levelParams,
    orchConfig,
  );
  const mpMs = Date.now() - mpT0;

  // Build enriched embeddings map from MP result
  const enrichedEmbs = new Map<string, number[]>();
  const H_final = mpResult.H;
  for (let i = 0; i < graph.l0Ids.length; i++) {
    enrichedEmbs.set(graph.l0Ids[i], H_final[i]);
  }
  for (const [level, ids] of graph.nodeIdsByLevel) {
    const E_level = mpResult.E.get(level) ?? [];
    for (let i = 0; i < ids.length; i++) {
      enrichedEmbs.set(ids[i], E_level[i]);
    }
  }

  // ---- Pre-compute K projections for KL scoring (epoch-level cache) ----
  // projectedKeys[h] = H_final @ W_k[h].T  →  [numL0, headDim]
  // Constant over the epoch: H_final fixed (MP once/epoch), W_k changes only
  // slightly per-batch (Adam step ≪ weight magnitude). Replaces O(batchSize ×
  // sparseTargets × numHeads) matVec calls with O(1) index lookups.
  // Cost: 1932 × 64 × 16 heads ≈ 8MB (negligible).
  const klPrecomputeT0 = Date.now();
  const projectedKeysPerHead: number[][][] = new Array(config.numHeads);
  for (let h = 0; h < config.numHeads; h++) {
    projectedKeysPerHead[h] = math.matmulTranspose(H_final, headParams[h].W_k);
  }
  const klPrecomputeMs = Date.now() - klPrecomputeT0;

  console.log(`\n${C.bold}━━━ Epoch ${epoch + 1}/${EPOCHS}${C.reset} ${C.dim}LR=${epochLR.toFixed(5)} τ=${tau.toFixed(4)} klW=${klWeight.toFixed(3)} MP=${formatDuration(mpMs)} KL-precompute=${formatDuration(klPrecomputeMs)}${C.reset}`);

  // ---- Epoch-level accumulators ----
  let epochAccCorrect = 0, epochAccTotal = 0;
  let epochGradNormSqSum = 0, epochGradBatches = 0;
  // Zero epoch-level dH and dE accumulators (gradients from all batches flow here)
  zeroDH(_epochDH);
  zeroDE(_epochDE);

  // ---- InfoNCE batches (prod) ----
  let infoLossSum = 0, infoBatches = 0;
  const prodShuffled = [...ds.prodTrain];
  shuffleInPlace(prodShuffled);
  const numInfoBatches = Math.ceil(prodShuffled.length / BATCH_SIZE);

  for (let b = 0; b < numInfoBatches; b++) {
    const batch = prodShuffled.slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE);
    if (batch.length === 0) continue;

    resetMultiLevelKHeadGradients(grads, levelParams, headParams, config);

    let batchLoss = 0;
    let batchCorrect = 0;

    // ---- Batch contrastive: in-batch negatives with symmetric CE ----
    const intentsProjected: number[][] = [];
    const positiveEmbs: number[][] = [];
    for (const ex of batch) {
      intentsProjected.push(math.matVecBlas(W_intent, ex.intentEmbedding));
      positiveEmbs.push(enrichedEmbs.get(ex.targetToolId) ?? []);
    }

    const { loss, cache } = batchContrastiveForward(
      intentsProjected, positiveEmbs, headParams, config, tau,
    );
    batchLoss = loss * batch.length;

    for (let i = 0; i < batch.length; i++) {
      let maxLogit = -Infinity;
      for (let j = 0; j < batch.length; j++) {
        if (cache.logits[i][j] > maxLogit) maxLogit = cache.logits[i][j];
      }
      if (cache.logits[i][i] >= maxLogit) batchCorrect++;
    }

    // Backward (K-head gradients)
    const { dIntentsProjected, dNodeEmbeddings } = batchContrastiveBackward(
      cache, headParams, grads.khead, config,
    );

    // Accumulate dNodeEmbeddings into _epochDH (epoch-level MP accumulator)
    for (let i = 0; i < batch.length; i++) {
      const l0Idx = l0IdxMap.get(batch[i].targetToolId);
      if (l0Idx !== undefined) {
        for (let d = 0; d < ds.embeddingDim; d++) {
          _epochDH[l0Idx][d] += dNodeEmbeddings[i][d];
        }
      }
    }

    // W_intent backward
    for (let i = 0; i < batch.length; i++) {
      backpropWIntent(dIntentsProjected[i], batch[i].intentEmbedding, grads, config);
    }

    // K-head gradient norms (MP norms computed at epoch level)
    const batchGN = computeGradNorms(grads);
    epochGradNormSqSum += batchGN.total * batchGN.total;
    epochGradBatches++;

    // ---- Adam step: K-head + W_intent only (MP deferred to epoch end) ----
    for (let h = 0; h < NUM_HEADS; h++) {
      adam.step(`W_q_${h}`, headParams[h].W_q, grads.khead.dW_q[h]);
      adam.step(`W_k_${h}`, headParams[h].W_k, grads.khead.dW_k[h]);
    }
    adam.step("W_intent", W_intent, grads.dW_intent);

    infoLossSum += batchLoss / batch.length;
    infoBatches++;
    epochAccCorrect += batchCorrect;
    epochAccTotal += batch.length;

    // ---- Batch log ----
    const batchAcc = (batchCorrect / batch.length * 100).toFixed(1);
    if ((b + 1) % 5 === 0 || b === numInfoBatches - 1) {
      const batchElapsed = Date.now() - t0 - mpMs;
      const batchEta = batchElapsed / (b + 1) * (numInfoBatches - b - 1);
      writeInPlace(
        `  ${progressBar(b + 1, numInfoBatches, 15)} ` +
        `${C.red}loss=${(batchLoss / batch.length).toFixed(3)}${C.reset} ` +
        `${C.green}acc=${batchAcc}%${C.reset} ` +
        `${C.dim}|∇|=${batchGN.total.toFixed(3)} ${formatMem()} ETA ${formatDuration(batchEta)}${C.reset}`);
    }
  }
  if (numInfoBatches > 0) console.log();

  // ---- Contrastive L1+ batches (recursive hierarchy-level contrastive) ----
  // For each hierarchy level above L0, compute InfoNCE(intent → ancestor_embedding)
  // and accumulate dE gradients. This gives the MP a DIRECT loss signal — not just
  // indirect dH from L0 scoring. Recursive: iterates all levels automatically.
  let hierLossSum = 0, hierBatches = 0;
  const hierWeight = 0.5; // scale relative to InfoNCE (balances L1+ vs L0 signal)
  const hierBatchesByLevel = new Map<number, number>(); // per-level batch count for normalization

  for (let orchLevel = 0; orchLevel <= graph.maxLevel; orchLevel++) {
    const levelNodeIds = graph.nodeIdsByLevel.get(orchLevel);
    if (!levelNodeIds || levelNodeIds.length === 0) {
      console.log(`  ${C.dim}HIER-L${orchLevel + 1}: skipped (no nodeIds)${C.reset}`);
      continue;
    }

    const levelEmbs = mpResult.E.get(orchLevel);
    if (!levelEmbs || levelEmbs.length === 0) {
      console.log(`  ${C.dim}HIER-L${orchLevel + 1}: skipped (no embs, E keys=[${[...mpResult.E.keys()]}])${C.reset}`);
      continue;
    }

    // Collect examples (prod + n8n) that have a valid ancestor at this level.
    // Prod tools are often orphans (no L1 parent), but n8n tools have hierarchy
    // from workflow groupings → n8n data provides the bulk of HIER examples.
    const levelExamples: { intentEmbedding: number[]; ancestorIdxs: number[] }[] = [];
    let noL0 = 0, noAnc = 0, noMap = 0;
    // Helper to collect from any example source with targetToolId + intentEmbedding
    const collectFromExamples = (examples: { targetToolId: string; intentEmbedding: number[] }[]) => {
      for (const ex of examples) {
        const l0Idx = l0IdxMap.get(ex.targetToolId);
        if (l0Idx === undefined) { noL0++; continue; }
        const ancestors = graph.l0Ancestors[l0Idx];
        if (!ancestors || ancestors.size === 0) { noMap++; continue; }
        const ancestorIdxs = ancestors.get(orchLevel);
        if (ancestorIdxs && ancestorIdxs.length > 0) {
          levelExamples.push({ intentEmbedding: ex.intentEmbedding, ancestorIdxs });
        } else {
          noAnc++;
        }
      }
    };
    collectFromExamples(prodShuffled);
    collectFromExamples(ds.n8nTrain); // n8n tools have L1+ parents via workflow groupings
    shuffleInPlace(levelExamples); // mix prod + n8n before batching
    if (levelExamples.length < BATCH_SIZE) {
      console.log(`  ${C.dim}HIER-L${orchLevel + 1}: skipped (${levelExamples.length} ex < ${BATCH_SIZE}, noL0=${noL0}, noMap=${noMap}, noAnc=${noAnc})${C.reset}`);
      continue;
    }

    let levelLossSum = 0, levelBatchCount = 0;
    const numLevelBatches = Math.ceil(levelExamples.length / BATCH_SIZE);

    for (let b = 0; b < numLevelBatches; b++) {
      const batch = levelExamples.slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE);
      if (batch.length < 2) continue; // need ≥2 for in-batch negatives

      resetMultiLevelKHeadGradients(grads, levelParams, headParams, config);

      const intentsProjected: number[][] = [];
      const positiveEmbs: number[][] = [];
      for (const ex of batch) {
        intentsProjected.push(math.matVecBlas(W_intent, ex.intentEmbedding));
        positiveEmbs.push(levelEmbs[ex.ancestorIdxs[0]]);
      }

      const { loss, cache: hierCache } = batchContrastiveForward(
        intentsProjected, positiveEmbs, headParams, config, tau,
      );

      const { dIntentsProjected, dNodeEmbeddings } = batchContrastiveBackward(
        hierCache, headParams, grads.khead, config,
      );

      // Track grad norms for epoch summary (scaled by hierWeight)
      const hierGN = computeGradNorms(grads);
      epochGradNormSqSum += hierGN.total * hierGN.total * hierWeight * hierWeight;
      epochGradBatches++;

      // Accumulate dNodeEmbeddings into _epochDE at this level (DIRECT MP gradient)
      const levelDE = _epochDE.get(orchLevel)!;
      for (let i = 0; i < batch.length; i++) {
        const ancestorIdx = batch[i].ancestorIdxs[0];
        for (let d = 0; d < ds.embeddingDim; d++) {
          levelDE[ancestorIdx][d] += dNodeEmbeddings[i][d] * hierWeight;
        }
      }

      // W_intent backward (scaled by hierWeight to balance with L0 InfoNCE)
      for (let i = 0; i < batch.length; i++) {
        const scaled = dIntentsProjected[i].map(v => v * hierWeight);
        backpropWIntent(scaled, batch[i].intentEmbedding, grads, config);
      }

      // Scale K-head gradients by hierWeight (same scaling as W_intent and dE)
      for (let h = 0; h < NUM_HEADS; h++) {
        for (const row of grads.khead.dW_q[h]) { for (let d = 0; d < row.length; d++) row[d] *= hierWeight; }
        for (const row of grads.khead.dW_k[h]) { for (let d = 0; d < row.length; d++) row[d] *= hierWeight; }
      }

      // Adam step: K-head + W_intent (MP deferred to epoch end)
      for (let h = 0; h < NUM_HEADS; h++) {
        adam.step(`W_q_${h}`, headParams[h].W_q, grads.khead.dW_q[h]);
        adam.step(`W_k_${h}`, headParams[h].W_k, grads.khead.dW_k[h]);
      }
      adam.step("W_intent", W_intent, grads.dW_intent);

      levelLossSum += loss;
      levelBatchCount++;
      hierLossSum += loss;
      hierBatches++;
      hierBatchesByLevel.set(orchLevel, (hierBatchesByLevel.get(orchLevel) ?? 0) + 1);
    }

    if (levelBatchCount > 0) {
      console.log(
        `  ${C.yellow}HIER-L${orchLevel + 1}${C.reset} ${levelExamples.length} ex, ` +
        `${levelBatchCount} batches, ${C.red}loss=${(levelLossSum / levelBatchCount).toFixed(3)}${C.reset} ${C.dim}w=${hierWeight.toFixed(2)}${C.reset}`);
    }
  }

  // ---- KL batches (n8n soft targets) — BATCHED BACKWARD ----
  //
  // Gradient flow: dLogit → K-head → dNodeEmbedding → _epochDH (accumulated)
  // MP backward deferred to epoch end (same as InfoNCE path).
  //
  // ---- Batched backward design ----
  //
  // PROBLEM: The original per-target backward calls ~768 individual
  // backpropMultiHeadKHeadLogit() per batch (128 examples × ~7.5 sparse targets
  // × ~80% nonzero filter). Each call does per-head: 2 outerProductAdd (rank-1
  // updates on [64,1024]) + 2 matVecTransposeBlas ([64,1024]^T @ [64]). Total:
  // ~73K small BLAS ops per batch = 61% of KL cost.
  //
  // SOLUTION: Collect all backward tuples across the batch, then for each head
  // do 4 large matmuls instead of T*4 small vector ops:
  //
  //   T = total tuples with nonzero gradient (~768)
  //   headDim = 64, embDim = 1024
  //
  //   Per-tuple scalars:
  //     dHeadLogit[t] = dLogit[t] / (numHeads × √headDim)
  //
  //   Per-tuple vectors (from forward cache):
  //     Q_cache[t][h] = W_q[h] @ intentProj[t]       [headDim]
  //     K_cache[t][h] = projectedKeysPerHead[h][l0Idx] [headDim]
  //     intentProj_batch[t]  [embDim]
  //     nodeEmb_batch[t]     [embDim]
  //
  //   For each head h:
  //     dQ_batch[t][d] = K_cache[t][h][d] × dHeadLogit[t]    [T, headDim]
  //     dK_batch[t][d] = Q_cache[t][h][d] × dHeadLogit[t]    [T, headDim]
  //
  //     Weight gradients (sum of outer products = matmul):
  //       dW_q[h] += dQ_batch^T @ intentProj_batch    [headDim,T] @ [T,embDim] = [headDim,embDim]
  //       dW_k[h] += dK_batch^T @ nodeEmb_batch        [headDim,T] @ [T,embDim] = [headDim,embDim]
  //
  //     Input gradients (batch matmul):
  //       dIntentBatch_h = dQ_batch @ W_q[h]           [T,headDim] @ [headDim,embDim] = [T,embDim]
  //       dNodeEmbBatch_h = dK_batch @ W_k[h]          [T,headDim] @ [headDim,embDim] = [T,embDim]
  //
  //   Scatter-accumulate dIntentBatch → per-example totalDIntentProj
  //   Scatter-accumulate dNodeEmbBatch → _epochDH
  //   Per-example: backpropWIntent(totalDIntentProj, intentOriginal)
  //
  // Mathematical equivalence:
  //   outerProductAdd(dW, dQ_i, v_i) adds dW[r][c] += dQ_i[r] × v_i[c]
  //   sum_i(dQ_i ⊗ v_i) = [dQ_0; dQ_1; ...]^T @ [v_0; v_1; ...]
  //                       = dQ_batch^T @ v_batch     (matmul)
  //   where dQ_batch is [T, headDim] and v_batch is [T, embDim]
  //   result shape: [headDim, embDim] ✓
  //
  //   matVecTransposeBlas(W, dQ_i) computes W^T @ dQ_i = [embDim]
  //   For all tuples: [dQ_0; dQ_1; ...] @ W = dQ_batch @ W   (matmul)
  //   where dQ_batch is [T, headDim] and W is [headDim, embDim]
  //   result shape: [T, embDim] ✓
  //
  // Memory budget (T=960 worst case):
  //   dQ_batch/dK_batch: [960, 64] = 480KB (reused per head)
  //   intentProj_batch/nodeEmb_batch: [960, 1024] = 7.5MB each
  //   dIntentBatch/dNodeEmbBatch: [960, 1024] = 7.5MB each (output of matmul)
  //   Total peak: ~30MB — well within 12GB limit
  //
  let klLossSum = 0, klBatches = 0;
  if (klWeight > 0 && ds.n8nTrain.length > 0) {
    const n8nShuffled = [...ds.n8nTrain];
    shuffleInPlace(n8nShuffled);
    const n8nSample = KL_SUBSAMPLE > 0 ? n8nShuffled.slice(0, KL_SUBSAMPLE) : n8nShuffled;
    const numKLBatches = Math.ceil(n8nSample.length / KL_BATCH_SIZE);

    const scoringDim = headParams[0].W_q.length;     // headDim = 64
    const invScale = 1.0 / Math.sqrt(scoringDim);    // 1/√headDim
    const invHeads = 1.0 / config.numHeads;           // 1/numHeads
    const dHeadLogitScale = invHeads * invScale;      // combined: 1/(numHeads × √headDim)

    // Gradient accumulation: reset grads once, accumulate over KL_ACCUM_STEPS
    // batches, then do ONE Adam step. Reduces Adam calls by KL_ACCUM_STEPS×.
    // Gradients are normalized by accumSteps at Adam step time.
    let accumCount = 0;
    resetMultiLevelKHeadGradients(grads, levelParams, headParams, config);

    for (let b = 0; b < numKLBatches; b++) {
      const batch = n8nSample.slice(b * KL_BATCH_SIZE, (b + 1) * KL_BATCH_SIZE);
      if (batch.length === 0) continue;

      let batchKL = 0;

      // ================================================================
      // Phase 1: Batched forward pass + collect backward tuples
      // ================================================================
      //
      // Batched projections (BLAS-accelerated):
      //   intentProjBatch = matmulTranspose(intentEmbBatch, W_intent)  [B,1024]@[1024,1024]^T=[B,1024]
      //   Q_batch[h]      = matmulTranspose(intentProjBatch, W_q[h])   [B,1024]@[64,1024]^T=[B,64]
      // This replaces B×matVecBlas(W_intent,x) + B×16×matVecBlas(W_q,x) (JS fallback for W_q)
      // with 1 BLAS matmul + 16 BLAS matmul.
      //
      // Scoring (dot Q·K) remains per-example because sparse targets differ per example.

      // Backward tuple: everything needed for one (example, sparse_target) pair
      interface KLBackwardTuple {
        dLogit: number;                   // scalar gradient from KL
        Q_perHead: number[][];            // [numHeads][headDim] — Q vectors
        K_perHead: number[][];            // [numHeads][headDim] — K vectors (pre-computed)
        intentProjected: number[];        // [embDim] — W_intent @ intentEmbedding
        nodeEmb: number[];                // [embDim] — H_final[l0Idx]
        exIdx: number;                    // index into batch (for scatter-add dIntentProj)
        l0Idx: number;                    // leaf index (for scatter-add _epochDH)
      }

      const backwardTuples: KLBackwardTuple[] = [];
      // Track which examples have valid tuples (for W_intent backprop)
      const exampleIntentOriginal: (number[] | null)[] = new Array(batch.length).fill(null);

      // --- Pre-filter valid examples and resolve sparse targets ---
      interface ValidExample {
        exIdx: number;
        intentEmb: number[];
        sparseL0Idxs: number[];
        sparseProbs: number[];
      }
      const validExamples: ValidExample[] = [];
      for (let exIdx = 0; exIdx < batch.length; exIdx++) {
        const ex = batch[exIdx];
        if (!ex.softTargetSparse || ex.softTargetSparse.length === 0) continue;
        const sparseL0Idxs: number[] = [];
        const sparseProbs: number[] = [];
        for (const [l0Idx, prob] of ex.softTargetSparse) {
          if (l0Idx >= 0 && l0Idx < ds.leafIds.length) {
            sparseL0Idxs.push(l0Idx);
            sparseProbs.push(prob);
          }
        }
        if (sparseL0Idxs.length === 0) continue;
        validExamples.push({ exIdx, intentEmb: ex.intentEmbedding, sparseL0Idxs, sparseProbs });
      }

      if (validExamples.length > 0) {
        const V = validExamples.length;

        // --- Batched intent projection: [V, embDim] ---
        // matmulTranspose(A, B) = A @ B^T
        // intentEmbBatch[V, 1024] @ W_intent[1024, 1024]^T = [V, 1024]
        // Hits BLAS: V >= 10 (usually ~120+), dim=1024 >= 64 ✓
        const intentEmbBatch: number[][] = new Array(V);
        for (let i = 0; i < V; i++) intentEmbBatch[i] = validExamples[i].intentEmb;
        const intentProjBatch = math.matmulTranspose(intentEmbBatch, W_intent); // [V, 1024]

        // --- Batched Q projection per head: [V, headDim] ---
        // Q_batch[h] = matmulTranspose(intentProjBatch, W_q[h])
        //   [V, 1024] @ [64, 1024]^T = [V, 64]
        // Hits BLAS: V >= 10, dim=1024 >= 64 ✓
        // Replaces V × 16 individual matVecBlas calls (which fall back to JS for W_q[64,1024])
        const Q_allHeads: number[][][] = new Array(config.numHeads); // [numHeads][V][headDim]
        for (let h = 0; h < config.numHeads; h++) {
          Q_allHeads[h] = math.matmulTranspose(intentProjBatch, headParams[h].W_q); // [V, headDim]
        }

        // --- Per-example scoring + KL + backward tuple collection ---
        for (let vi = 0; vi < V; vi++) {
          const { exIdx, sparseL0Idxs, sparseProbs } = validExamples[vi];
          const intentProjected = intentProjBatch[vi]; // [embDim], from batched result

          // Collect Q vectors for this example from batched results
          const Q_perHead: number[][] = new Array(config.numHeads);
          for (let h = 0; h < config.numHeads; h++) {
            Q_perHead[h] = Q_allHeads[h][vi]; // [headDim]
          }

          // Forward: compute logits for all sparse targets (per-example, targets differ)
          const logits: number[] = [];
          const K_perHead_all: number[][][] = []; // [target][head][headDim]

          for (const l0Idx of sparseL0Idxs) {
            const K_target: number[][] = [];
            let avgLogit = 0;
            for (let h = 0; h < config.numHeads; h++) {
              const K = projectedKeysPerHead[h][l0Idx];
              const dotQK = math.dot(Q_perHead[h], K);
              avgLogit += dotQK * invScale; // logit = dot / √dim
              K_target.push(K);
            }
            logits.push(avgLogit * invHeads); // average over heads
            K_perHead_all.push(K_target);
          }

          // Softmax with temperature + KL loss
          let maxL = -Infinity;
          for (const l of logits) if (l > maxL) maxL = l;
          const expL = logits.map(l => Math.exp((l - maxL) / tau));
          const sumE = expL.reduce((a, b_) => a + b_, 0);
          const q = expL.map(e => e / sumE);

          let kl = 0;
          for (let j = 0; j < sparseProbs.length; j++) {
            if (sparseProbs[j] > 1e-8 && q[j] > 1e-8) {
              kl += sparseProbs[j] * Math.log(sparseProbs[j] / q[j]);
            }
          }
          batchKL += kl;

          // Collect backward tuples for targets with nonzero gradient
          let hasValidTuple = false;
          for (let j = 0; j < sparseL0Idxs.length; j++) {
            const dLogit = (q[j] - sparseProbs[j]) * klWeight / tau;
            if (Math.abs(dLogit) < 1e-10) continue;

            backwardTuples.push({
              dLogit,
              Q_perHead,          // shared across targets of same example
              K_perHead: K_perHead_all[j],
              intentProjected,    // shared across targets of same example
              nodeEmb: H_final[sparseL0Idxs[j]],
              exIdx,
              l0Idx: sparseL0Idxs[j],
            });
            hasValidTuple = true;
          }
          if (hasValidTuple) {
            exampleIntentOriginal[exIdx] = validExamples[vi].intentEmb;
          }
        }
      } // end if validExamples.length > 0

      // ================================================================
      // Phase 2: Batched backward (all tuples at once)
      // ================================================================

      const T = backwardTuples.length;

      if (T > 0) {
        const embDim = ds.embeddingDim;
        const headDim = scoringDim;

        // Build batch matrices: intentProj_batch [T, embDim], nodeEmb_batch [T, embDim]
        const intentProjBatch: number[][] = new Array(T);
        const nodeEmbBatch: number[][] = new Array(T);
        for (let t = 0; t < T; t++) {
          intentProjBatch[t] = backwardTuples[t].intentProjected;
          nodeEmbBatch[t] = backwardTuples[t].nodeEmb;
        }

        // Accumulator for dIntentProjected per tuple (sum across all heads)
        // and dNodeEmbedding per tuple (sum across all heads).
        // [T, embDim] — allocated once, accumulated across heads.
        const dIntentProjAll: number[][] = Array.from({ length: T }, () => new Array(embDim).fill(0));
        const dNodeEmbAll: number[][] = Array.from({ length: T }, () => new Array(embDim).fill(0));

        for (let h = 0; h < config.numHeads; h++) {
          // Build dQ_batch and dK_batch for this head: [T, headDim]
          const dQ_batch: number[][] = new Array(T);
          const dK_batch: number[][] = new Array(T);

          for (let t = 0; t < T; t++) {
            const tuple = backwardTuples[t];
            const d = tuple.dLogit * dHeadLogitScale;
            const K_h = tuple.K_perHead[h];
            const Q_h = tuple.Q_perHead[h];

            // dQ[t][dim] = K[t][h][dim] × dHeadLogit
            // dK[t][dim] = Q[t][h][dim] × dHeadLogit
            const dQ = new Array(headDim);
            const dK = new Array(headDim);
            for (let s = 0; s < headDim; s++) {
              dQ[s] = K_h[s] * d;
              dK[s] = Q_h[s] * d;
            }
            dQ_batch[t] = dQ;
            dK_batch[t] = dK;
          }

          // Weight gradient: dW_q[h] += dQ_batch^T @ intentProjBatch
          // dW_k[h] += dK_batch^T @ nodeEmbBatch
          // When KL_ISOLATE_KHEAD=true, skip W_q/W_k weight updates from KL.
          // This prevents KL's "smooth distribution" gradient from polluting
          // the scoring heads, while still allowing gradient flow to W_intent
          // and MP weights via the input gradients below.
          if (!KL_ISOLATE_KHEAD) {
            const dQ_T = math.transpose(dQ_batch);      // [headDim, T]
            const dK_T = math.transpose(dK_batch);      // [headDim, T]

            const dWq_h = math.matmul(dQ_T, intentProjBatch); // [headDim, embDim]
            const dWk_h = math.matmul(dK_T, nodeEmbBatch);    // [headDim, embDim]

            const gradDWq = grads.khead.dW_q[h];
            const gradDWk = grads.khead.dW_k[h];
            for (let r = 0; r < headDim; r++) {
              const gqr = gradDWq[r], dqr = dWq_h[r];
              const gkr = gradDWk[r], dkr = dWk_h[r];
              for (let c = 0; c < embDim; c++) {
                gqr[c] += dqr[c];
                gkr[c] += dkr[c];
              }
            }
          }

          // Input gradient: dIntentBatch_h = dQ_batch @ W_q[h]
          //   [T, headDim] @ [headDim, embDim] = [T, embDim]
          // This hits BLAS when T >= 10.
          const dIntentBatch_h = math.matmul(dQ_batch, headParams[h].W_q); // [T, embDim]
          const dNodeEmbBatch_h = math.matmul(dK_batch, headParams[h].W_k); // [T, embDim]

          // Accumulate into per-tuple totals (summing across heads)
          for (let t = 0; t < T; t++) {
            const diAll = dIntentProjAll[t], diH = dIntentBatch_h[t];
            const dnAll = dNodeEmbAll[t], dnH = dNodeEmbBatch_h[t];
            for (let d = 0; d < embDim; d++) {
              diAll[d] += diH[d];
              dnAll[d] += dnH[d];
            }
          }
        } // end for each head

        // ================================================================
        // Phase 3: Scatter-accumulate results
        // ================================================================

        // Scatter dNodeEmbAll → _epochDH
        for (let t = 0; t < T; t++) {
          const l0Idx = backwardTuples[t].l0Idx;
          const dnAll = dNodeEmbAll[t];
          const dhRow = _epochDH[l0Idx];
          for (let d = 0; d < embDim; d++) {
            dhRow[d] += dnAll[d];
          }
        }

        // Scatter-add dIntentProjAll per example, then backprop W_intent
        // Group tuples by exIdx to sum their dIntentProjected
        const perExDIntent = new Map<number, number[]>();
        for (let t = 0; t < T; t++) {
          const exIdx = backwardTuples[t].exIdx;
          let acc = perExDIntent.get(exIdx);
          if (!acc) {
            acc = new Array(embDim).fill(0);
            perExDIntent.set(exIdx, acc);
          }
          const diAll = dIntentProjAll[t];
          for (let d = 0; d < embDim; d++) {
            acc[d] += diAll[d];
          }
        }

        for (const [exIdx, totalDIntentProj] of perExDIntent) {
          const intentOrig = exampleIntentOriginal[exIdx];
          if (intentOrig) {
            backpropWIntent(totalDIntentProj, intentOrig, grads, config);
          }
        }
      } // end if T > 0

      klLossSum += batchKL / batch.length;
      klBatches++;
      accumCount++;

      // Gradient accumulation: Adam step every KL_ACCUM_STEPS batches or at end
      const isLastBatch = (b === numKLBatches - 1);
      if (accumCount >= KL_ACCUM_STEPS || isLastBatch) {
        // Normalize accumulated gradients by number of accumulated batches
        if (accumCount > 1) {
          const invAccum = 1 / accumCount;
          if (!KL_ISOLATE_KHEAD) {
            for (let h = 0; h < NUM_HEADS; h++) {
              for (const row of grads.khead.dW_q[h]) for (let d = 0; d < row.length; d++) row[d] *= invAccum;
              for (const row of grads.khead.dW_k[h]) for (let d = 0; d < row.length; d++) row[d] *= invAccum;
            }
          }
          for (const row of grads.dW_intent) for (let d = 0; d < row.length; d++) row[d] *= invAccum;
        }

        // K-head gradient norms (on normalized grads)
        const klGN = computeGradNorms(grads);
        epochGradNormSqSum += klGN.total * klGN.total;
        epochGradBatches++;

        // Adam step: W_intent always; W_q/W_k only if KL is allowed to update them
        if (!KL_ISOLATE_KHEAD) {
          for (let h = 0; h < NUM_HEADS; h++) {
            adam.step(`W_q_${h}`, headParams[h].W_q, grads.khead.dW_q[h]);
            adam.step(`W_k_${h}`, headParams[h].W_k, grads.khead.dW_k[h]);
          }
        }
        adam.step("W_intent", W_intent, grads.dW_intent);

        // Reset for next accumulation window
        accumCount = 0;
        if (!isLastBatch) {
          resetMultiLevelKHeadGradients(grads, levelParams, headParams, config);
        }
      }

      if ((b + 1) % 4 === 0 || b === numKLBatches - 1) {
        writeInPlace(
          `  ${C.magenta}KL${C.reset} ${progressBar(b + 1, numKLBatches, 10)} ` +
          `${C.red}loss=${(klLossSum / klBatches).toFixed(4)}${C.reset} ` +
          `${C.dim}w=${klWeight.toFixed(3)} T=${T} ${formatMem()}${C.reset}`);
      }
    }
    if (numKLBatches > 0) console.log();
  }

  // ---- Epoch-level MP backward ("autoroute") ----
  // All batch dH/dE gradients accumulated in _epochDH/_epochDE. ONE backward pass
  // through the full graph instead of ~100 per-batch passes.
  const mpBackT0 = Date.now();
  // Normalize dH by num_batches (InfoNCE + KL sources)
  const numBatchesDH = infoBatches + klBatches;
  if (numBatchesDH > 0) {
    const scale = 1 / numBatchesDH;
    for (const row of _epochDH) {
      for (let d = 0; d < row.length; d++) row[d] *= scale;
    }
  }
  // Normalize dE per-level by the number of batches that contributed to each level.
  // Global normalization would bias levels with fewer batches (over-normalized).
  for (const [orchLevel, rows] of _epochDE) {
    const levelBatches = hierBatchesByLevel.get(orchLevel) ?? 0;
    if (levelBatches > 0) {
      const scale = 1 / levelBatches;
      for (const row of rows) {
        for (let d = 0; d < row.length; d++) row[d] *= scale;
      }
    }
  }
  const mpGrads = orchestrator.backwardMultiLevel(
    _epochDE, _epochDH, mpBackwardCache, levelParams,
  );
  const mpBackMs = Date.now() - mpBackT0;

  // Adam step for MP params (once per epoch, with reduced LR)
  const savedLr = adam.lr;
  adam.lr = epochLR * MP_LR_SCALE;
  for (const [level, lp] of levelParams) {
    const lg = mpGrads.levelGrads.get(level);
    if (!lg) continue;
    for (let h = 0; h < NUM_HEADS; h++) {
      adam.step(`W_child_L${level}_H${h}`, lp.W_child[h], lg.dW_child[h]);
      adam.step(`W_parent_L${level}_H${h}`, lp.W_parent[h], lg.dW_parent[h]);
    }
    adam.step(`a_up_L${level}`, lp.a_upward, lg.da_upward);
    adam.step(`a_down_L${level}`, lp.a_downward, lg.da_downward);
  }
  adam.lr = savedLr;

  // MP gradient norms (exponential notation to see real magnitude)
  const mpGN = computeMPGradNorms(mpGrads);
  console.log(
    `  ${C.blue}MP backward${C.reset} ${formatDuration(mpBackMs)} ` +
    `${C.dim}|∇W_c|=${mpGN.wChild.toExponential(2)} |∇W_p|=${mpGN.wParent.toExponential(2)} |∇a|=${mpGN.atten.toExponential(2)}${C.reset}`);

  // Release caches for GC between epoch forward passes
  mpBackwardCache = null!;
  mpResult = null!;

  // ---- Epoch summary (enriched) ----
  const infoLoss = infoBatches > 0 ? infoLossSum / infoBatches : 0;
  const klLoss = klBatches > 0 ? klLossSum / klBatches : 0;
  const epochAcc = epochAccTotal > 0 ? (epochAccCorrect / epochAccTotal * 100) : 0;
  const epochGradNorm = epochGradBatches > 0
    ? Math.sqrt(epochGradNormSqSum / epochGradBatches)
    : 0;
  const elapsedMs = Date.now() - t0;
  epochDurationsMs.push(elapsedMs);
  // ETA: average of past epoch durations * remaining epochs
  const avgEpochMs = epochDurationsMs.reduce((a, b) => a + b, 0) / epochDurationsMs.length;
  const remainingEpochs = EPOCHS - (epoch + 1);
  const etaMs = avgEpochMs * remainingEpochs;
  const etaMin = (etaMs / 60000).toFixed(0);

  const hierLoss = hierBatches > 0 ? hierLossSum / hierBatches : 0;
  const lossStr = `${C.red}loss=${infoLoss.toFixed(3)}${C.reset}` +
    (hierBatches > 0 ? `${C.yellow}+hier=${hierLoss.toFixed(3)}${C.reset}` : "") +
    (klBatches > 0 ? `${C.magenta}+kl=${klLoss.toFixed(3)}${C.reset}` : "");
  const accColor = epochAcc >= 80 ? C.green : epochAcc >= 50 ? C.yellow : C.red;
  const accStr = `${accColor}acc=${epochAcc.toFixed(1)}%${C.reset}`;
  const etaStr = remainingEpochs > 0 ? `${C.cyan}ETA ${etaMin}min${C.reset}` : `${C.green}DONE${C.reset}`;
  console.log(
    `${C.bold}  ✓ Epoch ${epoch + 1}/${EPOCHS}${C.reset} ` +
    `${lossStr} ${accStr} ` +
    `${C.dim}|∇|=${epochGradNorm.toFixed(3)} mpBack=${formatDuration(mpBackMs)} ${formatDuration(elapsedMs)} ${formatMem()}${C.reset} ${etaStr}`,
  );

  // ---- Eval ----
  const shouldEval = (epoch + 1) % EVAL_EVERY === 0 || epoch === EPOCHS - 1;
  let testHit1 = 0, testHit3 = 0, testHit5 = 0, testMRR = 0;

  if (shouldEval && ds.prodTest.length > 0) {
    const evalT0 = Date.now();
    const testSample = ds.prodTest.slice(0, Math.min(ds.prodTest.length, 500));
    let hit1 = 0, hit3 = 0, hit5 = 0, rr = 0;
    let hierHit1 = 0, hierTotal = 0, orphHit1 = 0, orphTotal = 0;

    // --- Batched eval: precompute K projections for ALL L0 nodes per head ---
    // AllL0Embs: [numL0 × embDim] matrix
    const numL0 = graph.l0Ids.length;
    const embDim = ds.embeddingDim;
    const AllL0Embs: number[][] = new Array(numL0);
    for (let i = 0; i < numL0; i++) {
      AllL0Embs[i] = enrichedEmbs.get(graph.l0Ids[i]) ?? new Array(embDim).fill(0);
    }
    // K_all_h[h] = W_k[h] @ AllL0Embs^T → [headDim × numL0]
    const AllL0EmbsT = math.transpose(AllL0Embs); // [embDim × numL0]
    const K_all: number[][][] = new Array(config.numHeads);
    for (let h = 0; h < config.numHeads; h++) {
      K_all[h] = math.matmul(headParams[h].W_k, AllL0EmbsT); // [headDim × numL0]
    }

    const scale = 1.0 / Math.sqrt(config.headDim);
    let validCount = 0;

    // Extended metrics accumulators
    let ndcg5Sum = 0;
    let scoreMeanSum = 0, scoreVarSum = 0;
    const top1Predictions = new Map<number, number>(); // l0Idx → count

    for (const ex of testSample) {
      const intentProjected = math.matVecBlas(W_intent, ex.intentEmbedding);
      const targetIdx = graph.l0Ids.indexOf(ex.targetToolId);
      if (targetIdx < 0) continue;
      validCount++;

      // Compute Q_h for each head, then dot with all K columns → scores[numL0]
      const scores = new Float64Array(numL0); // accumulates across heads
      for (let h = 0; h < config.numHeads; h++) {
        const Q_h = math.matVecBlas(headParams[h].W_q, intentProjected); // [headDim]
        const K_h = K_all[h]; // [headDim × numL0]
        // scores[i] += Q_h · K_h[:,i] * scale
        for (let i = 0; i < numL0; i++) {
          let dot = 0;
          for (let d = 0; d < config.headDim; d++) {
            dot += Q_h[d] * K_h[d][i];
          }
          scores[i] += dot * scale;
        }
      }
      // Average across heads
      const invHeads = 1.0 / config.numHeads;
      for (let i = 0; i < numL0; i++) scores[i] *= invHeads;

      // Find rank of target
      const targetScore = scores[targetIdx];
      let rank = 1;
      for (let i = 0; i < numL0; i++) {
        if (i !== targetIdx && scores[i] > targetScore) rank++;
      }

      if (rank <= 1) hit1++;
      if (rank <= 3) hit3++;
      if (rank <= 5) hit5++;
      rr += 1 / rank;

      // NDCG@5: DCG = 1/log2(rank+1) if rank<=5, IDCG = 1/log2(2) = 1.0
      if (rank <= 5) ndcg5Sum += 1.0 / Math.log2(rank + 1);

      // Score distribution stats (collect mean/variance online)
      let sMean = 0;
      for (let i = 0; i < numL0; i++) sMean += scores[i];
      sMean /= numL0;
      let sVar = 0;
      for (let i = 0; i < numL0; i++) { const d = scores[i] - sMean; sVar += d * d; }
      sVar /= numL0;
      scoreMeanSum += sMean;
      scoreVarSum += sVar;

      // Mode collapse detection: track top-1 predicted tool
      let top1Idx = 0;
      for (let i = 1; i < numL0; i++) { if (scores[i] > scores[top1Idx]) top1Idx = i; }
      top1Predictions.set(top1Idx, (top1Predictions.get(top1Idx) || 0) + 1);

      // Track R@1 by hierarchy status
      const hasAnc = graph.l0Ancestors[targetIdx] && graph.l0Ancestors[targetIdx].size > 0;
      if (hasAnc) { hierTotal++; if (rank <= 1) hierHit1++; }
      else { orphTotal++; if (rank <= 1) orphHit1++; }
    }

    const count = validCount || 1;
    testHit1 = hit1 / count;
    testHit3 = hit3 / count;
    testHit5 = hit5 / count;
    testMRR = rr / count;
    const testNDCG5 = ndcg5Sum / count;
    const avgScoreMean = scoreMeanSum / count;
    const avgScoreStd = Math.sqrt(scoreVarSum / count);

    // Mode collapse: most predicted tool + unique count
    let top1ModeCount = 0;
    let top1ModeToolIdx = -1;
    top1Predictions.forEach((cnt, idx) => { if (cnt > top1ModeCount) { top1ModeCount = cnt; top1ModeToolIdx = idx; } });
    const top1UniqueTools = top1Predictions.size;

    const isNewBest = testHit1 > bestHit1;
    if (isNewBest) { bestHit1 = testHit1; bestEpoch = epoch + 1; }
    if (testMRR > bestMRR) bestMRR = testMRR;

    const evalMs = Date.now() - evalT0;
    const r1Color = isNewBest ? C.green : C.yellow;
    console.log(
      `  ${C.blue}📊 EVAL${C.reset} ` +
      `${r1Color}R@1=${(testHit1 * 100).toFixed(1)}%${C.reset} ` +
      `R@3=${(testHit3 * 100).toFixed(1)}% R@5=${(testHit5 * 100).toFixed(1)}% ` +
      `MRR=${testMRR.toFixed(3)} NDCG@5=${testNDCG5.toFixed(3)} ${C.dim}(${validCount} test, ${formatDuration(evalMs)})${C.reset}`,
    );
    console.log(
      `  ${C.dim}   Best R@1=${(bestHit1 * 100).toFixed(1)}% MRR=${bestMRR.toFixed(3)} (epoch ${bestEpoch})${C.reset}`,
    );
    // Score distribution + mode collapse
    const modeToolName = top1ModeToolIdx >= 0 ? graph.l0Ids[top1ModeToolIdx] : "?";
    const collapseWarning = top1ModeCount > validCount * 0.3 ? ` ${C.red}⚠ MODE COLLAPSE${C.reset}` : "";
    console.log(
      `  ${C.dim}   Scores: μ=${avgScoreMean.toFixed(4)} σ=${avgScoreStd.toFixed(4)} | ` +
      `Top-1 mode: ${modeToolName} (${top1ModeCount}/${validCount}=${(top1ModeCount / validCount * 100).toFixed(0)}%), ` +
      `${top1UniqueTools} unique tools${C.reset}${collapseWarning}`,
    );

    // ---- Metric 1: MP delta norm (H_final vs H_init) ----
    // How much did MP change embeddings? Split by hierarchical vs orphan.
    let evalMpDeltaHier = 0, evalMpDeltaOrph = 0;
    {
      let hierDeltaSum = 0, hierCount = 0;
      let orphDeltaSum = 0, orphCount = 0;
      for (let i = 0; i < numL0; i++) {
        let normSq = 0;
        const hf = H_final[i], hi = graph.H_init[i];
        for (let d = 0; d < embDim; d++) {
          const diff = hf[d] - hi[d];
          normSq += diff * diff;
        }
        const norm = Math.sqrt(normSq);
        const hasAnc = graph.l0Ancestors[i] && graph.l0Ancestors[i].size > 0;
        if (hasAnc) { hierDeltaSum += norm; hierCount++; }
        else { orphDeltaSum += norm; orphCount++; }
      }
      const hierAvg = hierCount > 0 ? hierDeltaSum / hierCount : 0;
      const orphAvg = orphCount > 0 ? orphDeltaSum / orphCount : 0;
      evalMpDeltaHier = hierAvg;
      evalMpDeltaOrph = orphAvg;
      console.log(
        `  ${C.dim}   MP Δ: hier=${hierAvg.toFixed(4)} (${hierCount}) orph=${orphAvg.toFixed(4)} (${orphCount})${C.reset}`,
      );
    }

    // ---- Metric 2: R@1 split by hierarchical vs orphan (from eval loop above) ----
    {
      const hierR1 = hierTotal > 0 ? (hierHit1 / hierTotal * 100).toFixed(1) : "n/a";
      const orphR1 = orphTotal > 0 ? (orphHit1 / orphTotal * 100).toFixed(1) : "n/a";
      console.log(
        `  ${C.dim}   R@1 split: hier=${hierR1}% (${hierTotal}) orph=${orphR1}% (${orphTotal})${C.reset}`,
      );
    }

    // ---- Metric 3: Silhouette intra-capability ----
    // For L0 tools with L1 ancestors: avg cosine sim to siblings under same cap
    // vs avg cosine sim to random tools from other caps.
    let evalSilhouette = 0;
    {
      const cosine = math.cosineSimilarity;
      // Group hierarchical L0 by their first L1 ancestor
      const capGroups = new Map<number, number[]>(); // l1Idx → [l0Idxs]
      for (let i = 0; i < numL0; i++) {
        const anc = graph.l0Ancestors[i];
        if (!anc || anc.size === 0) continue;
        // Get first L1 ancestor (level 1 in the map)
        for (const [, ancIdxs] of anc) {
          if (ancIdxs.length > 0) {
            const capIdx = ancIdxs[0];
            let group = capGroups.get(capIdx);
            if (!group) { group = []; capGroups.set(capIdx, group); }
            group.push(i);
            break; // first ancestor only
          }
        }
      }

      // Only evaluate caps with ≥2 tools (need siblings)
      let intraSim = 0, interSim = 0, pairCount = 0;
      const multiGroups = [...capGroups.values()].filter(g => g.length >= 2);
      const allHierTools = multiGroups.flat();

      for (const group of multiGroups.slice(0, 50)) { // cap at 50 groups for speed
        for (let a = 0; a < group.length; a++) {
          const ea = H_final[group[a]];
          // Intra: avg sim to siblings
          for (let b = a + 1; b < group.length; b++) {
            const eb = H_final[group[b]];
            intraSim += cosine(ea, eb);
            pairCount++;
          }
          // Inter: sim to one random tool from another group
          if (allHierTools.length > group.length) {
            let rIdx = allHierTools[Math.floor(Math.random() * allHierTools.length)];
            while (group.includes(rIdx)) {
              rIdx = allHierTools[Math.floor(Math.random() * allHierTools.length)];
            }
            interSim += cosine(ea, H_final[rIdx]);
          }
        }
      }
      const nGroups = multiGroups.length;
      const avgIntra = pairCount > 0 ? intraSim / pairCount : 0;
      const totalInter = multiGroups.reduce((s, g) => s + g.length, 0);
      const avgInter = totalInter > 0 ? interSim / totalInter : 0;
      const silhouette = avgIntra - avgInter; // higher = better clustering
      evalSilhouette = silhouette;
      console.log(
        `  ${C.dim}   Silhouette: intra=${avgIntra.toFixed(4)} inter=${avgInter.toFixed(4)} ` +
        `Δ=${silhouette.toFixed(4)} (${nGroups} caps ≥2 tools)${C.reset}`,
      );
    }

    // ---- Metric 4: Weight norms (detect instability / explosion) ----
    {
      // W_intent: [embDim, embDim]
      let wIntentNormSq = 0;
      for (const row of W_intent) for (const v of row) wIntentNormSq += v * v;
      const wIntentNorm = Math.sqrt(wIntentNormSq);

      // W_q / W_k: average norm across heads
      let wqNormSum = 0, wkNormSum = 0;
      for (let h = 0; h < NUM_HEADS; h++) {
        let qSq = 0, kSq = 0;
        for (const row of headParams[h].W_q) for (const v of row) qSq += v * v;
        for (const row of headParams[h].W_k) for (const v of row) kSq += v * v;
        wqNormSum += Math.sqrt(qSq);
        wkNormSum += Math.sqrt(kSq);
      }
      const wqNorm = wqNormSum / NUM_HEADS;
      const wkNorm = wkNormSum / NUM_HEADS;

      console.log(
        `  ${C.dim}   Weights: |W_intent|=${wIntentNorm.toFixed(2)} |W_q|=${wqNorm.toFixed(2)} |W_k|=${wkNorm.toFixed(2)}${C.reset}`,
      );

      // ---- Build epoch log entry ----
      epochLog.push({
        epoch: epoch + 1,
        lr: epochLR,
        tau,
        klWeight,
        infoLoss,
        hierLoss,
        klLoss,
        trainAcc: epochAcc,
        testHit1: testHit1 * 100,
        testHit3: testHit3 * 100,
        testHit5: testHit5 * 100,
        testMRR,
        testNDCG5,
        hierR1: hierTotal > 0 ? hierHit1 / hierTotal * 100 : 0,
        orphR1: orphTotal > 0 ? orphHit1 / orphTotal * 100 : 0,
        gradNorm: epochGradNorm,
        weightNormIntent: wIntentNorm,
        weightNormWq: wqNorm,
        weightNormWk: wkNorm,
        mpDeltaHier: evalMpDeltaHier,
        mpDeltaOrph: evalMpDeltaOrph,
        silhouette: evalSilhouette,
        scoreMean: avgScoreMean,
        scoreStd: avgScoreStd,
        top1ModeCount,
        top1UniqueTools,
        durationMs: elapsedMs,
      });
    }

    // Save best model checkpoint (overwrite previous best)
    if (isNewBest) {
      const bestPath = resolve(GRU_DATA_DIR, "shgat-params-ob-best.json");
      Deno.writeTextFileSync(bestPath, JSON.stringify(serializeParams()));
      console.log(`  ${C.green}💾 Best model saved${C.reset} ${C.dim}→ ${bestPath}${C.reset}`);
    }
  } else {
    // Non-eval epoch: still log loss/grad data
    epochLog.push({
      epoch: epoch + 1, lr: epochLR, tau, klWeight,
      infoLoss, hierLoss, klLoss, trainAcc: epochAcc,
      testHit1: -1, testHit3: -1, testHit5: -1, testMRR: -1, testNDCG5: -1,
      hierR1: -1, orphR1: -1, gradNorm: epochGradNorm,
      weightNormIntent: 0, weightNormWq: 0, weightNormWk: 0,
      mpDeltaHier: 0, mpDeltaOrph: 0, silhouette: 0,
      scoreMean: 0, scoreStd: 0, top1ModeCount: 0, top1UniqueTools: 0,
      durationMs: elapsedMs,
    });
  }

  // ---- MP Health Metrics (every epoch) ----
  // Track gradient signal reaching MP and MP param magnitudes
  {
    // dH norm: how much gradient signal flows FROM scoring heads TO MP
    let dhNormSq = 0;
    for (const row of _epochDH) for (const v of row) dhNormSq += v * v;
    const dhNorm = Math.sqrt(dhNormSq);

    // dE norm per level: gradient signal from HIER contrastive
    const deNorms: string[] = [];
    _epochDE.forEach((rows, level) => {
      let normSq = 0;
      for (const row of rows) for (const v of row) normSq += v * v;
      deNorms.push(`L${level}=${Math.sqrt(normSq).toExponential(2)}`);
    });

    // MP param norms per level
    const mpParamNorms: string[] = [];
    for (const [level, lp] of levelParams) {
      let wcSq = 0, wpSq = 0;
      for (let h = 0; h < NUM_HEADS; h++) {
        for (const row of lp.W_child[h]) for (const v of row) wcSq += v * v;
        for (const row of lp.W_parent[h]) for (const v of row) wpSq += v * v;
      }
      let aUpSq = 0;
      for (const row of lp.a_upward) for (const v of row) aUpSq += v * v;
      const aUpNorm = Math.sqrt(aUpSq);
      let aDownSq = 0;
      for (const row of lp.a_downward) for (const v of row) aDownSq += v * v;
      const aDownNorm = Math.sqrt(aDownSq);
      mpParamNorms.push(
        `L${level}:|Wc|=${Math.sqrt(wcSq / NUM_HEADS).toFixed(2)} |Wp|=${Math.sqrt(wpSq / NUM_HEADS).toFixed(2)} ` +
        `|a↑|=${aUpNorm.toFixed(3)} |a↓|=${aDownNorm.toFixed(3)}`
      );
    }

    // H_final norm stats (are embeddings collapsing or exploding?)
    const hNorms: number[] = [];
    for (let i = 0; i < graph.l0Ids.length; i++) {
      let sq = 0;
      const hf = H_final[i];
      for (let d = 0; d < hf.length; d++) sq += hf[d] * hf[d];
      hNorms.push(Math.sqrt(sq));
    }
    hNorms.sort((a, b) => a - b);
    const hMin = hNorms[0];
    const hMedian = hNorms[Math.floor(hNorms.length / 2)];
    const hMax = hNorms[hNorms.length - 1];
    const hMean = hNorms.reduce((s, v) => s + v, 0) / hNorms.length;

    console.log(
      `  ${C.cyan}MP health${C.reset} ${C.dim}|dH|=${dhNorm.toExponential(2)} |dE|=[${deNorms.join(",")}]${C.reset}`
    );
    console.log(
      `  ${C.dim}   ${mpParamNorms.join("  ")}${C.reset}`
    );
    console.log(
      `  ${C.dim}   H_final norms: min=${hMin.toFixed(3)} med=${hMedian.toFixed(3)} mean=${hMean.toFixed(3)} max=${hMax.toFixed(3)}${C.reset}`
    );
  }
}

// ==========================================================================
// Report + Export
// ==========================================================================

const totalMs = Date.now() - trainingStartMs;
console.log(`\n${C.bold}${C.bgBlue} TRAINING REPORT ${C.reset}`);
console.log(`  ${C.dim}Time${C.reset}       ${formatDuration(totalMs)} total (${formatDuration(totalMs / EPOCHS)}/epoch)`);
console.log(`  ${C.dim}Peak RSS${C.reset}   ${formatMem()}`);
console.log(`  ${C.green}Best R@1${C.reset}   ${(bestHit1 * 100).toFixed(1)}% ${C.dim}(epoch ${bestEpoch})${C.reset}`);
console.log(`  ${C.green}Best MRR${C.reset}   ${bestMRR.toFixed(3)}`);

// Export final params (last epoch — best model already saved as shgat-params-ob-best.json)
const runId = new Date().toISOString().replace(/[:.]/g, "-");
const outputPath = resolve(GRU_DATA_DIR, `shgat-params-ob-${runId}.json`);
Deno.writeTextFileSync(outputPath, JSON.stringify(serializeParams()));
console.log(`\nParams (last epoch) \u2192 ${outputPath}`);
console.log(`Params (best R@1)   \u2192 ${resolve(GRU_DATA_DIR, "shgat-params-ob-best.json")}`);

const report = {
  timestamp: new Date().toISOString(),
  mode: "ob-manual-backward",
  config: { EPOCHS, BATCH_SIZE, KL_BATCH_SIZE, KL_ACCUM_STEPS, LEARNING_RATE, LR_WARMUP, TAU_START, TAU_END, SEED, USE_KL, KL_WARMUP, KL_WEIGHT_PLATEAU, KL_ISOLATE_KHEAD, MP_LR_SCALE, EVAL_EVERY, NUM_HEADS, HEAD_DIM },
  dataset: { nodes: ds.nodes.length, leaves: ds.leafIds.length, embDim: ds.embeddingDim, prodTrain: ds.prodTrain.length, prodTest: ds.prodTest.length, n8nTrain: ds.n8nTrain.length, n8nEval: ds.n8nEval.length },
  results: { bestHit1, bestMRR, bestEpoch, totalTimeSec: +(totalMs / 1000).toFixed(1), peakRssMB: Math.round(Deno.memoryUsage().rss / 1024 / 1024) },
  epochLog,
};
const reportPath = resolve(GRU_DATA_DIR, `shgat-training-report-ob-${runId}.json`);
Deno.writeTextFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(`Report \u2192 ${reportPath}`);
console.log("\n=== OB Training complete ===");
