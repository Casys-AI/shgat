#!/usr/bin/env npx tsx
/**
 * SHGAT-TF Retrainer — Train on expanded vocab (bench dataset)
 *
 * Re-trains SHGAT on the same data used by benchmark-e2e.ts:
 * - 1,901 tools (BGE-M3 embeddings)
 * - 7,234 n8n workflows as L1 capabilities (description embeddings)
 * - 44K traces (prod + n8n soft targets)
 *
 * Uses SHGATBuilder + AutogradTrainer with PER + curriculum + annealing.
 * Saves params via exportParams() for benchmark-e2e.ts consumption.
 *
 * ──────────────────────────────────────────────────────────────────────
 * TRAINING STRATEGY (v2, 2026-02-13)
 * ──────────────────────────────────────────────────────────────────────
 *
 * 1. LR schedule: linear warmup (0→peak over 3 epochs) + cosine decay
 * 2. InfoNCE: 32 negatives (up from 8) for harder contrastive learning
 * 3. KL (ON by default): sampled softmax (128 negs instead of 1901 full)
 *    with weight scheduling: 0 during warmup, then ramp to 0.2
 * 4. Subgraph MP: Ancestral Path Sampling (K=16) in the tape
 *    so W_up/W_down receive gradients without OOM
 * 5. MP LR scale: 0.1x for MP weights (noisy subgraph gradients)
 *
 * Usage:
 *   cd lib/shgat-tf
 *   NODE_OPTIONS="--max-old-space-size=8192 --expose-gc" npx tsx tools/train-from-bench.ts \
 *     --epochs 15 --lr 0.01 --kl --kl-negs 128
 *
 * @module shgat-tf/tools/train-from-bench
 */

import { writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { encode as msgpackEncode } from "@msgpack/msgpack";
import pako from "pako";
import postgres from "postgres";

import { tf } from "../dist-node/src/tf/backend.ts";
import { SHGATBuilder } from "../dist-node/src/core/builder.ts";
import type { SoftTargetExample, TrainingExample } from "../dist-node/src/core/types.ts";
import { annealBeta, annealTemperature, PERBuffer } from "../dist-node/src/training/per-buffer.ts";
import {
  type BenchDataset,
  type CompactExample,
  loadBenchDataset,
} from "../../gru/src/shgat/bench-dataset.ts";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const GRU_DATA_DIR = resolve(__dirname, "../../gru/data");
const N8N_SHGAT_PAIRS_PATH = resolve(GRU_DATA_DIR, "n8n-shgat-contrastive-pairs.json");
const EXPANDED_VOCAB_PATH = resolve(GRU_DATA_DIR, "expanded-vocab.json");

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

const cliArgs = process.argv.slice(2);

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
SHGAT-TF Retrainer — expanded vocab

Options:
  --epochs <n>        Training epochs (default: 15)
  --batch-size <n>    Batch size (default: 32)
  --lr <n>            Peak learning rate (default: 0.01)
  --lr-warmup <n>     LR warmup epochs (0→lr, default: 3)
  --temperature <n>   InfoNCE temperature start (default: 0.10)
  --num-negatives <n> Negatives per InfoNCE example (default: 32)
  --seed <n>          Random seed (default: 42)
  --per / --no-per    PER sampling (default: on)
  --curriculum / --no-curriculum  Curriculum learning (default: on)
  --kl / --no-kl      KL divergence on n8n soft targets (default: ON)
  --kl-warmup <n>     KL warmup epochs (weight=0 during warmup, default: 3)
  --kl-weight <n>     KL loss weight at plateau (default: 0.2)
  --kl-negs <n>       Sampled negatives for KL (0 = full vocab, default: 128)
  --max-n8n <n>       Cap n8n training examples (default: 0 = all)
  --max-workflows <n> Cap n8n workflow graph nodes (default: 99999 = all)
  --eval-chunk <n>    Tools per eval scoring chunk (default: 256, 0 = full)
  --eval-every <n>    Run full eval every N epochs (default: 5)
  --mp-lr-scale <n>   LR scale for MP weights W_up/W_down (default: 0.1)
  --write-db          Save params to DB shgat_params
  --help              Show this help
`);
  process.exit(0);
}

const EPOCHS = parseInt(getArg("epochs", "15"), 10);
const BATCH_SIZE = parseInt(getArg("batch-size", "32"), 10);
const LEARNING_RATE = parseFloat(getArg("lr", "0.01"));
const LR_WARMUP = parseInt(getArg("lr-warmup", "3"), 10);
const TAU_START = parseFloat(getArg("temperature", "0.10"));
const TAU_END = 0.06;
const INFO_NUM_NEGATIVES = parseInt(getArg("num-negatives", "32"), 10);
const SEED = parseInt(getArg("seed", "42"), 10);
const USE_PER = boolArg("per", true);
const USE_CURRICULUM = boolArg("curriculum", true);
const WRITE_DB = boolArg("write-db", false);
const USE_KL = boolArg("kl", true); // ON by default — sampled KL with weight scheduling
const KL_WARMUP = parseInt(getArg("kl-warmup", "3"), 10);
const KL_WEIGHT_PLATEAU = parseFloat(getArg("kl-weight", "0.2"));
const KL_NUM_NEGS = parseInt(getArg("kl-negs", "128"), 10); // 0 = full vocab
const MAX_N8N = parseInt(getArg("max-n8n", "0"), 10); // 0 = no cap
const MAX_WORKFLOWS = parseInt(getArg("max-workflows", "99999"), 10); // cap n8n workflow nodes (99999 = all)
const EVAL_CHUNK = parseInt(getArg("eval-chunk", "256"), 10); // tools per scoring chunk (0 = full)
const EVAL_EVERY = Math.max(1, parseInt(getArg("eval-every", "5"), 10));
const MP_LR_SCALE = parseFloat(getArg("mp-lr-scale", "0.1"));

// ---------------------------------------------------------------------------
// Seeded RNG (for negative sampling)
// ---------------------------------------------------------------------------

function seededRng(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = seededRng(SEED);

function shuffleInPlace<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ---------------------------------------------------------------------------
// LR and KL weight scheduling
// ---------------------------------------------------------------------------

/**
 * Learning rate with linear warmup + cosine decay.
 *
 * Epoch 0..warmup-1:  LR ramps linearly from lr_min to lr_peak
 * Epoch warmup..total: LR decays via cosine from lr_peak to lr_min
 */
function scheduleLR(epoch: number, totalEpochs: number, lrPeak: number, warmupEpochs: number): number {
  const lrMin = lrPeak * 0.01; // 1% of peak as floor
  if (epoch < warmupEpochs) {
    // Linear warmup: lrMin → lrPeak
    const progress = (epoch + 1) / Math.max(warmupEpochs, 1);
    return lrMin + (lrPeak - lrMin) * progress;
  }
  // Cosine decay: lrPeak → lrMin
  const decayEpochs = totalEpochs - warmupEpochs;
  const decayProgress = Math.min((epoch - warmupEpochs) / Math.max(decayEpochs - 1, 1), 1.0);
  return lrMin + (lrPeak - lrMin) * 0.5 * (1 + Math.cos(Math.PI * decayProgress));
}

/**
 * KL weight scheduling: 0 during warmup, then linear ramp to plateau.
 *
 * Epoch 0..warmup-1:     weight = 0 (InfoNCE only)
 * Epoch warmup..2*warmup: weight ramps linearly from 0 to plateau
 * Epoch 2*warmup+:       weight = plateau
 */
function scheduleKLWeight(epoch: number, warmupEpochs: number, plateau: number): number {
  if (epoch < warmupEpochs) return 0;
  const rampStart = warmupEpochs;
  const rampEnd = warmupEpochs * 2;
  if (epoch >= rampEnd) return plateau;
  const progress = (epoch - rampStart) / Math.max(rampEnd - rampStart, 1);
  return plateau * progress;
}

// ---------------------------------------------------------------------------
// Eval helpers (chunked scoring to reduce peak memory)
// ---------------------------------------------------------------------------

type ScoreFn = (intentEmb: number[], nodeIds: string[]) => number[];

function computeRankFromScores(scores: number[], targetIdx: number): number {
  const targetScore = scores[targetIdx];
  let rank = 1;
  for (let i = 0; i < scores.length; i++) {
    if (i !== targetIdx && scores[i] > targetScore) rank++;
  }
  return rank;
}

function computeRankChunked(
  scoreFn: ScoreFn,
  intentEmbedding: number[],
  toolIds: string[],
  targetIdx: number,
  chunkSize: number,
): number {
  if (targetIdx < 0) return -1;
  if (chunkSize <= 0 || chunkSize >= toolIds.length) {
    const scores = scoreFn(intentEmbedding, toolIds);
    return computeRankFromScores(scores, targetIdx);
  }

  let targetScore: number | null = null;
  for (let start = 0; start < toolIds.length; start += chunkSize) {
    const chunkIds = toolIds.slice(start, start + chunkSize);
    const scores = scoreFn(intentEmbedding, chunkIds);
    const idxInChunk = targetIdx - start;
    if (idxInChunk >= 0 && idxInChunk < scores.length) {
      targetScore = scores[idxInChunk];
      break;
    }
  }
  if (targetScore === null || !Number.isFinite(targetScore)) return -1;

  let rank = 1;
  for (let start = 0; start < toolIds.length; start += chunkSize) {
    const chunkIds = toolIds.slice(start, start + chunkSize);
    const scores = scoreFn(intentEmbedding, chunkIds);
    for (let i = 0; i < scores.length; i++) {
      const globalIdx = start + i;
      if (globalIdx !== targetIdx && scores[i] > targetScore) rank++;
    }
  }
  return rank;
}

function computeKLDivergenceFromScores(
  scores: number[],
  softTargetSparse: [number, number][],
): number {
  let maxS = -Infinity;
  for (const s of scores) maxS = Math.max(maxS, s);
  let sumExp = 0;
  for (const s of scores) sumExp += Math.exp(s - maxS);
  if (sumExp <= 0) return 0;

  let kl = 0;
  for (const [idx, p] of softTargetSparse) {
    if (p > 1e-8 && idx < scores.length) {
      const q = Math.max(Math.exp(scores[idx] - maxS) / sumExp, 1e-8);
      kl += p * Math.log(p / q);
    }
  }
  return kl;
}

function computeKLDivergenceChunked(
  scoreFn: ScoreFn,
  intentEmbedding: number[],
  toolIds: string[],
  softTargetSparse: [number, number][],
  chunkSize: number,
): number {
  if (chunkSize <= 0 || chunkSize >= toolIds.length) {
    const scores = scoreFn(intentEmbedding, toolIds);
    return computeKLDivergenceFromScores(scores, softTargetSparse);
  }

  const targetIdxSet = new Set<number>(softTargetSparse.map(([idx]) => idx));
  const sparseScores = new Map<number, number>();

  let maxS = -Infinity;
  for (let start = 0; start < toolIds.length; start += chunkSize) {
    const chunkIds = toolIds.slice(start, start + chunkSize);
    const scores = scoreFn(intentEmbedding, chunkIds);
    for (let i = 0; i < scores.length; i++) {
      const globalIdx = start + i;
      const s = scores[i];
      if (s > maxS) maxS = s;
      if (targetIdxSet.has(globalIdx)) sparseScores.set(globalIdx, s);
    }
  }

  let sumExp = 0;
  for (let start = 0; start < toolIds.length; start += chunkSize) {
    const chunkIds = toolIds.slice(start, start + chunkSize);
    const scores = scoreFn(intentEmbedding, chunkIds);
    for (const s of scores) sumExp += Math.exp(s - maxS);
  }
  if (sumExp <= 0) return 0;

  let kl = 0;
  for (const [idx, p] of softTargetSparse) {
    if (p > 1e-8) {
      const s = sparseScores.get(idx);
      if (s !== undefined) {
        const q = Math.max(Math.exp(s - maxS) / sumExp, 1e-8);
        kl += p * Math.log(p / q);
      }
    }
  }
  return kl;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) throw new Error("DATABASE_URL env var required");

console.log("=== SHGAT-TF Retrainer (expanded vocab) ===");
console.log(`    Epochs: ${EPOCHS}, Batch: ${BATCH_SIZE}, LR: ${LEARNING_RATE} (warmup: ${LR_WARMUP}ep)`);
console.log(
  `    τ: ${TAU_START}→${TAU_END}, PER: ${USE_PER}, Curriculum: ${USE_CURRICULUM}`,
);
console.log(
  `    KL: ${USE_KL}, KL warmup: ${KL_WARMUP}ep, KL weight: ${KL_WEIGHT_PLATEAU}, KL negs: ${KL_NUM_NEGS || "full"}`,
);
console.log(
  `    InfoNCE negatives: ${INFO_NUM_NEGATIVES}, MP LR scale: ${MP_LR_SCALE}`,
);
console.log(
  `    Seed: ${SEED}, Write DB: ${WRITE_DB}, EvalChunk: ${EVAL_CHUNK}, EvalEvery: ${EVAL_EVERY}\n`,
);

// 1. Load dataset via shared loader
const sql = postgres(DATABASE_URL);
const ds: BenchDataset = await loadBenchDataset(sql, {
  expandedVocabPath: EXPANDED_VOCAB_PATH,
  n8nPairsPath: N8N_SHGAT_PAIRS_PATH,
  n8nDataDir: GRU_DATA_DIR,
  maxN8nWorkflows: MAX_WORKFLOWS,
  prodOversample: 3,
  splitSeed: SEED,
});

const totalProd = ds.prodTrain.length + ds.prodTest.length;
const totalN8n = ds.n8nTrain.length + ds.n8nEval.length;
const prodSplitPct = ((ds.prodTrain.length / totalProd) * 100).toFixed(0);
const n8nSplitPct = ((ds.n8nTrain.length / totalN8n) * 100).toFixed(0);

console.log(`\n${"=".repeat(60)}`);
console.log("  DATASET SUMMARY");
console.log("=".repeat(60));
console.log(
  `  Graph:        ${ds.nodes.length} nodes (${ds.leafIds.length} leaves, ${
    ds.nodes.length - ds.leafIds.length
  } non-leaf)`,
);
console.log(`  Embedding:    ${ds.embeddingDim}D (BGE-M3)`);
console.log(`  ─── Production traces ───`);
console.log(
  `  Total:        ${totalProd} examples (${ds.prodTrain.length / 3} raw × 3x oversample)`,
);
console.log(
  `  Train/Test:   ${ds.prodTrain.length} / ${ds.prodTest.length}  (${prodSplitPct}/${
    100 - +prodSplitPct
  } split)`,
);
console.log(`  ─── N8n soft targets ───`);
console.log(`  Total:        ${totalN8n} examples (from ${ds.workflowToolLists.length} workflows)`);
console.log(
  `  Train/Eval:   ${ds.n8nTrain.length} / ${ds.n8nEval.length}  (${n8nSplitPct}/${
    100 - +n8nSplitPct
  } split)`,
);
console.log("=".repeat(60));

// 2. Build SHGAT via builder (backend auto-selected for training)
console.log("\n[SHGAT] Building graph...");
const shgat = await SHGATBuilder.create()
  .nodes(ds.nodes)
  .training({ learningRate: LEARNING_RATE, batchSize: BATCH_SIZE, temperature: TAU_START })
  .architecture({ numHeads: 16, headDim: 64, hiddenDim: 1024, embeddingDim: ds.embeddingDim, mpLearningRateScale: MP_LR_SCALE })
  .backend("training")
  .build();

// Configure subgraph sampling K for mini-batch MP
shgat.setSubgraphK(16);
console.log(`  Message passing: ${shgat.hasMessagePassing ? "ON" : "OFF"} (subgraph K=16)`);

// 3. Convert prod examples → TrainingExample format
//    Shared loader returns TransitionExample; we need TrainingExample for SHGAT-TF
function toTrainingExample(
  ex: { intentEmbedding: number[]; contextToolIds: string[]; targetToolId: string },
  allLeafIds: string[],
  tier: "easy" | "medium" | "hard",
): TrainingExample {
  // Pick negatives from tier
  const negatives: string[] = [];
  const pool = [...allLeafIds];
  const exclude = new Set([ex.targetToolId, ...ex.contextToolIds]);
  const filtered = pool.filter((id) => !exclude.has(id));
  shuffleInPlace(filtered);

  if (USE_CURRICULUM) {
    const third = Math.floor(filtered.length / 3);
    let slice: string[];
    if (tier === "easy") slice = filtered.slice(third * 2); // last third
    else if (tier === "hard") slice = filtered.slice(0, third); // first third
    else slice = filtered.slice(third, third * 2); // middle
    shuffleInPlace(slice);
    for (let i = 0; i < INFO_NUM_NEGATIVES && i < slice.length; i++) {
      negatives.push(slice[i]);
    }
  } else {
    for (let i = 0; i < INFO_NUM_NEGATIVES && i < filtered.length; i++) {
      negatives.push(filtered[i]);
    }
  }

  return {
    intentEmbedding: ex.intentEmbedding,
    contextTools: ex.contextToolIds,
    candidateId: ex.targetToolId,
    outcome: 1,
    negativeCapIds: negatives,
  };
}

/** Convert CompactExample (already sparse) to SoftTargetExample for trainBatchKL. */
function compactToSoftTarget(ex: CompactExample): SoftTargetExample {
  return {
    intentEmbedding: Array.from(ex.intentEmbedding),
    softTargetSparse: ex.softTargetSparse,
  };
}

// 4. Build combined example pool
console.log("\n[Training] Preparing examples...");

// Prod → TrainingExample
const prodTrainingExamples = ds.prodTrain.map((ex) => toTrainingExample(ex, ds.leafIds, "medium"));
console.log(`  Prod training examples: ${prodTrainingExamples.length}`);

// N8n CompactExamples (already sparse — no need to convert upfront)
let n8nCompact = ds.n8nTrain;
if (MAX_N8N > 0 && n8nCompact.length > MAX_N8N) {
  shuffleInPlace(n8nCompact);
  n8nCompact = n8nCompact.slice(0, MAX_N8N);
  console.log(`  N8n capped to ${MAX_N8N} examples`);
}
console.log(`  N8n soft target examples: ${n8nCompact.length} (compact/sparse)`);

// All InfoNCE examples (prod only — n8n uses KL path)
const allInfoNCE: TrainingExample[] = [...prodTrainingExamples];

// Initialize PER buffer for InfoNCE only
let perInfoNCE: PERBuffer<TrainingExample> | null = null;
if (USE_PER) {
  perInfoNCE = new PERBuffer(allInfoNCE, { alpha: 0.6, beta: 0.4, epsilon: 0.01, maxPriority: 25 });
}

// 5. Training loop
console.log(`\n=== Training: ${EPOCHS} epochs ===\n`);

interface EpochReport {
  epoch: number;
  // InfoNCE metrics (prod examples)
  infoLoss: number;
  infoAcc: number;
  infoGradNorm: number;
  // KL metrics (n8n examples)
  klLoss: number;
  klGradNorm: number;
  // Combined
  avgLoss: number;
  // Test metrics (full-vocab ranking on prod test set)
  testHit1: number;
  testHit3: number;
  testHit5: number;
  testMRR: number;
  testTermAcc: number;
  // N8n eval metrics
  n8nEvalKL: number;
  // Hyperparams
  tau: number;
  beta: number;
  tier: string;
  lr: number;
  klWeight: number;
  // Runtime
  elapsedMs: number;
  rssMB: number;
}
const epochReports: EpochReport[] = [];
const trainingStartMs = Date.now();

let bestTestHit1 = 0;
let bestMRR = 0;
let bestEpoch = 0;
let currTier: "easy" | "medium" | "hard" = "easy";
let lastTestHit1 = 0;
let lastTestHit3 = 0;
let lastTestHit5 = 0;
let lastTestMRR = 0;
let lastTestTermAcc = 0;
let lastN8nEvalKL = 0;

for (let epoch = 0; epoch < EPOCHS; epoch++) {
  const t0 = Date.now();
  const beta = USE_PER ? annealBeta(epoch, EPOCHS, 0.4) : 1.0;
  const tau = annealTemperature(epoch, EPOCHS, TAU_START, TAU_END);
  shgat.setTemperature(tau);

  // LR scheduling: warmup + cosine decay
  const epochLR = scheduleLR(epoch, EPOCHS, LEARNING_RATE, LR_WARMUP);
  shgat.setLearningRate(epochLR);

  // KL weight scheduling: 0 during warmup, then ramp to plateau
  const klWeight = USE_KL ? scheduleKLWeight(epoch, KL_WARMUP, KL_WEIGHT_PLATEAU) : 0;

  // Diagnostic: track tensor count at each phase
  const mem0 = tf.memory();
  console.log(
    `  [diag] epoch ${epoch + 1} start: ${mem0.numTensors} tensors, ${
      (mem0.numBytes / 1e6).toFixed(0)
    }MB tf | LR=${epochLR.toFixed(5)} klW=${klWeight.toFixed(3)}`,
  );

  // NOTE: precomputeEnrichedEmbeddings is called AFTER InfoNCE (not before)
  // so InfoNCE can use full MP-in-tape without competing for RAM.

  // --- Separate metric accumulators ---
  let infoLossSum = 0, infoAccSum = 0, infoGradSum = 0, infoBatches = 0;
  let klLossSum = 0, klGradSum = 0, klBatches = 0;

  // --- InfoNCE batches (prod examples) ---
  const numInfoNCE = Math.ceil(allInfoNCE.length / BATCH_SIZE);
  for (let b = 0; b < numInfoNCE; b++) {
    let batch: TrainingExample[];
    if (perInfoNCE) {
      const sample = perInfoNCE.sample(BATCH_SIZE, beta);
      batch = sample.items.map((ex) =>
        toTrainingExample(
          {
            intentEmbedding: ex.intentEmbedding,
            contextToolIds: ex.contextTools,
            targetToolId: ex.candidateId,
          },
          ds.leafIds,
          currTier,
        )
      );
      // C4 fix: scope catches leaked tensors from trainBatch (subgraph, MP intermediates)
      tf.engine().startScope();
      const metrics = await shgat.trainBatch(batch);
      tf.engine().endScope();
      infoLossSum += metrics.loss;
      infoAccSum += metrics.accuracy;
      infoGradSum += metrics.gradientNorm;
      infoBatches++;
      const tdErrors = batch.map(() => metrics.loss);
      perInfoNCE.updatePriorities(sample.indices, tdErrors);
    } else {
      const start = b * BATCH_SIZE;
      batch = allInfoNCE.slice(start, start + BATCH_SIZE);
      if (batch.length === 0) continue;
      batch = batch.map((ex) =>
        toTrainingExample(
          {
            intentEmbedding: ex.intentEmbedding,
            contextToolIds: ex.contextTools,
            targetToolId: ex.candidateId,
          },
          ds.leafIds,
          currTier,
        )
      );
      // C4 fix: scope catches leaked tensors from trainBatch (subgraph, MP intermediates)
      tf.engine().startScope();
      const metrics = await shgat.trainBatch(batch);
      tf.engine().endScope();
      infoLossSum += metrics.loss;
      infoAccSum += metrics.accuracy;
      infoGradSum += metrics.gradientNorm;
      infoBatches++;
    }
    if ((b + 1) % 10 === 0 || b === numInfoNCE - 1) {
      const mem = (process.memoryUsage().rss / 1024 / 1024).toFixed(0);
      process.stdout.write(
        `\r    InfoNCE ${b + 1}/${numInfoNCE} | loss=${(infoLossSum / infoBatches).toFixed(4)} ` +
          `acc=${(infoAccSum / infoBatches * 100).toFixed(1)}% ` +
          `∇=${(infoGradSum / infoBatches).toFixed(2)} | ${mem}MB`,
      );
    }
  }
  if (numInfoNCE > 0) process.stdout.write("\n");
  const memInfo = tf.memory();
  console.log(
    `  [diag] after InfoNCE: ${memInfo.numTensors} tensors (+${
      memInfo.numTensors - mem0.numTensors
    })`,
  );

  // NOTE: precompute enriched embeddings is deferred to AFTER KL (before eval only).
  // Running MP forward here would OOM on top of InfoNCE's 18GB RSS footprint.
  // KL trains with raw embeddings (only W_k/W_intent get gradients — same as v4).

  // --- KL batches (n8n soft targets — converted lazily per batch) ---
  // Uses sampled KL: score positive + KL_NUM_NEGS negatives instead of full vocab.
  // KL weight scheduling: 0 during warmup, then ramp to KL_WEIGHT_PLATEAU.
  if (klWeight > 0 && n8nCompact.length > 0) {
    shuffleInPlace(n8nCompact);
    const numKL = Math.ceil(n8nCompact.length / BATCH_SIZE);

    // Pre-select sampled tool IDs for this epoch (if using sampled KL)
    // For sampled KL, we score against a subset of tools per batch instead of full vocab.
    const useSampledKL = KL_NUM_NEGS > 0 && KL_NUM_NEGS < ds.leafIds.length;

    for (let b = 0; b < numKL; b++) {
      const start = b * BATCH_SIZE;
      const slice = n8nCompact.slice(start, start + BATCH_SIZE);
      if (slice.length === 0) continue;
      const batch: SoftTargetExample[] = slice.map(compactToSoftTarget);

      let metrics;
      if (useSampledKL) {
        // Sampled KL: for each example, keep the non-zero soft target tools + sample KL_NUM_NEGS random negatives
        // Build a common subset of tool indices for this batch
        const nonZeroIdxs = new Set<number>();
        for (const ex of batch) {
          for (const [idx] of ex.softTargetSparse) {
            nonZeroIdxs.add(idx);
          }
        }
        // Sample additional negatives
        const negPool: number[] = [];
        for (let i = 0; i < ds.leafIds.length; i++) {
          if (!nonZeroIdxs.has(i)) negPool.push(i);
        }
        shuffleInPlace(negPool);
        const sampledNegs = negPool.slice(0, KL_NUM_NEGS);
        const subsetIdxs = [...nonZeroIdxs, ...sampledNegs].sort((a, b) => a - b);

        // Build mapping: original index → position in subset
        const idxMap = new Map<number, number>();
        for (let i = 0; i < subsetIdxs.length; i++) {
          idxMap.set(subsetIdxs[i], i);
        }

        // Remap soft targets to subset indices
        const remappedBatch: SoftTargetExample[] = batch.map((ex) => ({
          intentEmbedding: ex.intentEmbedding,
          softTargetSparse: ex.softTargetSparse
            .filter(([idx]) => idxMap.has(idx))
            .map(([idx, prob]) => [idxMap.get(idx)!, prob] as [number, number]),
        }));

        // Build subset tool IDs
        const subsetToolIds = subsetIdxs.map((i) => ds.leafIds[i]);

        // C4 fix: scope catches leaked tensors from trainBatchKL
        tf.engine().startScope();
        metrics = await shgat.trainBatchKL(remappedBatch, subsetToolIds, tau, klWeight);
        tf.engine().endScope();
      } else {
        // Full vocab KL (original behavior, weighted)
        // C4 fix: scope catches leaked tensors from trainBatchKL
        tf.engine().startScope();
        metrics = await shgat.trainBatchKL(batch, ds.leafIds, tau, klWeight);
        tf.engine().endScope();
      }

      klLossSum += metrics.klLoss;
      klGradSum += metrics.gradientNorm;
      klBatches++;
      if ((b + 1) % 100 === 0 || b === numKL - 1) {
        const mem = (process.memoryUsage().rss / 1024 / 1024).toFixed(0);
        const eta = klBatches > 0
          ? ((Date.now() - t0) / klBatches * (numKL - b - 1) / 1000).toFixed(0)
          : "?";
        process.stdout.write(
          `\r    KL ${b + 1}/${numKL} | klLoss=${(klLossSum / klBatches).toFixed(4)} ` +
            `∇=${(klGradSum / klBatches).toFixed(2)} w=${klWeight.toFixed(3)} | ${mem}MB | ETA ~${eta}s`,
        );
      }

      // Force GC every 200 batches to prevent RSS creep
      if ((b + 1) % 200 === 0) {
        // deno-lint-ignore no-explicit-any
        const gc = (global as any).gc;
        if (typeof gc === "function") gc();
      }
    }
    process.stdout.write("\n");
  }

  const memKL = tf.memory();
  console.log(
    `  [diag] after KL: ${memKL.numTensors} tensors (+${memKL.numTensors - memInfo.numTensors})`,
  );

  // Decay PER priorities
  if (perInfoNCE) perInfoNCE.decayPriorities(0.95);

  const infoLoss = infoBatches > 0 ? infoLossSum / infoBatches : 0;
  const infoAcc = infoBatches > 0 ? infoAccSum / infoBatches : 0;
  const infoGrad = infoBatches > 0 ? infoGradSum / infoBatches : 0;
  const klLoss = klBatches > 0 ? klLossSum / klBatches : 0;
  const klGrad = klBatches > 0 ? klGradSum / klBatches : 0;
  const totalBatches = infoBatches + klBatches;
  const avgLoss = totalBatches > 0 ? (infoLossSum + klLossSum) / totalBatches : 0;

  const shouldEval = (epoch + 1) % EVAL_EVERY === 0 || epoch === EPOCHS - 1;

  // --- Full-vocab test evaluation: Hit@1, Hit@3, Hit@5, MRR, termAcc ---
  let testHit1 = lastTestHit1;
  let testHit3 = lastTestHit3;
  let testHit5 = lastTestHit5;
  let testMRR = lastTestMRR;
  let testTermAcc = lastTestTermAcc;
  let n8nEvalKL = lastN8nEvalKL;

  if (shouldEval) {
    // Force GC before eval to reclaim KL-phase memory (RSS ~20GB at this point)
    // deno-lint-ignore no-explicit-any
    const gc = (global as any).gc;
    if (typeof gc === "function") gc();

    // Re-compute enriched embeddings with UPDATED params before eval
    shgat.precomputeEnrichedEmbeddings();

    const scoreFn: ScoreFn = (intent, ids) => shgat.score(intent, ids);

    if (ds.prodTest.length > 0) {
      let hit1 = 0, hit3 = 0, hit5 = 0, rr = 0, termCorrect = 0, termTotal = 0;
      const testSample = ds.prodTest.slice(0, Math.min(ds.prodTest.length, 500));
      for (const ex of testSample) {
        const targetIdx = ds.leafIds.indexOf(ex.targetToolId);
        const rank = computeRankChunked(
          scoreFn,
          ex.intentEmbedding,
          ds.leafIds,
          targetIdx,
          EVAL_CHUNK,
        );
        if (rank < 0) continue;
        if (rank <= 1) hit1++;
        if (rank <= 3) hit3++;
        if (rank <= 5) hit5++;
        rr += 1 / rank;
        if ("isTerminal" in ex && (ex as { isTerminal: number }).isTerminal === 1) {
          termTotal++;
          if (rank <= 1) termCorrect++;
        }
      }
      testHit1 = hit1 / testSample.length;
      testHit3 = hit3 / testSample.length;
      testHit5 = hit5 / testSample.length;
      testMRR = rr / testSample.length;
      testTermAcc = termTotal > 0 ? termCorrect / termTotal : 0;
    }

    // --- N8n eval: KL divergence on held-out eval set (score-only, no training) ---
    if (ds.n8nEval.length > 0) {
      const evalSample = ds.n8nEval.slice(0, Math.min(ds.n8nEval.length, 200));
      let klSum = 0;
      for (const ex of evalSample) {
        const st = compactToSoftTarget(ex);
        const kl = computeKLDivergenceChunked(
          scoreFn,
          st.intentEmbedding,
          ds.leafIds,
          st.softTargetSparse,
          EVAL_CHUNK,
        );
        klSum += kl;
      }
      n8nEvalKL = klSum / evalSample.length;
    }

    lastTestHit1 = testHit1;
    lastTestHit3 = testHit3;
    lastTestHit5 = testHit5;
    lastTestMRR = testMRR;
    lastTestTermAcc = testTermAcc;
    lastN8nEvalKL = n8nEvalKL;
  }

  // Curriculum tier update
  if (USE_CURRICULUM) {
    if (testHit1 < 0.35) currTier = "easy";
    else if (testHit1 > 0.55) currTier = "hard";
    else currTier = "medium";
  }

  if (testHit1 > bestTestHit1) {
    bestTestHit1 = testHit1;
    bestEpoch = epoch + 1;
  }
  if (testMRR > bestMRR) bestMRR = testMRR;

  const elapsedMs = Date.now() - t0;
  const rssMB = Math.round(process.memoryUsage().rss / 1024 / 1024);
  epochReports.push({
    epoch: epoch + 1,
    infoLoss,
    infoAcc,
    infoGradNorm: infoGrad,
    klLoss,
    klGradNorm: klGrad,
    avgLoss,
    testHit1,
    testHit3,
    testHit5,
    testMRR,
    testTermAcc,
    n8nEvalKL,
    tau,
    beta,
    tier: currTier,
    lr: epochLR,
    klWeight,
    elapsedMs,
    rssMB,
  });

  const elapsed = (elapsedMs / 1000).toFixed(0);
  const totalElapsed = ((Date.now() - trainingStartMs) / 1000).toFixed(0);
  const eta = epoch > 0
    ? (((Date.now() - trainingStartMs) / (epoch + 1)) * (EPOCHS - epoch - 1) / 1000 / 60).toFixed(1)
    : "?";

  console.log(
    `\n  ┌─ Epoch ${
      String(epoch + 1).padStart(2)
    }/${EPOCHS}  (${elapsed}s, total ${totalElapsed}s, ETA ~${eta}min)  τ=${tau.toFixed(4)} β=${
      beta.toFixed(2)
    } LR=${epochLR.toFixed(5)} tier=${currTier}  ${rssMB}MB`,
  );
  console.log(
    `  │ InfoNCE  loss=${infoLoss.toFixed(4)}  acc=${(infoAcc * 100).toFixed(1)}%  ∇=${
      infoGrad.toFixed(2)
    }  (${infoBatches} batches, ${INFO_NUM_NEGATIVES} negs)`,
  );
  console.log(
    `  │ KL       loss=${klLoss.toFixed(4)}  ∇=${
      klGrad.toFixed(2)
    }  (${klBatches} batches)  w=${klWeight.toFixed(3)}  n8nEval=${n8nEvalKL.toFixed(4)}`,
  );
  console.log(
    `  │ Test     Hit@1=${(testHit1 * 100).toFixed(1)}%  Hit@3=${
      (testHit3 * 100).toFixed(1)
    }%  Hit@5=${(testHit5 * 100).toFixed(1)}%  MRR=${testMRR.toFixed(3)}  termAcc=${
      (testTermAcc * 100).toFixed(1)
    }%`,
  );
  console.log(
    `  └─ Best    Hit@1=${(bestTestHit1 * 100).toFixed(1)}%  MRR=${
      bestMRR.toFixed(3)
    }  (epoch ${bestEpoch})`,
  );
}

const totalTrainingMs = Date.now() - trainingStartMs;

// 6. Training report
const report = {
  timestamp: new Date().toISOString(),
  config: {
    epochs: EPOCHS,
    batchSize: BATCH_SIZE,
    learningRate: LEARNING_RATE,
    lrWarmup: LR_WARMUP,
    tauStart: TAU_START,
    tauEnd: TAU_END,
    numNegatives: INFO_NUM_NEGATIVES,
    seed: SEED,
    per: USE_PER,
    curriculum: USE_CURRICULUM,
    kl: USE_KL,
    klWarmup: KL_WARMUP,
    klWeightPlateau: KL_WEIGHT_PLATEAU,
    klNumNegs: KL_NUM_NEGS,
    mpLrScale: MP_LR_SCALE,
  },
  dataset: {
    totalNodes: ds.nodes.length,
    leaves: ds.leafIds.length,
    embeddingDim: ds.embeddingDim,
    prodTrain: ds.prodTrain.length,
    prodTest: ds.prodTest.length,
    n8nTrain: ds.n8nTrain.length,
    n8nEval: ds.n8nEval.length,
    infonceExamples: allInfoNCE.length,
    klExamples: n8nCompact.length,
  },
  results: {
    bestTestHit1,
    bestMRR,
    bestEpoch,
    finalLoss: epochReports[epochReports.length - 1]?.avgLoss ?? NaN,
    finalInfoLoss: epochReports[epochReports.length - 1]?.infoLoss ?? NaN,
    finalInfoAcc: epochReports[epochReports.length - 1]?.infoAcc ?? NaN,
    finalKLLoss: epochReports[epochReports.length - 1]?.klLoss ?? NaN,
    finalHit1: epochReports[epochReports.length - 1]?.testHit1 ?? NaN,
    finalHit3: epochReports[epochReports.length - 1]?.testHit3 ?? NaN,
    finalHit5: epochReports[epochReports.length - 1]?.testHit5 ?? NaN,
    finalMRR: epochReports[epochReports.length - 1]?.testMRR ?? NaN,
    finalTermAcc: epochReports[epochReports.length - 1]?.testTermAcc ?? NaN,
    finalN8nEvalKL: epochReports[epochReports.length - 1]?.n8nEvalKL ?? NaN,
    totalTrainingSec: +(totalTrainingMs / 1000).toFixed(1),
    avgEpochSec: +(totalTrainingMs / EPOCHS / 1000).toFixed(1),
  },
  epochs: epochReports,
};

const last = epochReports[epochReports.length - 1];
console.log("\n" + "=".repeat(70));
console.log("  TRAINING REPORT");
console.log("=".repeat(70));
console.log(`  Date:           ${report.timestamp}`);
console.log(`  Epochs:         ${EPOCHS}`);
console.log(`  Batch size:     ${BATCH_SIZE}`);
console.log(`  Learning rate:  ${LEARNING_RATE}`);
console.log(`  τ:              ${TAU_START} → ${TAU_END} (cosine)`);
console.log(`  PER:            ${USE_PER}`);
console.log(`  Curriculum:     ${USE_CURRICULUM}`);
console.log(`  Seed:           ${SEED}`);
console.log("-".repeat(70));
console.log(`  Nodes:          ${report.dataset.totalNodes} (${report.dataset.leaves} leaves)`);
console.log(
  `  Prod:           ${report.dataset.prodTrain} train / ${report.dataset.prodTest} test  (${prodSplitPct}/${
    100 - +prodSplitPct
  })`,
);
console.log(
  `  N8n:            ${report.dataset.n8nTrain} train / ${report.dataset.n8nEval} eval  (${n8nSplitPct}/${
    100 - +n8nSplitPct
  })`,
);
console.log(
  `  InfoNCE pool:   ${report.dataset.infonceExamples}  |  KL pool: ${report.dataset.klExamples}`,
);
console.log("-".repeat(70));
console.log(`  ─── Final metrics (epoch ${EPOCHS}) ───`);
console.log(
  `  InfoNCE loss:   ${last?.infoLoss.toFixed(4) ?? "N/A"}   acc: ${
    last ? (last.infoAcc * 100).toFixed(1) + "%" : "N/A"
  }   ∇: ${last?.infoGradNorm.toFixed(2) ?? "N/A"}`,
);
console.log(
  `  KL loss:        ${last?.klLoss.toFixed(4) ?? "N/A"}   ∇: ${
    last?.klGradNorm.toFixed(2) ?? "N/A"
  }   n8nEval: ${last?.n8nEvalKL.toFixed(4) ?? "N/A"}`,
);
console.log(`  Hit@1:          ${last ? (last.testHit1 * 100).toFixed(1) + "%" : "N/A"}`);
console.log(`  Hit@3:          ${last ? (last.testHit3 * 100).toFixed(1) + "%" : "N/A"}`);
console.log(`  Hit@5:          ${last ? (last.testHit5 * 100).toFixed(1) + "%" : "N/A"}`);
console.log(`  MRR:            ${last?.testMRR.toFixed(3) ?? "N/A"}`);
console.log(`  Terminal acc:   ${last ? (last.testTermAcc * 100).toFixed(1) + "%" : "N/A"}`);
console.log("-".repeat(70));
console.log(`  ─── Best across epochs ───`);
console.log(`  Best Hit@1:     ${(bestTestHit1 * 100).toFixed(1)}%  (epoch ${bestEpoch})`);
console.log(`  Best MRR:       ${bestMRR.toFixed(3)}`);
console.log("-".repeat(70));
console.log(
  `  Total time:     ${report.results.totalTrainingSec}s  (${report.results.avgEpochSec}s/epoch)`,
);
console.log(`  Peak RSS:       ${last?.rssMB ?? "?"}MB`);
console.log("=".repeat(70));

// 7. Export params + report
console.log("\n=== Saving SHGAT params ===");
const runId = new Date().toISOString().replace(/[:.]/g, "-");
const params = shgat.exportParams();
const outputPath = resolve(__dirname, `../../gru/data/shgat-params-expanded-${runId}.json`);
writeFileSync(outputPath, JSON.stringify(params));
console.log(`  Params → ${outputPath}`);

const reportPath = resolve(__dirname, `../../gru/data/shgat-training-report-${runId}.json`);
writeFileSync(reportPath, JSON.stringify(report, null, 2));
console.log(`  Report → ${reportPath}`);

// Optional: write to DB
if (WRITE_DB) {
  console.log("  Writing params to DB...");
  try {
    const msgpackBytes = msgpackEncode(params);
    const compressed = pako.gzip(msgpackBytes, { level: 6 });
    const base64 = btoa(String.fromCharCode(...compressed));

    const wrapper = {
      compressed: true,
      format: "msgpack+gzip+base64",
      size: msgpackBytes.length,
      compressedSize: compressed.length,
      data: base64,
    };

    await sql`
      INSERT INTO shgat_params (params, updated_at)
      SELECT ${JSON.stringify(wrapper)}::jsonb, NOW()
      WHERE NOT EXISTS (SELECT 1 FROM shgat_params)
    `;
    await sql`
      UPDATE shgat_params SET params = ${JSON.stringify(wrapper)}::jsonb, updated_at = NOW()
    `;
    console.log("  Params saved to DB");
  } catch (e) {
    console.warn(`  Failed to save to DB: ${e}`);
  }
}

// Cleanup
shgat.dispose();
await sql.end();

console.log("\n=== SHGAT retraining complete ===");
