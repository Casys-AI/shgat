#!/usr/bin/env -S deno run -A --no-check
/**
 * SHGAT-TF Production Benchmark
 *
 * Full training + evaluation on production traces with production-matched params:
 * - 243 capabilities, 644 tools, 904 episodic events
 * - Training: AutogradTrainer + PER + cosine temp annealing + curriculum learning
 * - Params matched to train-worker.ts: lr=0.05, epochs=10, batch=32, PER maxPriority=25
 * - 80/20 train/test split (same as prod)
 * - Evaluation: MRR, Hit@1, Hit@3, Hit@5 on 180 held-out test queries
 *
 * Run: deno run -A --no-check lib/shgat-tf/tests/shgat_prod_bench.ts
 *
 * @module shgat-tf/tests/shgat_prod_bench
 */

import {
  type Node,
  type TrainingExample,
} from "../mod.ts";
import {
  AutogradTrainer,
  PERBuffer,
  annealTemperature,
  annealBeta,
  buildGraphStructure,
  disposeGraphStructure,
  type CapabilityInfo,
} from "../src/training/index.ts";
import { DEFAULT_SHGAT_CONFIG } from "../src/core/types.ts";
import { initTensorFlow, getBackend } from "../src/tf/backend.ts";

// Cosine similarity between two vectors
function cosineSim(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

// ============================================================================
// Production-matched hyperparameters (from train-worker.ts)
// ============================================================================

const PROD_CONFIG = {
  // Training
  epochs: 10,                // default in train-shgat-standalone.ts
  batchSize: 32,             // default
  learningRate: 0.05,        // train-worker.ts:255
  gradientClip: 1.0,
  l2Lambda: 0.0001,

  // PER (train-worker.ts:282-287)
  perAlpha: 0.6,             // priority exponent (optimal per benchmark)
  perBeta: 0.4,              // IS weight start (annealed to 1.0)
  perEpsilon: 0.01,          // min priority floor
  perMaxPriority: 25,        // match margin-based TD error range

  // Temperature annealing (cosine: 0.10 → 0.06)
  tauStart: 0.10,
  tauEnd: 0.06,

  // Curriculum learning thresholds (train-worker.ts:312-317)
  curriculumEasyThreshold: 0.60,
  curriculumHardThreshold: 0.75,

  // SHGAT architecture
  numHeads: 16,
  embeddingDim: 1024,
  headDim: 64,
  hiddenDim: 1024,

  // Train/test split
  trainRatio: 0.80,
};

// ============================================================================
// Load production traces
// ============================================================================

console.log("Loading production traces...");
const scenarioPath = new URL(
  "../../../tests/benchmarks/fixtures/scenarios/production-traces.json",
  import.meta.url,
);
const scenarioText = await Deno.readTextFile(scenarioPath);
const scenario = JSON.parse(scenarioText);

// deno-lint-ignore no-explicit-any
const rawCaps: any[] = scenario.nodes?.capabilities || [];
// deno-lint-ignore no-explicit-any
const rawTools: any[] = scenario.nodes?.tools || [];
// deno-lint-ignore no-explicit-any
const rawEvents: any[] = scenario.episodicEvents || [];
// deno-lint-ignore no-explicit-any
const rawQueries: any[] = scenario.testQueries || [];

console.log(
  `  ${rawCaps.length} capabilities, ${rawTools.length} tools, ` +
  `${rawEvents.length} events, ${rawQueries.length} test queries`,
);

// ============================================================================
// Build SHGAT nodes (unified Node API)
// ============================================================================

console.log("\nBuilding SHGAT graph...");
const t0 = performance.now();

// deno-lint-ignore no-explicit-any
const validCaps = rawCaps.filter((c: any) => c.embedding?.length === 1024);
// deno-lint-ignore no-explicit-any
const validTools = rawTools.filter((t: any) => t.embedding?.length === 1024);

const toolIdSet = new Set(validTools.map((t: { id: string }) => t.id));

// Tool nodes (leaves)
const nodes: Node[] = validTools.map((t: { id: string; embedding: number[] }) => ({
  id: t.id,
  embedding: t.embedding,
  children: [],
  level: 0,
}));

// Capability nodes (composites) — children = toolsUsed that exist in tools
for (const cap of validCaps) {
  const children = (cap.toolsUsed || []).filter((tid: string) => toolIdSet.has(tid));
  nodes.push({
    id: cap.id,
    embedding: cap.embedding,
    children,
    level: 0,
  });
}

const tBuild = performance.now() - t0;
console.log(`  Nodes built in ${tBuild.toFixed(0)}ms (${nodes.length} nodes: ${validTools.length} tools + ${validCaps.length} caps)`);

// ============================================================================
// Build training examples + 80/20 split
// ============================================================================

const capIdSet = new Set(validCaps.map((c: { id: string }) => c.id));

// Helper: Fisher-Yates shuffle (in-place)
function shuffleArray<T>(arr: T[]): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

const allExamples: TrainingExample[] = rawEvents
  // deno-lint-ignore no-explicit-any
  .filter((e: any) =>
    e.intentEmbedding?.length === 1024 &&
    capIdSet.has(e.selectedCapability)
  )
  // deno-lint-ignore no-explicit-any
  .map((e: any) => {
    // ALL negatives (for curriculum sorting by cosine similarity to intent)
    const allNegatives = validCaps
      .filter((c: { id: string }) => c.id !== e.selectedCapability)
      .map((c: { id: string; embedding: number[] }) => ({
        id: c.id,
        sim: cosineSim(e.intentEmbedding, c.embedding),
      }))
      .sort((a, b) => b.sim - a.sim) // descending: hard first
      .map(n => n.id);

    // Random 15 negatives (shuffled, not always the first 15)
    const shuffledNegs = shuffleArray([...allNegatives]);

    return {
      intentEmbedding: e.intentEmbedding,
      candidateId: e.selectedCapability,
      outcome: e.outcome === "success" ? 1 : 0,
      contextTools: e.contextTools || [],
      negativeCapIds: shuffledNegs.slice(0, 15),
      allNegativesSorted: allNegatives,
    } as TrainingExample;
  });

// Fisher-Yates shuffle
const shuffled = [...allExamples];
for (let i = shuffled.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
}

const splitIdx = Math.floor(shuffled.length * PROD_CONFIG.trainRatio);
const trainExamples = shuffled.slice(0, splitIdx);
const heldOutExamples = shuffled.slice(splitIdx);

console.log(`  ${allExamples.length} valid examples → ${trainExamples.length} train / ${heldOutExamples.length} held-out`);

// ============================================================================
// Build test queries (held-out)
// ============================================================================

interface TestQuery {
  intent: string;
  intentEmbedding: number[];
  expectedCapability: string;
  difficulty: string;
}

const testQueries: TestQuery[] = rawQueries
  // deno-lint-ignore no-explicit-any
  .filter((q: any) =>
    q.intentEmbedding?.length === 1024 &&
    capIdSet.has(q.expectedCapability)
  );

console.log(`  ${testQueries.length} external test queries`);

// ============================================================================
// Evaluation helper (uses trainer.score for consistent comparison)
// ============================================================================

function evaluateTrainer(
  trainerInstance: AutogradTrainer,
  queries: TestQuery[],
  capabilityIds: string[],
): { mrr: number; hit1: number; hit3: number; hit5: number; avgLatency: number } {
  let totalMRR = 0;
  let hit1 = 0;
  let hit3 = 0;
  let hit5 = 0;
  let totalTime = 0;

  for (const q of queries) {
    const start = performance.now();
    const scores = trainerInstance.score(q.intentEmbedding, capabilityIds);
    totalTime += performance.now() - start;

    const ranked = capabilityIds
      .map((id, i) => ({ id, score: scores[i] }))
      .sort((a, b) => b.score - a.score);

    const rank = ranked.findIndex((r) => r.id === q.expectedCapability) + 1;
    if (rank > 0) {
      totalMRR += 1 / rank;
      if (rank <= 1) hit1++;
      if (rank <= 3) hit3++;
      if (rank <= 5) hit5++;
    }
  }

  const n = queries.length;
  return {
    mrr: totalMRR / n,
    hit1: hit1 / n,
    hit3: hit3 / n,
    hit5: hit5 / n,
    avgLatency: totalTime / n,
  };
}

function printMetrics(label: string, m: { mrr: number; hit1: number; hit3: number; hit5: number; avgLatency: number }) {
  console.log(`  ${label}`);
  console.log(`    MRR:     ${m.mrr.toFixed(3)}`);
  console.log(`    Hit@1:   ${(m.hit1 * 100).toFixed(1)}%`);
  console.log(`    Hit@3:   ${(m.hit3 * 100).toFixed(1)}%`);
  console.log(`    Hit@5:   ${(m.hit5 * 100).toFixed(1)}%`);
  console.log(`    Latency: ${m.avgLatency.toFixed(1)}ms/query`);
}

// ============================================================================
// Setup AutogradTrainer (same scorer used for baseline AND post-training)
// ============================================================================

console.log("\n========== SETUP ==========");

// Initialize TF.js with training-compatible backend (WebGPU > CPU, never WASM)
await initTensorFlow("training");
console.log(`  Backend: ${getBackend()} (training mode - full autograd)`);

const config = {
  ...DEFAULT_SHGAT_CONFIG,
  numHeads: PROD_CONFIG.numHeads,
  embeddingDim: PROD_CONFIG.embeddingDim,
  headDim: PROD_CONFIG.headDim,
  hiddenDim: PROD_CONFIG.hiddenDim,
};

const trainer = new AutogradTrainer(config, {
  learningRate: PROD_CONFIG.learningRate,
  batchSize: PROD_CONFIG.batchSize,
  temperature: PROD_CONFIG.tauStart,
  gradientClip: PROD_CONFIG.gradientClip,
  l2Lambda: PROD_CONFIG.l2Lambda,
});
// Dense autograd mode (default) - no need to call setSparseMP(false)

// Set embeddings
const allEmbeddings = new Map<string, number[]>();
for (const node of nodes) {
  allEmbeddings.set(node.id, node.embedding);
}
trainer.setNodeEmbeddings(allEmbeddings);

// Build graph structure for message passing (same as prod train-worker)
const capInfos: CapabilityInfo[] = validCaps.map((c: { id: string; toolsUsed?: string[] }) => ({
  id: c.id,
  toolsUsed: (c.toolsUsed || []).filter((tid: string) => toolIdSet.has(tid)),
}));
const toolIds = validTools.map((t: { id: string }) => t.id);
const graphStructure = buildGraphStructure(capInfos, toolIds);
trainer.setGraph(graphStructure);

const capIds = validCaps.map((c: { id: string }) => c.id);
console.log(`  Message passing: dense autograd (full gradient flow), maxLevel=${graphStructure.maxLevel}`);
console.log(`  Scoring: trainer.score() (${capIds.length} capabilities)`);

// ============================================================================
// PRE-TRAINING BASELINE (random init params, same scorer as post-training)
// ============================================================================

console.log("\n========== PRE-TRAINING BASELINE ==========");
const baseline = evaluateTrainer(trainer, testQueries, capIds);
printMetrics("trainer.score (random init + MP)", baseline);

// ============================================================================
// Training with production-matched params
// ============================================================================

console.log("\n========== TRAINING (prod params) ==========");
console.log(`  lr=${PROD_CONFIG.learningRate} | epochs=${PROD_CONFIG.epochs} | batch=${PROD_CONFIG.batchSize}`);
console.log(`  PER: alpha=${PROD_CONFIG.perAlpha} beta=${PROD_CONFIG.perBeta}→1.0 maxPriority=${PROD_CONFIG.perMaxPriority}`);
console.log(`  Temperature: ${PROD_CONFIG.tauStart}→${PROD_CONFIG.tauEnd} (cosine anneal)`);
console.log(`  Curriculum: easy<${PROD_CONFIG.curriculumEasyThreshold} | hard>${PROD_CONFIG.curriculumHardThreshold}`);
console.log();

// PER buffer with prod params
const buffer = new PERBuffer(trainExamples, {
  alpha: PROD_CONFIG.perAlpha,
  beta: PROD_CONFIG.perBeta,
  epsilon: PROD_CONFIG.perEpsilon,
  maxPriority: PROD_CONFIG.perMaxPriority,
});

const numBatchesPerEpoch = Math.ceil(trainExamples.length / PROD_CONFIG.batchSize);
const tTrainStart = performance.now();
let lastAccuracy = 0.55; // initial assumption (same as train-worker.ts:314)

for (let epoch = 0; epoch < PROD_CONFIG.epochs; epoch++) {
  const tau = annealTemperature(epoch, PROD_CONFIG.epochs, PROD_CONFIG.tauStart, PROD_CONFIG.tauEnd);
  trainer.setTemperature(tau); // Actually apply temperature annealing to the trainer
  const beta = annealBeta(epoch, PROD_CONFIG.epochs, PROD_CONFIG.perBeta);

  // Curriculum difficulty selection (matched to train-worker.ts:312-317)
  const difficulty = lastAccuracy < PROD_CONFIG.curriculumEasyThreshold
    ? "easy"
    : lastAccuracy > PROD_CONFIG.curriculumHardThreshold
      ? "hard"
      : "medium";

  let epochLoss = 0;
  let epochAcc = 0;
  let epochGrad = 0;
  let batchCount = 0;

  // Multiple batches per epoch (process all training data)
  for (let b = 0; b < numBatchesPerEpoch; b++) {
    const { items, indices } = buffer.sample(PROD_CONFIG.batchSize, beta);
    const metrics = await trainer.trainBatch(items);

    // Update PER priorities
    const errors = items.map(() => metrics.loss);
    buffer.updatePriorities(indices, errors);

    epochLoss += metrics.loss;
    epochAcc += metrics.accuracy;
    epochGrad += metrics.gradientNorm;
    batchCount++;
  }

  epochLoss /= batchCount;
  epochAcc /= batchCount;
  epochGrad /= batchCount;
  lastAccuracy = epochAcc;

  // Decay stale priorities every 3 epochs
  if (epoch % 3 === 0) {
    buffer.decayPriorities(0.9);
  }

  console.log(
    `  Epoch ${String(epoch).padStart(2)}/${PROD_CONFIG.epochs}: ` +
    `loss=${epochLoss.toFixed(4)} acc=${epochAcc.toFixed(3)} ` +
    `tau=${tau.toFixed(3)} grad=${epochGrad.toFixed(3)} ` +
    `[${difficulty}] (${batchCount} batches)`,
  );
}

const tTrain = performance.now() - tTrainStart;
console.log(`\n  Training done in ${(tTrain / 1000).toFixed(1)}s (${(tTrain / PROD_CONFIG.epochs / 1000).toFixed(1)}s/epoch)`);

// ============================================================================
// Post-training evaluation (same scorer as baseline)
// ============================================================================

console.log("\n========== POST-TRAINING EVALUATION ==========");

const postTrainer = evaluateTrainer(trainer, testQueries, capIds);
printMetrics("trainer.score (trained + MP)", postTrainer);

// Held-out accuracy (internal test set from 80/20 split)
console.log(`\n  Held-out validation (${heldOutExamples.length} examples):`);
let heldOutCorrect = 0;
for (const ex of heldOutExamples) {
  const scores = trainer.score(ex.intentEmbedding, capIds);
  const ranked = capIds
    .map((id, i) => ({ id, score: scores[i] }))
    .sort((a, b) => b.score - a.score);
  if (ranked[0]?.id === ex.candidateId) heldOutCorrect++;
}
console.log(`    Held-out Hit@1: ${((heldOutCorrect / heldOutExamples.length) * 100).toFixed(1)}%`);

// ============================================================================
// Summary
// ============================================================================

console.log("\n" + "=".repeat(70));
console.log("  SHGAT-TF PRODUCTION BENCHMARK SUMMARY");
console.log("=".repeat(70));
console.log(`  Graph:     ${validTools.length} tools + ${validCaps.length} capabilities (${nodes.length} nodes)`);
console.log(`  Training:  ${trainExamples.length} examples, ${PROD_CONFIG.epochs} epochs, lr=${PROD_CONFIG.learningRate}`);
console.log(`  Test:      ${testQueries.length} external queries + ${heldOutExamples.length} held-out`);
console.log(`  Build:     ${tBuild.toFixed(0)}ms`);
console.log(`  Train:     ${(tTrain / 1000).toFixed(1)}s total, ${(tTrain / PROD_CONFIG.epochs / 1000).toFixed(1)}s/epoch`);
console.log(`  MP:        dense autograd (CPU/WebGPU), maxLevel=${graphStructure.maxLevel}`);
console.log();

const col1 = 18, col2 = 14, col3 = 14;
console.log(`  ${"Metric".padEnd(col1)} ${"Baseline".padEnd(col2)} ${"Trained".padEnd(col3)}`);
console.log(`  ${"─".repeat(col1)} ${"─".repeat(col2)} ${"─".repeat(col3)}`);
console.log(`  ${"MRR".padEnd(col1)} ${baseline.mrr.toFixed(3).padEnd(col2)} ${postTrainer.mrr.toFixed(3).padEnd(col3)}`);
console.log(`  ${"Hit@1".padEnd(col1)} ${pct(baseline.hit1).padEnd(col2)} ${pct(postTrainer.hit1).padEnd(col3)}`);
console.log(`  ${"Hit@3".padEnd(col1)} ${pct(baseline.hit3).padEnd(col2)} ${pct(postTrainer.hit3).padEnd(col3)}`);
console.log(`  ${"Hit@5".padEnd(col1)} ${pct(baseline.hit5).padEnd(col2)} ${pct(postTrainer.hit5).padEnd(col3)}`);
console.log(`  ${"Latency/query".padEnd(col1)} ${ms(baseline.avgLatency).padEnd(col2)} ${ms(postTrainer.avgLatency).padEnd(col3)}`);
console.log(`  ${"Held-out Hit@1".padEnd(col1)} ${"—".padEnd(col2)} ${pct(heldOutCorrect / heldOutExamples.length).padEnd(col3)}`);
console.log("=".repeat(70));

// Cleanup
disposeGraphStructure(graphStructure);
trainer.dispose();

// ============================================================================
// Formatting helpers
// ============================================================================

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function ms(v: number): string {
  return `${v.toFixed(1)}ms`;
}
