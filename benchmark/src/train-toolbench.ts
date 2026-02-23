/**
 * SHGAT Option B Training Script (ToolBench)
 *
 * Trains SHGAT attention weights using contrastive learning on ToolBench data.
 * 49 categories, ~3400 collections, ~14000 APIs, 1100 queries.
 *
 * Hierarchy: Category → Collection → API (3 levels)
 *
 * Uses hard negative mining, curriculum learning, and temperature annealing.
 *
 * Usage:
 *   npx tsx src/train-toolbench.ts                        # 5-fold CV, default
 *   npx tsx src/train-toolbench.ts --epochs 10 --lr 0.001 # Custom params
 *   npx tsx src/train-toolbench.ts --overfit               # Train+eval on all data
 *   npx tsx src/train-toolbench.ts --flat-only             # Only flat model
 *   npx tsx src/train-toolbench.ts --hier-only             # Only hierarchical model
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { evaluate, printResults, type RelevanceLabel, type EvalResults } from "./metrics.ts";
import { batchScoreCosine } from "./cosine-baseline.ts";
import { batchScoreSHGAT } from "./shgat-scorer.ts";
import {
  SHGATBuilder,
  type NodeInput,
  type TrainingExample,
} from "../../dist-node/mod.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DATA_DIR = resolve(ROOT, "data", "toolbench");

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);

function parseArg(flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

function hasFlag(flag: string): boolean {
  return args.includes(flag);
}

const EPOCHS = parseInt(parseArg("--epochs") ?? "10", 10);
const LR = parseFloat(parseArg("--lr") ?? "0.001");
const BATCH_SIZE = parseInt(parseArg("--batch-size") ?? "16", 10);
const FOLDS = parseInt(parseArg("--folds") ?? "5", 10);
const SEED = parseInt(parseArg("--seed") ?? "42", 10);
const FLAT_ONLY = hasFlag("--flat-only");
const HIER_ONLY = hasFlag("--hier-only");
const NO_CURRICULUM = hasFlag("--no-curriculum");
const OVERFIT_MODE = hasFlag("--overfit");
const NUM_NEGATIVES = 8;

const TEMP_MAX = 0.10;
const TEMP_MIN = 0.06;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

interface ToolBenchTool {
  id: string;
  name: string;
  description: string;
  text: string;
  category: string;
  collection: string;
  embedding: number[];
}

interface ToolBenchQuery {
  id: string;
  query: string;
  labels: Array<{ id: string; relevance: number }>;
  embedding: number[];
}

interface Hierarchy {
  categories: Record<string, string[]>;
  collections: Record<string, string[]>;
  api_to_collection: Record<string, string>;
  collection_to_category: Record<string, string>;
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

function loadData(): { tools: ToolBenchTool[]; queries: ToolBenchQuery[]; hierarchy: Hierarchy } {
  const toolsPath = resolve(DATA_DIR, "tools.json");
  const queriesPath = resolve(DATA_DIR, "queries.json");
  const hierPath = resolve(DATA_DIR, "hierarchy.json");

  if (!existsSync(toolsPath) || !existsSync(queriesPath)) {
    console.error(
      "ToolBench data not found. Run:\n" +
      "  python3 lib/shgat-tf/benchmark/scripts/process-toolbench.py\n" +
      "  deno run --allow-all --config deno.json lib/shgat-tf/benchmark/scripts/embed-toolbench.ts",
    );
    process.exit(1);
  }

  const tools: ToolBenchTool[] = JSON.parse(readFileSync(toolsPath, "utf-8"));
  const queries: ToolBenchQuery[] = JSON.parse(readFileSync(queriesPath, "utf-8"));
  const hierarchy: Hierarchy = JSON.parse(readFileSync(hierPath, "utf-8"));

  return { tools, queries, hierarchy };
}

// ---------------------------------------------------------------------------
// Ground truth (ToolRet labels have tool IDs directly)
// ---------------------------------------------------------------------------

function buildGroundTruth(
  queries: ToolBenchQuery[],
  toolIdSet: Set<string>,
): Map<string, RelevanceLabel[]> {
  const groundTruth = new Map<string, RelevanceLabel[]>();
  let resolved = 0, unresolved = 0;

  for (const q of queries) {
    const labels: RelevanceLabel[] = [];
    for (const l of q.labels) {
      if (toolIdSet.has(l.id)) {
        labels.push({ id: l.id, relevance: l.relevance });
        resolved++;
      } else {
        unresolved++;
      }
    }
    groundTruth.set(q.id, labels);
  }

  console.log(`[ground-truth] ${resolved} resolved, ${unresolved} unresolved`);
  return groundTruth;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function cosineAnneal(epoch: number, totalEpochs: number, tempMax: number, tempMin: number): number {
  return tempMin + 0.5 * (tempMax - tempMin) * (1 + Math.cos(Math.PI * epoch / totalEpochs));
}

function meanEmbedding(embeddings: number[][], dim: number): number[] {
  const mean = new Array(dim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) mean[i] += emb[i];
  }
  const n = embeddings.length || 1;
  for (let i = 0; i < dim; i++) mean[i] /= n;
  return mean;
}

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(array: T[], rng: () => number): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// ---------------------------------------------------------------------------
// Build contrastive training examples
// ---------------------------------------------------------------------------

function buildTrainingExamples(
  queries: ToolBenchQuery[],
  tools: ToolBenchTool[],
  groundTruth: Map<string, RelevanceLabel[]>,
  numNegatives: number = NUM_NEGATIVES,
): TrainingExample[] {
  const examples: TrainingExample[] = [];
  const toolEmbeddings = new Map<string, number[]>();
  for (const t of tools) toolEmbeddings.set(t.id, t.embedding);
  const allToolIds = tools.map((t) => t.id);

  for (const query of queries) {
    const labels = groundTruth.get(query.id);
    if (!labels || labels.length === 0) continue;

    const positiveIds = new Set(labels.map((l) => l.id));

    // Cosine similarity for hard negative mining
    const sims: Array<{ id: string; sim: number }> = [];
    for (const toolId of allToolIds) {
      if (positiveIds.has(toolId)) continue;
      const emb = toolEmbeddings.get(toolId)!;
      sims.push({ id: toolId, sim: cosineSim(query.embedding, emb) });
    }
    sims.sort((a, b) => b.sim - a.sim);
    const allNegativesSorted = sims.map((s) => s.id);

    for (const positiveLabel of labels) {
      const negativeCapIds = allNegativesSorted.slice(0, numNegatives);
      examples.push({
        intentEmbedding: query.embedding,
        contextTools: [],
        candidateId: positiveLabel.id,
        outcome: 1.0,
        negativeCapIds,
        allNegativesSorted,
      });
    }
  }

  return examples;
}

// ---------------------------------------------------------------------------
// K-fold cross-validation split
// ---------------------------------------------------------------------------

interface FoldSplit {
  trainQueryIds: Set<string>;
  testQueryIds: Set<string>;
}

function kFoldSplit(queries: ToolBenchQuery[], k: number, seed: number): FoldSplit[] {
  const rng = mulberry32(seed);
  const queryIds = queries.map((q) => q.id);
  const shuffled = seededShuffle(queryIds, rng);
  const foldSize = Math.ceil(shuffled.length / k);
  const folds: FoldSplit[] = [];

  for (let fold = 0; fold < k; fold++) {
    const testStart = fold * foldSize;
    const testEnd = Math.min(testStart + foldSize, shuffled.length);
    const testQueryIds = new Set(shuffled.slice(testStart, testEnd));
    const trainQueryIds = new Set(shuffled.filter((id) => !testQueryIds.has(id)));
    folds.push({ trainQueryIds, testQueryIds });
  }

  return folds;
}

// ---------------------------------------------------------------------------
// Build hierarchy nodes
// ---------------------------------------------------------------------------

function buildHierarchyNodes(
  tools: ToolBenchTool[],
  hierarchy: Hierarchy,
): { nodes: NodeInput[]; leafIds: string[] } {
  const dim = tools[0]?.embedding.length ?? 1024;
  const toolById = new Map(tools.map((t) => [t.id, t]));
  const nodes: NodeInput[] = [];
  const leafIds: string[] = [];

  // L0: APIs (leaves)
  for (const tool of tools) {
    nodes.push({ id: tool.id, embedding: tool.embedding, children: [] });
    leafIds.push(tool.id);
  }

  // L1: Collections
  for (const [collKey, apiIds] of Object.entries(hierarchy.collections)) {
    const childEmbs = apiIds
      .map((aid) => toolById.get(aid)?.embedding)
      .filter((e): e is number[] => !!e);
    if (childEmbs.length === 0) continue;
    nodes.push({
      id: `coll:${collKey}`,
      embedding: meanEmbedding(childEmbs, dim),
      children: apiIds.filter((aid) => toolById.has(aid)),
    });
  }

  // L2: Categories
  for (const [catName, collKeys] of Object.entries(hierarchy.categories)) {
    const collIds: string[] = [];
    const collEmbs: number[][] = [];
    for (const ck of collKeys) {
      const apiIds = hierarchy.collections[ck] ?? [];
      const childEmbs = apiIds
        .map((aid) => toolById.get(aid)?.embedding)
        .filter((e): e is number[] => !!e);
      if (childEmbs.length > 0) {
        collIds.push(`coll:${ck}`);
        collEmbs.push(meanEmbedding(childEmbs, dim));
      }
    }
    if (collEmbs.length === 0) continue;
    nodes.push({
      id: `category:${catName}`,
      embedding: meanEmbedding(collEmbs, dim),
      children: collIds,
    });
  }

  return { nodes, leafIds };
}

// ---------------------------------------------------------------------------
// Training loop for a single fold
// ---------------------------------------------------------------------------

interface FoldResult {
  foldIdx: number;
  evalResult: EvalResults;
  finalLoss: number;
  finalAccuracy: number;
}

async function trainFold(
  foldIdx: number,
  totalFolds: number,
  trainQueries: ToolBenchQuery[],
  testQueries: ToolBenchQuery[],
  tools: ToolBenchTool[],
  nodes: NodeInput[],
  leafIds: string[],
  groundTruth: Map<string, RelevanceLabel[]>,
  modelLabel: string,
  isHierarchical: boolean,
): Promise<FoldResult> {
  const trainQCount = trainQueries.length;
  const testQCount = testQueries.length;
  console.log(`\n--- Fold ${foldIdx + 1}/${totalFolds} (${trainQCount} train, ${testQCount} test) [${modelLabel}] ---`);

  const trainExamples = buildTrainingExamples(trainQueries, tools, groundTruth, NUM_NEGATIVES);
  console.log(`  ${trainExamples.length} training examples`);

  if (trainExamples.length === 0) {
    console.log("  WARNING: No training examples, skipping...");
    return { foldIdx, evalResult: { name: modelLabel, numQueries: 0, metrics: {} }, finalLoss: NaN, finalAccuracy: 0 };
  }

  // Build SHGAT with training enabled
  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .training({ learningRate: LR, temperature: TEMP_MAX })
    .architecture({
      seed: SEED,
      downwardResidual: isHierarchical ? parseFloat(parseArg("--dr") ?? "0.5") : 0,
      preserveDimResiduals: isHierarchical
        ? (parseArg("--pdr") ?? "0.5,0.3,0.1").split(",").map(Number)
        : undefined,
    })
    .backend("training")
    .build();

  const rng = mulberry32(SEED + foldIdx);
  let lastLoss = 0;
  let lastAcc = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    const temp = cosineAnneal(epoch, EPOCHS, TEMP_MAX, TEMP_MIN);
    shgat.setTemperature(temp);

    // Curriculum learning
    if (!NO_CURRICULUM && epoch > 0 && trainExamples[0].allNegativesSorted) {
      for (const ex of trainExamples) {
        if (!ex.allNegativesSorted || ex.allNegativesSorted.length === 0) continue;
        const totalNegs = ex.allNegativesSorted.length;
        const thirdSize = Math.floor(totalNegs / 3);

        let tierStart: number, tierEnd: number;
        if (lastAcc < 0.35) {
          tierStart = totalNegs - thirdSize; tierEnd = totalNegs;
        } else if (lastAcc > 0.55) {
          tierStart = 0; tierEnd = thirdSize;
        } else {
          tierStart = thirdSize; tierEnd = thirdSize * 2;
        }

        const tierSlice = ex.allNegativesSorted.slice(tierStart, tierEnd);
        ex.negativeCapIds = seededShuffle(tierSlice, rng).slice(0, NUM_NEGATIVES);
      }
    }

    const shuffled = seededShuffle(trainExamples, rng);

    let epochLoss = 0, epochAcc = 0, epochGrad = 0, numBatches = 0;
    for (let batchStart = 0; batchStart < shuffled.length; batchStart += BATCH_SIZE) {
      const batch = shuffled.slice(batchStart, batchStart + BATCH_SIZE);
      const metrics = await shgat.trainBatch(batch);
      epochLoss += metrics.loss;
      epochAcc += metrics.accuracy;
      epochGrad += metrics.gradientNorm;
      numBatches++;
    }

    epochLoss /= numBatches;
    epochAcc /= numBatches;
    epochGrad /= numBatches;
    lastLoss = epochLoss;
    lastAcc = epochAcc;

    if (epoch === 0 || epoch === EPOCHS - 1 || (epoch + 1) % 5 === 0) {
      console.log(
        `  Epoch ${String(epoch + 1).padStart(3)}/${EPOCHS}: ` +
        `loss=${epochLoss.toFixed(4)}  acc=${epochAcc.toFixed(2)}  ` +
        `grad=${epochGrad.toFixed(2)}  temp=${temp.toFixed(3)}`,
      );
    }
  }

  // Evaluate on test queries
  const testGT = new Map<string, RelevanceLabel[]>();
  for (const q of testQueries) {
    const labels = groundTruth.get(q.id);
    if (labels) testGT.set(q.id, labels);
  }

  const rankings = batchScoreSHGAT(shgat, testQueries, leafIds, modelLabel.toLowerCase().replace(/\s+/g, "-"));
  const evalResult = evaluate(modelLabel, rankings, testGT);

  console.log(
    `  Fold ${foldIdx + 1} test: ` +
    `R@1=${((evalResult.metrics["recall@1"] ?? 0) * 100).toFixed(1)}  ` +
    `R@3=${((evalResult.metrics["recall@3"] ?? 0) * 100).toFixed(1)}  ` +
    `R@5=${((evalResult.metrics["recall@5"] ?? 0) * 100).toFixed(1)}  ` +
    `NDCG@5=${((evalResult.metrics["ndcg@5"] ?? 0) * 100).toFixed(1)}`,
  );

  shgat.dispose();
  return { foldIdx, evalResult, finalLoss: lastLoss, finalAccuracy: lastAcc };
}

// ---------------------------------------------------------------------------
// Aggregate fold results
// ---------------------------------------------------------------------------

function aggregateFolds(foldResults: FoldResult[], name: string): EvalResults {
  const validFolds = foldResults.filter((f) => f.evalResult.numQueries > 0);
  if (validFolds.length === 0) return { name, numQueries: 0, metrics: {} };

  const allKeys = new Set<string>();
  for (const f of validFolds) {
    for (const key of Object.keys(f.evalResult.metrics)) allKeys.add(key);
  }

  const metrics: Record<string, number> = {};
  for (const key of allKeys) {
    let sum = 0, count = 0;
    for (const f of validFolds) {
      const val = f.evalResult.metrics[key];
      if (val !== undefined) { sum += val; count++; }
    }
    metrics[key] = count > 0 ? sum / count : 0;
  }

  const totalQueries = validFolds.reduce((acc, f) => acc + f.evalResult.numQueries, 0);
  return { name, numQueries: totalQueries, metrics };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== SHGAT-TF Option B Training (ToolBench) ===\n");
  console.log("Hierarchy: Category (49) -> Collection (~3400) -> API (~14000)\n");

  const { tools, queries, hierarchy } = loadData();
  const toolIdSet = new Set(tools.map(t => t.id));
  const groundTruth = buildGroundTruth(queries, toolIdSet);

  const queriesWithGT = queries.filter((q) => {
    const labels = groundTruth.get(q.id);
    return labels && labels.length > 0;
  });

  const totalExamples = buildTrainingExamples(queriesWithGT, tools, groundTruth).length;
  console.log(
    `Data: ${queries.length} queries, ${tools.length} tools, ~${totalExamples} examples`,
  );
  console.log(
    `Config: epochs=${EPOCHS}, lr=${LR}, batch=${BATCH_SIZE}, ` +
    `folds=${OVERFIT_MODE ? "overfit" : FOLDS}, seed=${SEED}`,
  );
  if (FLAT_ONLY) console.log("Mode: flat-only");
  if (HIER_ONLY) console.log("Mode: hier-only");

  const toolIds = tools.map((t) => t.id);
  const results: EvalResults[] = [];

  // ---- 1. Cosine baseline ----
  console.log("\n--- Cosine Baseline ---");
  const t0 = performance.now();
  const cosineRankings = batchScoreCosine(queries, tools);
  const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
  results.push(cosineResult);
  console.log(`  Time: ${((performance.now() - t0) / 1000).toFixed(2)}s`);
  console.log(
    `  R@1=${((cosineResult.metrics["recall@1"] ?? 0) * 100).toFixed(1)}  ` +
    `R@5=${((cosineResult.metrics["recall@5"] ?? 0) * 100).toFixed(1)}`,
  );

  // ---- 2. Option A Flat reference ----
  if (!HIER_ONLY) {
    console.log("\n--- Option A (Flat, no training) ---");
    const flatNodes: NodeInput[] = tools.map((t) => ({
      id: t.id, embedding: t.embedding, children: [],
    }));
    const flatScorer = await SHGATBuilder.create()
      .nodes(flatNodes)
      .architecture({ seed: SEED })
      .build();
    const flatRankings = batchScoreSHGAT(flatScorer, queries, toolIds, "option-a-flat");
    const flatResult = evaluate("Option A (Flat)", flatRankings, groundTruth);
    results.push(flatResult);
    console.log(
      `  R@1=${((flatResult.metrics["recall@1"] ?? 0) * 100).toFixed(1)}  ` +
      `R@5=${((flatResult.metrics["recall@5"] ?? 0) * 100).toFixed(1)}`,
    );
    flatScorer.dispose();
  }

  // ---- 3. Build nodes ----
  const { nodes: hierNodes, leafIds } = buildHierarchyNodes(tools, hierarchy);
  const flatNodes: NodeInput[] = tools.map((t) => ({
    id: t.id, embedding: t.embedding, children: [],
  }));

  const nColls = Object.keys(hierarchy.collections).length;
  const nCats = Object.keys(hierarchy.categories).length;
  console.log(`\n[hier] ${hierNodes.length} nodes: ${leafIds.length} L0 + ${nColls} L1 + ${nCats} L2`);

  // ---- 4. Option B: Trained models ----
  if (OVERFIT_MODE) {
    console.log("\n========== OVERFIT MODE ==========");

    if (!HIER_ONLY) {
      const fr = await trainFold(0, 1, queriesWithGT, queriesWithGT, tools, flatNodes, toolIds, groundTruth, "Option B (Flat, overfit)", false);
      results.push(fr.evalResult);
    }
    if (!FLAT_ONLY) {
      const fr = await trainFold(0, 1, queriesWithGT, queriesWithGT, tools, hierNodes, leafIds, groundTruth, "Option B (Hier, overfit)", true);
      results.push(fr.evalResult);
    }
  } else {
    const folds = kFoldSplit(queriesWithGT, FOLDS, SEED);

    if (!HIER_ONLY) {
      console.log(`\n========== Option B Flat (${FOLDS}-fold CV) ==========`);
      const flatFoldResults: FoldResult[] = [];
      for (let fi = 0; fi < folds.length; fi++) {
        const fold = folds[fi];
        const trainQs = queries.filter((q) => fold.trainQueryIds.has(q.id));
        const testQs = queries.filter((q) => fold.testQueryIds.has(q.id));
        const result = await trainFold(fi, folds.length, trainQs, testQs, tools, flatNodes, toolIds, groundTruth, "Option B (Flat)", false);
        flatFoldResults.push(result);
      }
      results.push(aggregateFolds(flatFoldResults, "Option B (Flat)"));
    }

    if (!FLAT_ONLY) {
      console.log(`\n========== Option B Hier (${FOLDS}-fold CV) ==========`);
      const hierFoldResults: FoldResult[] = [];
      for (let fi = 0; fi < folds.length; fi++) {
        const fold = folds[fi];
        const trainQs = queries.filter((q) => fold.trainQueryIds.has(q.id));
        const testQs = queries.filter((q) => fold.testQueryIds.has(q.id));
        const result = await trainFold(fi, folds.length, trainQs, testQs, tools, hierNodes, leafIds, groundTruth, "Option B (Hier)", true);
        hierFoldResults.push(result);
      }
      results.push(aggregateFolds(hierFoldResults, "Option B (Hier)"));
    }
  }

  // ---- Results ----
  console.log("\n");
  printResults(results);
  console.log(`${queriesWithGT.length} queries with GT, ${tools.length} tools, seed=${SEED}`);
  console.log(`Config: epochs=${EPOCHS}, lr=${LR}, batch=${BATCH_SIZE}`);
}

main().catch((err) => {
  console.error("Training failed:", err);
  process.exit(1);
});
