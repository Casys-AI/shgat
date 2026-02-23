/**
 * SHGAT Mixed Training: InfoNCE (LiveMCPBench) + KL (n8n soft targets)
 *
 * Extends Option B training with n8n workflow description soft targets.
 * K-heads learn to match cosine similarity distributions via KL divergence,
 * augmenting the 282 LiveMCPBench examples with thousands of n8n examples.
 *
 * Usage:
 *   npx tsx src/train-livemcp-mixed.ts                                   # defaults
 *   npx tsx src/train-livemcp-mixed.ts --kl-weight 0.3 --kl-temp 0.01    # custom
 *   npx tsx src/train-livemcp-mixed.ts --flat-only --epochs 25 --lr 0.001
 *   npx tsx src/train-livemcp-mixed.ts --overfit                          # train+eval all
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
  type SHGATTrainerScorer,
  type TrainingExample,
  type SoftTargetExample,
} from "../../dist-node/mod.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DATA_DIR = resolve(ROOT, "data", "livemcp");

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

const EPOCHS = parseInt(parseArg("--epochs") ?? "25", 10);
const LR = parseFloat(parseArg("--lr") ?? "0.001");
const BATCH_SIZE = parseInt(parseArg("--batch-size") ?? "16", 10);
const FOLDS = parseInt(parseArg("--folds") ?? "5", 10);
const SEED = parseInt(parseArg("--seed") ?? "42", 10);
const FLAT_ONLY = hasFlag("--flat-only");
const HIER_ONLY = hasFlag("--hier-only");
const NO_CURRICULUM = hasFlag("--no-curriculum");
const OVERFIT_MODE = hasFlag("--overfit");
const NUM_NEGATIVES = 8;

// KL-specific params
const KL_WEIGHT = parseFloat(parseArg("--kl-weight") ?? "0.3");
const KL_TEMP = parseFloat(parseArg("--kl-temp") ?? "0.01");
const MAX_N8N = parseInt(parseArg("--max-n8n") ?? "5000", 10);
const N8N_DATA_PATH = resolve(DATA_DIR, parseArg("--n8n-data") ?? "n8n-kl-targets.json");
const KL_BATCH_SIZE = parseInt(parseArg("--kl-batch") ?? "16", 10);

const TEMP_MAX = 0.10;
const TEMP_MIN = 0.06;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

interface LiveMCPTool {
  id: string;
  name: string;
  description: string;
  text: string;
  server_name: string;
  server_display_name: string;
  category: string;
  embedding: number[];
}

interface LiveMCPQuery {
  id: string;
  query: string;
  category: string;
  ground_truth_tools: string[];
  embedding: number[];
}

interface Hierarchy {
  categories: Record<string, string[]>;
  servers: Record<string, string[]>;
  server_category: Record<string, string>;
  tool_server: Record<string, string>;
}

interface N8nKLData {
  toolIds: string[];
  temperature: number;
  topK: number;
  numExamples: number;
  examples: {
    intentEmbedding: number[];
    sp: [number, number][];
    wfId: number;
    wfName: string;
  }[];
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

function loadData(): { tools: LiveMCPTool[]; queries: LiveMCPQuery[]; hierarchy: Hierarchy } {
  const toolsPath = resolve(DATA_DIR, "tools.json");
  const queriesPath = resolve(DATA_DIR, "queries.json");
  const hierPath = resolve(DATA_DIR, "hierarchy.json");

  if (!existsSync(toolsPath) || !existsSync(queriesPath)) {
    console.error(
      "LiveMCPBench data not found. Run:\n" +
      "  python3 scripts/download-livemcp.py\n" +
      "  npx tsx scripts/embed-livemcp.ts",
    );
    process.exit(1);
  }

  const tools: LiveMCPTool[] = JSON.parse(readFileSync(toolsPath, "utf-8"));
  const queries: LiveMCPQuery[] = JSON.parse(readFileSync(queriesPath, "utf-8"));
  const hierarchy: Hierarchy = JSON.parse(readFileSync(hierPath, "utf-8"));

  return { tools, queries, hierarchy };
}

function loadN8nKLData(): SoftTargetExample[] {
  if (!existsSync(N8N_DATA_PATH)) {
    console.error(
      `n8n KL data not found at ${N8N_DATA_PATH}\n` +
      `Run: npx tsx src/build-n8n-soft-targets.ts`,
    );
    process.exit(1);
  }

  const raw: N8nKLData = JSON.parse(readFileSync(N8N_DATA_PATH, "utf-8"));
  console.log(`Loaded ${raw.numExamples} n8n KL examples (T=${raw.temperature}, topK=${raw.topK})`);

  // Convert to SoftTargetExample format
  const examples: SoftTargetExample[] = raw.examples.map((ex) => ({
    intentEmbedding: ex.intentEmbedding,
    softTargetSparse: ex.sp,
  }));

  return examples;
}

// ---------------------------------------------------------------------------
// Ground truth resolution
// ---------------------------------------------------------------------------

function resolveGroundTruth(
  queries: LiveMCPQuery[],
  tools: LiveMCPTool[],
): Map<string, RelevanceLabel[]> {
  const nameToIds = new Map<string, string[]>();
  for (const t of tools) {
    const name = t.name.toLowerCase();
    if (!nameToIds.has(name)) nameToIds.set(name, []);
    nameToIds.get(name)!.push(t.id);
  }

  const groundTruth = new Map<string, RelevanceLabel[]>();
  let resolved = 0;
  let unresolved = 0;

  for (const q of queries) {
    const labels: RelevanceLabel[] = [];
    for (const toolName of q.ground_truth_tools) {
      const ids = nameToIds.get(toolName.toLowerCase());
      if (ids) {
        for (const id of ids) {
          labels.push({ id, relevance: 1 });
        }
        resolved++;
      } else {
        const normalized = toolName.toLowerCase().replace(/[-_]/g, "");
        let found = false;
        for (const [name, ids2] of nameToIds) {
          if (name.replace(/[-_]/g, "") === normalized) {
            for (const id of ids2) {
              labels.push({ id, relevance: 1 });
            }
            resolved++;
            found = true;
            break;
          }
        }
        if (!found) {
          unresolved++;
        }
      }
    }
    groundTruth.set(q.id, labels);
  }

  console.log(`[ground-truth] Resolved ${resolved} tool references, ${unresolved} unresolved`);
  return groundTruth;
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
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
  queries: LiveMCPQuery[],
  tools: LiveMCPTool[],
  groundTruth: Map<string, RelevanceLabel[]>,
  numNegatives: number = NUM_NEGATIVES,
): TrainingExample[] {
  const examples: TrainingExample[] = [];
  const toolEmbeddings = new Map<string, number[]>();
  for (const t of tools) {
    toolEmbeddings.set(t.id, t.embedding);
  }
  const allToolIds = tools.map((t) => t.id);

  for (const query of queries) {
    const labels = groundTruth.get(query.id);
    if (!labels || labels.length === 0) continue;

    const positiveIds = new Set(labels.map((l) => l.id));
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
// K-fold split
// ---------------------------------------------------------------------------

interface FoldSplit {
  trainQueryIds: Set<string>;
  testQueryIds: Set<string>;
}

function kFoldSplit(queries: LiveMCPQuery[], k: number, seed: number): FoldSplit[] {
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
  tools: LiveMCPTool[],
  hierarchy: Hierarchy,
): { nodes: NodeInput[]; leafIds: string[] } {
  const dim = tools[0]?.embedding.length ?? 1024;
  const toolById = new Map(tools.map((t) => [t.id, t]));
  const nodes: NodeInput[] = [];
  const leafIds: string[] = [];

  for (const tool of tools) {
    nodes.push({ id: tool.id, embedding: tool.embedding, children: [] });
    leafIds.push(tool.id);
  }

  for (const [serverName, toolIds] of Object.entries(hierarchy.servers)) {
    const serverId = `server:${serverName}`;
    const childEmbs = toolIds
      .map((tid) => toolById.get(tid)?.embedding)
      .filter((e): e is number[] => !!e);
    if (childEmbs.length === 0) continue;
    nodes.push({
      id: serverId,
      embedding: meanEmbedding(childEmbs, dim),
      children: toolIds.filter((tid) => toolById.has(tid)),
    });
  }

  for (const [catName, serverNames] of Object.entries(hierarchy.categories)) {
    const catId = `category:${catName}`;
    const serverIds: string[] = [];
    const serverEmbs: number[][] = [];

    for (const sName of serverNames) {
      const sId = `server:${sName}`;
      const toolIds2 = hierarchy.servers[sName] ?? [];
      const childEmbs = toolIds2
        .map((tid) => toolById.get(tid)?.embedding)
        .filter((e): e is number[] => !!e);
      if (childEmbs.length > 0) {
        serverIds.push(sId);
        serverEmbs.push(meanEmbedding(childEmbs, dim));
      }
    }
    if (serverEmbs.length === 0) continue;
    nodes.push({
      id: catId,
      embedding: meanEmbedding(serverEmbs, dim),
      children: serverIds,
    });
  }

  return { nodes, leafIds };
}

// ---------------------------------------------------------------------------
// Mixed training loop (InfoNCE + KL)
// ---------------------------------------------------------------------------

interface FoldResult {
  foldIdx: number;
  evalResult: EvalResults;
  finalLoss: number;
  finalAccuracy: number;
  finalKLLoss: number;
}

async function trainFoldMixed(
  foldIdx: number,
  totalFolds: number,
  trainQueries: LiveMCPQuery[],
  testQueries: LiveMCPQuery[],
  tools: LiveMCPTool[],
  nodes: NodeInput[],
  leafIds: string[],
  groundTruth: Map<string, RelevanceLabel[]>,
  klExamples: SoftTargetExample[],
  toolIds: string[],
  modelLabel: string,
  isHierarchical: boolean,
): Promise<FoldResult> {
  const trainQCount = trainQueries.length;
  const testQCount = testQueries.length;
  console.log(`\n--- Fold ${foldIdx + 1}/${totalFolds} (${trainQCount} train, ${testQCount} test, ${klExamples.length} KL) [${modelLabel}] ---`);

  // Build training examples from train queries only
  const trainExamples = buildTrainingExamples(trainQueries, tools, groundTruth, NUM_NEGATIVES);
  console.log(`  InfoNCE: ${trainExamples.length} examples, KL: ${klExamples.length} examples`);

  if (trainExamples.length === 0) {
    console.log("  WARNING: No training examples for this fold, skipping...");
    return {
      foldIdx,
      evalResult: { name: modelLabel, numQueries: 0, metrics: {} },
      finalLoss: NaN,
      finalAccuracy: 0,
      finalKLLoss: NaN,
    };
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
  let lastKLLoss = 0;

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Temperature annealing (InfoNCE)
    const temp = cosineAnneal(epoch, EPOCHS, TEMP_MAX, TEMP_MIN);
    shgat.setTemperature(temp);

    // Curriculum learning for InfoNCE
    if (!NO_CURRICULUM && epoch > 0 && trainExamples[0].allNegativesSorted) {
      for (const ex of trainExamples) {
        if (!ex.allNegativesSorted || ex.allNegativesSorted.length === 0) continue;
        const totalNegs = ex.allNegativesSorted.length;
        const thirdSize = Math.floor(totalNegs / 3);
        let tierStart: number;
        let tierEnd: number;

        if (lastAcc < 0.35) {
          tierStart = totalNegs - thirdSize;
          tierEnd = totalNegs;
        } else if (lastAcc > 0.55) {
          tierStart = 0;
          tierEnd = thirdSize;
        } else {
          tierStart = thirdSize;
          tierEnd = thirdSize * 2;
        }

        const tierSlice = ex.allNegativesSorted.slice(tierStart, tierEnd);
        ex.negativeCapIds = seededShuffle(tierSlice, rng).slice(0, NUM_NEGATIVES);
      }
    }

    // Shuffle both example sets
    const shuffledInfoNCE = seededShuffle(trainExamples, rng);
    const shuffledKL = seededShuffle(klExamples, rng).slice(0, MAX_N8N);

    // --- InfoNCE batches ---
    let epochLoss = 0;
    let epochAcc = 0;
    let numBatchesNCE = 0;

    for (let batchStart = 0; batchStart < shuffledInfoNCE.length; batchStart += BATCH_SIZE) {
      const batch = shuffledInfoNCE.slice(batchStart, batchStart + BATCH_SIZE);
      const metrics = await shgat.trainBatch(batch);
      epochLoss += metrics.loss;
      epochAcc += metrics.accuracy;
      numBatchesNCE++;
    }

    epochLoss /= numBatchesNCE || 1;
    epochAcc /= numBatchesNCE || 1;

    // --- KL batches ---
    let epochKLLoss = 0;
    let numBatchesKL = 0;

    for (let batchStart = 0; batchStart < shuffledKL.length; batchStart += KL_BATCH_SIZE) {
      const batch = shuffledKL.slice(batchStart, batchStart + KL_BATCH_SIZE);
      const klMetrics = await shgat.trainBatchKL(batch, toolIds, KL_TEMP);
      epochKLLoss += klMetrics.klLoss;
      numBatchesKL++;
    }

    epochKLLoss /= numBatchesKL || 1;

    lastLoss = epochLoss;
    lastAcc = epochAcc;
    lastKLLoss = epochKLLoss;

    // Combined loss for logging
    const combinedLoss = epochLoss + KL_WEIGHT * epochKLLoss;

    // Log at selected epochs
    if (epoch === 0 || epoch === 4 || epoch === 9 || epoch === 14 || epoch === 19 || epoch === 24 ||
        epoch === EPOCHS - 1 || (epoch + 1) % 10 === 0) {
      console.log(
        `  Epoch ${String(epoch + 1).padStart(3)}/${EPOCHS}: ` +
        `nce=${epochLoss.toFixed(4)}  kl=${epochKLLoss.toFixed(4)}  ` +
        `combined=${combinedLoss.toFixed(4)}  acc=${epochAcc.toFixed(2)}  ` +
        `temp=${temp.toFixed(3)}`,
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

  return { foldIdx, evalResult, finalLoss: lastLoss, finalAccuracy: lastAcc, finalKLLoss: lastKLLoss };
}

// ---------------------------------------------------------------------------
// Aggregate fold results
// ---------------------------------------------------------------------------

function aggregateFolds(foldResults: FoldResult[], name: string): EvalResults {
  const validFolds = foldResults.filter((f) => f.evalResult.numQueries > 0);
  if (validFolds.length === 0) {
    return { name, numQueries: 0, metrics: {} };
  }

  const allKeys = new Set<string>();
  for (const f of validFolds) {
    for (const key of Object.keys(f.evalResult.metrics)) {
      allKeys.add(key);
    }
  }

  const metrics: Record<string, number> = {};
  for (const key of allKeys) {
    let sum = 0;
    let count = 0;
    for (const f of validFolds) {
      const val = f.evalResult.metrics[key];
      if (val !== undefined) {
        sum += val;
        count++;
      }
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
  console.log("=== SHGAT-TF Mixed Training: InfoNCE + KL (LiveMCPBench + n8n) ===\n");

  const { tools, queries, hierarchy } = loadData();
  const groundTruth = resolveGroundTruth(queries, tools);
  const klExamples = loadN8nKLData();

  const queriesWithGT = queries.filter((q) => {
    const labels = groundTruth.get(q.id);
    return labels && labels.length > 0;
  });

  const totalExamples = buildTrainingExamples(queriesWithGT, tools, groundTruth).length;
  console.log(
    `\nData: ${queries.length} queries, ${tools.length} tools, ~${totalExamples} InfoNCE examples, ${klExamples.length} KL examples`,
  );
  console.log(
    `Config: epochs=${EPOCHS}, lr=${LR}, batch=${BATCH_SIZE}, ` +
    `folds=${OVERFIT_MODE ? "overfit" : FOLDS}, seed=${SEED}`,
  );
  console.log(
    `KL config: weight=${KL_WEIGHT}, temp=${KL_TEMP}, max_n8n=${MAX_N8N}, kl_batch=${KL_BATCH_SIZE}`,
  );
  if (NO_CURRICULUM) console.log("Curriculum learning: DISABLED");
  if (FLAT_ONLY) console.log("Mode: flat-only");
  if (HIER_ONLY) console.log("Mode: hier-only");

  const toolIds = tools.map((t) => t.id);
  const results: EvalResults[] = [];

  // ---- 1. Cosine baseline ----
  console.log("\n--- Cosine Baseline ---");
  const cosineRankings = batchScoreCosine(queries, tools);
  const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
  results.push(cosineResult);
  console.log(
    `  R@1=${((cosineResult.metrics["recall@1"] ?? 0) * 100).toFixed(1)}  ` +
    `R@3=${((cosineResult.metrics["recall@3"] ?? 0) * 100).toFixed(1)}  ` +
    `R@5=${((cosineResult.metrics["recall@5"] ?? 0) * 100).toFixed(1)}`,
  );

  // ---- 2. Build nodes ----
  const { nodes: hierNodes, leafIds } = buildHierarchyNodes(tools, hierarchy);
  const flatNodes: NodeInput[] = tools.map((t) => ({
    id: t.id, embedding: t.embedding, children: [],
  }));

  // ---- 3. Mixed training ----
  if (OVERFIT_MODE) {
    console.log("\n========== OVERFIT MODE (no split) ==========");

    if (!HIER_ONLY) {
      const result = await trainFoldMixed(
        0, 1,
        queriesWithGT, queriesWithGT,
        tools, flatNodes, toolIds, groundTruth,
        klExamples, toolIds,
        "Mixed (Flat, overfit)", false,
      );
      results.push(result.evalResult);
    }

    if (!FLAT_ONLY) {
      const result = await trainFoldMixed(
        0, 1,
        queriesWithGT, queriesWithGT,
        tools, hierNodes, leafIds, groundTruth,
        klExamples, toolIds,
        "Mixed (Hier, overfit)", true,
      );
      results.push(result.evalResult);
    }
  } else {
    // K-fold cross-validation
    const folds = kFoldSplit(queriesWithGT, FOLDS, SEED);

    if (!HIER_ONLY) {
      console.log(`\n========== Mixed Flat (${FOLDS}-fold CV) ==========`);
      const flatFoldResults: FoldResult[] = [];

      for (let fi = 0; fi < folds.length; fi++) {
        const fold = folds[fi];
        const trainQs = queries.filter((q) => fold.trainQueryIds.has(q.id));
        const testQs = queries.filter((q) => fold.testQueryIds.has(q.id));

        const result = await trainFoldMixed(
          fi, folds.length,
          trainQs, testQs,
          tools, flatNodes, toolIds, groundTruth,
          klExamples, toolIds,  // KL examples in ALL folds (augmentation)
          "Mixed (Flat)", false,
        );
        flatFoldResults.push(result);
      }

      const aggFlat = aggregateFolds(flatFoldResults, "Mixed (Flat)");
      results.push(aggFlat);
    }

    if (!FLAT_ONLY) {
      console.log(`\n========== Mixed Hier (${FOLDS}-fold CV) ==========`);
      const hierFoldResults: FoldResult[] = [];

      for (let fi = 0; fi < folds.length; fi++) {
        const fold = folds[fi];
        const trainQs = queries.filter((q) => fold.trainQueryIds.has(q.id));
        const testQs = queries.filter((q) => fold.testQueryIds.has(q.id));

        const result = await trainFoldMixed(
          fi, folds.length,
          trainQs, testQs,
          tools, hierNodes, leafIds, groundTruth,
          klExamples, toolIds,
          "Mixed (Hier)", true,
        );
        hierFoldResults.push(result);
      }

      const aggHier = aggregateFolds(hierFoldResults, "Mixed (Hier)");
      results.push(aggHier);
    }
  }

  // ---- Final results ----
  console.log("\n");
  printResults(results);

  console.log(`Evaluated on ${queriesWithGT.length} queries with ground truth, ${tools.length} tools`);
  console.log(`Config: epochs=${EPOCHS}, lr=${LR}, batch=${BATCH_SIZE}, seed=${SEED}`);
  console.log(`KL: weight=${KL_WEIGHT}, temp=${KL_TEMP}, max_n8n=${MAX_N8N}`);
  if (!OVERFIT_MODE) {
    console.log(`Split: ${FOLDS}-fold cross-validation`);
  } else {
    console.log("Mode: overfit (train+eval on all data)");
  }
}

main().catch((err) => {
  console.error("Training failed:", err);
  process.exit(1);
});
