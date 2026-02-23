/**
 * SHGAT Option B Training Script (LiveMCPBench)
 *
 * Trains SHGAT attention weights using contrastive learning on LiveMCPBench data.
 * Uses hard negative mining, curriculum learning, and temperature annealing.
 *
 * Option B: Learn attention weights via InfoNCE contrastive loss.
 * Compared against Option A (orthogonal projection, no training) and cosine baseline.
 *
 * Usage:
 *   npx tsx src/train-livemcp.ts                       # 5-fold CV, default params
 *   npx tsx src/train-livemcp.ts --epochs 50 --lr 0.01 # Custom epochs/lr
 *   npx tsx src/train-livemcp.ts --overfit              # Train+eval on all data
 *   npx tsx src/train-livemcp.ts --flat-only            # Only flat model
 *   npx tsx src/train-livemcp.ts --hier-only            # Only hierarchical model
 *   npx tsx src/train-livemcp.ts --no-curriculum        # Disable curriculum learning
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
const LR = parseFloat(parseArg("--lr") ?? "0.01");
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
// Data types (same as run-livemcp.ts)
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

// ---------------------------------------------------------------------------
// Data loading (cloned from run-livemcp.ts)
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

// ---------------------------------------------------------------------------
// Ground truth resolution (cloned from run-livemcp.ts)
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

/** Seeded pseudo-random number generator (Mulberry32) */
function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fisher-Yates shuffle with seeded RNG */
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

  // Pre-compute tool embedding index
  const toolEmbeddings = new Map<string, number[]>();
  for (const t of tools) {
    toolEmbeddings.set(t.id, t.embedding);
  }
  const allToolIds = tools.map((t) => t.id);

  for (const query of queries) {
    const labels = groundTruth.get(query.id);
    if (!labels || labels.length === 0) continue;

    const positiveIds = new Set(labels.map((l) => l.id));

    // Compute cosine similarity between query and ALL tools
    const sims: Array<{ id: string; sim: number }> = [];
    for (const toolId of allToolIds) {
      if (positiveIds.has(toolId)) continue;
      const emb = toolEmbeddings.get(toolId)!;
      sims.push({ id: toolId, sim: cosineSim(query.embedding, emb) });
    }

    // Sort descending (hardest negatives first)
    sims.sort((a, b) => b.sim - a.sim);
    const allNegativesSorted = sims.map((s) => s.id);

    for (const positiveLabel of labels) {
      // Top-N hardest negatives
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

function kFoldSplit(queries: LiveMCPQuery[], k: number, seed: number): FoldSplit[] {
  const rng = mulberry32(seed);

  // Only queries with ground truth
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
// Build hierarchy nodes (cloned from run-livemcp.ts)
// ---------------------------------------------------------------------------

function buildHierarchyNodes(
  tools: LiveMCPTool[],
  hierarchy: Hierarchy,
): { nodes: NodeInput[]; leafIds: string[] } {
  const dim = tools[0]?.embedding.length ?? 1024;
  const toolById = new Map(tools.map((t) => [t.id, t]));

  const nodes: NodeInput[] = [];
  const leafIds: string[] = [];

  // L0: Tools (leaves)
  for (const tool of tools) {
    nodes.push({ id: tool.id, embedding: tool.embedding, children: [] });
    leafIds.push(tool.id);
  }

  // L1: Servers (mean of child tool embeddings)
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

  // L2: Categories (mean of child server embeddings)
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
  trainQueries: LiveMCPQuery[],
  testQueries: LiveMCPQuery[],
  tools: LiveMCPTool[],
  nodes: NodeInput[],
  leafIds: string[],
  groundTruth: Map<string, RelevanceLabel[]>,
  modelLabel: string,
  isHierarchical: boolean,
): Promise<FoldResult> {
  const trainQCount = trainQueries.length;
  const testQCount = testQueries.length;
  console.log(`\n--- Fold ${foldIdx + 1}/${totalFolds} (${trainQCount} train, ${testQCount} test queries) [${modelLabel}] ---`);

  // Build training examples from train queries only
  const trainExamples = buildTrainingExamples(trainQueries, tools, groundTruth, NUM_NEGATIVES);
  console.log(`  ${trainExamples.length} training examples`);

  if (trainExamples.length === 0) {
    console.log("  WARNING: No training examples for this fold, skipping...");
    return {
      foldIdx,
      evalResult: { name: modelLabel, numQueries: 0, metrics: {} },
      finalLoss: NaN,
      finalAccuracy: 0,
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

  for (let epoch = 0; epoch < EPOCHS; epoch++) {
    // Temperature annealing
    const temp = cosineAnneal(epoch, EPOCHS, TEMP_MAX, TEMP_MIN);
    shgat.setTemperature(temp);

    // Curriculum learning: adjust negatives based on accuracy
    if (!NO_CURRICULUM && epoch > 0 && trainExamples[0].allNegativesSorted) {
      for (const ex of trainExamples) {
        if (!ex.allNegativesSorted || ex.allNegativesSorted.length === 0) continue;
        const totalNegs = ex.allNegativesSorted.length;
        const thirdSize = Math.floor(totalNegs / 3);

        let tierStart: number;
        let tierEnd: number;

        if (lastAcc < 0.35) {
          // Easy negatives (last third)
          tierStart = totalNegs - thirdSize;
          tierEnd = totalNegs;
        } else if (lastAcc > 0.55) {
          // Hard negatives (first third)
          tierStart = 0;
          tierEnd = thirdSize;
        } else {
          // Medium negatives (middle third)
          tierStart = thirdSize;
          tierEnd = thirdSize * 2;
        }

        // Sample NUM_NEGATIVES from the tier
        const tierSlice = ex.allNegativesSorted.slice(tierStart, tierEnd);
        const sampled = seededShuffle(tierSlice, rng).slice(0, NUM_NEGATIVES);
        ex.negativeCapIds = sampled;
      }
    }

    // Shuffle training examples
    const shuffled = seededShuffle(trainExamples, rng);

    // Process in batches
    let epochLoss = 0;
    let epochAcc = 0;
    let epochGrad = 0;
    let numBatches = 0;

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

    // Log at selected epochs
    if (epoch === 0 || epoch === 4 || epoch === 9 || epoch === 14 || epoch === 19 || epoch === 24 ||
        epoch === EPOCHS - 1 || (epoch + 1) % 10 === 0) {
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
  if (validFolds.length === 0) {
    return { name, numQueries: 0, metrics: {} };
  }

  // Collect all metric keys
  const allKeys = new Set<string>();
  for (const f of validFolds) {
    for (const key of Object.keys(f.evalResult.metrics)) {
      allKeys.add(key);
    }
  }

  // Average across folds
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
  console.log("=== SHGAT-TF Option B Training (LiveMCPBench) ===\n");

  const { tools, queries, hierarchy } = loadData();
  const groundTruth = resolveGroundTruth(queries, tools);

  const queriesWithGT = queries.filter((q) => {
    const labels = groundTruth.get(q.id);
    return labels && labels.length > 0;
  });

  const totalExamples = buildTrainingExamples(queriesWithGT, tools, groundTruth).length;
  console.log(
    `\nData: ${queries.length} queries, ${tools.length} tools, ~${totalExamples} examples`,
  );
  console.log(
    `Config: epochs=${EPOCHS}, lr=${LR}, batch=${BATCH_SIZE}, ` +
    `folds=${OVERFIT_MODE ? "overfit" : FOLDS}, seed=${SEED}`,
  );
  if (NO_CURRICULUM) console.log("Curriculum learning: DISABLED");
  if (FLAT_ONLY) console.log("Mode: flat-only");
  if (HIER_ONLY) console.log("Mode: hier-only");

  const toolIds = tools.map((t) => t.id);
  const results: EvalResults[] = [];

  // ---- 1. Cosine baseline (always) ----
  console.log("\n--- Cosine Baseline ---");
  const t0 = performance.now();
  const cosineRankings = batchScoreCosine(queries, tools);
  const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
  results.push(cosineResult);
  console.log(`  Time: ${((performance.now() - t0) / 1000).toFixed(2)}s`);
  console.log(
    `  R@1=${((cosineResult.metrics["recall@1"] ?? 0) * 100).toFixed(1)}  ` +
    `R@3=${((cosineResult.metrics["recall@3"] ?? 0) * 100).toFixed(1)}  ` +
    `R@5=${((cosineResult.metrics["recall@5"] ?? 0) * 100).toFixed(1)}`,
  );

  // ---- 2. SHGAT Option A Flat (reference) ----
  if (!HIER_ONLY) {
    console.log("\n--- SHGAT Option A (Flat, no training) ---");
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
      `R@3=${((flatResult.metrics["recall@3"] ?? 0) * 100).toFixed(1)}  ` +
      `R@5=${((flatResult.metrics["recall@5"] ?? 0) * 100).toFixed(1)}`,
    );
    flatScorer.dispose();
  }

  // ---- 3. Build hierarchy nodes (shared for hier model + flat reuse) ----
  const { nodes: hierNodes, leafIds } = buildHierarchyNodes(tools, hierarchy);
  const flatNodes: NodeInput[] = tools.map((t) => ({
    id: t.id, embedding: t.embedding, children: [],
  }));

  // ---- 4. Option B: Trained models ----
  if (OVERFIT_MODE) {
    // Overfit mode: train and evaluate on ALL data
    console.log("\n========== OVERFIT MODE (no split) ==========");

    if (!HIER_ONLY) {
      const foldResult = await trainFold(
        0, 1,
        queriesWithGT, queriesWithGT,
        tools, flatNodes, toolIds, groundTruth,
        "Option B (Flat, overfit)", false,
      );
      results.push(foldResult.evalResult);
    }

    if (!FLAT_ONLY) {
      const foldResult = await trainFold(
        0, 1,
        queriesWithGT, queriesWithGT,
        tools, hierNodes, leafIds, groundTruth,
        "Option B (Hier, overfit)", true,
      );
      results.push(foldResult.evalResult);
    }
  } else {
    // K-fold cross-validation
    const folds = kFoldSplit(queriesWithGT, FOLDS, SEED);
    const queryById = new Map(queries.map((q) => [q.id, q]));

    if (!HIER_ONLY) {
      console.log(`\n========== Option B Flat (${FOLDS}-fold CV) ==========`);
      const flatFoldResults: FoldResult[] = [];

      for (let fi = 0; fi < folds.length; fi++) {
        const fold = folds[fi];
        const trainQs = queries.filter((q) => fold.trainQueryIds.has(q.id));
        const testQs = queries.filter((q) => fold.testQueryIds.has(q.id));

        const result = await trainFold(
          fi, folds.length,
          trainQs, testQs,
          tools, flatNodes, toolIds, groundTruth,
          "Option B (Flat)", false,
        );
        flatFoldResults.push(result);
      }

      const aggFlat = aggregateFolds(flatFoldResults, "Option B (Flat)");
      results.push(aggFlat);
    }

    if (!FLAT_ONLY) {
      console.log(`\n========== Option B Hier (${FOLDS}-fold CV) ==========`);
      const hierFoldResults: FoldResult[] = [];

      for (let fi = 0; fi < folds.length; fi++) {
        const fold = folds[fi];
        const trainQs = queries.filter((q) => fold.trainQueryIds.has(q.id));
        const testQs = queries.filter((q) => fold.testQueryIds.has(q.id));

        const result = await trainFold(
          fi, folds.length,
          trainQs, testQs,
          tools, hierNodes, leafIds, groundTruth,
          "Option B (Hier)", true,
        );
        hierFoldResults.push(result);
      }

      const aggHier = aggregateFolds(hierFoldResults, "Option B (Hier)");
      results.push(aggHier);
    }
  }

  // ---- Final results ----
  console.log("\n");
  printResults(results);

  console.log(`Evaluated on ${queriesWithGT.length} queries with ground truth, ${tools.length} tools`);
  console.log(`Config: epochs=${EPOCHS}, lr=${LR}, batch=${BATCH_SIZE}, seed=${SEED}`);
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
