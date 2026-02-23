/**
 * ToolBench Benchmark Runner
 *
 * Evaluates SHGAT on ToolBench/ToolRet data:
 * 49 categories, ~3400 collections, ~14000 APIs, 1100 queries.
 *
 * 3-level hierarchy: Category → Collection → API
 *
 * Scorers:
 * 1. Cosine: Raw dot product of BGE-M3 embeddings (baseline)
 * 2. SHGAT-Flat: 16-head orthogonal projection, all APIs as leaves
 * 3. SHGAT-Hier: 16-head + 3-level hierarchy
 *
 * Usage:
 *   npx tsx src/run-toolbench.ts                          # Full benchmark
 *   npx tsx src/run-toolbench.ts --cosine-only            # Cosine only
 *   npx tsx src/run-toolbench.ts --sweep                  # Residual sweep
 *   npx tsx src/run-toolbench.ts --dr 0.85 --pdr 0.95,0.6,0.4
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { evaluate, printResults, type RelevanceLabel, type EvalResults } from "./metrics.ts";
import { batchScoreCosine } from "./cosine-baseline.ts";
import { batchScoreSHGAT, type ToolData } from "./shgat-scorer.ts";
import { SHGATBuilder, type NodeInput } from "../../dist-node/mod.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DATA_DIR = resolve(ROOT, "data", "toolbench");

const args = process.argv.slice(2);
const cosineOnly = args.includes("--cosine-only");
const sweepMode = args.includes("--sweep");

function parseArg(flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}
const drArg = parseArg("--dr");
const pdrArg = parseArg("--pdr");
const seedArg = parseArg("--seed");
const SEED = seedArg !== undefined ? parseInt(seedArg, 10) : 42;

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
  categories: Record<string, string[]>;             // category → [coll_key, ...]
  collections: Record<string, string[]>;            // coll_key → [api_id, ...]
  api_to_collection: Record<string, string>;        // api_id → coll_key
  collection_to_category: Record<string, string>;   // coll_key → category
}

interface HierConfig {
  label: string;
  downwardResidual: number;
  preserveDimResiduals?: number[];
}

// ---------------------------------------------------------------------------
// Load data
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
// Ground truth (ToolRet labels already have tool IDs directly)
// ---------------------------------------------------------------------------

function buildGroundTruth(
  queries: ToolBenchQuery[],
  toolIdSet: Set<string>,
): Map<string, RelevanceLabel[]> {
  const groundTruth = new Map<string, RelevanceLabel[]>();
  let resolved = 0, unresolvedTools = 0, queriesWithGT = 0;

  for (const q of queries) {
    const labels: RelevanceLabel[] = [];
    for (const l of q.labels) {
      if (toolIdSet.has(l.id)) {
        labels.push({ id: l.id, relevance: l.relevance });
        resolved++;
      } else {
        unresolvedTools++;
      }
    }
    if (labels.length > 0) queriesWithGT++;
    groundTruth.set(q.id, labels);
  }

  console.log(`[ground-truth] ${resolved} resolved, ${unresolvedTools} unresolved, ${queriesWithGT}/${queries.length} queries with GT`);
  return groundTruth;
}

// ---------------------------------------------------------------------------
// Build hierarchy nodes
// ---------------------------------------------------------------------------

function meanEmbedding(embeddings: number[][], dim: number): number[] {
  const mean = new Array(dim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) mean[i] += emb[i];
  }
  const n = embeddings.length || 1;
  for (let i = 0; i < dim; i++) mean[i] /= n;
  return mean;
}

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

  // L1: Collections (mean of child API embeddings)
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

  // L2: Categories (mean of child collection embeddings)
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
// Run a hierarchical config
// ---------------------------------------------------------------------------

async function runHierConfig(
  config: HierConfig,
  nodes: NodeInput[],
  leafIds: string[],
  queries: ToolBenchQuery[],
  groundTruth: Map<string, RelevanceLabel[]>,
): Promise<EvalResults> {
  console.log(`\n--- ${config.label} ---`);
  console.log(`  DR=${config.downwardResidual}, PDR=${JSON.stringify(config.preserveDimResiduals ?? "default")}`);

  const t0 = performance.now();
  const builder = SHGATBuilder.create().nodes(nodes);
  const archOpts: Record<string, unknown> = { seed: SEED };
  if (config.downwardResidual !== undefined) archOpts.downwardResidual = config.downwardResidual;
  if (config.preserveDimResiduals) archOpts.preserveDimResiduals = config.preserveDimResiduals;
  builder.architecture(archOpts as any);

  const scorer = await builder.build();
  const buildTime = performance.now() - t0;
  console.log(`  Build: ${(buildTime / 1000).toFixed(2)}s, ${nodes.length} nodes, MP=${scorer.hasMessagePassing}`);

  const t1 = performance.now();
  const rankings = batchScoreSHGAT(scorer, queries, leafIds, config.label.toLowerCase().replace(/\s+/g, "-"));
  const scoreTime = performance.now() - t1;
  const result = evaluate(config.label, rankings, groundTruth);
  console.log(`  Score: ${(scoreTime / 1000).toFixed(2)}s`);
  scorer.dispose();
  return result;
}

// ---------------------------------------------------------------------------
// Per-category analysis
// ---------------------------------------------------------------------------

function analyzePerCategory(
  queries: ToolBenchQuery[],
  tools: ToolBenchTool[],
  rankings: Map<string, { id: string; score: number }[]>,
  groundTruth: Map<string, RelevanceLabel[]>,
): void {
  // Group queries by the category of their ground truth tools
  const toolCat = new Map(tools.map(t => [t.id, t.category]));
  const catQueries = new Map<string, string[]>();

  for (const q of queries) {
    const labels = groundTruth.get(q.id) ?? [];
    if (labels.length === 0) continue;
    // Use first GT tool's category
    const cat = toolCat.get(labels[0].id) ?? "unknown";
    if (!catQueries.has(cat)) catQueries.set(cat, []);
    catQueries.get(cat)!.push(q.id);
  }

  console.log("\n  Per-category R@5:");
  const catResults: Array<{ cat: string; n: number; r5: number }> = [];

  for (const [cat, qids] of catQueries) {
    if (qids.length < 3) continue; // skip tiny categories
    let r5sum = 0;
    for (const qid of qids) {
      const ranked = rankings.get(qid) ?? [];
      const labels = groundTruth.get(qid) ?? [];
      const relevant = new Set(labels.filter(l => l.relevance > 0).map(l => l.id));
      let found = 0;
      for (let i = 0; i < Math.min(5, ranked.length); i++) {
        if (relevant.has(ranked[i].id)) found++;
      }
      r5sum += relevant.size > 0 ? found / relevant.size : 0;
    }
    catResults.push({ cat, n: qids.length, r5: r5sum / qids.length });
  }

  catResults.sort((a, b) => b.r5 - a.r5);
  for (const { cat, n, r5 } of catResults.slice(0, 10)) {
    console.log(`    ${cat.padEnd(35)} ${n} queries  R@5=${(r5 * 100).toFixed(1)}%`);
  }
  if (catResults.length > 10) {
    console.log(`    ... and ${catResults.length - 10} more categories`);
  }
}

// ---------------------------------------------------------------------------
// Sweep configs
// ---------------------------------------------------------------------------

const SWEEP_CONFIGS: HierConfig[] = [
  { label: "Hier-PDR[.99,.5,.3]", downwardResidual: 0, preserveDimResiduals: [0.99, 0.5, 0.3] },
  { label: "Hier-PDR[.95,.7,.5]", downwardResidual: 0, preserveDimResiduals: [0.95, 0.7, 0.5] },
  { label: "Hier-DR=0.85", downwardResidual: 0.85 },
  { label: "Hier-DR=0.95", downwardResidual: 0.95 },
  { label: "Hier-DR=0.85+PDR[.95,.5,.3]", downwardResidual: 0.85, preserveDimResiduals: [0.95, 0.5, 0.3] },
  { label: "Hier-DR=1.0 (ctrl)", downwardResidual: 1.0 },
];

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== SHGAT-TF ToolBench Benchmark ===\n");
  console.log("Hierarchy: Category (49) -> Collection (~3400) -> API (~14000)\n");

  const { tools, queries, hierarchy } = loadData();
  const toolIdSet = new Set(tools.map(t => t.id));

  console.log(`[data] ${tools.length} tools, ${queries.length} queries`);
  console.log(
    `[data] ${Object.keys(hierarchy.categories).length} categories, ` +
    `${Object.keys(hierarchy.collections).length} collections`,
  );

  const groundTruth = buildGroundTruth(queries, toolIdSet);
  const leafIds = tools.map((t) => t.id);
  const results: EvalResults[] = [];

  // ---- 1. Cosine baseline ----
  console.log("\n--- Cosine Baseline ---");
  const t0 = performance.now();
  const cosineRankings = batchScoreCosine(queries, tools);
  const cosineTime = performance.now() - t0;
  const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
  results.push(cosineResult);
  console.log(`  Time: ${(cosineTime / 1000).toFixed(2)}s`);

  // Per-category analysis for cosine
  analyzePerCategory(queries, tools, cosineRankings, groundTruth);

  if (cosineOnly) {
    printResults(results);
    return;
  }

  // ---- 2. SHGAT Flat ----
  console.log("\n--- SHGAT Flat ---");
  const t1 = performance.now();
  const flatNodes: NodeInput[] = tools.map((t) => ({
    id: t.id, embedding: t.embedding, children: [],
  }));
  const flatScorer = await SHGATBuilder.create().nodes(flatNodes).architecture({ seed: SEED }).build();
  const buildFlatTime = performance.now() - t1;
  console.log(`  Build: ${(buildFlatTime / 1000).toFixed(2)}s, ${flatNodes.length} nodes, MP=${flatScorer.hasMessagePassing}, seed=${SEED}`);

  const t2 = performance.now();
  const flatRankings = batchScoreSHGAT(flatScorer, queries, leafIds, "shgat-flat");
  const flatScoringTime = performance.now() - t2;
  const flatResult = evaluate("SHGAT-Flat", flatRankings, groundTruth);
  results.push(flatResult);
  console.log(`  Scoring time: ${(flatScoringTime / 1000).toFixed(2)}s`);
  flatScorer.dispose();

  // ---- 3. Hierarchical ----
  const { nodes: hierNodes, leafIds: hierLeafIds } = buildHierarchyNodes(tools, hierarchy);
  const nColls = Object.keys(hierarchy.collections).length;
  const nCats = Object.keys(hierarchy.categories).length;
  console.log(`\n[hier] ${hierNodes.length} nodes: ${hierLeafIds.length} L0 + ${nColls} L1 + ${nCats} L2`);

  if (sweepMode) {
    console.log("\n========== RESIDUAL SWEEP ==========");

    const baselineResult = await runHierConfig(
      { label: "Hier-Baseline (DR=0)", downwardResidual: 0 },
      hierNodes, hierLeafIds, queries, groundTruth,
    );
    results.push(baselineResult);

    for (const config of SWEEP_CONFIGS) {
      const result = await runHierConfig(config, hierNodes, hierLeafIds, queries, groundTruth);
      results.push(result);
    }
  } else if (drArg !== undefined || pdrArg !== undefined) {
    const dr = drArg !== undefined ? parseFloat(drArg) : 0;
    const pdr = pdrArg ? pdrArg.split(",").map(Number) : undefined;
    const label = `Hier-DR=${dr}${pdr ? `+PDR[${pdr.join(",")}]` : ""}`;
    const result = await runHierConfig(
      { label, downwardResidual: dr, preserveDimResiduals: pdr },
      hierNodes, hierLeafIds, queries, groundTruth,
    );
    results.push(result);
  } else {
    // Default: run the most interesting config
    const defaultResult = await runHierConfig(
      { label: "SHGAT-Hier PDR[.99,.5,.3]", downwardResidual: 0, preserveDimResiduals: [0.99, 0.5, 0.3] },
      hierNodes, hierLeafIds, queries, groundTruth,
    );
    results.push(defaultResult);
  }

  // Print results
  printResults(results);
  console.log(`\n${queries.length} queries, ${tools.length} tools, seed=${SEED}`);
}

main().catch((err) => {
  console.error("FATAL:", err);
  process.exit(1);
});
