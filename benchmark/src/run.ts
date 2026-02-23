/**
 * SHGAT-TF ToolRet Benchmark Runner
 *
 * Evaluates SHGAT Option A (orthogonal projection, no training)
 * against raw cosine similarity on the ToolRet benchmark.
 *
 * Three scorers:
 * 1. Cosine: Raw dot product of BGE-M3 embeddings (baseline)
 * 2. SHGAT-Flat: 16-head orthogonal projection, all tools as leaves
 * 3. SHGAT-Hier: 16-head + 3-level hierarchy (Category → Server → API)
 *
 * Usage:
 *   tsx src/run.ts                  # Full benchmark (requires data/)
 *   tsx src/run.ts --synthetic      # Synthetic data (no download needed)
 *   tsx src/run.ts --cosine-only    # Only cosine baseline
 *   tsx src/run.ts --limit 100      # Limit to first 100 queries
 */

import { readFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { evaluate, printResults, type RelevanceLabel, type EvalResults } from "./metrics.ts";
import { batchScoreCosine } from "./cosine-baseline.ts";
import { buildFlatScorer, buildHierarchicalScorer, batchScoreSHGAT, type ToolData } from "./shgat-scorer.ts";
import { generateSyntheticData, type SyntheticQuery } from "./synthetic.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------
const args = process.argv.slice(2);
const useSynthetic = args.includes("--synthetic");
const cosineOnly = args.includes("--cosine-only");
const limitIdx = args.indexOf("--limit");
const limit = limitIdx >= 0 ? parseInt(args[limitIdx + 1], 10) : Infinity;

// ---------------------------------------------------------------------------
// Load data
// ---------------------------------------------------------------------------

interface QueryData {
  id: string;
  query: string;
  embedding: number[];
  labels: RelevanceLabel[];
}

function loadRealData(): { tools: ToolData[]; queries: QueryData[] } {
  const toolsPath = resolve(ROOT, "data/tools.json");
  const queriesPath = resolve(ROOT, "data/queries.json");

  if (!existsSync(toolsPath) || !existsSync(queriesPath)) {
    console.error(
      "Data not found. Run 'python3 scripts/prepare-data.py' first.\n" +
      "Or use --synthetic for testing without real data.",
    );
    process.exit(1);
  }

  console.log("[load] Reading tools...");
  const tools: ToolData[] = JSON.parse(readFileSync(toolsPath, "utf-8"));

  console.log("[load] Reading queries...");
  const rawQueries = JSON.parse(readFileSync(queriesPath, "utf-8"));
  const queries: QueryData[] = rawQueries.map((q: any) => ({
    id: q.id,
    query: q.query,
    embedding: q.embedding,
    labels: q.labels.map((l: any) => ({ id: l.id, relevance: l.relevance })),
  }));

  // Validate embeddings exist
  const toolsWithEmb = tools.filter((t) => t.embedding && t.embedding.length > 0);
  const queriesWithEmb = queries.filter((q) => q.embedding && q.embedding.length > 0);

  if (toolsWithEmb.length === 0 || queriesWithEmb.length === 0) {
    console.error(
      "Embeddings missing. Re-run 'python3 scripts/prepare-data.py' without --skip-embed.",
    );
    process.exit(1);
  }

  return { tools: toolsWithEmb, queries: queriesWithEmb };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== SHGAT-TF ToolRet Benchmark ===\n");

  let tools: ToolData[];
  let queries: QueryData[];

  if (useSynthetic) {
    console.log("[mode] Synthetic data (for pipeline testing)\n");
    const synth = generateSyntheticData();
    tools = synth.tools;
    queries = synth.queries;
  } else {
    console.log("[mode] Real ToolRet data\n");
    const data = loadRealData();
    tools = data.tools;
    queries = data.queries;
  }

  // Apply limit
  if (limit < queries.length) {
    queries = queries.slice(0, limit);
    console.log(`[limit] Using first ${limit} queries\n`);
  }

  // Build ground truth map
  const groundTruth = new Map<string, RelevanceLabel[]>();
  for (const q of queries) {
    groundTruth.set(q.id, q.labels);
  }

  const dim = tools[0]?.embedding.length ?? 1024;
  const toolIds = tools.map((t) => t.id);

  console.log(`[data] ${tools.length} tools, ${queries.length} queries, dim=${dim}`);

  // Count hierarchy
  const categories = new Set(tools.map((t) => t.category_name).filter(Boolean));
  const servers = new Set(tools.map((t) => t.tool_name).filter(Boolean));
  console.log(`[data] Hierarchy: ${categories.size} categories, ${servers.size} tool servers\n`);

  const results: EvalResults[] = [];

  // ---- 1. Cosine baseline ----
  console.log("--- Cosine Baseline ---");
  const t0 = performance.now();
  const cosineRankings = batchScoreCosine(queries, tools);
  const cosineTime = performance.now() - t0;
  const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
  results.push(cosineResult);
  console.log(`  Time: ${(cosineTime / 1000).toFixed(1)}s`);

  if (!cosineOnly) {
    // ---- 2. SHGAT Flat ----
    console.log("\n--- SHGAT Option A (Flat) ---");
    const t1 = performance.now();
    const flatScorer = await buildFlatScorer(tools);
    const buildFlatTime = performance.now() - t1;
    console.log(`  Build time: ${(buildFlatTime / 1000).toFixed(1)}s`);

    const t2 = performance.now();
    const flatRankings = batchScoreSHGAT(flatScorer, queries, toolIds, "shgat-flat");
    const flatScoringTime = performance.now() - t2;
    const flatResult = evaluate("SHGAT-Flat", flatRankings, groundTruth);
    results.push(flatResult);
    console.log(`  Scoring time: ${(flatScoringTime / 1000).toFixed(1)}s`);
    flatScorer.dispose();

    // ---- 3. SHGAT Hierarchical ----
    console.log("\n--- SHGAT Option A (Hierarchical) ---");
    const t3 = performance.now();
    const { scorer: hierScorer, leafIds } = await buildHierarchicalScorer(tools);
    const buildHierTime = performance.now() - t3;
    console.log(`  Build time: ${(buildHierTime / 1000).toFixed(1)}s`);

    const t4 = performance.now();
    const hierRankings = batchScoreSHGAT(hierScorer, queries, leafIds, "shgat-hier");
    const hierScoringTime = performance.now() - t4;
    const hierResult = evaluate("SHGAT-Hier", hierRankings, groundTruth);
    results.push(hierResult);
    console.log(`  Scoring time: ${(hierScoringTime / 1000).toFixed(1)}s`);
    hierScorer.dispose();
  }

  // ---- Results ----
  printResults(results);

  // Print summary
  console.log("Legend: R@k = Recall@k (%), N@k = NDCG@k (%)");
  console.log(`Evaluated on ${queries.length} queries, ${tools.length} tools`);
  if (useSynthetic) {
    console.log("(synthetic data — results are for pipeline validation only)");
  }
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
