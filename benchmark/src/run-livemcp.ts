/**
 * LiveMCPBench Benchmark Runner
 *
 * Evaluates SHGAT Option A (orthogonal projection, no training) on LiveMCPBench:
 * 69 MCP servers, 525 tools, 8 categories, 3-level hierarchy.
 *
 * Scorers:
 * 1. Cosine: Raw dot product of BGE-M3 embeddings (baseline)
 * 2. SHGAT-Flat: 16-head orthogonal projection, all tools as leaves
 * 3. SHGAT-Hier: 16-head + 3-level hierarchy (Category -> Server -> Tool)
 *
 * Usage:
 *   tsx src/run-livemcp.ts                         # Full benchmark (3 scorers)
 *   tsx src/run-livemcp.ts --cosine-only           # Cosine only
 *   tsx src/run-livemcp.ts --sweep                 # 6-run residual sweep (hier only)
 *   tsx src/run-livemcp.ts --dr 0.85 --pdr 0.95,0.6,0.4  # Single hier config
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
const DATA_DIR = resolve(ROOT, "data", "livemcp");

const args = process.argv.slice(2);
const cosineOnly = args.includes("--cosine-only");
const sweepMode = args.includes("--sweep");

// Parse --dr (downwardResidual) and --pdr (preserveDimResiduals)
function parseArg(flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}
const drArg = parseArg("--dr");
const pdrArg = parseArg("--pdr");
const seedArg = parseArg("--seed");
const SEED = seedArg !== undefined ? parseInt(seedArg, 10) : 42;  // Default seed=42 for reproducibility

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
  categories: Record<string, string[]>;      // category -> [server_name, ...]
  servers: Record<string, string[]>;          // server_name -> [tool_id, ...]
  server_category: Record<string, string>;    // server_name -> category
  tool_server: Record<string, string>;        // tool_id -> server_name
}

interface HierConfig {
  label: string;
  downwardResidual: number;
  preserveDimResiduals?: number[];
}

// ---------------------------------------------------------------------------
// Load data
// ---------------------------------------------------------------------------

function loadData(): { tools: LiveMCPTool[]; queries: LiveMCPQuery[]; hierarchy: Hierarchy } {
  const toolsPath = resolve(DATA_DIR, "tools.json");
  const queriesPath = resolve(DATA_DIR, "queries.json");
  const hierPath = resolve(DATA_DIR, "hierarchy.json");

  if (!existsSync(toolsPath) || !existsSync(queriesPath)) {
    console.error(
      "LiveMCPBench data not found. Run:\n" +
      "  python3 scripts/download-livemcp.py\n" +
      "  deno run --allow-all --config deno.json scripts/embed-livemcp.ts",
    );
    process.exit(1);
  }

  const tools: LiveMCPTool[] = JSON.parse(readFileSync(toolsPath, "utf-8"));
  const queries: LiveMCPQuery[] = JSON.parse(readFileSync(queriesPath, "utf-8"));
  const hierarchy: Hierarchy = JSON.parse(readFileSync(hierPath, "utf-8"));

  return { tools, queries, hierarchy };
}

// ---------------------------------------------------------------------------
// Resolve ground truth: tool names -> tool IDs
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
        for (const [name, ids] of nameToIds) {
          if (name.replace(/[-_]/g, "") === normalized) {
            for (const id of ids) {
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
// Build hierarchy nodes (reusable across runs)
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
      const toolIds = hierarchy.servers[sName] ?? [];
      const childEmbs = toolIds
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
// Run a single hierarchical config
// ---------------------------------------------------------------------------

async function runHierConfig(
  config: HierConfig,
  nodes: NodeInput[],
  leafIds: string[],
  queries: LiveMCPQuery[],
  groundTruth: Map<string, RelevanceLabel[]>,
): Promise<EvalResults> {
  console.log(`\n--- ${config.label} ---`);
  console.log(`  downwardResidual=${config.downwardResidual}, preserveDimResiduals=${JSON.stringify(config.preserveDimResiduals ?? "default")}`);

  const t0 = performance.now();
  const builder = SHGATBuilder.create().nodes(nodes);

  const archOpts: Record<string, unknown> = { seed: SEED };
  if (config.downwardResidual !== undefined) archOpts.downwardResidual = config.downwardResidual;
  if (config.preserveDimResiduals) archOpts.preserveDimResiduals = config.preserveDimResiduals;

  builder.architecture(archOpts as any);

  const scorer = await builder.build();
  const buildTime = performance.now() - t0;
  console.log(`  Build: ${(buildTime / 1000).toFixed(2)}s, MP=${scorer.hasMessagePassing}`);

  const t1 = performance.now();
  const rankings = batchScoreSHGAT(scorer, queries, leafIds, config.label.toLowerCase().replace(/\s+/g, "-"));
  const scoreTime = performance.now() - t1;
  const result = evaluate(config.label, rankings, groundTruth);
  console.log(`  Score: ${(scoreTime / 1000).toFixed(2)}s`);
  scorer.dispose();
  return result;
}

/** Compute mean embedding */
function meanEmbedding(embeddings: number[][], dim: number): number[] {
  const mean = new Array(dim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) mean[i] += emb[i];
  }
  const n = embeddings.length || 1;
  for (let i = 0; i < dim; i++) mean[i] /= n;
  return mean;
}

// ---------------------------------------------------------------------------
// Sweep configurations
// ---------------------------------------------------------------------------

const SWEEP_CONFIGS: HierConfig[] = [
  // Run 1: PerLevel seul, L0 protege a 95%
  { label: "Hier-PDR[.95,.7,.5]", downwardResidual: 0, preserveDimResiduals: [0.95, 0.7, 0.5] },
  // Run 2: PerLevel seul, L0 quasi-parfait
  { label: "Hier-PDR[.99,.5,.3]", downwardResidual: 0, preserveDimResiduals: [0.99, 0.5, 0.3] },
  // Run 3: downwardResidual global seul
  { label: "Hier-DR=0.85", downwardResidual: 0.85 },
  // Run 4: downwardResidual agressif
  { label: "Hier-DR=0.95", downwardResidual: 0.95 },
  // Run 5: Double residual
  { label: "Hier-DR=0.85+PDR[.95,.5,.3]", downwardResidual: 0.85, preserveDimResiduals: [0.95, 0.5, 0.3] },
  // Run 6: Controle (= SHGAT-Flat equivalent, MP desactive)
  { label: "Hier-DR=1.0 (ctrl)", downwardResidual: 1.0 },
];

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main(): Promise<void> {
  console.log("=== SHGAT-TF LiveMCPBench Benchmark ===\n");
  console.log("Hierarchy: Category (8) -> MCP Server (69) -> Tool (525)\n");

  const { tools, queries, hierarchy } = loadData();

  console.log(`[data] ${tools.length} tools, ${queries.length} queries`);
  console.log(
    `[data] ${Object.keys(hierarchy.categories).length} categories, ` +
    `${Object.keys(hierarchy.servers).length} servers`,
  );

  const groundTruth = resolveGroundTruth(queries, tools);
  const queriesWithGT = [...groundTruth.entries()].filter(([, labels]) => labels.length > 0);
  console.log(`[data] ${queriesWithGT.length}/${queries.length} queries have resolved ground truth\n`);

  const toolIds = tools.map((t) => t.id);
  const results: EvalResults[] = [];

  // ---- 1. Cosine baseline (always) ----
  console.log("--- Cosine Baseline ---");
  const t0 = performance.now();
  const cosineRankings = batchScoreCosine(queries, tools);
  const cosineTime = performance.now() - t0;
  const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
  results.push(cosineResult);
  console.log(`  Time: ${(cosineTime / 1000).toFixed(2)}s`);

  if (cosineOnly) {
    console.log("\n");
    printResults(results);
    return;
  }

  // ---- 2. SHGAT Flat (reference) ----
  console.log("\n--- SHGAT Option A (Flat) ---");
  const t1 = performance.now();
  const flatNodes: NodeInput[] = tools.map((t) => ({
    id: t.id, embedding: t.embedding, children: [],
  }));
  const flatScorer = await SHGATBuilder.create().nodes(flatNodes).architecture({ seed: SEED }).build();
  const buildFlatTime = performance.now() - t1;
  console.log(`  Build time: ${(buildFlatTime / 1000).toFixed(2)}s, MP=${flatScorer.hasMessagePassing}, seed=${SEED}`);
  const t2 = performance.now();
  const flatRankings = batchScoreSHGAT(flatScorer, queries, toolIds, "shgat-flat");
  const flatScoringTime = performance.now() - t2;
  const flatResult = evaluate("SHGAT-Flat", flatRankings, groundTruth);
  results.push(flatResult);
  console.log(`  Scoring time: ${(flatScoringTime / 1000).toFixed(2)}s`);
  flatScorer.dispose();

  // ---- 3. Hierarchical configs ----
  const { nodes: hierNodes, leafIds } = buildHierarchyNodes(tools, hierarchy);
  const nServers = Object.keys(hierarchy.servers).length;
  const nCats = Object.keys(hierarchy.categories).length;
  console.log(`\n[hier] ${hierNodes.length} nodes: ${leafIds.length} L0 + ${nServers} L1 + ${nCats} L2`);

  if (sweepMode) {
    // Sweep mode: run all 6 configs + baseline (no residual)
    console.log("\n========== RESIDUAL SWEEP (6 configs) ==========");

    // Baseline (collapse): no residual
    const baselineResult = await runHierConfig(
      { label: "Hier-Baseline (DR=0)", downwardResidual: 0 },
      hierNodes, leafIds, queries, groundTruth,
    );
    results.push(baselineResult);

    for (const config of SWEEP_CONFIGS) {
      const result = await runHierConfig(config, hierNodes, leafIds, queries, groundTruth);
      results.push(result);
    }
  } else if (drArg !== undefined || pdrArg !== undefined) {
    // Single custom config
    const dr = drArg !== undefined ? parseFloat(drArg) : 0;
    const pdr = pdrArg ? pdrArg.split(",").map(Number) : undefined;
    const label = `Hier-DR=${dr}${pdr ? `+PDR[${pdr.join(",")}]` : ""}`;
    const result = await runHierConfig(
      { label, downwardResidual: dr, preserveDimResiduals: pdr },
      hierNodes, leafIds, queries, groundTruth,
    );
    results.push(result);
  } else {
    // Default: run baseline hier (DR=0, no PDR)
    const defaultResult = await runHierConfig(
      { label: "SHGAT-Hier", downwardResidual: 0 },
      hierNodes, leafIds, queries, groundTruth,
    );
    results.push(defaultResult);
  }

  // ---- Results ----
  console.log("\n");
  printResults(results);

  console.log(`\nEvaluated on ${queriesWithGT.length} queries with ground truth, ${tools.length} tools`);
  console.log("Hierarchy: Category (8) -> MCP Server (69) -> Tool (525)");
}

main().catch((err) => {
  console.error("Benchmark failed:", err);
  process.exit(1);
});
