#!/usr/bin/env -S deno run --allow-read --allow-env
/**
 * LiveMCPBench benchmark using the PRODUCTION SHGAT (src/graphrag/algorithms/shgat.ts).
 *
 * Compares:
 * 1. Cosine baseline
 * 2. Prod SHGAT Flat (tools only, no capabilities)
 * 3. Prod SHGAT Hier (tools + server caps + category caps)
 *
 * Usage:
 *   deno run --allow-read --allow-env src/run-livemcp-prod.ts
 */

import { SHGAT } from "../../../../src/graphrag/algorithms/shgat.ts";
import type { CapabilityNode, ToolNode } from "../../../../src/graphrag/algorithms/shgat/types.ts";

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

interface RelevanceLabel {
  id: string;
  relevance: number;
}

interface ScoredResult {
  id: string;
  score: number;
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

const DATA_DIR = new URL("../data/livemcp/", import.meta.url).pathname;

function loadData() {
  const tools: LiveMCPTool[] = JSON.parse(Deno.readTextFileSync(`${DATA_DIR}/tools.json`));
  const queries: LiveMCPQuery[] = JSON.parse(Deno.readTextFileSync(`${DATA_DIR}/queries.json`));
  const hierarchy: Hierarchy = JSON.parse(Deno.readTextFileSync(`${DATA_DIR}/hierarchy.json`));
  return { tools, queries, hierarchy };
}

// ---------------------------------------------------------------------------
// Ground truth resolution (same as run-livemcp.ts)
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
  let resolved = 0, unresolved = 0;

  for (const q of queries) {
    const labels: RelevanceLabel[] = [];
    for (const toolName of q.ground_truth_tools) {
      const ids = nameToIds.get(toolName.toLowerCase());
      if (ids) {
        for (const id of ids) labels.push({ id, relevance: 1 });
        resolved++;
      } else {
        const normalized = toolName.toLowerCase().replace(/[-_]/g, "");
        let found = false;
        for (const [name, ids2] of nameToIds) {
          if (name.replace(/[-_]/g, "") === normalized) {
            for (const id of ids2) labels.push({ id, relevance: 1 });
            resolved++;
            found = true;
            break;
          }
        }
        if (!found) unresolved++;
      }
    }
    groundTruth.set(q.id, labels);
  }

  console.log(`[ground-truth] Resolved ${resolved}, ${unresolved} unresolved`);
  return groundTruth;
}

// ---------------------------------------------------------------------------
// Metrics (inline — avoid Node import)
// ---------------------------------------------------------------------------

function recallAtK(ranked: ScoredResult[], labels: RelevanceLabel[], k: number): number {
  const relevant = new Set(labels.filter(l => l.relevance > 0).map(l => l.id));
  if (relevant.size === 0) return 0;
  let found = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i++) {
    if (relevant.has(ranked[i].id)) found++;
  }
  return found / relevant.size;
}

function ndcgAtK(ranked: ScoredResult[], labels: RelevanceLabel[], k: number): number {
  const relMap = new Map(labels.map(l => [l.id, l.relevance]));
  let dcg = 0;
  for (let i = 0; i < Math.min(k, ranked.length); i++) {
    const rel = relMap.get(ranked[i].id) ?? 0;
    dcg += rel / Math.log2(i + 2);
  }
  const sortedRels = labels.map(l => l.relevance).sort((a, b) => b - a);
  let idcg = 0;
  for (let i = 0; i < Math.min(k, sortedRels.length); i++) {
    idcg += sortedRels[i] / Math.log2(i + 2);
  }
  return idcg > 0 ? dcg / idcg : 0;
}

function evaluate(
  name: string,
  rankings: Map<string, ScoredResult[]>,
  groundTruth: Map<string, RelevanceLabel[]>,
) {
  const ks = [1, 3, 5, 10, 20];
  const metrics: Record<string, number> = {};
  for (const k of ks) { metrics[`R@${k}`] = 0; metrics[`N@${k}`] = 0; }

  let n = 0;
  for (const [qid, ranked] of rankings) {
    const labels = groundTruth.get(qid);
    if (!labels || labels.length === 0) continue;
    n++;
    for (const k of ks) {
      metrics[`R@${k}`] += recallAtK(ranked, labels, k);
      metrics[`N@${k}`] += ndcgAtK(ranked, labels, k);
    }
  }

  if (n > 0) {
    for (const key of Object.keys(metrics)) metrics[key] /= n;
  }

  return { name, numQueries: n, metrics };
}

// ---------------------------------------------------------------------------
// Cosine baseline
// ---------------------------------------------------------------------------

function cosineSim(a: number[], b: number[]): number {
  let dot = 0, nA = 0, nB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; nA += a[i] * a[i]; nB += b[i] * b[i];
  }
  return dot / (Math.sqrt(nA) * Math.sqrt(nB) + 1e-8);
}

function batchScoreCosine(
  queries: LiveMCPQuery[],
  tools: LiveMCPTool[],
): Map<string, ScoredResult[]> {
  const rankings = new Map<string, ScoredResult[]>();
  for (const q of queries) {
    const results: ScoredResult[] = tools.map(t => ({
      id: t.id, score: cosineSim(q.embedding, t.embedding),
    }));
    results.sort((a, b) => b.score - a.score);
    rankings.set(q.id, results);
  }
  return rankings;
}

// ---------------------------------------------------------------------------
// Helper: mean embedding
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

// ---------------------------------------------------------------------------
// Build Prod SHGAT (Flat: tools only)
// ---------------------------------------------------------------------------

function buildProdFlat(tools: LiveMCPTool[]): SHGAT {
  const shgat = new SHGAT({
    preserveDim: true,
    preserveDimResidual: 0.3,
    numHeads: 16,
    hiddenDim: 1024,
    headDim: 64,
    embeddingDim: 1024,
  });

  for (const t of tools) {
    const node: ToolNode = {
      id: t.id,
      embedding: t.embedding,
    };
    shgat.registerTool(node);
  }

  return shgat;
}

// ---------------------------------------------------------------------------
// Build Prod SHGAT (Hier: tools + server caps + category caps)
// ---------------------------------------------------------------------------

function buildProdHier(tools: LiveMCPTool[], hierarchy: Hierarchy): SHGAT {
  const dim = tools[0]?.embedding.length ?? 1024;
  const toolById = new Map(tools.map(t => [t.id, t]));

  const shgat = new SHGAT({
    preserveDim: true,
    preserveDimResidual: 0.3,
    numHeads: 16,
    hiddenDim: 1024,
    headDim: 64,
    embeddingDim: 1024,
  });

  // L0: Register tools
  for (const t of tools) {
    shgat.registerTool({ id: t.id, embedding: t.embedding });
  }

  // L1: Server capabilities (contain tools)
  for (const [serverName, toolIds] of Object.entries(hierarchy.servers)) {
    const validToolIds = toolIds.filter(tid => toolById.has(tid));
    if (validToolIds.length === 0) continue;

    const childEmbs = validToolIds
      .map(tid => toolById.get(tid)?.embedding)
      .filter((e): e is number[] => !!e);

    const cap: CapabilityNode = {
      id: `server:${serverName}`,
      embedding: meanEmbedding(childEmbs, dim),
      members: validToolIds.map(tid => ({ type: "tool" as const, id: tid })),
      hierarchyLevel: 0, // computed by rebuildHierarchy
      successRate: 0.8,
    };
    shgat.registerCapability(cap);
  }

  // L2: Category capabilities (contain servers)
  for (const [catName, serverNames] of Object.entries(hierarchy.categories)) {
    const serverIds: string[] = [];
    const serverEmbs: number[][] = [];

    for (const sName of serverNames) {
      const tIds = hierarchy.servers[sName] ?? [];
      const childEmbs = tIds
        .map(tid => toolById.get(tid)?.embedding)
        .filter((e): e is number[] => !!e);
      if (childEmbs.length > 0) {
        serverIds.push(`server:${sName}`);
        serverEmbs.push(meanEmbedding(childEmbs, dim));
      }
    }

    if (serverEmbs.length === 0) continue;

    const cap: CapabilityNode = {
      id: `category:${catName}`,
      embedding: meanEmbedding(serverEmbs, dim),
      members: serverIds.map(sid => ({ type: "capability" as const, id: sid })),
      hierarchyLevel: 0, // computed by rebuildHierarchy
      successRate: 0.8,
    };
    shgat.registerCapability(cap);
  }

  return shgat;
}

// ---------------------------------------------------------------------------
// Score queries with Prod SHGAT
// ---------------------------------------------------------------------------

function batchScoreProd(
  shgat: SHGAT,
  queries: LiveMCPQuery[],
  toolIds: string[],
  label: string,
): Map<string, ScoredResult[]> {
  const toolIdSet = new Set(toolIds);
  const rankings = new Map<string, ScoredResult[]>();

  for (let qi = 0; qi < queries.length; qi++) {
    const q = queries[qi];
    const toolScores = shgat.scoreAllTools(q.embedding);

    // Convert to ScoredResult[], filter to toolIds only
    const results: ScoredResult[] = toolScores
      .filter(ts => toolIdSet.has(ts.toolId))
      .map(ts => ({ id: ts.toolId, score: ts.score }));

    // Add any tools not scored with score 0
    const scored = new Set(results.map(r => r.id));
    for (const tid of toolIds) {
      if (!scored.has(tid)) results.push({ id: tid, score: 0 });
    }

    results.sort((a, b) => b.score - a.score);
    rankings.set(q.id, results);

    if ((qi + 1) % 50 === 0) {
      console.log(`  [${label}] ${qi + 1}/${queries.length} queries`);
    }
  }
  return rankings;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function printTable(results: Array<{ name: string; metrics: Record<string, number> }>) {
  const ks = [1, 3, 5, 10, 20];
  console.log("\n" + "Scorer".padEnd(25) + ks.map(k => `R@${k}`.padStart(8) + `N@${k}`.padStart(8)).join(""));
  console.log("─".repeat(25 + ks.length * 16));
  for (const r of results) {
    const cells = r.name.padEnd(25) +
      ks.map(k =>
        ((r.metrics[`R@${k}`] ?? 0) * 100).toFixed(1).padStart(8) +
        ((r.metrics[`N@${k}`] ?? 0) * 100).toFixed(1).padStart(8)
      ).join("");
    console.log(cells);
  }
  console.log("");
}

console.log("=== LiveMCPBench: Prod SHGAT vs Cosine ===\n");

const { tools, queries, hierarchy } = loadData();
const groundTruth = resolveGroundTruth(queries, tools);
const toolIds = tools.map(t => t.id);
const results: Array<{ name: string; numQueries: number; metrics: Record<string, number> }> = [];

// 1. Cosine baseline
console.log("\n--- Cosine Baseline ---");
const t0 = performance.now();
const cosineRankings = batchScoreCosine(queries, tools);
const cosineResult = evaluate("Cosine", cosineRankings, groundTruth);
results.push(cosineResult);
console.log(`  Time: ${((performance.now() - t0) / 1000).toFixed(2)}s`);
console.log(`  R@1=${(cosineResult.metrics["R@1"]! * 100).toFixed(1)}`);

// 2. Prod SHGAT Flat (tools only, no capabilities)
console.log("\n--- Prod SHGAT Flat ---");
const t1 = performance.now();
const shgatFlat = buildProdFlat(tools);
console.log(`  Built: ${tools.length} tools in ${((performance.now() - t1) / 1000).toFixed(2)}s`);
const flatRankings = batchScoreProd(shgatFlat, queries, toolIds, "prod-flat");
const flatResult = evaluate("Prod SHGAT Flat", flatRankings, groundTruth);
results.push(flatResult);
console.log(`  R@1=${(flatResult.metrics["R@1"]! * 100).toFixed(1)}`);

// 3. Prod SHGAT Hier (tools + server caps + category caps)
console.log("\n--- Prod SHGAT Hier ---");
const t2 = performance.now();
const shgatHier = buildProdHier(tools, hierarchy);
console.log(`  Built in ${((performance.now() - t2) / 1000).toFixed(2)}s`);
const hierRankings = batchScoreProd(shgatHier, queries, toolIds, "prod-hier");
const hierResult = evaluate("Prod SHGAT Hier", hierRankings, groundTruth);
results.push(hierResult);
console.log(`  R@1=${(hierResult.metrics["R@1"]! * 100).toFixed(1)}`);

// Results table
printTable(results);
console.log(`${queries.length} queries, ${tools.length} tools`);
