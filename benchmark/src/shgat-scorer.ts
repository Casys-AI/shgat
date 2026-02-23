/**
 * SHGAT Option A scorer for ToolRet benchmark.
 *
 * Option A: Orthogonal projection (Gram-Schmidt QR), no training.
 * - W_q = W_k = shared orthogonal matrix
 * - 16 heads × 64 dim = 1024 (matches BGE-M3)
 * - JL lemma guarantees cosine similarity preservation
 *
 * Two variants:
 * 1. SHGAT-Flat: All tools as leaves, no hierarchy (tests K-head attention only)
 * 2. SHGAT-Hier: Category → Tool/Server → API hierarchy (tests message passing)
 */

import type { ScoredResult } from "./metrics.ts";

// Import from dist-node (built by ../scripts/build-node.sh)
import { SHGATBuilder, type SHGATScorer, type NodeInput } from "../../dist-node/mod.ts";

export interface ToolData {
  id: string;
  embedding: number[];
  category_name?: string | null;
  tool_name?: string | null;
  api_name?: string | null;
}

/**
 * Build a flat SHGAT scorer (all tools as leaves, no hierarchy).
 *
 * Tests whether K-head orthogonal projection outperforms raw cosine.
 */
export async function buildFlatScorer(tools: ToolData[]): Promise<SHGATScorer> {
  console.log(`  [shgat-flat] Building with ${tools.length} leaf nodes...`);

  const nodes: NodeInput[] = tools.map((t) => ({
    id: t.id,
    embedding: t.embedding,
    children: [],
  }));

  const scorer = await SHGATBuilder.create()
    .nodes(nodes)
    .build();

  console.log(`  [shgat-flat] Built. MP=${scorer.hasMessagePassing}`);
  return scorer;
}

/**
 * Build a hierarchical SHGAT scorer from ToolBench structure.
 *
 * Hierarchy:
 * - L0: API endpoints (leaves) — the actual tools to score
 * - L1: Tool/Server composites — group APIs by tool
 * - L2: Category composites — group tools by category
 *
 * Tests whether hierarchical message passing adds value.
 */
export async function buildHierarchicalScorer(
  tools: ToolData[],
): Promise<{ scorer: SHGATScorer; leafIds: string[] }> {
  // Group tools by hierarchy
  const byToolServer = new Map<string, ToolData[]>();
  const toolServerCategory = new Map<string, string>();
  const noHierarchy: ToolData[] = [];

  for (const tool of tools) {
    if (tool.tool_name && tool.category_name) {
      const key = tool.tool_name;
      if (!byToolServer.has(key)) byToolServer.set(key, []);
      byToolServer.get(key)!.push(tool);
      toolServerCategory.set(key, tool.category_name);
    } else {
      noHierarchy.push(tool);
    }
  }

  const byCategory = new Map<string, Set<string>>();
  for (const [toolName, cat] of toolServerCategory) {
    if (!byCategory.has(cat)) byCategory.set(cat, new Set());
    byCategory.get(cat)!.add(toolName);
  }

  console.log(
    `  [shgat-hier] ${tools.length} APIs, ${byToolServer.size} tool servers, ` +
    `${byCategory.size} categories, ${noHierarchy.length} orphans`,
  );

  const dim = tools[0]?.embedding.length ?? 1024;
  const nodes: NodeInput[] = [];
  const leafIds: string[] = [];

  // L0: API endpoints (leaves)
  for (const tool of tools) {
    nodes.push({
      id: tool.id,
      embedding: tool.embedding,
      children: [],
    });
    leafIds.push(tool.id);
  }

  // L1: Tool/Server composites (mean of child embeddings)
  for (const [toolName, apis] of byToolServer) {
    const serverId = `server:${toolName}`;
    const meanEmb = meanEmbedding(apis.map((a) => a.embedding), dim);
    nodes.push({
      id: serverId,
      embedding: meanEmb,
      children: apis.map((a) => a.id),
    });
  }

  // L2: Category composites (mean of server embeddings)
  for (const [catName, serverNames] of byCategory) {
    const catId = `category:${catName}`;
    const serverIds = Array.from(serverNames).map((s) => `server:${s}`);

    // Mean of server embeddings (which are already means of API embeddings)
    const serverEmbs: number[][] = [];
    for (const sName of serverNames) {
      const apis = byToolServer.get(sName);
      if (apis) {
        serverEmbs.push(meanEmbedding(apis.map((a) => a.embedding), dim));
      }
    }
    const catEmb = serverEmbs.length > 0 ? meanEmbedding(serverEmbs, dim) : new Array(dim).fill(0);

    nodes.push({
      id: catId,
      embedding: catEmb,
      children: serverIds,
    });
  }

  console.log(`  [shgat-hier] Total nodes: ${nodes.length} (${leafIds.length} L0 + ${byToolServer.size} L1 + ${byCategory.size} L2)`);

  const scorer = await SHGATBuilder.create()
    .nodes(nodes)
    .build();

  console.log(`  [shgat-hier] Built. MP=${scorer.hasMessagePassing}`);
  return { scorer, leafIds };
}

/**
 * Score all tools against a query using a SHGAT scorer.
 */
function scoreSHGAT(
  scorer: SHGATScorer,
  queryEmbedding: number[],
  toolIds: string[],
): ScoredResult[] {
  const scores = scorer.score(queryEmbedding, toolIds);
  const results: ScoredResult[] = toolIds.map((id, i) => ({ id, score: scores[i] }));
  results.sort((a, b) => b.score - a.score);
  return results;
}

/**
 * Batch score all queries using a SHGAT scorer.
 */
export function batchScoreSHGAT(
  scorer: SHGATScorer,
  queries: Array<{ id: string; embedding: number[] }>,
  toolIds: string[],
  label: string = "shgat",
): Map<string, ScoredResult[]> {
  const rankings = new Map<string, ScoredResult[]>();

  for (let qi = 0; qi < queries.length; qi++) {
    const q = queries[qi];
    rankings.set(q.id, scoreSHGAT(scorer, q.embedding, toolIds));
    if ((qi + 1) % 50 === 0) {
      process.stdout.write(`\r  [${label}] ${qi + 1}/${queries.length} queries`);
    }
  }
  if (queries.length >= 50) console.log("");

  return rankings;
}

/** Compute mean embedding from a set of embeddings */
function meanEmbedding(embeddings: number[][], dim: number): number[] {
  const mean = new Array(dim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      mean[i] += emb[i];
    }
  }
  const n = embeddings.length || 1;
  for (let i = 0; i < dim; i++) {
    mean[i] /= n;
  }
  return mean;
}
