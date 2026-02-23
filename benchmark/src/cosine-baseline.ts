/**
 * Cosine similarity baseline scorer.
 *
 * Pure dot-product scoring (no SHGAT, no message passing).
 * Since BGE-M3 embeddings are L2-normalized, dot product = cosine similarity.
 *
 * This is the baseline to beat — SHGAT should outperform this
 * to justify the overhead of orthogonal projection + message passing.
 */

import type { ScoredResult } from "./metrics.ts";

/**
 * Score all tools against a query using cosine similarity.
 *
 * @param queryEmbedding - L2-normalized query embedding (1024-dim)
 * @param tools - Array of {id, embedding} objects
 * @returns Scored results sorted by score descending
 */
export function scoreCosine(
  queryEmbedding: number[],
  tools: Array<{ id: string; embedding: number[] }>,
): ScoredResult[] {
  const results: ScoredResult[] = [];

  for (const tool of tools) {
    let score = 0;
    for (let i = 0; i < queryEmbedding.length; i++) {
      score += queryEmbedding[i] * tool.embedding[i];
    }
    results.push({ id: tool.id, score });
  }

  results.sort((a, b) => b.score - a.score);
  return results;
}

/**
 * Batch score all queries against all tools.
 *
 * @param queries - Array of {id, embedding} objects
 * @param tools - Array of {id, embedding} objects
 * @returns Map of query ID → ranked results
 */
export function batchScoreCosine(
  queries: Array<{ id: string; embedding: number[] }>,
  tools: Array<{ id: string; embedding: number[] }>,
): Map<string, ScoredResult[]> {
  const rankings = new Map<string, ScoredResult[]>();

  for (let qi = 0; qi < queries.length; qi++) {
    const q = queries[qi];
    rankings.set(q.id, scoreCosine(q.embedding, tools));
    if ((qi + 1) % 100 === 0) {
      process.stdout.write(`\r  [cosine] ${qi + 1}/${queries.length} queries`);
    }
  }
  if (queries.length >= 100) console.log("");

  return rankings;
}
