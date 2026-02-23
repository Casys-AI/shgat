/**
 * Synthetic data generator for testing the benchmark pipeline.
 *
 * Generates a small synthetic ToolBench-like dataset with:
 * - 5 categories, 20 tool servers, 100 API endpoints
 * - 50 queries with ground-truth labels
 * - Random 1024-dim embeddings (L2-normalized)
 *
 * Use this to test the benchmark pipeline without downloading real data.
 */

import type { RelevanceLabel } from "./metrics.ts";
import type { ToolData } from "./shgat-scorer.ts";

/** Synthetic query with embedding and ground truth */
export interface SyntheticQuery {
  id: string;
  query: string;
  embedding: number[];
  labels: RelevanceLabel[];
}

/** Generate a random L2-normalized vector */
function randomNormalized(dim: number): number[] {
  const v = new Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    v[i] = Math.random() * 2 - 1; // [-1, 1]
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) {
    v[i] /= norm;
  }
  return v;
}

/** Generate embedding near a centroid (with noise) */
function nearEmbedding(centroid: number[], noise: number = 0.3): number[] {
  const dim = centroid.length;
  const v = new Array(dim);
  let norm = 0;
  for (let i = 0; i < dim; i++) {
    v[i] = centroid[i] + (Math.random() * 2 - 1) * noise;
    norm += v[i] * v[i];
  }
  norm = Math.sqrt(norm);
  for (let i = 0; i < dim; i++) {
    v[i] /= norm;
  }
  return v;
}

/**
 * Generate synthetic ToolBench-like data.
 *
 * Structure:
 * - 5 categories (Movies, Weather, Finance, Music, Sports)
 * - 4 tool servers per category (20 total)
 * - 5 API endpoints per tool server (100 total)
 * - 50 queries, each targeting 1-3 relevant API endpoints
 *
 * Tool embeddings within the same category cluster together,
 * so hierarchical scoring should outperform flat scoring.
 */
export function generateSyntheticData(dim: number = 1024): {
  tools: ToolData[];
  queries: SyntheticQuery[];
} {
  const categories = ["Movies", "Weather", "Finance", "Music", "Sports"];
  const serversPerCat = 4;
  const apisPerServer = 5;
  const numQueries = 50;

  const tools: ToolData[] = [];

  // Generate category centroids (spread apart)
  const catCentroids = categories.map(() => randomNormalized(dim));

  // Generate tools
  let toolIdx = 0;
  for (let ci = 0; ci < categories.length; ci++) {
    const catName = categories[ci];
    const catCentroid = catCentroids[ci];

    for (let si = 0; si < serversPerCat; si++) {
      const serverName = `${catName.toLowerCase()}_tool_${si + 1}`;
      // Server embedding is near category centroid
      const serverCentroid = nearEmbedding(catCentroid, 0.2);

      for (let ai = 0; ai < apisPerServer; ai++) {
        const apiId = `toolbench_tool_${toolIdx}`;
        // API embedding is near server centroid (even tighter cluster)
        const apiEmb = nearEmbedding(serverCentroid, 0.1);

        tools.push({
          id: apiId,
          embedding: apiEmb,
          category_name: catName,
          tool_name: serverName,
          api_name: `api_${ai + 1}`,
        });
        toolIdx++;
      }
    }
  }

  // Generate queries
  const queries: SyntheticQuery[] = [];
  for (let qi = 0; qi < numQueries; qi++) {
    // Pick 1-3 random relevant tools
    const numRelevant = 1 + Math.floor(Math.random() * 3);
    const relevantIndices = new Set<number>();
    while (relevantIndices.size < numRelevant) {
      relevantIndices.add(Math.floor(Math.random() * tools.length));
    }

    // Query embedding = mean of relevant tool embeddings + noise
    const relevantTools = Array.from(relevantIndices).map((i) => tools[i]);
    const meanEmb = new Array(dim).fill(0);
    for (const t of relevantTools) {
      for (let i = 0; i < dim; i++) {
        meanEmb[i] += t.embedding[i];
      }
    }
    for (let i = 0; i < dim; i++) {
      meanEmb[i] /= relevantTools.length;
    }
    const queryEmb = nearEmbedding(meanEmb, 0.05); // Very close to relevant tools

    queries.push({
      id: `query_${qi}`,
      query: `Synthetic query ${qi} about ${relevantTools.map((t) => t.tool_name).join(", ")}`,
      embedding: queryEmb,
      labels: relevantTools.map((t) => ({ id: t.id, relevance: 1 })),
    });
  }

  return { tools, queries };
}
