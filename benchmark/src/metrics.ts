/**
 * Information Retrieval evaluation metrics for ToolRet benchmark.
 *
 * Implements Recall@k, NDCG@k, MAP@k, Precision@k.
 * Follows pytrec_eval conventions used in the ToolRet paper.
 */

/** A single scored result: tool ID + score */
export interface ScoredResult {
  id: string;
  score: number;
}

/** Ground truth label: tool ID + relevance (typically 0 or 1) */
export interface RelevanceLabel {
  id: string;
  relevance: number;
}

/**
 * Compute Recall@k: fraction of relevant items found in top-k
 */
export function recallAtK(
  ranked: ScoredResult[],
  labels: RelevanceLabel[],
  k: number,
): number {
  const relevant = new Set(labels.filter((l) => l.relevance > 0).map((l) => l.id));
  if (relevant.size === 0) return 0;

  const topK = ranked.slice(0, k);
  let found = 0;
  for (const r of topK) {
    if (relevant.has(r.id)) found++;
  }
  return found / relevant.size;
}

/**
 * Compute NDCG@k: Normalized Discounted Cumulative Gain
 */
export function ndcgAtK(
  ranked: ScoredResult[],
  labels: RelevanceLabel[],
  k: number,
): number {
  const relMap = new Map(labels.map((l) => [l.id, l.relevance]));

  // DCG
  let dcg = 0;
  const topK = ranked.slice(0, k);
  for (let i = 0; i < topK.length; i++) {
    const rel = relMap.get(topK[i].id) ?? 0;
    dcg += (Math.pow(2, rel) - 1) / Math.log2(i + 2); // log2(rank+1), rank is 1-indexed
  }

  // Ideal DCG
  const idealRels = labels
    .map((l) => l.relevance)
    .sort((a, b) => b - a)
    .slice(0, k);
  let idcg = 0;
  for (let i = 0; i < idealRels.length; i++) {
    idcg += (Math.pow(2, idealRels[i]) - 1) / Math.log2(i + 2);
  }

  return idcg === 0 ? 0 : dcg / idcg;
}

/**
 * Compute MAP@k: Mean Average Precision at k
 */
export function mapAtK(
  ranked: ScoredResult[],
  labels: RelevanceLabel[],
  k: number,
): number {
  const relevant = new Set(labels.filter((l) => l.relevance > 0).map((l) => l.id));
  if (relevant.size === 0) return 0;

  const topK = ranked.slice(0, k);
  let sumPrecision = 0;
  let hits = 0;
  for (let i = 0; i < topK.length; i++) {
    if (relevant.has(topK[i].id)) {
      hits++;
      sumPrecision += hits / (i + 1);
    }
  }
  return sumPrecision / relevant.size;
}

/**
 * Compute Precision@k: fraction of top-k that are relevant
 */
export function precisionAtK(
  ranked: ScoredResult[],
  labels: RelevanceLabel[],
  k: number,
): number {
  const relevant = new Set(labels.filter((l) => l.relevance > 0).map((l) => l.id));
  const topK = ranked.slice(0, k);
  let hits = 0;
  for (const r of topK) {
    if (relevant.has(r.id)) hits++;
  }
  return hits / Math.min(k, topK.length || 1);
}

/**
 * Evaluation results for a single scorer across all queries
 */
export interface EvalResults {
  /** Scorer name */
  name: string;
  /** Number of queries evaluated */
  numQueries: number;
  /** Metrics at various k values */
  metrics: {
    [metricAtK: string]: number; // e.g. "recall@5": 0.42
  };
}

/**
 * Evaluate a scorer's rankings across all queries.
 *
 * @param name - Scorer name
 * @param rankings - Map of query ID → ranked results
 * @param groundTruth - Map of query ID → relevance labels
 * @param ks - k values to evaluate at (default: [1, 3, 5, 10, 20])
 */
export function evaluate(
  name: string,
  rankings: Map<string, ScoredResult[]>,
  groundTruth: Map<string, RelevanceLabel[]>,
  ks: number[] = [1, 3, 5, 10, 20],
): EvalResults {
  const metrics: Record<string, number> = {};

  // Initialize accumulators
  for (const k of ks) {
    metrics[`recall@${k}`] = 0;
    metrics[`ndcg@${k}`] = 0;
    metrics[`map@${k}`] = 0;
    metrics[`precision@${k}`] = 0;
  }

  let numQueries = 0;
  for (const [qid, ranked] of rankings) {
    const labels = groundTruth.get(qid);
    if (!labels || labels.length === 0) continue;

    numQueries++;
    for (const k of ks) {
      metrics[`recall@${k}`] += recallAtK(ranked, labels, k);
      metrics[`ndcg@${k}`] += ndcgAtK(ranked, labels, k);
      metrics[`map@${k}`] += mapAtK(ranked, labels, k);
      metrics[`precision@${k}`] += precisionAtK(ranked, labels, k);
    }
  }

  // Average
  if (numQueries > 0) {
    for (const key of Object.keys(metrics)) {
      metrics[key] /= numQueries;
    }
  }

  return { name, numQueries, metrics };
}

/**
 * Print evaluation results as a formatted table.
 */
export function printResults(results: EvalResults[]): void {
  const ks = [1, 3, 5, 10, 20];

  // Header
  const header = ["Scorer", ...ks.flatMap((k) => [`R@${k}`, `N@${k}`])];
  const widths = header.map((h) => Math.max(h.length, 7));
  widths[0] = Math.max(widths[0], ...results.map((r) => r.name.length));

  const pad = (s: string, w: number) => s.padEnd(w);
  const num = (n: number, w: number) => (n * 100).toFixed(1).padStart(w);

  console.log("\n" + header.map((h, i) => pad(h, widths[i])).join(" │ "));
  console.log(widths.map((w) => "─".repeat(w)).join("─┼─"));

  for (const r of results) {
    const cells = [pad(r.name, widths[0])];
    for (let ki = 0; ki < ks.length; ki++) {
      const k = ks[ki];
      cells.push(num(r.metrics[`recall@${k}`] ?? 0, widths[1 + ki * 2]));
      cells.push(num(r.metrics[`ndcg@${k}`] ?? 0, widths[2 + ki * 2]));
    }
    console.log(cells.join(" │ "));
  }

  console.log("");
}
