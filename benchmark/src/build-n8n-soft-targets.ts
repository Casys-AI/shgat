/**
 * Build n8n soft targets for SHGAT KL divergence training.
 *
 * For each n8n workflow with a description embedding:
 * 1. Compute cosine similarity between description embedding and all 525 LiveMCPBench tool embeddings
 * 2. Apply softmax with temperature T to get a probability distribution
 * 3. Keep sparse top-K probabilities (re-normalized)
 *
 * No hard mapping n8n→PML tools needed — the cosine similarity IS the training signal.
 * K-heads learn to reproduce and surpass this distribution.
 *
 * Usage:
 *   npx tsx src/build-n8n-soft-targets.ts                          # defaults
 *   npx tsx src/build-n8n-soft-targets.ts --temp 0.005 --top-k 15  # custom
 *   npx tsx src/build-n8n-soft-targets.ts --use-names              # embed wf names as fallback
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(__dirname, "..");
const DATA_DIR = resolve(ROOT, "data", "livemcp");
const GRU_DATA_DIR = resolve(__dirname, "../../../../lib/gru/data");

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const args = process.argv.slice(2);

function parseArg(flag: string): string | undefined {
  const idx = args.indexOf(flag);
  return idx >= 0 && idx + 1 < args.length ? args[idx + 1] : undefined;
}

function hasFlag(flag: string): boolean {
  return args.includes(flag);
}

const TEMPERATURE = parseFloat(parseArg("--temp") ?? "0.01");
const TOP_K = parseInt(parseArg("--top-k") ?? "10", 10);
const MIN_MAX_SIM = parseFloat(parseArg("--min-sim") ?? "0.3");
const OUTPUT_PATH = resolve(DATA_DIR, parseArg("--output") ?? "n8n-kl-targets.json");

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface LiveMCPTool {
  id: string;
  name: string;
  description: string;
  embedding: number[];
}

interface N8nWorkflow {
  id: number;
  name: string;
  description?: string;
  nodes: { type: string; displayName: string; operation?: string }[];
}

interface SoftTargetOutput {
  toolIds: string[];
  temperature: number;
  topK: number;
  numExamples: number;
  stats: {
    totalWorkflows: number;
    withEmbedding: number;
    aboveMinSim: number;
    avgMaxSim: number;
    medianMaxSim: number;
  };
  examples: {
    intentEmbedding: number[];
    sp: [number, number][];
    wfId: number;
    wfName: string;
  }[];
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

function softmax(values: number[], temperature: number): number[] {
  const scaled = values.map(v => v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  console.log("=== Build n8n Soft Targets for SHGAT KL Training ===\n");
  console.log(`Temperature: ${TEMPERATURE}`);
  console.log(`Top-K: ${TOP_K}`);
  console.log(`Min max sim: ${MIN_MAX_SIM}`);

  // 1. Load LiveMCPBench tools
  const toolsPath = resolve(DATA_DIR, "tools.json");
  if (!existsSync(toolsPath)) {
    console.error("LiveMCPBench tools.json not found at", toolsPath);
    process.exit(1);
  }
  const tools: LiveMCPTool[] = JSON.parse(readFileSync(toolsPath, "utf-8"));
  const toolIds = tools.map(t => t.id);
  const toolEmbs = tools.map(t => t.embedding);
  console.log(`\nLoaded ${tools.length} LiveMCPBench tools`);

  // 2. Load n8n workflow description embeddings
  const wfEmbPath = resolve(GRU_DATA_DIR, "n8n-workflow-description-embeddings.json");
  let wfEmbeddings: Record<string, number[]> = {};

  if (existsSync(wfEmbPath)) {
    wfEmbeddings = JSON.parse(readFileSync(wfEmbPath, "utf-8"));
    console.log(`Loaded ${Object.keys(wfEmbeddings).length} workflow description embeddings`);
  } else {
    console.error(
      `\nWARNING: ${wfEmbPath} not found.\n` +
      `Run: cd lib/gru && npx tsx src/n8n/embed-n8n-nodes.ts\n` +
      `to generate workflow description embeddings.\n`
    );
    process.exit(1);
  }

  // 3. Load n8n workflows (for metadata)
  const wfPath = resolve(GRU_DATA_DIR, "n8n-workflows.json");
  if (!existsSync(wfPath)) {
    console.error("n8n-workflows.json not found at", wfPath);
    process.exit(1);
  }
  const workflows: N8nWorkflow[] = JSON.parse(readFileSync(wfPath, "utf-8"));
  const wfById = new Map<number, N8nWorkflow>();
  for (const wf of workflows) {
    wfById.set(wf.id, wf);
  }
  console.log(`Loaded ${workflows.length} workflow metadata`);

  // 4. Compute soft targets
  console.log("\nComputing soft targets...");

  const maxSims: number[] = [];
  const examples: SoftTargetOutput["examples"] = [];
  let skippedLowSim = 0;

  for (const [wfIdStr, intentEmb] of Object.entries(wfEmbeddings)) {
    const wfId = parseInt(wfIdStr, 10);
    const wf = wfById.get(wfId);
    const wfName = wf?.name ?? `workflow-${wfId}`;

    // Compute cosine similarity against all 525 tools
    const similarities = toolEmbs.map(toolEmb => cosineSimilarity(intentEmb, toolEmb));
    const maxSim = Math.max(...similarities);
    maxSims.push(maxSim);

    // Skip if max similarity is too low (no meaningful match)
    if (maxSim < MIN_MAX_SIM) {
      skippedLowSim++;
      continue;
    }

    // Softmax with temperature
    const probs = softmax(similarities, TEMPERATURE);

    // Sparse top-K
    const indexed: [number, number][] = probs.map((p, i) => [i, p]);
    indexed.sort((a, b) => b[1] - a[1]);
    const sparse = indexed.slice(0, TOP_K);

    // Re-normalize
    const total = sparse.reduce((s, [, p]) => s + p, 0);
    if (total > 0) {
      for (const entry of sparse) {
        entry[1] /= total;
      }
    }

    examples.push({
      intentEmbedding: intentEmb,
      sp: sparse,
      wfId,
      wfName,
    });
  }

  // 5. Statistics
  maxSims.sort((a, b) => a - b);
  const avgMaxSim = maxSims.reduce((a, b) => a + b, 0) / maxSims.length;
  const medianMaxSim = maxSims[Math.floor(maxSims.length / 2)];

  console.log(`\n--- Statistics ---`);
  console.log(`Total workflows with embeddings: ${Object.keys(wfEmbeddings).length}`);
  console.log(`Skipped (max sim < ${MIN_MAX_SIM}): ${skippedLowSim}`);
  console.log(`Kept: ${examples.length}`);
  console.log(`Max cosine sim distribution:`);
  console.log(`  min:    ${maxSims[0]?.toFixed(4)}`);
  console.log(`  p25:    ${maxSims[Math.floor(maxSims.length * 0.25)]?.toFixed(4)}`);
  console.log(`  median: ${medianMaxSim.toFixed(4)}`);
  console.log(`  p75:    ${maxSims[Math.floor(maxSims.length * 0.75)]?.toFixed(4)}`);
  console.log(`  max:    ${maxSims[maxSims.length - 1]?.toFixed(4)}`);
  console.log(`  avg:    ${avgMaxSim.toFixed(4)}`);

  // Show top-5 examples
  console.log(`\n--- Top 5 examples (highest max sim) ---`);
  const sorted = [...examples].sort((a, b) => {
    const simA = Math.max(...toolEmbs.map(te => cosineSimilarity(a.intentEmbedding, te)));
    const simB = Math.max(...toolEmbs.map(te => cosineSimilarity(b.intentEmbedding, te)));
    return simB - simA;
  }).slice(0, 5);

  for (const ex of sorted) {
    const topTools = ex.sp.slice(0, 3).map(([idx, prob]) =>
      `${toolIds[idx]} (${(prob * 100).toFixed(1)}%)`
    );
    console.log(`  "${ex.wfName}" → ${topTools.join(", ")}`);
  }

  // 6. Write output
  const output: SoftTargetOutput = {
    toolIds,
    temperature: TEMPERATURE,
    topK: TOP_K,
    numExamples: examples.length,
    stats: {
      totalWorkflows: Object.keys(wfEmbeddings).length,
      withEmbedding: Object.keys(wfEmbeddings).length,
      aboveMinSim: examples.length,
      avgMaxSim,
      medianMaxSim,
    },
    examples,
  };

  writeFileSync(OUTPUT_PATH, JSON.stringify(output));
  const sizeMB = (Buffer.byteLength(JSON.stringify(output)) / 1024 / 1024).toFixed(1);
  console.log(`\nOutput: ${OUTPUT_PATH} (${sizeMB} MB, ${examples.length} examples)`);
}

main();
