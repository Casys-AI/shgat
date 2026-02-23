#!/usr/bin/env -S deno run --allow-all
/**
 * Embed ToolBench data with BGE-M3.
 *
 * Reads processed JSON files from process-toolbench.py and adds 1024-dim
 * BGE-M3 embeddings using the project's EmbeddingModel.
 *
 * Usage (from project root):
 *   deno run --allow-all --config deno.json lib/shgat-tf/benchmark/scripts/embed-toolbench.ts
 *
 * Input:  data/toolbench/tools-raw.json, data/toolbench/queries-raw.json
 * Output: data/toolbench/tools.json, data/toolbench/queries.json (with embedding field)
 *
 * Supports resuming: if tools.json already exists with some embeddings,
 * it will skip those and continue from where it left off.
 */

import { EmbeddingModel } from "../../../../src/vector/embeddings.ts";
import { resolve, dirname, fromFileUrl } from "jsr:@std/path@1";

const __dirname = dirname(fromFileUrl(import.meta.url));
const DATA_DIR = resolve(__dirname, "..", "data", "toolbench");

// ---------------------------------------------------------------------------
// Embedding with resume support
// ---------------------------------------------------------------------------

async function embedAll(
  texts: string[],
  model: EmbeddingModel,
  label: string,
  existingEmbeddings?: (number[] | null)[],
): Promise<number[][]> {
  const total = texts.length;
  const embeddings: number[][] = existingEmbeddings?.map(e => e ?? []) ?? new Array(total).fill([]);
  const t0 = performance.now();
  let skipped = 0;

  for (let i = 0; i < total; i++) {
    // Skip if already embedded (resume support)
    if (embeddings[i] && embeddings[i].length > 0) {
      skipped++;
      continue;
    }

    const emb = await model.encode(texts[i]);
    embeddings[i] = emb;

    const done = i + 1 - skipped;
    if (done % 100 === 0 || i === total - 1) {
      const elapsed = (performance.now() - t0) / 1000;
      const rate = done / elapsed;
      const remaining = total - i - 1;
      const eta = remaining / rate;
      console.log(
        `[embed:${label}] ${i + 1}/${total} (${skipped} cached) ` +
          `${rate.toFixed(1)} texts/s, ETA ${eta.toFixed(0)}s`,
      );
    }

    // Checkpoint every 1000 items
    if (done > 0 && done % 1000 === 0) {
      const checkpointPath = resolve(DATA_DIR, `${label}-checkpoint.json`);
      await Deno.writeTextFile(checkpointPath, JSON.stringify(embeddings));
      console.log(`  [checkpoint] Saved ${i + 1} embeddings`);
    }
  }

  return embeddings;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const toolsRawPath = resolve(DATA_DIR, "tools-raw.json");
  const queriesRawPath = resolve(DATA_DIR, "queries-raw.json");

  try {
    await Deno.stat(toolsRawPath);
    await Deno.stat(queriesRawPath);
  } catch {
    console.error(
      "Raw data not found. Run processing first:\n" +
        "  python3 lib/shgat-tf/benchmark/scripts/process-toolbench.py",
    );
    Deno.exit(1);
  }

  // Load raw data
  console.log("[load] Reading raw tools...");
  const tools: Array<Record<string, unknown>> = JSON.parse(
    await Deno.readTextFile(toolsRawPath),
  );
  console.log(`[load] ${tools.length} tools`);

  console.log("[load] Reading raw queries...");
  const queries: Array<Record<string, unknown>> = JSON.parse(
    await Deno.readTextFile(queriesRawPath),
  );
  console.log(`[load] ${queries.length} queries`);

  // Check for existing embeddings (resume support)
  const toolsOutPath = resolve(DATA_DIR, "tools.json");
  const queriesOutPath = resolve(DATA_DIR, "queries.json");

  let existingToolEmbs: (number[] | null)[] | undefined;
  try {
    const existing = JSON.parse(await Deno.readTextFile(toolsOutPath));
    if (Array.isArray(existing) && existing.length === tools.length) {
      existingToolEmbs = existing.map((t: Record<string, unknown>) =>
        (t.embedding as number[]) ?? null
      );
      const cached = existingToolEmbs.filter(e => e && e.length > 0).length;
      console.log(`[resume] Found ${cached}/${tools.length} cached tool embeddings`);
    }
  } catch {
    // No existing file, start fresh
  }

  // Check for checkpoint
  const checkpointPath = resolve(DATA_DIR, "tools-checkpoint.json");
  if (!existingToolEmbs) {
    try {
      const checkpoint: number[][] = JSON.parse(
        await Deno.readTextFile(checkpointPath),
      );
      if (checkpoint.length === tools.length) {
        existingToolEmbs = checkpoint;
        const cached = existingToolEmbs.filter(e => e && e.length > 0).length;
        console.log(`[resume] Found checkpoint with ${cached}/${tools.length} embeddings`);
      }
    } catch {
      // No checkpoint
    }
  }

  // Load model
  console.log("[model] Loading BGE-M3...");
  const model = new EmbeddingModel();
  await model.load();
  console.log("[model] Ready");

  // Embed tools
  console.log(`\n[embed] Embedding ${tools.length} tools...`);
  const toolTexts = tools.map((t) => t.text as string);
  const toolEmbeddings = await embedAll(toolTexts, model, "tools", existingToolEmbs);

  const toolsWithEmb = tools.map((t, i) => ({
    ...t,
    embedding: toolEmbeddings[i],
  }));

  // Save tools immediately
  console.log("\n[save] Writing tools with embeddings...");
  await Deno.writeTextFile(toolsOutPath, JSON.stringify(toolsWithEmb));
  const toolsSize = (await Deno.stat(toolsOutPath)).size / 1e6;
  console.log(`[save] Tools → ${toolsOutPath} (${toolsSize.toFixed(1)} MB)`);

  // Embed queries
  console.log(`\n[embed] Embedding ${queries.length} queries...`);
  const queryTexts = queries.map((q) => q.query as string);
  const queryEmbeddings = await embedAll(queryTexts, model, "queries");

  const queriesWithEmb = queries.map((q, i) => ({
    ...q,
    embedding: queryEmbeddings[i],
  }));

  await model.dispose();

  // Save queries
  console.log("[save] Writing queries with embeddings...");
  await Deno.writeTextFile(queriesOutPath, JSON.stringify(queriesWithEmb));
  const queriesSize = (await Deno.stat(queriesOutPath)).size / 1e6;
  console.log(`[save] Queries → ${queriesOutPath} (${queriesSize.toFixed(1)} MB)`);

  // Cleanup checkpoint
  try {
    await Deno.remove(checkpointPath);
  } catch { /* ok */ }

  console.log(`\n=== Done ===`);
  console.log(`Tools:   ${toolsWithEmb.length} (dim=${toolEmbeddings[0].length})`);
  console.log(`Queries: ${queriesWithEmb.length}`);
  console.log(`\nNext: run the ToolBench benchmark`);
}

main().catch((err) => {
  console.error("Failed:", err);
  Deno.exit(1);
});
