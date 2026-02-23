#!/usr/bin/env -S deno run --allow-all
/**
 * Embed LiveMCPBench data with BGE-M3.
 *
 * Reads raw JSON files produced by download-livemcp.py and adds 1024-dim
 * BGE-M3 embeddings using the project's EmbeddingModel.
 *
 * Usage (from project root):
 *   deno run --allow-all --config deno.json lib/shgat-tf/benchmark/scripts/embed-livemcp.ts
 *
 * Input:  data/livemcp/tools-raw.json, data/livemcp/queries-raw.json
 * Output: data/livemcp/tools.json, data/livemcp/queries.json (with embedding field)
 */

import { EmbeddingModel } from "../../../../src/vector/embeddings.ts";
import { resolve, dirname, fromFileUrl } from "jsr:@std/path@1";

const __dirname = dirname(fromFileUrl(import.meta.url));
const DATA_DIR = resolve(__dirname, "..", "data", "livemcp");

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

async function embedAll(
  texts: string[],
  model: EmbeddingModel,
  label: string,
): Promise<number[][]> {
  const total = texts.length;
  const embeddings: number[][] = [];
  const t0 = performance.now();

  for (let i = 0; i < total; i++) {
    const emb = await model.encode(texts[i]);
    embeddings.push(emb);

    if ((i + 1) % 50 === 0 || i === total - 1) {
      const elapsed = (performance.now() - t0) / 1000;
      const rate = (i + 1) / elapsed;
      const eta = (total - i - 1) / rate;
      console.log(
        `[embed:${label}] ${i + 1}/${total} ` +
          `(${(((i + 1) / total) * 100).toFixed(1)}%) ` +
          `${rate.toFixed(1)} texts/s, ETA ${eta.toFixed(0)}s`,
      );
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
      "Raw data not found. Run download first:\n" +
        "  python3 lib/shgat-tf/benchmark/scripts/download-livemcp.py",
    );
    Deno.exit(1);
  }

  // Load raw data
  console.log("[load] Reading raw tools...");
  const tools = JSON.parse(await Deno.readTextFile(toolsRawPath));
  console.log(`[load] ${tools.length} tools`);

  console.log("[load] Reading raw queries...");
  const queries = JSON.parse(await Deno.readTextFile(queriesRawPath));
  console.log(`[load] ${queries.length} queries`);

  // Load model
  const model = new EmbeddingModel();
  await model.load();

  // Embed tools (use the pre-built text field which includes server+category context)
  console.log(`\n[embed] Embedding ${tools.length} tools...`);
  const toolTexts = tools.map((t: { text: string }) => t.text);
  const toolEmbeddings = await embedAll(toolTexts, model, "tools");

  const toolsWithEmb = tools.map(
    (t: Record<string, unknown>, i: number) => ({
      ...t,
      embedding: toolEmbeddings[i],
    }),
  );

  // Embed queries
  console.log(`\n[embed] Embedding ${queries.length} queries...`);
  const queryTexts = queries.map((q: { query: string }) => q.query);
  const queryEmbeddings = await embedAll(queryTexts, model, "queries");

  const queriesWithEmb = queries.map(
    (q: Record<string, unknown>, i: number) => ({
      ...q,
      embedding: queryEmbeddings[i],
    }),
  );

  await model.dispose();

  // Save
  const toolsOutPath = resolve(DATA_DIR, "tools.json");
  const queriesOutPath = resolve(DATA_DIR, "queries.json");

  console.log("\n[save] Writing tools with embeddings...");
  await Deno.writeTextFile(toolsOutPath, JSON.stringify(toolsWithEmb));
  const toolsSize = (await Deno.stat(toolsOutPath)).size / 1e6;
  console.log(`[save] Tools → ${toolsOutPath} (${toolsSize.toFixed(1)} MB)`);

  console.log("[save] Writing queries with embeddings...");
  await Deno.writeTextFile(queriesOutPath, JSON.stringify(queriesWithEmb));
  const queriesSize = (await Deno.stat(queriesOutPath)).size / 1e6;
  console.log(`[save] Queries → ${queriesOutPath} (${queriesSize.toFixed(1)} MB)`);

  console.log(`\n=== Done ===`);
  console.log(`Tools:   ${toolsWithEmb.length} (dim=${toolEmbeddings[0].length})`);
  console.log(`Queries: ${queriesWithEmb.length}`);
  console.log(`\nNext: run the benchmark`);
}

main().catch((err) => {
  console.error("Failed:", err);
  Deno.exit(1);
});
