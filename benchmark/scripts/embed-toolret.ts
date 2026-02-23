#!/usr/bin/env -S deno run --allow-all --config deno.json
/**
 * Embed ToolRet data with BGE-M3 (Xenova/bge-m3 via @huggingface/transformers).
 *
 * Reads raw JSON files produced by download-toolret.py and adds 1024-dim
 * BGE-M3 embeddings using the project's EmbeddingModel.
 *
 * Usage (from project root):
 *   deno run --allow-all --config deno.json lib/shgat-tf/benchmark/scripts/embed-toolret.ts
 *
 * Input:  data/tools-raw.json, data/queries-raw.json
 * Output: data/tools.json, data/queries.json (with embedding field)
 */

import { EmbeddingModel } from "../../../../src/vector/embeddings.ts";

const DATA_DIR = new URL("../data/", import.meta.url).pathname;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ToolRaw {
  id: string;
  documentation: string;
  category_name: string | null;
  tool_name: string | null;
  api_name: string | null;
}

interface QueryRaw {
  id: string;
  query: string;
  labels: { id: string; relevance: number }[];
}

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

    if ((i + 1) % 100 === 0 || i === total - 1) {
      const elapsed = (performance.now() - t0) / 1000;
      const rate = (i + 1) / elapsed;
      const eta = (total - i - 1) / rate;
      console.log(
        `[embed:${label}] ${i + 1}/${total} ` +
        `(${((i + 1) / total * 100).toFixed(1)}%) ` +
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
  // Check input files exist
  const toolsRawPath = DATA_DIR + "tools-raw.json";
  const queriesRawPath = DATA_DIR + "queries-raw.json";

  try {
    await Deno.stat(toolsRawPath);
    await Deno.stat(queriesRawPath);
  } catch {
    console.error(
      "Raw data not found. Run download first:\n" +
      "  cd lib/shgat-tf/benchmark && python3 scripts/download-toolret.py",
    );
    Deno.exit(1);
  }

  // Load raw data
  console.log("[load] Reading raw tools...");
  const tools: ToolRaw[] = JSON.parse(await Deno.readTextFile(toolsRawPath));
  console.log(`[load] ${tools.length} tools`);

  console.log("[load] Reading raw queries...");
  const queries: QueryRaw[] = JSON.parse(await Deno.readTextFile(queriesRawPath));
  console.log(`[load] ${queries.length} queries`);

  // Load model
  const model = new EmbeddingModel();
  await model.load();

  // Embed tools
  console.log(`\n[embed] Embedding ${tools.length} tools...`);
  const toolTexts = tools.map((t) => t.documentation);
  const toolEmbeddings = await embedAll(toolTexts, model, "tools");

  const toolsWithEmb = tools.map((t, i) => ({
    ...t,
    embedding: toolEmbeddings[i],
  }));

  // Embed queries
  console.log(`\n[embed] Embedding ${queries.length} queries...`);
  const queryTexts = queries.map((q) => q.query);
  const queryEmbeddings = await embedAll(queryTexts, model, "queries");

  const queriesWithEmb = queries.map((q, i) => ({
    ...q,
    embedding: queryEmbeddings[i],
  }));

  await model.dispose();

  // Save
  const toolsOutPath = DATA_DIR + "tools.json";
  const queriesOutPath = DATA_DIR + "queries.json";

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
  console.log(`\nNext: cd lib/shgat-tf/benchmark && npm run bench`);
}

main().catch((err) => {
  console.error("Failed:", err);
  Deno.exit(1);
});
