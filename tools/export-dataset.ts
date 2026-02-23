#!/usr/bin/env npx tsx
/**
 * Export bench dataset to Parquet files for SHGAT OB training.
 *
 * Writes each section to a separate .parquet file (streamed, low peak memory)
 * plus a bench-metadata.json for small scalar data.
 *
 * Also writes msgpack.gz as legacy fallback (use --no-msgpack to skip).
 *
 * Run once (or whenever data changes):
 *   cd lib/shgat-tf
 *   DATABASE_URL=... npx tsx tools/export-dataset.ts
 *
 * Output (in ../../gru/data/):
 *   bench-nodes.parquet       — nodes (id, embedding, children, level)
 *   bench-prod-train.parquet  — prod training examples
 *   bench-prod-test.parquet   — prod test examples
 *   bench-n8n-train.parquet   — n8n training examples
 *   bench-n8n-eval.parquet    — n8n eval examples
 *   bench-metadata.json       — leafIds, embeddingDim, workflowToolLists
 *   bench-dataset-export.msgpack.gz (legacy, unless --no-msgpack)
 *
 * @module shgat-tf/tools/export-dataset
 */

import { writeFileSync } from "node:fs";
import { statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { encode as msgpackEncode } from "@msgpack/msgpack";
import pako from "pako";
import postgres from "postgres";
import * as arrow from "apache-arrow";
import {
  Compression,
  Table as WasmTable,
  writeParquet,
  WriterPropertiesBuilder,
} from "parquet-wasm";

import { loadBenchDataset } from "../../gru/src/shgat/bench-dataset.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const GRU_DATA_DIR = resolve(__dirname, "../../gru/data");

const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) throw new Error("DATABASE_URL env var required");

const SEED = parseInt(process.argv.includes("--seed") ? process.argv[process.argv.indexOf("--seed") + 1] : "42", 10);
const OVERSAMPLE = parseInt(process.argv.includes("--oversample") ? process.argv[process.argv.indexOf("--oversample") + 1] : "3", 10);
const SKIP_MSGPACK = process.argv.includes("--no-msgpack");

// ==========================================================================
// Helpers
// ==========================================================================

function embeddingToBytes(emb: number[] | Float32Array): Uint8Array {
  const f32 = emb instanceof Float32Array ? emb : new Float32Array(emb);
  return new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
}

function softTargetToBytes(sparse: [number, number][]): { indicesBytes: Uint8Array; probsBytes: Uint8Array } {
  if (!sparse || sparse.length === 0) {
    return { indicesBytes: new Uint8Array(0), probsBytes: new Uint8Array(0) };
  }
  const indices = new Int32Array(sparse.map(([idx]) => idx));
  const probs = new Float32Array(sparse.map(([, prob]) => prob));
  return {
    indicesBytes: new Uint8Array(indices.buffer),
    probsBytes: new Uint8Array(probs.buffer),
  };
}

function writeParquetFile(table: arrow.Table, filePath: string): void {
  const ipcStream = arrow.tableToIPC(table, "stream");
  const wasmTable = WasmTable.fromIPCStream(ipcStream);
  const writerProps = new WriterPropertiesBuilder()
    .setCompression(Compression.SNAPPY)
    .build();
  const parquetBytes = writeParquet(wasmTable, writerProps);
  writeFileSync(filePath, parquetBytes);
}

function logWrite(name: string, filePath: string, t0: number, numRows: number): void {
  const stat = statSync(filePath);
  console.log(`  ${name}: ${numRows} rows → ${(stat.size / 1e6).toFixed(1)}MB (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
}

// ==========================================================================
// MAIN
// ==========================================================================

console.log(`=== Export bench dataset (seed=${SEED}, oversample=${OVERSAMPLE}) ===`);

const sql = postgres(DATABASE_URL);
const ds = await loadBenchDataset(sql, {
  expandedVocabPath: resolve(GRU_DATA_DIR, "expanded-vocab.json"),
  n8nPairsPath: resolve(GRU_DATA_DIR, "n8n-shgat-contrastive-pairs.json"),
  n8nDataDir: GRU_DATA_DIR,
  maxN8nWorkflows: 99999,
  prodOversample: OVERSAMPLE,
  splitSeed: SEED,
});

console.log(`  Nodes:  ${ds.nodes.length} (${ds.leafIds.length} leaves), EmbDim: ${ds.embeddingDim}`);
console.log(`  Prod:   ${ds.prodTrain.length} train / ${ds.prodTest.length} test`);
console.log(`  N8n:    ${ds.n8nTrain.length} train / ${ds.n8nEval.length} eval\n`);

// ---- Nodes ----
{
  const t0 = performance.now();
  const nodes = ds.nodes;
  const n = nodes.length;
  const embBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  const ids: string[] = new Array(n);
  const levels = new Int32Array(n);
  const childrenJson: string[] = new Array(n);
  for (let i = 0; i < n; i++) {
    ids[i] = nodes[i].id;
    levels[i] = nodes[i].level;
    childrenJson[i] = JSON.stringify(nodes[i].children);
    embBuilder.append(embeddingToBytes(nodes[i].embedding));
  }
  embBuilder.finish();
  const table = new arrow.Table({
    id: arrow.vectorFromArray(ids, new arrow.Utf8()),
    embedding: embBuilder.toVector(),
    children_json: arrow.vectorFromArray(childrenJson, new arrow.Utf8()),
    level: arrow.makeVector(levels),
  });
  const outPath = resolve(GRU_DATA_DIR, "bench-nodes.parquet");
  writeParquetFile(table, outPath);
  logWrite("Nodes", outPath, t0, table.numRows);
}

// ---- Prod examples helper ----
function writeProdParquet(examples: typeof ds.prodTrain, filename: string): void {
  const t0 = performance.now();
  const n = examples.length;
  const embBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  const targets: string[] = new Array(n);
  const isTerminals = new Int32Array(n);
  const traceIds: string[] = new Array(n);
  const contextJson: string[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const ex = examples[i];
    embBuilder.append(embeddingToBytes(ex.intentEmbedding));
    contextJson[i] = JSON.stringify(ex.contextToolIds);
    targets[i] = ex.targetToolId;
    isTerminals[i] = (ex as unknown as { isTerminal?: number }).isTerminal ?? 0;
    traceIds[i] = ex._traceId;
  }
  embBuilder.finish();
  const table = new arrow.Table({
    intent_embedding: embBuilder.toVector(),
    context_tool_ids_json: arrow.vectorFromArray(contextJson, new arrow.Utf8()),
    target_tool_id: arrow.vectorFromArray(targets, new arrow.Utf8()),
    is_terminal: arrow.makeVector(isTerminals),
    trace_id: arrow.vectorFromArray(traceIds, new arrow.Utf8()),
  });
  const outPath = resolve(GRU_DATA_DIR, filename);
  writeParquetFile(table, outPath);
  logWrite(filename, outPath, t0, table.numRows);
}

writeProdParquet(ds.prodTrain, "bench-prod-train.parquet");
writeProdParquet(ds.prodTest, "bench-prod-test.parquet");

// ---- N8n examples helper ----
function writeN8nParquet(examples: typeof ds.n8nTrain, filename: string): void {
  const t0 = performance.now();
  const n = examples.length;
  const embBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  const indicesBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  const probsBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  const targets: string[] = new Array(n);
  const isTerminals = new Int32Array(n);
  const contextJson: string[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const ex = examples[i];
    embBuilder.append(embeddingToBytes(ex.intentEmbedding));
    contextJson[i] = JSON.stringify(ex.contextToolIds);
    targets[i] = ex.targetToolId;
    isTerminals[i] = ex.isTerminal;
    const { indicesBytes, probsBytes } = softTargetToBytes(ex.softTargetSparse);
    indicesBuilder.append(indicesBytes);
    probsBuilder.append(probsBytes);
  }
  embBuilder.finish();
  indicesBuilder.finish();
  probsBuilder.finish();
  const table = new arrow.Table({
    intent_embedding: embBuilder.toVector(),
    context_tool_ids_json: arrow.vectorFromArray(contextJson, new arrow.Utf8()),
    target_tool_id: arrow.vectorFromArray(targets, new arrow.Utf8()),
    is_terminal: arrow.makeVector(isTerminals),
    soft_target_indices: indicesBuilder.toVector(),
    soft_target_probs: probsBuilder.toVector(),
  });
  const outPath = resolve(GRU_DATA_DIR, filename);
  writeParquetFile(table, outPath);
  logWrite(filename, outPath, t0, table.numRows);
}

writeN8nParquet(ds.n8nTrain, "bench-n8n-train.parquet");
writeN8nParquet(ds.n8nEval, "bench-n8n-eval.parquet");

// ---- Metadata ----
{
  const metadataPath = resolve(GRU_DATA_DIR, "bench-metadata.json");
  const metadata = {
    leafIds: ds.leafIds,
    embeddingDim: ds.embeddingDim,
    workflowToolLists: ds.workflowToolLists,
    exportedAt: new Date().toISOString(),
    source: "export-dataset.ts",
    counts: {
      nodes: ds.nodes.length,
      prodTrain: ds.prodTrain.length,
      prodTest: ds.prodTest.length,
      n8nTrain: ds.n8nTrain.length,
      n8nEval: ds.n8nEval.length,
    },
  };
  writeFileSync(metadataPath, JSON.stringify(metadata));
  console.log(`  Metadata: bench-metadata.json`);
}

// ---- Legacy msgpack.gz (optional) ----
if (!SKIP_MSGPACK) {
  console.log("\n[Legacy] Writing msgpack.gz...");
  const exportData = {
    nodes: ds.nodes,
    leafIds: ds.leafIds,
    embeddingDim: ds.embeddingDim,
    workflowToolLists: ds.workflowToolLists,
    prodTrain: ds.prodTrain.map((ex) => ({
      intentEmbedding: ex.intentEmbedding,
      contextToolIds: ex.contextToolIds,
      targetToolId: ex.targetToolId,
      isTerminal: (ex as unknown as { isTerminal?: number }).isTerminal ?? 0,
      _traceId: ex._traceId,
    })),
    prodTest: ds.prodTest.map((ex) => ({
      intentEmbedding: ex.intentEmbedding,
      contextToolIds: ex.contextToolIds,
      targetToolId: ex.targetToolId,
      isTerminal: (ex as unknown as { isTerminal?: number }).isTerminal ?? 0,
      _traceId: ex._traceId,
    })),
    n8nTrain: ds.n8nTrain.map((ex) => ({
      intentEmbedding: Array.from(ex.intentEmbedding),
      contextToolIds: ex.contextToolIds,
      targetToolId: ex.targetToolId,
      isTerminal: ex.isTerminal,
      softTargetSparse: ex.softTargetSparse,
    })),
    n8nEval: ds.n8nEval.map((ex) => ({
      intentEmbedding: Array.from(ex.intentEmbedding),
      contextToolIds: ex.contextToolIds,
      targetToolId: ex.targetToolId,
      isTerminal: ex.isTerminal,
      softTargetSparse: ex.softTargetSparse,
    })),
  };
  const encoded = msgpackEncode(exportData);
  console.log(`  Raw: ${(encoded.byteLength / 1e6).toFixed(1)}MB`);
  const compressed = pako.gzip(encoded, { level: 6 });
  const outPath = resolve(GRU_DATA_DIR, "bench-dataset-export.msgpack.gz");
  writeFileSync(outPath, compressed);
  console.log(`  msgpack.gz: ${(compressed.length / 1e6).toFixed(1)}MB`);
}

console.log(`\n=== Export complete ===`);
console.log(`  Nodes:  ${ds.nodes.length} (${ds.leafIds.length} leaves)`);
console.log(`  Prod:   ${ds.prodTrain.length} train / ${ds.prodTest.length} test`);
console.log(`  N8n:    ${ds.n8nTrain.length} train / ${ds.n8nEval.length} eval`);

await sql.end();
