#!/usr/bin/env -S deno run --allow-read --allow-write --allow-env --allow-net
/**
 * Export bench dataset from msgpack.gz → Parquet files.
 *
 * Reads the existing bench-dataset-export.msgpack.gz and writes:
 *   bench-nodes.parquet       — nodes (id, embedding bytes, children JSON, level)
 *   bench-prod-train.parquet  — prod training examples
 *   bench-prod-test.parquet   — prod test examples
 *   bench-n8n-train.parquet   — n8n training examples
 *   bench-n8n-eval.parquet    — n8n eval examples
 *   bench-metadata.json       — small JSON with leafIds, embeddingDim, workflowToolLists
 *
 * Embeddings are stored as raw Float32 bytes in a Binary column (1024 * 4 = 4096 bytes/row).
 * String arrays (children, contextToolIds) are stored as JSON-encoded strings.
 * softTargetSparse indices/probs are stored as separate Binary columns.
 *
 * Usage:
 *   cd lib/shgat-tf
 *   deno run --allow-read --allow-write --allow-env --allow-net \
 *     tools/export-to-parquet.ts [--data-path <path>]
 *
 * @module shgat-tf/tools/export-to-parquet
 */

import { dirname, resolve, fromFileUrl } from "https://deno.land/std@0.224.0/path/mod.ts";
import { decode as msgpackDecode } from "@msgpack/msgpack";
import pako from "pako";
import * as arrow from "apache-arrow";
// parquet-wasm: Node entry point auto-initializes WASM (no initWasm needed).
// ESM entry point requires calling the default export first.
// Deno npm: resolves to the Node entry point.
import parquetWasmModule from "parquet-wasm";
import {
  Compression,
  Table as WasmTable,
  writeParquet,
  WriterPropertiesBuilder,
} from "parquet-wasm";

// ==========================================================================
// Types (same as train-ob.ts)
// ==========================================================================

interface ExportedNode {
  id: string;
  embedding: number[];
  children: string[];
  level: number;
}

interface ProdExample {
  intentEmbedding: number[];
  contextToolIds: string[];
  targetToolId: string;
  isTerminal: number;
  _traceId: string;
}

interface N8nExample {
  intentEmbedding: number[];
  contextToolIds: string[];
  targetToolId: string;
  isTerminal: number;
  softTargetSparse: [number, number][];
}

interface ExportedDataset {
  nodes: ExportedNode[];
  leafIds: string[];
  embeddingDim: number;
  workflowToolLists: string[][];
  prodTrain: ProdExample[];
  prodTest: ProdExample[];
  n8nTrain: N8nExample[];
  n8nEval: N8nExample[];
}

// ==========================================================================
// CLI
// ==========================================================================

const cliArgs = Deno.args;

function getArg(name: string, def: string): string {
  const idx = cliArgs.indexOf(`--${name}`);
  if (idx === -1) return def;
  const next = cliArgs[idx + 1];
  return next && !next.startsWith("--") ? next : def;
}

const scriptDir = dirname(fromFileUrl(import.meta.url));
const GRU_DATA_DIR = resolve(scriptDir, "../../gru/data");
const dataPath = getArg("data-path", resolve(GRU_DATA_DIR, "bench-dataset-export.msgpack.gz"));

if (cliArgs.includes("--help")) {
  console.log(`
Export bench dataset from msgpack.gz to Parquet files.

Options:
  --data-path <path>   Path to msgpack.gz dataset (default: auto)
  --help               Show this help
`);
  Deno.exit(0);
}

// ==========================================================================
// Helpers
// ==========================================================================

/**
 * Convert a number[] embedding to raw Float32 bytes.
 * Returns a Uint8Array of length embDim * 4.
 */
function embeddingToBytes(emb: number[]): Uint8Array {
  const f32 = new Float32Array(emb);
  return new Uint8Array(f32.buffer);
}

/**
 * Convert a sparse target array [[idx, prob], ...] to two separate byte arrays.
 * Indices: Int32Array bytes. Probs: Float32Array bytes.
 */
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

/**
 * Write an Arrow Table to a Parquet file using parquet-wasm with Snappy compression.
 */
function writeParquetFile(table: arrow.Table, filePath: string): void {
  const ipcStream = arrow.tableToIPC(table, "stream");
  const wasmTable = WasmTable.fromIPCStream(ipcStream);
  const writerProps = new WriterPropertiesBuilder()
    .setCompression(Compression.SNAPPY)
    .build();
  const parquetBytes = writeParquet(wasmTable, writerProps);
  Deno.writeFileSync(filePath, parquetBytes);
}

// ==========================================================================
// Build Arrow tables
// ==========================================================================

/**
 * Build nodes table: id (Utf8), embedding (Binary), children (Utf8 JSON), level (Int32)
 */
function buildNodesTable(nodes: ExportedNode[]): arrow.Table {
  const n = nodes.length;
  const ids: string[] = new Array(n);
  const levels = new Int32Array(n);
  const childrenJson: string[] = new Array(n);
  // For embeddings, we'll build a Binary vector via makeBuilder
  const embBuilderType = new arrow.Binary();
  const embBuilder = arrow.makeBuilder({ type: embBuilderType, nullValues: [null] });

  for (let i = 0; i < n; i++) {
    ids[i] = nodes[i].id;
    levels[i] = nodes[i].level;
    childrenJson[i] = JSON.stringify(nodes[i].children);
    embBuilder.append(embeddingToBytes(nodes[i].embedding));
  }
  embBuilder.finish();
  const embVector = embBuilder.toVector();

  // Build table from columns
  const idVector = arrow.vectorFromArray(ids, new arrow.Utf8());
  const levelVector = arrow.makeVector(levels);
  const childrenVector = arrow.vectorFromArray(childrenJson, new arrow.Utf8());

  return new arrow.Table({
    id: idVector,
    embedding: embVector,
    children_json: childrenVector,
    level: levelVector,
  });
}

/**
 * Build prod examples table:
 *   intent_embedding (Binary), context_tool_ids (Utf8 JSON),
 *   target_tool_id (Utf8), is_terminal (Int32), trace_id (Utf8)
 */
function buildProdTable(examples: ProdExample[]): arrow.Table {
  const n = examples.length;
  const targetToolIds: string[] = new Array(n);
  const isTerminals = new Int32Array(n);
  const traceIds: string[] = new Array(n);
  const contextToolIdsJson: string[] = new Array(n);

  const embBuilderType = new arrow.Binary();
  const embBuilder = arrow.makeBuilder({ type: embBuilderType, nullValues: [null] });

  for (let i = 0; i < n; i++) {
    const ex = examples[i];
    embBuilder.append(embeddingToBytes(ex.intentEmbedding));
    contextToolIdsJson[i] = JSON.stringify(ex.contextToolIds);
    targetToolIds[i] = ex.targetToolId;
    isTerminals[i] = ex.isTerminal;
    traceIds[i] = ex._traceId;
  }
  embBuilder.finish();

  return new arrow.Table({
    intent_embedding: embBuilder.toVector(),
    context_tool_ids_json: arrow.vectorFromArray(contextToolIdsJson, new arrow.Utf8()),
    target_tool_id: arrow.vectorFromArray(targetToolIds, new arrow.Utf8()),
    is_terminal: arrow.makeVector(isTerminals),
    trace_id: arrow.vectorFromArray(traceIds, new arrow.Utf8()),
  });
}

/**
 * Build n8n examples table:
 *   intent_embedding (Binary), context_tool_ids (Utf8 JSON),
 *   target_tool_id (Utf8), is_terminal (Int32),
 *   soft_target_indices (Binary), soft_target_probs (Binary)
 */
function buildN8nTable(examples: N8nExample[]): arrow.Table {
  const n = examples.length;
  const targetToolIds: string[] = new Array(n);
  const isTerminals = new Int32Array(n);
  const contextToolIdsJson: string[] = new Array(n);

  const embType = new arrow.Binary();
  const embBuilder = arrow.makeBuilder({ type: embType, nullValues: [null] });
  const indicesBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  const probsBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });

  for (let i = 0; i < n; i++) {
    const ex = examples[i];
    embBuilder.append(embeddingToBytes(ex.intentEmbedding));
    contextToolIdsJson[i] = JSON.stringify(ex.contextToolIds);
    targetToolIds[i] = ex.targetToolId;
    isTerminals[i] = ex.isTerminal;

    const { indicesBytes, probsBytes } = softTargetToBytes(ex.softTargetSparse);
    indicesBuilder.append(indicesBytes);
    probsBuilder.append(probsBytes);
  }
  embBuilder.finish();
  indicesBuilder.finish();
  probsBuilder.finish();

  return new arrow.Table({
    intent_embedding: embBuilder.toVector(),
    context_tool_ids_json: arrow.vectorFromArray(contextToolIdsJson, new arrow.Utf8()),
    target_tool_id: arrow.vectorFromArray(targetToolIds, new arrow.Utf8()),
    is_terminal: arrow.makeVector(isTerminals),
    soft_target_indices: indicesBuilder.toVector(),
    soft_target_probs: probsBuilder.toVector(),
  });
}

// ==========================================================================
// MAIN
// ==========================================================================

console.log("=== Export Parquet from msgpack.gz ===\n");

// ---- Init parquet-wasm ----
// Node entry point: auto-initialized (parquetWasmModule is the module itself).
// ESM entry point: parquetWasmModule is the async init function.
// We call it if it's a function (ESM path), otherwise it's already initialized (Node path).
console.log("[WASM] Initializing parquet-wasm...");
if (typeof parquetWasmModule === "function") {
  await (parquetWasmModule as unknown as () => Promise<void>)();
}
console.log("[WASM] Ready.\n");

// ---- Load msgpack.gz dataset (staged to minimize peak memory) ----
console.log(`[Data] Loading ${dataPath}...`);
let ds: ExportedDataset;
{
  let compressed: Uint8Array | null = Deno.readFileSync(dataPath);
  console.log(`  Compressed: ${(compressed.byteLength / 1e6).toFixed(1)}MB`);
  let raw: Uint8Array | null = pako.ungzip(compressed);
  compressed = null; // free compressed buffer
  console.log(`  Decompressed: ${(raw.byteLength / 1e6).toFixed(1)}MB`);
  ds = msgpackDecode(raw) as ExportedDataset;
  raw = null; // free decompressed buffer
}

console.log(`  Nodes: ${ds.nodes.length} (${ds.leafIds.length} leaves), EmbDim: ${ds.embeddingDim}`);
console.log(`  Prod: ${ds.prodTrain.length} train / ${ds.prodTest.length} test`);
console.log(`  N8n: ${ds.n8nTrain.length} train / ${ds.n8nEval.length} eval\n`);

// ---- Write nodes ----
{
  const t0 = performance.now();
  console.log("[Parquet] Building nodes table...");
  const table = buildNodesTable(ds.nodes);
  console.log(`  Schema: ${table.schema.fields.map((f: { name: string; type: unknown }) => `${f.name}:${f.type}`).join(", ")}`);
  console.log(`  Rows: ${table.numRows}`);

  const outPath = resolve(GRU_DATA_DIR, "bench-nodes.parquet");
  writeParquetFile(table, outPath);
  const stat = Deno.statSync(outPath);
  console.log(`  Written: ${outPath} (${(stat.size / 1e6).toFixed(1)}MB, ${(performance.now() - t0).toFixed(0)}ms)\n`);
}

// ---- Write prod train ----
{
  const t0 = performance.now();
  console.log("[Parquet] Building prod-train table...");
  const table = buildProdTable(ds.prodTrain);
  console.log(`  Rows: ${table.numRows}`);

  const outPath = resolve(GRU_DATA_DIR, "bench-prod-train.parquet");
  writeParquetFile(table, outPath);
  const stat = Deno.statSync(outPath);
  console.log(`  Written: ${outPath} (${(stat.size / 1e6).toFixed(1)}MB, ${(performance.now() - t0).toFixed(0)}ms)\n`);
}

// ---- Write prod test ----
{
  const t0 = performance.now();
  console.log("[Parquet] Building prod-test table...");
  const table = buildProdTable(ds.prodTest);
  console.log(`  Rows: ${table.numRows}`);

  const outPath = resolve(GRU_DATA_DIR, "bench-prod-test.parquet");
  writeParquetFile(table, outPath);
  const stat = Deno.statSync(outPath);
  console.log(`  Written: ${outPath} (${(stat.size / 1e6).toFixed(1)}MB, ${(performance.now() - t0).toFixed(0)}ms)\n`);
}

// ---- Write n8n train ----
{
  const t0 = performance.now();
  console.log("[Parquet] Building n8n-train table...");
  const table = buildN8nTable(ds.n8nTrain);
  console.log(`  Rows: ${table.numRows}`);

  const outPath = resolve(GRU_DATA_DIR, "bench-n8n-train.parquet");
  writeParquetFile(table, outPath);
  const stat = Deno.statSync(outPath);
  console.log(`  Written: ${outPath} (${(stat.size / 1e6).toFixed(1)}MB, ${(performance.now() - t0).toFixed(0)}ms)\n`);
}

// ---- Write n8n eval ----
{
  const t0 = performance.now();
  console.log("[Parquet] Building n8n-eval table...");
  const table = buildN8nTable(ds.n8nEval);
  console.log(`  Rows: ${table.numRows}`);

  const outPath = resolve(GRU_DATA_DIR, "bench-n8n-eval.parquet");
  writeParquetFile(table, outPath);
  const stat = Deno.statSync(outPath);
  console.log(`  Written: ${outPath} (${(stat.size / 1e6).toFixed(1)}MB, ${(performance.now() - t0).toFixed(0)}ms)\n`);
}

// ---- Write metadata JSON ----
{
  const metadataPath = resolve(GRU_DATA_DIR, "bench-metadata.json");
  const metadata = {
    leafIds: ds.leafIds,
    embeddingDim: ds.embeddingDim,
    workflowToolLists: ds.workflowToolLists,
    exportedAt: new Date().toISOString(),
    source: "export-to-parquet.ts",
    counts: {
      nodes: ds.nodes.length,
      prodTrain: ds.prodTrain.length,
      prodTest: ds.prodTest.length,
      n8nTrain: ds.n8nTrain.length,
      n8nEval: ds.n8nEval.length,
    },
  };
  Deno.writeTextFileSync(metadataPath, JSON.stringify(metadata));
  const stat = Deno.statSync(metadataPath);
  console.log(`[Metadata] Written: ${metadataPath} (${(stat.size / 1024).toFixed(1)}KB)`);
}

// ---- Summary ----
const mem = (Deno.memoryUsage().rss / 1024 / 1024).toFixed(0);
console.log(`\n=== Export complete (RSS: ${mem}MB) ===`);
console.log("Files:");
console.log("  bench-nodes.parquet");
console.log("  bench-prod-train.parquet");
console.log("  bench-prod-test.parquet");
console.log("  bench-n8n-train.parquet");
console.log("  bench-n8n-eval.parquet");
console.log("  bench-metadata.json");
