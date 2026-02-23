#!/usr/bin/env node
/**
 * Export bench dataset from msgpack.gz → Parquet files (Node.js version).
 *
 * Node.js handles large msgpack decodes better than Deno (lower GC overhead).
 * Run with: node --max-old-space-size=12288 tools/export-to-parquet-node.mjs
 */

import { readFileSync, writeFileSync, statSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { decode as msgpackDecode } from "@msgpack/msgpack";
import pako from "pako";
import * as arrow from "apache-arrow";
import parquetWasm, {
  Compression,
  Table as WasmTable,
  writeParquet,
  WriterPropertiesBuilder,
} from "parquet-wasm";

const __dirname = dirname(fileURLToPath(import.meta.url));
const GRU_DATA_DIR = resolve(__dirname, "../../gru/data");
const dataPath = process.argv.includes("--data-path")
  ? process.argv[process.argv.indexOf("--data-path") + 1]
  : resolve(GRU_DATA_DIR, "bench-dataset-export.msgpack.gz");

// Helpers
function embeddingToBytes(emb) {
  const f32 = new Float32Array(emb);
  return new Uint8Array(f32.buffer);
}

function softTargetToBytes(sparse) {
  if (!sparse || sparse.length === 0)
    return { indicesBytes: new Uint8Array(0), probsBytes: new Uint8Array(0) };
  const indices = new Int32Array(sparse.map(([idx]) => idx));
  const probs = new Float32Array(sparse.map(([, prob]) => prob));
  return {
    indicesBytes: new Uint8Array(indices.buffer),
    probsBytes: new Uint8Array(probs.buffer),
  };
}

function writeParquetFile(table, filePath) {
  const ipcStream = arrow.tableToIPC(table, "stream");
  const wasmTable = WasmTable.fromIPCStream(ipcStream);
  const writerProps = new WriterPropertiesBuilder()
    .setCompression(Compression.SNAPPY)
    .build();
  const parquetBytes = writeParquet(wasmTable, writerProps);
  writeFileSync(filePath, parquetBytes);
}

function buildNodesTable(nodes) {
  const n = nodes.length;
  const ids = new Array(n);
  const levels = new Int32Array(n);
  const childrenJson = new Array(n);
  const embBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
  for (let i = 0; i < n; i++) {
    ids[i] = nodes[i].id;
    levels[i] = nodes[i].level;
    childrenJson[i] = JSON.stringify(nodes[i].children);
    embBuilder.append(embeddingToBytes(nodes[i].embedding));
  }
  embBuilder.finish();
  return new arrow.Table({
    id: arrow.vectorFromArray(ids, new arrow.Utf8()),
    embedding: embBuilder.toVector(),
    children_json: arrow.vectorFromArray(childrenJson, new arrow.Utf8()),
    level: arrow.makeVector(levels),
  });
}

function buildProdTable(examples) {
  const n = examples.length;
  const targetToolIds = new Array(n);
  const isTerminals = new Int32Array(n);
  const traceIds = new Array(n);
  const contextToolIdsJson = new Array(n);
  const embBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
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

function buildN8nTable(examples) {
  const n = examples.length;
  const targetToolIds = new Array(n);
  const isTerminals = new Int32Array(n);
  const contextToolIdsJson = new Array(n);
  const embBuilder = arrow.makeBuilder({ type: new arrow.Binary(), nullValues: [null] });
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

// ====== MAIN ======
console.log("=== Export Parquet from msgpack.gz (Node.js) ===\n");

console.log(`[Data] Loading ${dataPath}...`);
let compressed = readFileSync(dataPath);
console.log(`  Compressed: ${(compressed.byteLength / 1e6).toFixed(1)}MB`);
let raw = pako.ungzip(compressed);
compressed = null;
console.log(`  Decompressed: ${(raw.byteLength / 1e6).toFixed(1)}MB`);

console.log("  Decoding msgpack...");
const t0 = performance.now();
const ds = msgpackDecode(raw);
raw = null;
if (global.gc) global.gc();
console.log(`  Decoded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
console.log(`  Nodes: ${ds.nodes.length}, Prod: ${ds.prodTrain.length}+${ds.prodTest.length}, N8n: ${ds.n8nTrain.length}+${ds.n8nEval.length}\n`);

// Write each section, then free
function writeSection(name, buildFn, data, outFile) {
  const t = performance.now();
  console.log(`[Parquet] ${name}...`);
  const table = buildFn(data);
  console.log(`  Rows: ${table.numRows}`);
  const outPath = resolve(GRU_DATA_DIR, outFile);
  writeParquetFile(table, outPath);
  const stat = statSync(outPath);
  console.log(`  Written: ${outFile} (${(stat.size / 1e6).toFixed(1)}MB, ${((performance.now() - t) / 1000).toFixed(1)}s)\n`);
}

writeSection("Nodes", buildNodesTable, ds.nodes, "bench-nodes.parquet");
ds.nodes = null;
if (global.gc) global.gc();

writeSection("Prod train", buildProdTable, ds.prodTrain, "bench-prod-train.parquet");
ds.prodTrain = null;
if (global.gc) global.gc();

writeSection("Prod test", buildProdTable, ds.prodTest, "bench-prod-test.parquet");
ds.prodTest = null;
if (global.gc) global.gc();

writeSection("N8n train", buildN8nTable, ds.n8nTrain, "bench-n8n-train.parquet");
ds.n8nTrain = null;
if (global.gc) global.gc();

writeSection("N8n eval", buildN8nTable, ds.n8nEval, "bench-n8n-eval.parquet");
ds.n8nEval = null;
if (global.gc) global.gc();

// Metadata
const metadataPath = resolve(GRU_DATA_DIR, "bench-metadata.json");
writeFileSync(metadataPath, JSON.stringify({
  leafIds: ds.leafIds,
  embeddingDim: ds.embeddingDim,
  workflowToolLists: ds.workflowToolLists,
  exportedAt: new Date().toISOString(),
  source: "export-to-parquet-node.mjs",
}));
console.log(`[Metadata] Written: bench-metadata.json`);

const mem = (process.memoryUsage().rss / 1024 / 1024).toFixed(0);
console.log(`\n=== Export complete (RSS: ${mem}MB) ===`);
