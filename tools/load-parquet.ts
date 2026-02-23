/**
 * Lazy Parquet loading module for the SHGAT training pipeline.
 *
 * Provides typed loading functions that read individual .parquet files
 * on demand, avoiding the ~6GB peak memory of the monolithic msgpack.gz.
 *
 * Each function reads a single Parquet file and reconstructs the typed
 * objects expected by train-ob.ts.
 *
 * Embedding storage format:
 *   Binary column containing raw Float32 bytes (1024 * 4 = 4096 bytes/row).
 *   Reconstructed to number[] via Float32Array view on the underlying buffer.
 *
 * Usage:
 *   import { loadNodes, loadProdExamples, loadN8nExamples, loadMetadata } from "./load-parquet.ts";
 *
 *   const nodes = await loadNodes("../../gru/data/bench-nodes.parquet");
 *   const prodTrain = await loadProdExamples("../../gru/data/bench-prod-train.parquet");
 *   const n8nTrain = await loadN8nExamples("../../gru/data/bench-n8n-train.parquet");
 *   const meta = await loadMetadata("../../gru/data/bench-metadata.json");
 *
 * @module shgat-tf/tools/load-parquet
 */

import * as arrow from "apache-arrow";
// parquet-wasm: Node entry point auto-initializes WASM (no initWasm needed).
// ESM entry point requires calling the default export first.
// Deno npm: resolves to the Node entry point.
import parquetWasmModule from "parquet-wasm";
import {
  readParquet,
  Table as WasmTable,
} from "parquet-wasm";

// ==========================================================================
// Types (same as train-ob.ts / export-dataset.ts)
// ==========================================================================

export interface ExportedNode {
  id: string;
  embedding: number[];
  children: string[];
  level: number;
}

export interface ProdExample {
  intentEmbedding: number[];
  contextToolIds: string[];
  targetToolId: string;
  isTerminal: number;
  _traceId: string;
}

export interface N8nExample {
  intentEmbedding: number[];
  contextToolIds: string[];
  targetToolId: string;
  isTerminal: number;
  softTargetSparse: [number, number][];
}

export interface DatasetMetadata {
  leafIds: string[];
  embeddingDim: number;
  workflowToolLists: string[][];
}

// ==========================================================================
// WASM initialization (lazy, once)
// ==========================================================================

let wasmInitialized = false;

/**
 * Initialize parquet-wasm if needed.
 * Node entry: auto-initialized (parquetWasmModule is the module itself).
 * ESM entry: parquetWasmModule is the async init function, must be called once.
 */
async function ensureWasm(): Promise<void> {
  if (wasmInitialized) return;
  if (typeof parquetWasmModule === "function") {
    await (parquetWasmModule as unknown as () => Promise<void>)();
  }
  wasmInitialized = true;
}

// ==========================================================================
// Internal helpers
// ==========================================================================

/**
 * Read a .parquet file and return an Apache Arrow JS Table.
 */
async function readParquetToArrow(path: string): Promise<arrow.Table> {
  await ensureWasm();
  const parquetBytes = Deno.readFileSync(path);
  const wasmTable: WasmTable = readParquet(parquetBytes);
  const ipcStream = wasmTable.intoIPCStream();
  return arrow.tableFromIPC(ipcStream);
}

/**
 * Extract a number[] from raw Float32 bytes stored in a Binary column cell.
 * Returns a new number[] (not a Float32Array view, for safety with GC).
 */
function bytesToEmbedding(bytes: Uint8Array): number[] {
  // Ensure proper alignment for Float32Array
  const aligned = new Uint8Array(bytes.length);
  aligned.set(bytes);
  const f32 = new Float32Array(aligned.buffer, aligned.byteOffset, aligned.length / 4);
  return Array.from(f32);
}

/**
 * Reconstruct softTargetSparse from separate index/prob Binary columns.
 * Indices: Int32Array bytes. Probs: Float32Array bytes.
 */
function bytesToSoftTargetSparse(
  indicesBytes: Uint8Array,
  probsBytes: Uint8Array,
): [number, number][] {
  if (!indicesBytes || indicesBytes.length === 0) return [];

  // Ensure alignment
  const idxAligned = new Uint8Array(indicesBytes.length);
  idxAligned.set(indicesBytes);
  const indices = new Int32Array(idxAligned.buffer, idxAligned.byteOffset, idxAligned.length / 4);

  const probAligned = new Uint8Array(probsBytes.length);
  probAligned.set(probsBytes);
  const probs = new Float32Array(probAligned.buffer, probAligned.byteOffset, probAligned.length / 4);

  const result: [number, number][] = new Array(indices.length);
  for (let i = 0; i < indices.length; i++) {
    result[i] = [indices[i], probs[i]];
  }
  return result;
}

/**
 * Get a cell value from an Arrow column, handling the Binary type specially.
 * Arrow JS Binary columns return Uint8Array for each cell.
 */
function getColumnValue<T>(table: arrow.Table, colName: string, rowIdx: number): T {
  const col = table.getChild(colName);
  if (!col) {
    throw new Error(`[load-parquet] Column "${colName}" not found in table. Available: ${table.schema.fields.map((f: { name: string }) => f.name).join(", ")}`);
  }
  return col.get(rowIdx) as T;
}

// ==========================================================================
// Public API
// ==========================================================================

/**
 * Load nodes from bench-nodes.parquet.
 *
 * Columns: id (Utf8), embedding (Binary), children_json (Utf8), level (Int32)
 */
export async function loadNodes(path: string): Promise<ExportedNode[]> {
  const t0 = performance.now();
  const table = await readParquetToArrow(path);
  const n = table.numRows;
  const result: ExportedNode[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const embBytes = getColumnValue<Uint8Array>(table, "embedding", i);
    result[i] = {
      id: getColumnValue<string>(table, "id", i),
      embedding: bytesToEmbedding(embBytes),
      children: JSON.parse(getColumnValue<string>(table, "children_json", i)),
      level: getColumnValue<number>(table, "level", i),
    };
  }

  console.log(`[load-parquet] Loaded ${n} nodes from ${path} (${(performance.now() - t0).toFixed(0)}ms)`);
  return result;
}

/**
 * Load prod examples from a bench-prod-*.parquet file.
 *
 * Columns: intent_embedding (Binary), context_tool_ids_json (Utf8),
 *          target_tool_id (Utf8), is_terminal (Int32), trace_id (Utf8)
 */
export async function loadProdExamples(path: string): Promise<ProdExample[]> {
  const t0 = performance.now();
  const table = await readParquetToArrow(path);
  const n = table.numRows;
  const result: ProdExample[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const embBytes = getColumnValue<Uint8Array>(table, "intent_embedding", i);
    result[i] = {
      intentEmbedding: bytesToEmbedding(embBytes),
      contextToolIds: JSON.parse(getColumnValue<string>(table, "context_tool_ids_json", i)),
      targetToolId: getColumnValue<string>(table, "target_tool_id", i),
      isTerminal: getColumnValue<number>(table, "is_terminal", i),
      _traceId: getColumnValue<string>(table, "trace_id", i),
    };
  }

  console.log(`[load-parquet] Loaded ${n} prod examples from ${path} (${(performance.now() - t0).toFixed(0)}ms)`);
  return result;
}

/**
 * Load n8n examples from a bench-n8n-*.parquet file.
 *
 * Columns: intent_embedding (Binary), context_tool_ids_json (Utf8),
 *          target_tool_id (Utf8), is_terminal (Int32),
 *          soft_target_indices (Binary), soft_target_probs (Binary)
 */
export async function loadN8nExamples(path: string): Promise<N8nExample[]> {
  const t0 = performance.now();
  const table = await readParquetToArrow(path);
  const n = table.numRows;
  const result: N8nExample[] = new Array(n);

  for (let i = 0; i < n; i++) {
    const embBytes = getColumnValue<Uint8Array>(table, "intent_embedding", i);
    const indicesBytes = getColumnValue<Uint8Array>(table, "soft_target_indices", i);
    const probsBytes = getColumnValue<Uint8Array>(table, "soft_target_probs", i);

    result[i] = {
      intentEmbedding: bytesToEmbedding(embBytes),
      contextToolIds: JSON.parse(getColumnValue<string>(table, "context_tool_ids_json", i)),
      targetToolId: getColumnValue<string>(table, "target_tool_id", i),
      isTerminal: getColumnValue<number>(table, "is_terminal", i),
      softTargetSparse: bytesToSoftTargetSparse(indicesBytes, probsBytes),
    };
  }

  console.log(`[load-parquet] Loaded ${n} n8n examples from ${path} (${(performance.now() - t0).toFixed(0)}ms)`);
  return result;
}

/**
 * Load metadata from bench-metadata.json (plain JSON, no Parquet).
 */
export async function loadMetadata(path: string): Promise<DatasetMetadata> {
  const t0 = performance.now();
  const text = await Deno.readTextFile(path);
  const parsed = JSON.parse(text) as DatasetMetadata & Record<string, unknown>;
  const result: DatasetMetadata = {
    leafIds: parsed.leafIds,
    embeddingDim: parsed.embeddingDim,
    workflowToolLists: parsed.workflowToolLists,
  };
  console.log(`[load-parquet] Loaded metadata from ${path} (${(performance.now() - t0).toFixed(0)}ms)`);
  console.log(`  leafIds: ${result.leafIds.length}, embeddingDim: ${result.embeddingDim}, workflows: ${result.workflowToolLists.length}`);
  return result;
}

/**
 * Load the full dataset from Parquet files (convenience wrapper).
 *
 * Loads each table lazily in sequence to minimize peak memory.
 * Unlike the msgpack.gz loader, only one table is fully in memory at a time
 * during the loading phase (though all are retained in the returned object).
 *
 * @param dataDir - Directory containing the bench-*.parquet and bench-metadata.json files
 */
export async function loadFullDataset(dataDir: string): Promise<{
  nodes: ExportedNode[];
  leafIds: string[];
  embeddingDim: number;
  workflowToolLists: string[][];
  prodTrain: ProdExample[];
  prodTest: ProdExample[];
  n8nTrain: N8nExample[];
  n8nEval: N8nExample[];
}> {
  const { resolve } = await import("https://deno.land/std@0.224.0/path/mod.ts");

  const metadata = await loadMetadata(resolve(dataDir, "bench-metadata.json"));
  const nodes = await loadNodes(resolve(dataDir, "bench-nodes.parquet"));
  const prodTrain = await loadProdExamples(resolve(dataDir, "bench-prod-train.parquet"));
  const prodTest = await loadProdExamples(resolve(dataDir, "bench-prod-test.parquet"));
  const n8nTrain = await loadN8nExamples(resolve(dataDir, "bench-n8n-train.parquet"));
  const n8nEval = await loadN8nExamples(resolve(dataDir, "bench-n8n-eval.parquet"));

  return {
    nodes,
    leafIds: metadata.leafIds,
    embeddingDim: metadata.embeddingDim,
    workflowToolLists: metadata.workflowToolLists,
    prodTrain,
    prodTest,
    n8nTrain,
    n8nEval,
  };
}
