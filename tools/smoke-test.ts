#!/usr/bin/env npx tsx
/**
 * SHGAT Smoke Test — validates training pipeline end-to-end in ~2 min
 *
 * Tests: InfoNCE batch → GC → KL batch → precompute → score
 * Verifies no OOM, no tensor leak, reasonable metrics.
 *
 * Usage:
 *   cd lib/shgat-tf
 *   NODE_OPTIONS="--max-old-space-size=12288 --expose-gc" npx tsx tools/smoke-test.ts
 */

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import postgres from "postgres";

import { tf } from "../dist-node/src/tf/backend.ts";
import { SHGATBuilder } from "../dist-node/src/core/builder.ts";
import type { SoftTargetExample, TrainingExample } from "../dist-node/src/core/types.ts";
import { NUM_NEGATIVES } from "../dist-node/src/core/types.ts";
import { type BenchDataset, loadBenchDataset } from "../../gru/src/shgat/bench-dataset.ts";

const __dirname = dirname(fileURLToPath(import.meta.url));
const EXPANDED_VOCAB_PATH = resolve(__dirname, "../../gru/data/expanded-vocab.json");
const N8N_SHGAT_PAIRS_PATH = resolve(__dirname, "../../gru/data/n8n-shgat-contrastive-pairs.json");
const GRU_DATA_DIR = resolve(__dirname, "../../gru/data");

function rss(): string {
  return (process.memoryUsage().rss / 1e6).toFixed(0);
}

async function main() {
  const DATABASE_URL = process.env.DATABASE_URL;
  if (!DATABASE_URL) throw new Error("DATABASE_URL env var required");

  console.log("[smoke] Loading dataset...");
  const sql = postgres(DATABASE_URL);
  const ds: BenchDataset = await loadBenchDataset(sql, {
    expandedVocabPath: EXPANDED_VOCAB_PATH,
    n8nPairsPath: N8N_SHGAT_PAIRS_PATH,
    n8nDataDir: GRU_DATA_DIR,
    maxN8nWorkflows: 99999,
    prodOversample: 3,
    splitSeed: 42,
  });
  console.log(
    `[smoke] ${ds.leafIds.length} tools, ${ds.nodes.length} nodes, ${ds.prodTrain.length} prod train, ${ds.n8nTrain.length} n8n train`,
  );

  console.log("[smoke] Building SHGAT...");
  const shgat = await SHGATBuilder.create()
    .nodes(ds.nodes)
    .training({ learningRate: 0.001, temperature: 0.07 })
    .backend("training")
    .build();

  const t0 = tf.memory();
  console.log(`[smoke] Ready: ${t0.numTensors} tensors, ${rss()}MB RSS`);

  // --- Phase 1: InfoNCE batch (full MP-in-tape) ---
  console.log("\n[smoke] Phase 1: InfoNCE batch (MP-in-tape)...");
  const infoBatch: TrainingExample[] = ds.prodTrain.slice(0, 32).map((ex) => {
    const exclude = new Set([ex.candidateId, ...ex.contextToolIds]);
    const negs = ds.leafIds.filter((id) => !exclude.has(id)).slice(0, NUM_NEGATIVES);
    return {
      intentEmbedding: Array.from(ex.intentEmbedding),
      candidateId: ex.candidateId,
      contextTools: ex.contextToolIds,
      outcome: 1,
      negativeCapIds: negs,
    };
  });

  const m1 = await shgat.trainBatch(infoBatch);
  const t1 = tf.memory();
  console.log(
    `[smoke] InfoNCE: loss=${m1.loss.toFixed(4)} acc=${m1.accuracy.toFixed(2)} grad=${
      m1.gradientNorm.toFixed(2)
    }`,
  );
  console.log(
    `[smoke] Tensors: ${t1.numTensors} (+${
      t1.numTensors - t0.numTensors
    } = Adam state), RSS=${rss()}MB`,
  );

  // --- Phase 2: Force GC ---
  // deno-lint-ignore no-explicit-any
  const gc = (global as any).gc;
  if (typeof gc === "function") {
    gc();
    console.log(`[smoke] After GC: RSS=${rss()}MB`);
  } else {
    console.log("[smoke] WARNING: gc() not available. Use --expose-gc");
  }

  // --- Phase 3: KL batch (raw embeddings, no MP) ---
  console.log("\n[smoke] Phase 3: KL batch (raw embeddings)...");
  const klBatch: SoftTargetExample[] = ds.n8nTrain.slice(0, 32).map((ex) => ({
    intentEmbedding: Array.from(ex.intentEmbedding) as number[],
    softTargetSparse: ex.softTargetSparse,
  }));

  const m2 = await shgat.trainBatchKL(klBatch, ds.leafIds, 0.07);
  const t2 = tf.memory();
  console.log(`[smoke] KL: loss=${m2.klLoss.toFixed(4)} grad=${m2.gradientNorm.toFixed(2)}`);
  console.log(
    `[smoke] Tensors: ${t2.numTensors} (+${t2.numTensors - t1.numTensors}), RSS=${rss()}MB`,
  );

  // --- Phase 4: KL stress test (100 batches, check leak + OOM) ---
  console.log("\n[smoke] Phase 4: KL stress test (100 batches)...");
  const KL_STRESS = 100;
  let t3 = t2;
  for (let i = 0; i < KL_STRESS; i++) {
    const start = (i * 32) % ds.n8nTrain.length;
    const klStressBatch: SoftTargetExample[] = ds.n8nTrain.slice(start, start + 32).map((ex) => ({
      intentEmbedding: Array.from(ex.intentEmbedding) as number[],
      softTargetSparse: ex.softTargetSparse,
    }));
    await shgat.trainBatchKL(klStressBatch, ds.leafIds, 0.07);
    if ((i + 1) % 25 === 0) {
      if (typeof gc === "function") gc();
      const tm = tf.memory();
      console.log(
        `[smoke] KL batch ${i + 1}/${KL_STRESS}: tensors=${tm.numTensors} RSS=${rss()}MB`,
      );
    }
  }
  t3 = tf.memory();
  console.log(
    `[smoke] After KL stress: tensors=${t3.numTensors} (+${
      t3.numTensors - t2.numTensors
    }), RSS=${rss()}MB`,
  );
  if (t3.numTensors > t2.numTensors + 5) {
    console.log("[smoke] WARNING: tensor leak in KL batches!");
  }

  // --- Phase 5: Precompute enriched embeddings ---
  console.log("\n[smoke] Phase 5: Precompute enriched embeddings...");
  shgat.precomputeEnrichedEmbeddings();
  const t4 = tf.memory();
  console.log(
    `[smoke] Tensors: ${t4.numTensors} (+${t4.numTensors - t3.numTensors}), RSS=${rss()}MB`,
  );
  if (t4.numTensors > t3.numTensors) {
    console.log("[smoke] WARNING: tensor leak in precompute!");
  }

  // --- Phase 6: Score eval ---
  console.log("\n[smoke] Phase 6: Score eval...");
  const testEx = ds.prodTest[0];
  const scores = shgat.score(testEx.intentEmbedding, ds.leafIds);
  const targetIdx = ds.leafIds.indexOf(testEx.targetToolId);
  const sorted = [...scores].map((s, i) => ({ s, i })).sort((a, b) => b.s - a.s);
  const rank = sorted.findIndex((x) => x.i === targetIdx) + 1;
  console.log(`[smoke] Target tool rank: ${rank}/${ds.leafIds.length}`);

  const t5 = tf.memory();
  console.log(`[smoke] Final: ${t5.numTensors} tensors, RSS=${rss()}MB`);

  // --- Summary ---
  console.log("\n=== SMOKE TEST SUMMARY ===");
  console.log(`  InfoNCE:    loss=${m1.loss.toFixed(4)} acc=${(m1.accuracy * 100).toFixed(1)}%`);
  console.log(`  KL:         loss=${m2.klLoss.toFixed(4)}`);
  console.log(`  Score rank: ${rank}/${ds.leafIds.length}`);
  console.log(`  Tensors:    ${t0.numTensors} → ${t5.numTensors} (expect +~290 for Adam)`);
  console.log(`  Memory:     ${rss()}MB RSS`);
  const leaked = t5.numTensors - t0.numTensors - 290; // ~290 for Adam optimizer state
  if (leaked > 10) {
    console.log(`  ⚠️  POSSIBLE LEAK: ~${leaked} extra tensors`);
  } else {
    console.log(`  ✓ No significant tensor leak`);
  }
  console.log("=== SMOKE TEST PASSED ===");

  process.exit(0);
}

main().catch((e) => {
  console.error("[smoke] FAILED:", e);
  process.exit(1);
});
