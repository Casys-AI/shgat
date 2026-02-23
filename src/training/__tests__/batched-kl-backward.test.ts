/**
 * Tests for batched KL backward & forward optimizations.
 *
 * Verifies mathematical equivalence between:
 * - Per-target backward (backpropMultiHeadKHeadLogit + outerProductAdd)
 * - Batched backward (matmul on stacked dQ/dK matrices)
 *
 * And between:
 * - Per-example forward (matVecBlas per intent × per head)
 * - Batched forward (matmulTranspose on stacked intents)
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import * as math from "../../utils/math.ts";
import {
  backpropMultiHeadKHeadLogit,
  backpropWIntent,
  type HeadParams,
  type KHeadGradientAccumulators,
  type MultiLevelKHeadGradientAccumulators,
} from "../multi-level-trainer-khead.ts";
import type { SHGATConfig } from "../../core/types.ts";

// ============================================================================
// Helpers
// ============================================================================

const SEED = 42;
function seededRandom(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF;
    return (s >>> 0) / 0xFFFFFFFF;
  };
}

function makeConfig(numHeads: number, headDim: number, embDim: number): SHGATConfig {
  return {
    embeddingDim: embDim,
    hiddenDim: embDim,
    headDim,
    numHeads,
    numLayers: 1,
    mlpHiddenDim: embDim,
    learningRate: 0.001,
    batchSize: 32,
    maxContextLength: 10,
    maxBufferSize: 100,
    minTracesForTraining: 5,
    dropout: 0,
    l2Lambda: 0,
    leakyReluSlope: 0.2,
    depthDecay: 0.8,
  };
}

function randomMatrix(rows: number, cols: number, rng: () => number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => (rng() - 0.5) * 0.2)
  );
}

function randomVec(dim: number, rng: () => number): number[] {
  return Array.from({ length: dim }, () => (rng() - 0.5) * 2);
}

function zeros2D(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function clone2D(m: number[][]): number[][] {
  return m.map(row => [...row]);
}

function makeHeadParams(numHeads: number, headDim: number, embDim: number, rng: () => number): HeadParams[] {
  return Array.from({ length: numHeads }, () => ({
    W_q: randomMatrix(headDim, embDim, rng),
    W_k: randomMatrix(headDim, embDim, rng),
  }));
}

function makeKHeadGrads(numHeads: number, headDim: number, embDim: number): KHeadGradientAccumulators {
  return {
    dW_q: Array.from({ length: numHeads }, () => zeros2D(headDim, embDim)),
    dW_k: Array.from({ length: numHeads }, () => zeros2D(headDim, embDim)),
  };
}

function makeMultiLevelGrads(numHeads: number, headDim: number, embDim: number): MultiLevelKHeadGradientAccumulators {
  return {
    khead: makeKHeadGrads(numHeads, headDim, embDim),
    dW_intent: zeros2D(embDim, embDim),
    levelGradients: new Map(),
  };
}

function matrixMaxDiff(a: number[][], b: number[][]): number {
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < a[i].length; j++) {
      maxDiff = Math.max(maxDiff, Math.abs(a[i][j] - b[i][j]));
    }
  }
  return maxDiff;
}

function vecMaxDiff(a: number[], b: number[]): number {
  let maxDiff = 0;
  for (let i = 0; i < a.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(a[i] - b[i]));
  }
  return maxDiff;
}

// ============================================================================
// Batched Backward Tests
// ============================================================================

Deno.test("batched backward: single tuple matches per-target backward", () => {
  const rng = seededRandom(SEED);
  const numHeads = 2, headDim = 4, embDim = 8;
  const config = makeConfig(numHeads, headDim, embDim);
  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);

  const intentProjected = randomVec(embDim, rng);
  const nodeEmb = randomVec(embDim, rng);
  const dLogit = 0.05;

  // --- Per-target (reference) ---
  const Q_perHead: number[][] = [];
  const K_perHead: number[][] = [];
  for (let h = 0; h < numHeads; h++) {
    Q_perHead.push(math.matVecBlas(headParams[h].W_q, intentProjected));
    K_perHead.push(math.matVecBlas(headParams[h].W_k, nodeEmb));
  }
  const headCaches = Q_perHead.map((Q, h) => ({
    Q, K: K_perHead[h],
    dotQK: math.dot(Q, K_perHead[h]),
  }));

  const refGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const refResult = backpropMultiHeadKHeadLogit(
    dLogit, headCaches, intentProjected, nodeEmb,
    headParams, refGrads, config,
  );

  // --- Batched (1 tuple) ---
  const batchGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const T = 1;
  const invScale = 1.0 / Math.sqrt(headDim);
  const invHeads = 1.0 / numHeads;
  const dHeadLogitScale = invHeads * invScale;

  const dIntentProjAll = zeros2D(T, embDim);
  const dNodeEmbAll = zeros2D(T, embDim);

  for (let h = 0; h < numHeads; h++) {
    const d = dLogit * dHeadLogitScale;
    const dQ = K_perHead[h].map(k => k * d);
    const dK = Q_perHead[h].map(q => q * d);

    const dQ_batch = [dQ]; // [1, headDim]
    const dK_batch = [dK]; // [1, headDim]
    const intentProjBatch = [intentProjected]; // [1, embDim]
    const nodeEmbBatch = [nodeEmb]; // [1, embDim]

    // Weight gradients via matmul
    const dQ_T = math.transpose(dQ_batch); // [headDim, 1]
    const dK_T = math.transpose(dK_batch);
    const dWq_h = math.matmul(dQ_T, intentProjBatch); // [headDim, embDim]
    const dWk_h = math.matmul(dK_T, nodeEmbBatch);

    // Accumulate
    for (let r = 0; r < headDim; r++) {
      for (let c = 0; c < embDim; c++) {
        batchGrads.dW_q[h][r][c] += dWq_h[r][c];
        batchGrads.dW_k[h][r][c] += dWk_h[r][c];
      }
    }

    // Input gradients via matmul
    const dIntentBatch_h = math.matmul(dQ_batch, headParams[h].W_q); // [1, embDim]
    const dNodeEmbBatch_h = math.matmul(dK_batch, headParams[h].W_k);

    for (let d = 0; d < embDim; d++) {
      dIntentProjAll[0][d] += dIntentBatch_h[0][d];
      dNodeEmbAll[0][d] += dNodeEmbBatch_h[0][d];
    }
  }

  // Compare weight gradients
  for (let h = 0; h < numHeads; h++) {
    const diffWq = matrixMaxDiff(refGrads.dW_q[h], batchGrads.dW_q[h]);
    const diffWk = matrixMaxDiff(refGrads.dW_k[h], batchGrads.dW_k[h]);
    assert(diffWq < 1e-10, `dW_q[${h}] maxDiff=${diffWq} should be ~0`);
    assert(diffWk < 1e-10, `dW_k[${h}] maxDiff=${diffWk} should be ~0`);
  }

  // Compare input gradients
  const diffIntent = vecMaxDiff(refResult.dIntentProjected, dIntentProjAll[0]);
  const diffNode = vecMaxDiff(refResult.dNodeEmbedding, dNodeEmbAll[0]);
  assert(diffIntent < 1e-10, `dIntentProjected maxDiff=${diffIntent} should be ~0`);
  assert(diffNode < 1e-10, `dNodeEmbedding maxDiff=${diffNode} should be ~0`);
});

Deno.test("batched backward: multi-tuple matches sum of per-target backwards", () => {
  const rng = seededRandom(SEED);
  const numHeads = 4, headDim = 8, embDim = 16;
  const config = makeConfig(numHeads, headDim, embDim);
  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);

  // Create T=5 tuples (simulating 5 sparse targets with dLogit)
  const T = 5;
  const tuples: Array<{
    dLogit: number;
    intentProjected: number[];
    nodeEmb: number[];
    Q_perHead: number[][];
    K_perHead: number[][];
  }> = [];

  for (let t = 0; t < T; t++) {
    const intentProjected = randomVec(embDim, rng);
    const nodeEmb = randomVec(embDim, rng);
    const Q_perHead: number[][] = [];
    const K_perHead: number[][] = [];
    for (let h = 0; h < numHeads; h++) {
      Q_perHead.push(math.matVecBlas(headParams[h].W_q, intentProjected));
      K_perHead.push(math.matVecBlas(headParams[h].W_k, nodeEmb));
    }
    tuples.push({
      dLogit: (rng() - 0.5) * 0.1,
      intentProjected, nodeEmb, Q_perHead, K_perHead,
    });
  }

  // --- Reference: per-target backward ---
  const refGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const refDIntents: number[][] = [];
  const refDNodes: number[][] = [];

  for (const tuple of tuples) {
    const headCaches = tuple.Q_perHead.map((Q, h) => ({
      Q, K: tuple.K_perHead[h],
      dotQK: math.dot(Q, tuple.K_perHead[h]),
    }));
    const result = backpropMultiHeadKHeadLogit(
      tuple.dLogit, headCaches, tuple.intentProjected, tuple.nodeEmb,
      headParams, refGrads, config,
    );
    refDIntents.push(result.dIntentProjected);
    refDNodes.push(result.dNodeEmbedding);
  }

  // --- Batched backward ---
  const batchGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const invScale = 1.0 / Math.sqrt(headDim);
  const invHeads = 1.0 / numHeads;
  const dHeadLogitScale = invHeads * invScale;

  const intentProjBatch = tuples.map(t => t.intentProjected);
  const nodeEmbBatch = tuples.map(t => t.nodeEmb);
  const dIntentProjAll = zeros2D(T, embDim);
  const dNodeEmbAll = zeros2D(T, embDim);

  for (let h = 0; h < numHeads; h++) {
    const dQ_batch: number[][] = new Array(T);
    const dK_batch: number[][] = new Array(T);

    for (let t = 0; t < T; t++) {
      const d = tuples[t].dLogit * dHeadLogitScale;
      dQ_batch[t] = tuples[t].K_perHead[h].map(k => k * d);
      dK_batch[t] = tuples[t].Q_perHead[h].map(q => q * d);
    }

    const dQ_T = math.transpose(dQ_batch);
    const dK_T = math.transpose(dK_batch);
    const dWq_h = math.matmul(dQ_T, intentProjBatch);
    const dWk_h = math.matmul(dK_T, nodeEmbBatch);

    for (let r = 0; r < headDim; r++) {
      for (let c = 0; c < embDim; c++) {
        batchGrads.dW_q[h][r][c] += dWq_h[r][c];
        batchGrads.dW_k[h][r][c] += dWk_h[r][c];
      }
    }

    const dIntentBatch_h = math.matmul(dQ_batch, headParams[h].W_q);
    const dNodeEmbBatch_h = math.matmul(dK_batch, headParams[h].W_k);

    for (let t = 0; t < T; t++) {
      for (let d = 0; d < embDim; d++) {
        dIntentProjAll[t][d] += dIntentBatch_h[t][d];
        dNodeEmbAll[t][d] += dNodeEmbBatch_h[t][d];
      }
    }
  }

  // Compare weight gradients (accumulated over T tuples)
  for (let h = 0; h < numHeads; h++) {
    const diffWq = matrixMaxDiff(refGrads.dW_q[h], batchGrads.dW_q[h]);
    const diffWk = matrixMaxDiff(refGrads.dW_k[h], batchGrads.dW_k[h]);
    assert(diffWq < 1e-8, `dW_q[${h}] maxDiff=${diffWq} (T=${T} tuples)`);
    assert(diffWk < 1e-8, `dW_k[${h}] maxDiff=${diffWk} (T=${T} tuples)`);
  }

  // Compare input gradients per tuple
  for (let t = 0; t < T; t++) {
    const diffI = vecMaxDiff(refDIntents[t], dIntentProjAll[t]);
    const diffN = vecMaxDiff(refDNodes[t], dNodeEmbAll[t]);
    assert(diffI < 1e-8, `tuple[${t}] dIntentProjected maxDiff=${diffI}`);
    assert(diffN < 1e-8, `tuple[${t}] dNodeEmbedding maxDiff=${diffN}`);
  }
});

Deno.test("batched backward: shared Q_perHead (same example, multiple targets)", () => {
  // In KL, the same example can have multiple sparse targets.
  // Q_perHead and intentProjected are shared across targets of the same example.
  // The batched backward must produce the same accumulated dW and dIntent.

  const rng = seededRandom(SEED + 1);
  const numHeads = 2, headDim = 4, embDim = 8;
  const config = makeConfig(numHeads, headDim, embDim);
  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);

  // Shared intent for one example, 3 different targets
  const intentProjected = randomVec(embDim, rng);
  const Q_perHead: number[][] = [];
  for (let h = 0; h < numHeads; h++) {
    Q_perHead.push(math.matVecBlas(headParams[h].W_q, intentProjected));
  }

  const targets = [
    { nodeEmb: randomVec(embDim, rng), dLogit: 0.03 },
    { nodeEmb: randomVec(embDim, rng), dLogit: -0.02 },
    { nodeEmb: randomVec(embDim, rng), dLogit: 0.01 },
  ];

  // --- Reference: per-target ---
  const refGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const refDIntentSum = new Array(embDim).fill(0);

  for (const target of targets) {
    const K_perHead: number[][] = [];
    for (let h = 0; h < numHeads; h++) {
      K_perHead.push(math.matVecBlas(headParams[h].W_k, target.nodeEmb));
    }
    const headCaches = Q_perHead.map((Q, h) => ({
      Q, K: K_perHead[h], dotQK: math.dot(Q, K_perHead[h]),
    }));
    const result = backpropMultiHeadKHeadLogit(
      target.dLogit, headCaches, intentProjected, target.nodeEmb,
      headParams, refGrads, config,
    );
    for (let d = 0; d < embDim; d++) refDIntentSum[d] += result.dIntentProjected[d];
  }

  // --- Batched ---
  const batchGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const T = targets.length;
  const invScale = 1.0 / Math.sqrt(headDim);
  const invHeads = 1.0 / numHeads;
  const dHeadLogitScale = invHeads * invScale;

  const intentProjBatch = targets.map(() => intentProjected); // shared
  const nodeEmbBatch = targets.map(t => t.nodeEmb);
  const dIntentProjAll = zeros2D(T, embDim);

  for (let h = 0; h < numHeads; h++) {
    const K_batch: number[][] = targets.map(t => math.matVecBlas(headParams[h].W_k, t.nodeEmb));
    const dQ_batch: number[][] = new Array(T);
    const dK_batch: number[][] = new Array(T);

    for (let t = 0; t < T; t++) {
      const d = targets[t].dLogit * dHeadLogitScale;
      dQ_batch[t] = K_batch[t].map(k => k * d);
      dK_batch[t] = Q_perHead[h].map(q => q * d);
    }

    const dQ_T = math.transpose(dQ_batch);
    const dK_T = math.transpose(dK_batch);
    const dWq_h = math.matmul(dQ_T, intentProjBatch);
    const dWk_h = math.matmul(dK_T, nodeEmbBatch);

    for (let r = 0; r < headDim; r++) {
      for (let c = 0; c < embDim; c++) {
        batchGrads.dW_q[h][r][c] += dWq_h[r][c];
        batchGrads.dW_k[h][r][c] += dWk_h[r][c];
      }
    }

    const dIntentBatch_h = math.matmul(dQ_batch, headParams[h].W_q);
    for (let t = 0; t < T; t++) {
      for (let d = 0; d < embDim; d++) {
        dIntentProjAll[t][d] += dIntentBatch_h[t][d];
      }
    }
  }

  // Sum dIntentProjected across all targets of same example
  const batchDIntentSum = new Array(embDim).fill(0);
  for (let t = 0; t < T; t++) {
    for (let d = 0; d < embDim; d++) batchDIntentSum[d] += dIntentProjAll[t][d];
  }

  // Compare
  for (let h = 0; h < numHeads; h++) {
    assert(matrixMaxDiff(refGrads.dW_q[h], batchGrads.dW_q[h]) < 1e-10, `dW_q[${h}] mismatch`);
    assert(matrixMaxDiff(refGrads.dW_k[h], batchGrads.dW_k[h]) < 1e-10, `dW_k[${h}] mismatch`);
  }
  assert(vecMaxDiff(refDIntentSum, batchDIntentSum) < 1e-10, `dIntentProjected sum mismatch`);
});

Deno.test("batched backward: zero dLogit tuple produces zero gradients", () => {
  const rng = seededRandom(SEED + 2);
  const numHeads = 2, headDim = 4, embDim = 8;
  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);

  const intentProjected = randomVec(embDim, rng);
  const nodeEmb = randomVec(embDim, rng);
  const Q_perHead = Array.from({ length: numHeads }, (_, h) =>
    math.matVecBlas(headParams[h].W_q, intentProjected));
  const K_perHead = Array.from({ length: numHeads }, (_, h) =>
    math.matVecBlas(headParams[h].W_k, nodeEmb));

  const dLogit = 0; // zero gradient
  const invScale = 1.0 / Math.sqrt(headDim);
  const invHeads = 1.0 / numHeads;
  const dHeadLogitScale = invHeads * invScale;

  const grads = makeKHeadGrads(numHeads, headDim, embDim);

  for (let h = 0; h < numHeads; h++) {
    const d = dLogit * dHeadLogitScale;
    const dQ = K_perHead[h].map(k => k * d);
    const dK = Q_perHead[h].map(q => q * d);
    const dQ_T = math.transpose([dQ]);
    const dK_T = math.transpose([dK]);
    const dWq_h = math.matmul(dQ_T, [intentProjected]);
    const dWk_h = math.matmul(dK_T, [nodeEmb]);
    for (let r = 0; r < headDim; r++) {
      for (let c = 0; c < embDim; c++) {
        grads.dW_q[h][r][c] += dWq_h[r][c];
        grads.dW_k[h][r][c] += dWk_h[r][c];
      }
    }
  }

  for (let h = 0; h < numHeads; h++) {
    for (const row of grads.dW_q[h]) for (const v of row) assertAlmostEquals(v, 0, 1e-15);
    for (const row of grads.dW_k[h]) for (const v of row) assertAlmostEquals(v, 0, 1e-15);
  }
});

Deno.test("batched backward: large T (stress test, T=100)", () => {
  const rng = seededRandom(SEED + 3);
  const numHeads = 4, headDim = 16, embDim = 32;
  const config = makeConfig(numHeads, headDim, embDim);
  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);

  const T = 100;
  const tuples = Array.from({ length: T }, () => {
    const intentProjected = randomVec(embDim, rng);
    const nodeEmb = randomVec(embDim, rng);
    const Q_perHead = Array.from({ length: numHeads }, (_, h) =>
      math.matVecBlas(headParams[h].W_q, intentProjected));
    const K_perHead = Array.from({ length: numHeads }, (_, h) =>
      math.matVecBlas(headParams[h].W_k, nodeEmb));
    return {
      dLogit: (rng() - 0.5) * 0.1,
      intentProjected, nodeEmb, Q_perHead, K_perHead,
    };
  });

  // Reference: per-target
  const refGrads = makeKHeadGrads(numHeads, headDim, embDim);
  for (const tuple of tuples) {
    const headCaches = tuple.Q_perHead.map((Q, h) => ({
      Q, K: tuple.K_perHead[h], dotQK: math.dot(Q, tuple.K_perHead[h]),
    }));
    backpropMultiHeadKHeadLogit(
      tuple.dLogit, headCaches, tuple.intentProjected, tuple.nodeEmb,
      headParams, refGrads, config,
    );
  }

  // Batched
  const batchGrads = makeKHeadGrads(numHeads, headDim, embDim);
  const invScale = 1.0 / Math.sqrt(headDim);
  const invHeads = 1.0 / numHeads;
  const dHeadLogitScale = invHeads * invScale;

  const intentProjBatch = tuples.map(t => t.intentProjected);
  const nodeEmbBatch = tuples.map(t => t.nodeEmb);

  for (let h = 0; h < numHeads; h++) {
    const dQ_batch = tuples.map(t => {
      const d = t.dLogit * dHeadLogitScale;
      return t.K_perHead[h].map(k => k * d);
    });
    const dK_batch = tuples.map(t => {
      const d = t.dLogit * dHeadLogitScale;
      return t.Q_perHead[h].map(q => q * d);
    });

    const dWq_h = math.matmul(math.transpose(dQ_batch), intentProjBatch);
    const dWk_h = math.matmul(math.transpose(dK_batch), nodeEmbBatch);

    for (let r = 0; r < headDim; r++) {
      for (let c = 0; c < embDim; c++) {
        batchGrads.dW_q[h][r][c] += dWq_h[r][c];
        batchGrads.dW_k[h][r][c] += dWk_h[r][c];
      }
    }
  }

  // With T=100 and dim=32, floating point error accumulates more
  for (let h = 0; h < numHeads; h++) {
    const diffWq = matrixMaxDiff(refGrads.dW_q[h], batchGrads.dW_q[h]);
    const diffWk = matrixMaxDiff(refGrads.dW_k[h], batchGrads.dW_k[h]);
    assert(diffWq < 1e-6, `T=100: dW_q[${h}] maxDiff=${diffWq}`);
    assert(diffWk < 1e-6, `T=100: dW_k[${h}] maxDiff=${diffWk}`);
  }
});

// ============================================================================
// Batched Forward Tests
// ============================================================================

Deno.test("batched forward: matmulTranspose matches per-example matVecBlas for W_intent", () => {
  const rng = seededRandom(SEED + 10);
  const embDim = 16;
  const B = 8;

  const W_intent = randomMatrix(embDim, embDim, rng);
  const intents = Array.from({ length: B }, () => randomVec(embDim, rng));

  // Per-example: matVecBlas(W_intent, intent) for each
  const perExample = intents.map(intent => math.matVecBlas(W_intent, intent));

  // Batched: matmulTranspose(intentBatch, W_intent) = [B, embDim] @ [embDim, embDim]^T
  const batched = math.matmulTranspose(intents, W_intent);

  assertEquals(batched.length, B, "batch size");
  assertEquals(batched[0].length, embDim, "output dim");

  for (let i = 0; i < B; i++) {
    const diff = vecMaxDiff(perExample[i], batched[i]);
    assert(diff < 1e-10, `example[${i}] maxDiff=${diff}`);
  }
});

Deno.test("batched forward: matmulTranspose matches per-example matVecBlas for W_q", () => {
  const rng = seededRandom(SEED + 11);
  const headDim = 8, embDim = 16;
  const B = 10;

  const W_q = randomMatrix(headDim, embDim, rng); // [headDim, embDim]
  const intentProjs = Array.from({ length: B }, () => randomVec(embDim, rng));

  // Per-example: matVecBlas(W_q, intentProj) — JS fallback since headDim < 256
  const perExample = intentProjs.map(ip => math.matVecBlas(W_q, ip));

  // Batched: matmulTranspose(intentProjBatch, W_q) = [B, embDim] @ [headDim, embDim]^T = [B, headDim]
  const batched = math.matmulTranspose(intentProjs, W_q);

  assertEquals(batched.length, B, "batch size");
  assertEquals(batched[0].length, headDim, "head dim");

  for (let i = 0; i < B; i++) {
    const diff = vecMaxDiff(perExample[i], batched[i]);
    assert(diff < 1e-10, `Q[${i}] maxDiff=${diff}`);
  }
});

Deno.test("batched forward: Q computation matches per-example across all heads", () => {
  const rng = seededRandom(SEED + 12);
  const numHeads = 4, headDim = 8, embDim = 32;
  const B = 6;

  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);
  const W_intent = randomMatrix(embDim, embDim, rng);
  const intents = Array.from({ length: B }, () => randomVec(embDim, rng));

  // --- Per-example (reference) ---
  const refQ: number[][][] = []; // [B][numHeads][headDim]
  const refIntentProj: number[][] = [];
  for (let i = 0; i < B; i++) {
    const intentProj = math.matVecBlas(W_intent, intents[i]);
    refIntentProj.push(intentProj);
    const qPerHead: number[][] = [];
    for (let h = 0; h < numHeads; h++) {
      qPerHead.push(math.matVecBlas(headParams[h].W_q, intentProj));
    }
    refQ.push(qPerHead);
  }

  // --- Batched ---
  const intentProjBatch = math.matmulTranspose(intents, W_intent); // [B, embDim]
  const Q_allHeads: number[][][] = new Array(numHeads); // [numHeads][B][headDim]
  for (let h = 0; h < numHeads; h++) {
    Q_allHeads[h] = math.matmulTranspose(intentProjBatch, headParams[h].W_q); // [B, headDim]
  }

  // Compare intentProjected
  for (let i = 0; i < B; i++) {
    const diff = vecMaxDiff(refIntentProj[i], intentProjBatch[i]);
    assert(diff < 1e-10, `intentProj[${i}] maxDiff=${diff}`);
  }

  // Compare Q per head
  for (let i = 0; i < B; i++) {
    for (let h = 0; h < numHeads; h++) {
      const diff = vecMaxDiff(refQ[i][h], Q_allHeads[h][i]);
      assert(diff < 1e-10, `Q[${i}][head${h}] maxDiff=${diff}`);
    }
  }
});

Deno.test("batched forward: scoring (dot Q·K) invariant", () => {
  // Scoring is per-example (sparse targets differ), so we verify
  // that using Q from batched result gives same logits as per-example Q
  const rng = seededRandom(SEED + 13);
  const numHeads = 2, headDim = 4, embDim = 8;
  const B = 4, numTargets = 3;

  const headParams = makeHeadParams(numHeads, headDim, embDim, rng);
  const W_intent = randomMatrix(embDim, embDim, rng);
  const intents = Array.from({ length: B }, () => randomVec(embDim, rng));
  // Pre-projected keys (simulating projectedKeysPerHead)
  const keys = Array.from({ length: numTargets }, () =>
    Array.from({ length: numHeads }, () => randomVec(headDim, rng))
  );

  const invScale = 1.0 / Math.sqrt(headDim);
  const invHeads = 1.0 / numHeads;

  // Per-example scoring
  const refLogits: number[][] = [];
  for (let i = 0; i < B; i++) {
    const intentProj = math.matVecBlas(W_intent, intents[i]);
    const Q: number[][] = [];
    for (let h = 0; h < numHeads; h++) {
      Q.push(math.matVecBlas(headParams[h].W_q, intentProj));
    }
    const logits: number[] = [];
    for (let j = 0; j < numTargets; j++) {
      let avg = 0;
      for (let h = 0; h < numHeads; h++) {
        avg += math.dot(Q[h], keys[j][h]) * invScale;
      }
      logits.push(avg * invHeads);
    }
    refLogits.push(logits);
  }

  // Batched Q, same scoring
  const intentProjBatch = math.matmulTranspose(intents, W_intent);
  const Q_allHeads: number[][][] = Array.from({ length: numHeads }, (_, h) =>
    math.matmulTranspose(intentProjBatch, headParams[h].W_q)
  );

  const batchLogits: number[][] = [];
  for (let i = 0; i < B; i++) {
    const logits: number[] = [];
    for (let j = 0; j < numTargets; j++) {
      let avg = 0;
      for (let h = 0; h < numHeads; h++) {
        avg += math.dot(Q_allHeads[h][i], keys[j][h]) * invScale;
      }
      logits.push(avg * invHeads);
    }
    batchLogits.push(logits);
  }

  for (let i = 0; i < B; i++) {
    for (let j = 0; j < numTargets; j++) {
      assertAlmostEquals(refLogits[i][j], batchLogits[i][j], 1e-10,
        `logit[${i}][${j}] mismatch`);
    }
  }
});

// ============================================================================
// W_intent backward test
// ============================================================================

Deno.test("batched backward: W_intent gradient via scatter-sum matches per-target", () => {
  const rng = seededRandom(SEED + 20);
  const embDim = 8;

  // 2 examples, example 0 has 2 targets, example 1 has 1 target
  const intentEmbs = [randomVec(embDim, rng), randomVec(embDim, rng)];
  const dIntentProjs = [
    randomVec(embDim, rng), // ex0, target0
    randomVec(embDim, rng), // ex0, target1
    randomVec(embDim, rng), // ex1, target0
  ];
  const exIdxs = [0, 0, 1];

  const numHeads = 1, headDim = 4;
  const config = makeConfig(numHeads, headDim, embDim);

  // --- Reference: per-target backpropWIntent ---
  const refGrads = makeMultiLevelGrads(numHeads, headDim, embDim);
  // Example 0: sum dIntentProj of its 2 targets, then backpropWIntent
  const sum0 = new Array(embDim).fill(0);
  for (let d = 0; d < embDim; d++) sum0[d] = dIntentProjs[0][d] + dIntentProjs[1][d];
  backpropWIntent(sum0, intentEmbs[0], refGrads, config);
  // Example 1: single target
  backpropWIntent(dIntentProjs[2], intentEmbs[1], refGrads, config);

  // --- Batched scatter-sum ---
  const batchGrads = makeMultiLevelGrads(numHeads, headDim, embDim);
  const perExDIntent = new Map<number, number[]>();
  for (let t = 0; t < 3; t++) {
    let acc = perExDIntent.get(exIdxs[t]);
    if (!acc) {
      acc = new Array(embDim).fill(0);
      perExDIntent.set(exIdxs[t], acc);
    }
    for (let d = 0; d < embDim; d++) acc[d] += dIntentProjs[t][d];
  }
  for (const [exIdx, totalDIntentProj] of perExDIntent) {
    backpropWIntent(totalDIntentProj, intentEmbs[exIdx], batchGrads, config);
  }

  const diff = matrixMaxDiff(refGrads.dW_intent, batchGrads.dW_intent);
  assert(diff < 1e-14, `dW_intent maxDiff=${diff}`);
});

// ============================================================================
// Gradient accumulation test
// ============================================================================

Deno.test("gradient accumulation: N batches accumulated = N × single batch averaged", () => {
  const rng = seededRandom(SEED + 30);
  const numHeads = 2, headDim = 4, embDim = 8;
  const N = 4; // accumulation steps

  // Generate N fake gradient batches
  const batchGrads: KHeadGradientAccumulators[] = [];
  for (let i = 0; i < N; i++) {
    batchGrads.push({
      dW_q: Array.from({ length: numHeads }, () => randomMatrix(headDim, embDim, rng)),
      dW_k: Array.from({ length: numHeads }, () => randomMatrix(headDim, embDim, rng)),
    });
  }

  // --- Method A: Accumulate then normalize ---
  const accumGrads = makeKHeadGrads(numHeads, headDim, embDim);
  for (const bg of batchGrads) {
    for (let h = 0; h < numHeads; h++) {
      for (let r = 0; r < headDim; r++) {
        for (let c = 0; c < embDim; c++) {
          accumGrads.dW_q[h][r][c] += bg.dW_q[h][r][c];
          accumGrads.dW_k[h][r][c] += bg.dW_k[h][r][c];
        }
      }
    }
  }
  // Normalize by N
  const invN = 1 / N;
  for (let h = 0; h < numHeads; h++) {
    for (const row of accumGrads.dW_q[h]) for (let d = 0; d < row.length; d++) row[d] *= invN;
    for (const row of accumGrads.dW_k[h]) for (let d = 0; d < row.length; d++) row[d] *= invN;
  }

  // --- Method B: Average directly ---
  const avgGrads = makeKHeadGrads(numHeads, headDim, embDim);
  for (const bg of batchGrads) {
    for (let h = 0; h < numHeads; h++) {
      for (let r = 0; r < headDim; r++) {
        for (let c = 0; c < embDim; c++) {
          avgGrads.dW_q[h][r][c] += bg.dW_q[h][r][c] / N;
          avgGrads.dW_k[h][r][c] += bg.dW_k[h][r][c] / N;
        }
      }
    }
  }

  for (let h = 0; h < numHeads; h++) {
    const diffWq = matrixMaxDiff(accumGrads.dW_q[h], avgGrads.dW_q[h]);
    const diffWk = matrixMaxDiff(accumGrads.dW_k[h], avgGrads.dW_k[h]);
    assert(diffWq < 1e-12, `accum dW_q[${h}] vs avg maxDiff=${diffWq}`);
    assert(diffWk < 1e-12, `accum dW_k[${h}] vs avg maxDiff=${diffWk}`);
  }
});
