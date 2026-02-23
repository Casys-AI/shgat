/**
 * Integration tests for the OB trainer with InfoNCE + K-head scoring pipeline.
 *
 * Uses a tiny synthetic graph to verify:
 * 1. InfoNCE loss computes correctly through the K-head scoring pipeline
 * 2. Gradients flow back through K-head (W_q, W_k, W_intent)
 * 3. dNodeEmbedding is non-zero for all candidates in a contrastive batch
 */

import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { infoNCELossAndGradient } from "../infonce-loss.ts";
import {
  computeMultiHeadKHeadScoresWithCache,
  backpropMultiHeadKHeadLogit,
  initMultiLevelKHeadGradients,
} from "../multi-level-trainer-khead.ts";
import type { HeadParams } from "../../initialization/parameters.ts";
import type { SHGATConfig } from "../../core/types.ts";
import * as math from "../../utils/math.ts";

// ============================================================================
// Helpers: tiny synthetic setup
// ============================================================================

function makeTinyConfig(): SHGATConfig {
  return {
    embeddingDim: 8,
    hiddenDim: 4,
    headDim: 2,
    numHeads: 2,
    numLayers: 1,
    mlpHiddenDim: 8,
    learningRate: 0.01,
    batchSize: 4,
    maxContextLength: 10,
    maxBufferSize: 100,
    minTracesForTraining: 5,
    dropout: 0,
    l2Lambda: 0.0001,
    leakyReluSlope: 0.2,
    depthDecay: 0.8,
  };
}

function makeHeadParams(config: SHGATConfig): HeadParams[] {
  const heads: HeadParams[] = [];
  for (let h = 0; h < config.numHeads; h++) {
    const W_q: number[][] = [];
    const W_k: number[][] = [];
    const W_v: number[][] = [];
    for (let i = 0; i < config.hiddenDim; i++) {
      W_q.push(Array.from({ length: config.embeddingDim }, () => (Math.random() - 0.5) * 0.1));
      W_k.push(Array.from({ length: config.embeddingDim }, () => (Math.random() - 0.5) * 0.1));
      W_v.push(Array.from({ length: config.embeddingDim }, () => (Math.random() - 0.5) * 0.1));
    }
    const a = Array.from({ length: 2 * config.headDim }, () => (Math.random() - 0.5) * 0.1);
    heads.push({ W_q, W_k, W_v, a });
  }
  return heads;
}

function makeWIntent(config: SHGATConfig): number[][] {
  return Array.from({ length: config.hiddenDim }, () =>
    Array.from({ length: config.embeddingDim }, () => (Math.random() - 0.5) * 0.1)
  );
}

function makeLevelParams(config: SHGATConfig): Map<number, import("../../core/types.ts").LevelParams> {
  const params = new Map<number, import("../../core/types.ts").LevelParams>();
  const { numHeads } = config;
  const headDim = config.headDim;

  const makeW = () =>
    Array.from({ length: numHeads }, () =>
      Array.from({ length: headDim }, () =>
        Array.from({ length: config.embeddingDim }, () => (Math.random() - 0.5) * 0.1)
      )
    );
  const makeA = () =>
    Array.from({ length: numHeads }, () =>
      Array.from({ length: 2 * headDim }, () => (Math.random() - 0.5) * 0.1)
    );

  params.set(0, {
    W_child: makeW(),
    W_parent: makeW(),
    a_upward: makeA(),
    a_downward: makeA(),
  });

  return params;
}

function randomVec(dim: number): number[] {
  return Array.from({ length: dim }, () => (Math.random() - 0.5) * 2);
}

// ============================================================================
// Tests
// ============================================================================

Deno.test("InfoNCE + K-head: gradient flows through W_q, W_k, W_intent", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const W_intent = makeWIntent(config);
  const levelParams = makeLevelParams(config);

  const grads = initMultiLevelKHeadGradients(levelParams, headParams, config);

  // Simulate scoring: intent vs 1 positive + 3 negatives
  const intentEmb = randomVec(config.embeddingDim);
  const intentProjected = math.matVec(W_intent, intentEmb);

  const nodeEmbeddings = [
    randomVec(config.embeddingDim), // positive
    randomVec(config.embeddingDim), // neg 1
    randomVec(config.embeddingDim), // neg 2
    randomVec(config.embeddingDim), // neg 3
  ];

  // Forward: compute logits for each candidate
  const logits: number[] = [];
  const allHeadCaches: Array<{ Q: number[]; K: number[]; dotQK: number }>[] = [];

  for (const nodeEmb of nodeEmbeddings) {
    const { logits: headLogits, caches } = computeMultiHeadKHeadScoresWithCache(
      intentProjected,
      nodeEmb,
      headParams,
      config,
    );
    // Average logit across heads
    const avgLogit = headLogits.reduce((a, b) => a + b, 0) / config.numHeads;
    logits.push(avgLogit);
    allHeadCaches.push(caches);
  }

  // InfoNCE loss + gradient (positive at index 0)
  const tau = 0.07;
  const { loss, gradient: dLogits } = infoNCELossAndGradient(logits, 0, tau);

  assert(loss > 0, `Loss should be positive, got ${loss}`);

  // Backward through K-head for each candidate
  for (let j = 0; j < nodeEmbeddings.length; j++) {
    backpropMultiHeadKHeadLogit(
      dLogits[j],
      allHeadCaches[j],
      intentProjected,
      nodeEmbeddings[j],
      headParams,
      grads.khead,
      config,
    );
  }

  // Check K-head gradients are non-zero
  let wqGradNorm = 0;
  for (const h of grads.khead.dW_q) {
    for (const row of h) {
      for (const val of row) wqGradNorm += val * val;
    }
  }
  assert(wqGradNorm > 0, `W_q gradient norm should be > 0, got ${wqGradNorm}`);

  let wkGradNorm = 0;
  for (const h of grads.khead.dW_k) {
    for (const row of h) {
      for (const val of row) wkGradNorm += val * val;
    }
  }
  assert(wkGradNorm > 0, `W_k gradient norm should be > 0, got ${wkGradNorm}`);
});

Deno.test("InfoNCE K-head: logit gradient matches finite differences", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);

  const intentProjected = randomVec(config.hiddenDim);
  const nodeEmb = randomVec(config.embeddingDim);

  // Forward
  const { caches } = computeMultiHeadKHeadScoresWithCache(
    intentProjected,
    nodeEmb,
    headParams,
    config,
  );

  // Analytical gradient of avgLogit w.r.t. intentProjected[d]
  const dummyGrads = {
    dW_q: headParams.map(h => h.W_q.map(r => r.map(() => 0))),
    dW_k: headParams.map(h => h.W_k.map(r => r.map(() => 0))),
  };
  const { dIntentProjected } = backpropMultiHeadKHeadLogit(
    1.0, // dLoss/dAvgLogit = 1
    caches,
    intentProjected,
    nodeEmb,
    headParams,
    dummyGrads,
    config,
  );

  // Numerical gradient via finite differences
  const eps = 1e-5;
  for (let d = 0; d < config.hiddenDim; d++) {
    const ipPlus = [...intentProjected];
    ipPlus[d] += eps;
    const logitsPlus = computeMultiHeadKHeadScoresWithCache(ipPlus, nodeEmb, headParams, config).logits;
    const avgPlus = logitsPlus.reduce((a, b) => a + b, 0) / config.numHeads;

    const ipMinus = [...intentProjected];
    ipMinus[d] -= eps;
    const logitsMinus = computeMultiHeadKHeadScoresWithCache(ipMinus, nodeEmb, headParams, config).logits;
    const avgMinus = logitsMinus.reduce((a, b) => a + b, 0) / config.numHeads;

    const numerical = (avgPlus - avgMinus) / (2 * eps);
    assertAlmostEquals(
      dIntentProjected[d],
      numerical,
      1e-3,
      `Gradient mismatch at dim ${d}: analytical=${dIntentProjected[d]}, numerical=${numerical}`,
    );
  }
});

Deno.test("K-head dNodeEmbedding is non-zero for all candidates in contrastive batch", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const W_intent = makeWIntent(config);

  const intentProjected = math.matVec(W_intent, randomVec(config.embeddingDim));

  // 1 positive + 3 negatives
  const candidates = Array.from({ length: 4 }, () => randomVec(config.embeddingDim));

  const logits: number[] = [];
  const allCaches: Array<{ Q: number[]; K: number[]; dotQK: number }>[] = [];
  for (const cand of candidates) {
    const { logits: hl, caches } = computeMultiHeadKHeadScoresWithCache(intentProjected, cand, headParams, config);
    logits.push(hl.reduce((a, b) => a + b, 0) / config.numHeads);
    allCaches.push(caches);
  }

  const { gradient: dLogits } = infoNCELossAndGradient(logits, 0, 0.07);

  // All candidates should have non-zero dNodeEmbedding
  for (let j = 0; j < candidates.length; j++) {
    const dummyGrads = {
      dW_q: headParams.map(h => h.W_q.map(r => r.map(() => 0))),
      dW_k: headParams.map(h => h.W_k.map(r => r.map(() => 0))),
    };
    const { dNodeEmbedding } = backpropMultiHeadKHeadLogit(
      dLogits[j],
      allCaches[j],
      intentProjected,
      candidates[j],
      headParams,
      dummyGrads,
      config,
    );
    const norm = dNodeEmbedding.reduce((s, v) => s + v * v, 0);
    assert(norm > 0, `dNodeEmbedding[${j}] should be non-zero, got norm=${norm}`);
  }
});
