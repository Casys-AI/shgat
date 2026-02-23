/**
 * Tests for batch contrastive loss — gradient checking via finite differences.
 */

import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import {
  batchContrastiveForward,
  batchContrastiveBackward,
} from "../batch-contrastive-loss.ts";
import type { HeadParams } from "../../initialization/parameters.ts";
import type { SHGATConfig } from "../../core/types.ts";

// ============================================================================
// Helpers
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

function randomVec(dim: number): number[] {
  return Array.from({ length: dim }, () => (Math.random() - 0.5) * 2);
}

function makeZeroGrads(headParams: HeadParams[]): { dW_q: number[][][]; dW_k: number[][][] } {
  return {
    dW_q: headParams.map(h => h.W_q.map(r => r.map(() => 0))),
    dW_k: headParams.map(h => h.W_k.map(r => r.map(() => 0))),
  };
}

// ============================================================================
// Tests
// ============================================================================

const B = 4;
const tau = 0.07;

Deno.test("batch contrastive: loss is positive and finite", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  const { loss } = batchContrastiveForward(intents, nodes, headParams, config, tau);

  assert(Number.isFinite(loss), `Loss should be finite, got ${loss}`);
  assert(loss > 0, `Loss should be positive, got ${loss}`);
});

Deno.test("batch contrastive: gradient dW_q matches finite differences", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  // Analytical gradient
  const grads = makeZeroGrads(headParams);
  const { cache } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(cache, headParams, grads, config);

  // Numerical gradient via finite differences for W_q[0][0][0]
  const eps = 1e-5;
  const h = 0;
  for (let r = 0; r < Math.min(2, config.hiddenDim); r++) {
    for (let c = 0; c < Math.min(2, config.embeddingDim); c++) {
      const orig = headParams[h].W_q[r][c];

      headParams[h].W_q[r][c] = orig + eps;
      const { loss: lossPlus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

      headParams[h].W_q[r][c] = orig - eps;
      const { loss: lossMinus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

      headParams[h].W_q[r][c] = orig;

      const numerical = (lossPlus - lossMinus) / (2 * eps);
      assertAlmostEquals(
        grads.dW_q[h][r][c],
        numerical,
        1e-3,
        `dW_q[${h}][${r}][${c}]: analytical=${grads.dW_q[h][r][c]}, numerical=${numerical}`,
      );
    }
  }
});

Deno.test("batch contrastive: gradient dW_k matches finite differences", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  const grads = makeZeroGrads(headParams);
  const { cache } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(cache, headParams, grads, config);

  const eps = 1e-5;
  const h = 0;
  for (let r = 0; r < Math.min(2, config.hiddenDim); r++) {
    for (let c = 0; c < Math.min(2, config.embeddingDim); c++) {
      const orig = headParams[h].W_k[r][c];

      headParams[h].W_k[r][c] = orig + eps;
      const { loss: lossPlus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

      headParams[h].W_k[r][c] = orig - eps;
      const { loss: lossMinus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

      headParams[h].W_k[r][c] = orig;

      const numerical = (lossPlus - lossMinus) / (2 * eps);
      assertAlmostEquals(
        grads.dW_k[h][r][c],
        numerical,
        1e-3,
        `dW_k[${h}][${r}][${c}]: analytical=${grads.dW_k[h][r][c]}, numerical=${numerical}`,
      );
    }
  }
});

Deno.test("batch contrastive: gradient dIntentsProjected matches finite differences", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  const grads = makeZeroGrads(headParams);
  const { cache } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  const { dIntentsProjected } = batchContrastiveBackward(cache, headParams, grads, config);

  const eps = 1e-5;
  const i = 0; // first example
  for (let d = 0; d < Math.min(3, config.embeddingDim); d++) {
    const orig = intents[i][d];

    intents[i][d] = orig + eps;
    const { loss: lossPlus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

    intents[i][d] = orig - eps;
    const { loss: lossMinus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

    intents[i][d] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    assertAlmostEquals(
      dIntentsProjected[i][d],
      numerical,
      1e-3,
      `dIntentsProjected[${i}][${d}]: analytical=${dIntentsProjected[i][d]}, numerical=${numerical}`,
    );
  }
});

Deno.test("batch contrastive: gradient dNodeEmbeddings matches finite differences", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  const grads = makeZeroGrads(headParams);
  const { cache } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  const { dNodeEmbeddings } = batchContrastiveBackward(cache, headParams, grads, config);

  const eps = 1e-5;
  const j = 1; // second example
  for (let d = 0; d < Math.min(3, config.embeddingDim); d++) {
    const orig = nodes[j][d];

    nodes[j][d] = orig + eps;
    const { loss: lossPlus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

    nodes[j][d] = orig - eps;
    const { loss: lossMinus } = batchContrastiveForward(intents, nodes, headParams, config, tau);

    nodes[j][d] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    assertAlmostEquals(
      dNodeEmbeddings[j][d],
      numerical,
      1e-3,
      `dNodeEmbeddings[${j}][${d}]: analytical=${dNodeEmbeddings[j][d]}, numerical=${numerical}`,
    );
  }
});
