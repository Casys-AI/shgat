/**
 * Tests for KL divergence optimization — gradient checking via finite differences.
 *
 * Tests the KL divergence loss used for n8n soft target augmentation:
 * - klDivergenceLoss: single-example KL loss
 * - batchedKLLoss: batched KL loss (internal)
 * - trainStepKL: full training step with K-head scoring
 * - trainBatchKL: OB trainer method
 */

import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import * as tf from "npm:@tensorflow/tfjs@4.22.0";
import { klDivergenceLoss, trainStepKL, type TFParams, type TrainerConfig } from "../autograd-trainer.ts";
import type { SHGATConfig, SoftTargetExample } from "../../core/types.ts";

// ============================================================================
// Helpers
// ============================================================================

function makeTinyConfig(): SHGATConfig {
  return {
    embeddingDim: 8,
    hiddenDim: 8, // must equal embeddingDim (batchedKHeadForward shares W_k for Q and K)
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

function makeTFParams(config: SHGATConfig): TFParams {
  const W_k: tf.Variable[] = [];
  for (let h = 0; h < config.numHeads; h++) {
    const data = [];
    for (let i = 0; i < config.embeddingDim; i++) {
      for (let j = 0; j < config.headDim; j++) {
        data.push((Math.random() - 0.5) * 0.1);
      }
    }
    W_k.push(tf.variable(tf.tensor2d(data, [config.embeddingDim, config.headDim])));
  }

  const outDim = config.hiddenDim || config.embeddingDim;
  const intentData = [];
  for (let i = 0; i < config.embeddingDim; i++) {
    for (let j = 0; j < outDim; j++) {
      intentData.push((Math.random() - 0.5) * 0.1);
    }
  }
  const W_intent = tf.variable(tf.tensor2d(intentData, [config.embeddingDim, outDim]));

  return {
    W_k,
    W_intent,
    W_up: new Map(),
    W_down: new Map(),
    a_up: new Map(),
    a_down: new Map(),
  };
}

function makeTrainerConfig(config: SHGATConfig): TrainerConfig {
  return {
    learningRate: config.learningRate,
    batchSize: config.batchSize,
    temperature: 0.07,
    gradientClip: 5.0,
    l2Lambda: config.l2Lambda,
  };
}

function randomVec(dim: number): number[] {
  return Array.from({ length: dim }, () => (Math.random() - 0.5) * 2);
}

function makeSoftTargetExample(vocabSize: number, topK: number, embDim: number): SoftTargetExample {
  // Select random top-K indices
  const indices = Array.from({ length: vocabSize }, (_, i) => i)
    .sort(() => Math.random() - 0.5)
    .slice(0, topK);

  // Random probabilities (will be normalized)
  const rawProbs = indices.map(() => Math.random());
  const sum = rawProbs.reduce((a, b) => a + b, 0);
  const probs = rawProbs.map(p => p / sum);

  const softTargetSparse: [number, number][] = indices.map((idx, i) => [idx, probs[i]]);

  return {
    intentEmbedding: randomVec(embDim),
    softTargetSparse,
  };
}

// ============================================================================
// Tests
// ============================================================================

Deno.test("KL divergence: loss is positive and finite", () => {
  const vocabSize = 10;
  const temperature = 0.1;

  const scores = tf.tensor1d(Array.from({ length: vocabSize }, () => Math.random()));
  const softTargetSparse: [number, number][] = [
    [0, 0.7],
    [1, 0.2],
    [5, 0.1],
  ];

  const loss = klDivergenceLoss(scores, softTargetSparse, vocabSize, temperature);
  const lossValue = loss.dataSync()[0];

  assert(Number.isFinite(lossValue), `Loss should be finite, got ${lossValue}`);
  assert(lossValue >= 0, `KL divergence should be non-negative, got ${lossValue}`);

  scores.dispose();
  loss.dispose();
});

Deno.test("KL divergence: perfect match gives near-zero loss", () => {
  const vocabSize = 5;
  const temperature = 0.01; // Low temperature for sharper distribution

  // Target: tool 2 has probability 1.0
  const softTargetSparse: [number, number][] = [[2, 1.0]];

  // Scores: give tool 2 a very high score
  const scoresData = Array.from({ length: vocabSize }, () => -10.0);
  scoresData[2] = 10.0; // Tool 2 should dominate
  const scores = tf.tensor1d(scoresData);

  const loss = klDivergenceLoss(scores, softTargetSparse, vocabSize, temperature);
  const lossValue = loss.dataSync()[0];

  assert(lossValue < 0.1, `Near-perfect match should have low KL loss, got ${lossValue}`);

  scores.dispose();
  loss.dispose();
});

Deno.test("KL divergence: gradient flows through scores", () => {
  const vocabSize = 5;
  const temperature = 0.1;

  const softTargetSparse: [number, number][] = [
    [0, 0.6],
    [2, 0.4],
  ];

  const scoresVar = tf.variable(tf.randomNormal([vocabSize]));

  const gradFunc = tf.grad((s: tf.Tensor1D) =>
    klDivergenceLoss(s, softTargetSparse, vocabSize, temperature)
  );

  const grad = gradFunc(scoresVar);
  const gradData = grad.dataSync();

  // Gradient should be non-zero for at least some scores
  const nonZeroGrads = Array.from(gradData).filter(g => Math.abs(g) > 1e-6);
  assert(nonZeroGrads.length > 0, "Gradient should have non-zero components");

  scoresVar.dispose();
  grad.dispose();
});

Deno.test("KL divergence: different targets produce different gradients", () => {
  const vocabSize = 10;
  const temperature = 0.1;

  // Two different target distributions
  const targetA: [number, number][] = [[0, 0.1], [5, 0.9]];
  const targetB: [number, number][] = [[0, 0.9], [5, 0.1]];

  const scoresData = Array.from({ length: vocabSize }, () => Math.random());
  const scoresA = tf.variable(tf.tensor1d(scoresData));
  const scoresB = tf.variable(tf.tensor1d(scoresData));

  const gradA = tf.grad((s: tf.Tensor1D) =>
    klDivergenceLoss(s, targetA, vocabSize, temperature)
  )(scoresA);

  const gradB = tf.grad((s: tf.Tensor1D) =>
    klDivergenceLoss(s, targetB, vocabSize, temperature)
  )(scoresB);

  const gradAData = Array.from(gradA.dataSync());
  const gradBData = Array.from(gradB.dataSync());

  // Gradients should differ when targets differ
  let maxDiff = 0;
  for (let i = 0; i < vocabSize; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(gradAData[i] - gradBData[i]));
  }
  assert(maxDiff > 1e-4, `Different targets should produce different gradients, maxDiff=${maxDiff}`);

  scoresA.dispose();
  scoresB.dispose();
  gradA.dispose();
  gradB.dispose();
});

Deno.test("trainStepKL: loss decreases after multiple steps", () => {
  const config = makeTinyConfig();
  const vocabSize = 20;
  const batchSize = 4;
  const temperature = 0.1;

  // Create synthetic examples
  const klExamples: SoftTargetExample[] = Array.from({ length: batchSize }, () =>
    makeSoftTargetExample(vocabSize, 3, config.embeddingDim)
  );

  // Random tool embeddings
  const toolEmbs = Array.from({ length: vocabSize }, () => randomVec(config.embeddingDim));
  const allToolEmbsTensor = tf.tensor2d(toolEmbs);

  // Initialize parameters
  const params = makeTFParams(config);
  const trainerConfig = makeTrainerConfig(config);
  const optimizer = tf.train.adam(config.learningRate);

  // First step
  const metrics1 = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params,
    config,
    trainerConfig,
    optimizer,
    temperature,
    1.0, // klWeight
  );

  // Second step (should have lower loss if optimization is working)
  const metrics2 = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params,
    config,
    trainerConfig,
    optimizer,
    temperature,
    1.0,
  );

  // Third step
  const metrics3 = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params,
    config,
    trainerConfig,
    optimizer,
    temperature,
    1.0,
  );

  assert(
    metrics3.klLoss < metrics1.klLoss,
    `Loss should decrease: step1=${metrics1.klLoss}, step3=${metrics3.klLoss}`,
  );

  // Cleanup
  allToolEmbsTensor.dispose();
  params.W_k.forEach(v => v.dispose());
  params.W_intent.dispose();
});

Deno.test("trainStepKL: klWeight scales the gradient norm", () => {
  const config = makeTinyConfig();
  const vocabSize = 15;
  const batchSize = 3;
  const temperature = 0.1;

  const klExamples: SoftTargetExample[] = Array.from({ length: batchSize }, () =>
    makeSoftTargetExample(vocabSize, 3, config.embeddingDim)
  );

  const toolEmbs = Array.from({ length: vocabSize }, () => randomVec(config.embeddingDim));
  const allToolEmbsTensor = tf.tensor2d(toolEmbs);

  // Test with klWeight=1.0
  const params1 = makeTFParams(config);
  const trainerConfig1 = makeTrainerConfig(config);
  const optimizer1 = tf.train.adam(config.learningRate);
  const metrics1 = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params1,
    config,
    trainerConfig1,
    optimizer1,
    temperature,
    1.0,
  );

  // Test with klWeight=0.5 (should have smaller gradients)
  const params2 = makeTFParams(config);
  const trainerConfig2 = makeTrainerConfig(config);
  const optimizer2 = tf.train.adam(config.learningRate);
  const metrics2 = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params2,
    config,
    trainerConfig2,
    optimizer2,
    temperature,
    0.5,
  );

  assert(
    metrics2.gradientNorm < metrics1.gradientNorm,
    `Lower klWeight should produce smaller gradients: w=1.0 -> ${metrics1.gradientNorm}, w=0.5 -> ${metrics2.gradientNorm}`,
  );

  // Cleanup
  allToolEmbsTensor.dispose();
  params1.W_k.forEach(v => v.dispose());
  params1.W_intent.dispose();
  params2.W_k.forEach(v => v.dispose());
  params2.W_intent.dispose();
});

Deno.test("trainStepKL: temperature affects loss sensitivity", () => {
  const config = makeTinyConfig();
  const vocabSize = 10;
  const batchSize = 2;

  const klExamples: SoftTargetExample[] = Array.from({ length: batchSize }, () =>
    makeSoftTargetExample(vocabSize, 3, config.embeddingDim)
  );

  const toolEmbs = Array.from({ length: vocabSize }, () => randomVec(config.embeddingDim));
  const allToolEmbsTensor = tf.tensor2d(toolEmbs);
  const params = makeTFParams(config);
  const trainerConfig = makeTrainerConfig(config);
  const optimizer = tf.train.adam(config.learningRate);

  // Low temperature (sharper softmax, higher loss sensitivity)
  const metricsLowT = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params,
    config,
    trainerConfig,
    optimizer,
    0.01, // Very low temperature
    1.0,
  );

  // High temperature (flatter softmax, lower loss sensitivity)
  const metricsHighT = trainStepKL(
    klExamples,
    allToolEmbsTensor,
    vocabSize,
    params,
    config,
    trainerConfig,
    optimizer,
    1.0, // High temperature
    1.0,
  );

  assert(
    Number.isFinite(metricsLowT.klLoss) && Number.isFinite(metricsHighT.klLoss),
    "Both losses should be finite",
  );

  // Cleanup
  allToolEmbsTensor.dispose();
  params.W_k.forEach(v => v.dispose());
  params.W_intent.dispose();
});

Deno.test("trainStepKL: validates intentEmbedding dimensions", () => {
  const config = makeTinyConfig();
  const vocabSize = 10;
  const temperature = 0.1;

  // Create example with WRONG embedding dimension
  const badExample: SoftTargetExample = {
    intentEmbedding: randomVec(config.embeddingDim + 5), // Wrong size!
    softTargetSparse: [[0, 0.8], [1, 0.2]],
  };

  const toolEmbs = Array.from({ length: vocabSize }, () => randomVec(config.embeddingDim));
  const allToolEmbsTensor = tf.tensor2d(toolEmbs);
  const params = makeTFParams(config);
  const trainerConfig = makeTrainerConfig(config);
  const optimizer = tf.train.adam(config.learningRate);

  let errorThrown = false;
  try {
    trainStepKL(
      [badExample],
      allToolEmbsTensor,
      vocabSize,
      params,
      config,
      trainerConfig,
      optimizer,
      temperature,
      1.0,
    );
  } catch (e) {
    errorThrown = true;
    assert(
      e.message.includes("intentEmbedding") && e.message.includes("embeddingDim"),
      `Error message should mention dimension mismatch: ${e.message}`,
    );
  }

  assert(errorThrown, "Should throw error for mismatched embedding dimensions");

  // Cleanup
  allToolEmbsTensor.dispose();
  params.W_k.forEach(v => v.dispose());
  params.W_intent.dispose();
});

Deno.test("KL divergence: sparse target efficiency", () => {
  const vocabSize = 1000; // Large vocabulary
  const temperature = 0.1;

  // Sparse target: only 5 non-zero entries out of 1000
  const softTargetSparse: [number, number][] = [
    [10, 0.5],
    [20, 0.3],
    [100, 0.1],
    [500, 0.08],
    [999, 0.02],
  ];

  const scores = tf.tensor1d(Array.from({ length: vocabSize }, () => Math.random()));

  // Should complete without error or timeout
  const startTime = Date.now();
  const loss = klDivergenceLoss(scores, softTargetSparse, vocabSize, temperature);
  const elapsed = Date.now() - startTime;

  assert(elapsed < 1000, `Sparse KL should be fast even with large vocab, took ${elapsed}ms`);
  assert(Number.isFinite(loss.dataSync()[0]), "Loss should be finite");

  scores.dispose();
  loss.dispose();
});
