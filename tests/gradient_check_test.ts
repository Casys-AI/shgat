/**
 * SHGAT-TF Gradient Correctness Tests
 *
 * Verifies analytical gradients against numerical (finite-difference) gradients.
 * These tests run in seconds and catch gradient bugs that would take 35+ minutes
 * to discover through benchmarking.
 *
 * Run: deno test -A --no-check lib/shgat-tf/tests/gradient_check_test.ts
 *
 * @module shgat-tf/tests/gradient_check_test
 */

import { assertGreater, assertLess } from "@std/assert";
import { tf } from "../src/tf/backend.ts";
import {
  AutogradTrainer,
  forwardScoring,
  kHeadScoring,
  infoNCELoss,
  initTFParams,
  buildGraphStructure,
  disposeGraphStructure,
  type TFParams,
  type CapabilityInfo,
  type TrainingExample,
} from "../src/training/index.ts";
import { type SHGATConfig, DEFAULT_SHGAT_CONFIG } from "../src/core/types.ts";

// ============================================================================
// Small config for fast tests
// ============================================================================

const SMALL_CONFIG: SHGATConfig = {
  ...DEFAULT_SHGAT_CONFIG,
  embeddingDim: 32,
  numHeads: 2,
  headDim: 16,
  hiddenDim: 32,
  preserveDim: true,
};

const EPSILON = 5e-4;
// Float32 + small dims = expect ~10-15% relative error. >25% is a real bug.
const GRAD_TOL = 0.25;

// ============================================================================
// Helpers
// ============================================================================

function makeEmb(seed: number, dim: number): number[] {
  return Array.from({ length: dim }, (_, j) => Math.sin(seed * 0.7 + j * 0.3) * 0.5);
}

function relError(a: number, b: number): number {
  const denom = Math.max(Math.abs(a), Math.abs(b), 1e-8);
  return Math.abs(a - b) / denom;
}

function disposeParams(params: TFParams): void {
  for (const W of params.W_k) W.dispose();
  if (params.W_q) for (const W of params.W_q) W.dispose();
  params.W_intent.dispose();
  params.residualWeights?.dispose();
  for (const [, ws] of params.W_up) ws.forEach((w) => w.dispose());
  for (const [, ws] of params.W_down) ws.forEach((w) => w.dispose());
  for (const [, ws] of params.a_up) ws.forEach((w) => w.dispose());
  for (const [, ws] of params.a_down) ws.forEach((w) => w.dispose());
  if (params.projectionHead) {
    params.projectionHead.W1.dispose();
    params.projectionHead.b1.dispose();
    params.projectionHead.W2.dispose();
    params.projectionHead.b2.dispose();
  }
}

function createSmallGraph() {
  const dim = SMALL_CONFIG.embeddingDim;
  const toolIds = ["t0", "t1", "t2", "t3", "t4"];
  const capIds = ["c0", "c1", "c2"];
  const embeddings = new Map<string, number[]>();
  toolIds.forEach((id, i) => embeddings.set(id, makeEmb(i, dim)));
  capIds.forEach((id, i) => embeddings.set(id, makeEmb(10 + i, dim)));
  return { toolIds, capIds, embeddings };
}

// ============================================================================
// Test 1: K-head scoring dScore/dW_k
// ============================================================================

Deno.test("Gradient: K-head dScore/dW_k matches finite differences", () => {
  const params = initTFParams(SMALL_CONFIG, 1);
  try {
    const dim = SMALL_CONFIG.embeddingDim;
    const intentEmb = tf.tensor1d(makeEmb(50, dim));
    const nodeEmbs = tf.tensor2d([makeEmb(0, dim), makeEmb(1, dim), makeEmb(2, dim)]);

    const { grads, value } = tf.variableGrads(() => {
      const scores = kHeadScoring(intentEmb, nodeEmbs, params, SMALL_CONFIG);
      return tf.sum(scores) as tf.Scalar;
    });

    const W_k_0_grad = grads[params.W_k[0].name]?.arraySync() as number[][];
    const W_k_0_data = params.W_k[0].arraySync() as number[][];

    let maxRelErr = 0;
    let numChecked = 0;
    const spots = [[0, 0], [2, 5], [dim - 1, SMALL_CONFIG.headDim - 1]];

    for (const [i, j] of spots) {
      if (i >= W_k_0_data.length || j >= W_k_0_data[0].length) continue;

      const perturb = (delta: number): number => {
        const W = W_k_0_data.map((r) => [...r]);
        W[i][j] += delta;
        const t = tf.tensor2d(W);
        params.W_k[0].assign(t);
        t.dispose();
        return tf.tidy(() => {
          const s = kHeadScoring(intentEmb, nodeEmbs, params, SMALL_CONFIG);
          return tf.sum(s).arraySync() as number;
        });
      };

      const numGrad = (perturb(+EPSILON) - perturb(-EPSILON)) / (2 * EPSILON);
      // Restore
      const restore = tf.tensor2d(W_k_0_data);
      params.W_k[0].assign(restore);
      restore.dispose();

      const err = relError(W_k_0_grad[i][j], numGrad);
      if (err > maxRelErr) maxRelErr = err;
      numChecked++;
    }

    console.log(`  K-head dW_k: ${numChecked} checked, maxRelErr=${(maxRelErr * 100).toFixed(1)}%`);
    assertLess(maxRelErr, GRAD_TOL, `K-head gradient error ${(maxRelErr * 100).toFixed(1)}% > ${GRAD_TOL * 100}%`);

    Object.values(grads).forEach((g) => g.dispose());
    (value as tf.Tensor).dispose();
    intentEmb.dispose();
    nodeEmbs.dispose();
  } finally {
    disposeParams(params);
  }
});

// ============================================================================
// Test 2: InfoNCE loss gradient
// ============================================================================

Deno.test("Gradient: InfoNCE dLoss/dScores matches finite differences", () => {
  const temperature = 0.1;
  const posVar = tf.variable(tf.scalar(0.8), true, "test_pos_v2");
  const negVar = tf.variable(tf.tensor1d([0.3, 0.5, 0.1]), true, "test_neg_v2");

  try {
    const { grads, value } = tf.variableGrads(() =>
      infoNCELoss(posVar as unknown as tf.Scalar, negVar as unknown as tf.Tensor1D, temperature)
    );

    const dPos = grads["test_pos_v2"]?.arraySync() as number;
    const dNeg = grads["test_neg_v2"]?.arraySync() as number[];

    // Numerical: positive
    const posData = posVar.arraySync() as number;
    const lossAt = (pDelta: number): number => {
      const t = tf.scalar(posData + pDelta);
      posVar.assign(t);
      t.dispose();
      return tf.tidy(() =>
        (infoNCELoss(posVar as unknown as tf.Scalar, negVar as unknown as tf.Tensor1D, temperature).arraySync() as number)
      );
    };
    const numDPos = (lossAt(+EPSILON) - lossAt(-EPSILON)) / (2 * EPSILON);
    posVar.assign(tf.scalar(posData));

    const posErr = relError(dPos, numDPos);
    console.log(`  InfoNCE dPos: analytical=${dPos.toFixed(6)} numerical=${numDPos.toFixed(6)} err=${(posErr * 100).toFixed(1)}%`);
    assertLess(posErr, GRAD_TOL);

    // Numerical: negatives
    const negData = negVar.arraySync() as number[];
    let maxNegErr = 0;
    for (let i = 0; i < negData.length; i++) {
      const lossAtNeg = (delta: number): number => {
        const arr = [...negData];
        arr[i] += delta;
        const t = tf.tensor1d(arr);
        negVar.assign(t);
        t.dispose();
        return tf.tidy(() =>
          (infoNCELoss(posVar as unknown as tf.Scalar, negVar as unknown as tf.Tensor1D, temperature).arraySync() as number)
        );
      };
      const numG = (lossAtNeg(+EPSILON) - lossAtNeg(-EPSILON)) / (2 * EPSILON);
      const t = tf.tensor1d(negData);
      negVar.assign(t);
      t.dispose();
      const err = relError(dNeg[i], numG);
      if (err > maxNegErr) maxNegErr = err;
    }

    console.log(`  InfoNCE dNeg: maxRelErr=${(maxNegErr * 100).toFixed(1)}%`);
    assertLess(maxNegErr, GRAD_TOL);

    Object.values(grads).forEach((g) => g.dispose());
    (value as tf.Tensor).dispose();
  } finally {
    posVar.dispose();
    negVar.dispose();
  }
});

// ============================================================================
// Test 3: Training non-regression
// ============================================================================

Deno.test("Training: must not degrade scoring on training data", async () => {
  const { toolIds, capIds, embeddings } = createSmallGraph();

  const trainer = new AutogradTrainer(SMALL_CONFIG, {
    learningRate: 0.01,
    batchSize: 3,
    temperature: 0.1,
    gradientClip: 1.0,
    l2Lambda: 0.0001,
  }, 1);

  try {
    trainer.setNodeEmbeddings(embeddings);
    const capInfos: CapabilityInfo[] = [
      { id: "c0", toolsUsed: ["t0", "t1"] },
      { id: "c1", toolsUsed: ["t2", "t3"] },
      { id: "c2", toolsUsed: ["t1", "t4"] },
    ];
    const graph = buildGraphStructure(capInfos, toolIds);
    trainer.setGraph(graph);

    const examples: TrainingExample[] = capIds.map((capId, i) => ({
      intentEmbedding: makeEmb(100 + i, SMALL_CONFIG.embeddingDim),
      contextTools: [],
      candidateId: capId,
      outcome: 1,
      negativeCapIds: capIds.filter((c) => c !== capId),
    }));

    // Baseline
    let baselineCorrect = 0;
    for (const ex of examples) {
      const scores = trainer.score(ex.intentEmbedding, capIds);
      const maxIdx = scores.indexOf(Math.max(...scores));
      if (capIds[maxIdx] === ex.candidateId) baselineCorrect++;
    }
    console.log(`  Baseline: ${baselineCorrect}/${examples.length}`);

    // Train 30 epochs
    for (let epoch = 0; epoch < 30; epoch++) {
      await trainer.trainBatch(examples);
    }

    // Post-training
    let trainedCorrect = 0;
    for (const ex of examples) {
      const scores = trainer.score(ex.intentEmbedding, capIds);
      const maxIdx = scores.indexOf(Math.max(...scores));
      if (capIds[maxIdx] === ex.candidateId) trainedCorrect++;
    }
    console.log(`  Trained:  ${trainedCorrect}/${examples.length}`);
    console.log(`  Delta: ${trainedCorrect - baselineCorrect}`);

    // With only 3 examples and 30 epochs of training, we should at least not regress
    assertGreater(
      trainedCorrect,
      baselineCorrect - 1,
      `Training degraded: ${baselineCorrect} -> ${trainedCorrect}`,
    );

    disposeGraphStructure(graph);
  } finally {
    trainer.dispose();
  }
});

// ============================================================================
// Test 6: Temperature effect on loss
// ============================================================================

Deno.test("Temperature: lower temp produces sharper loss", () => {
  const posScore = tf.scalar(0.8);
  const negScores = tf.tensor1d([0.3, 0.5, 0.1]);

  const lossHigh = infoNCELoss(posScore, negScores, 0.10).arraySync() as number;
  const lossLow = infoNCELoss(posScore, negScores, 0.06).arraySync() as number;

  console.log(`  Loss tau=0.10: ${lossHigh.toFixed(6)}, tau=0.06: ${lossLow.toFixed(6)}`);
  assertGreater(Math.abs(lossHigh - lossLow), 0.001, "Temperature has no effect");

  posScore.dispose();
  negScores.dispose();
});

// ============================================================================
// Test 7: Gradient scale — batch-sum = N * single
// ============================================================================

Deno.test("Gradient scale: batch-sum grad = N * single-example grad", () => {
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const dim = SMALL_CONFIG.embeddingDim;
    const temperature = 0.1;
    const intent = tf.tensor1d(makeEmb(50, dim));
    const nodes = tf.tensor2d([makeEmb(0, dim), makeEmb(1, dim), makeEmb(2, dim)]);

    // Single
    const { grads: g1, value: v1 } = tf.variableGrads(() => {
      const scores = forwardScoring(intent, nodes, params, SMALL_CONFIG);
      const pos = scores.slice([0], [1]).squeeze() as tf.Scalar;
      const neg = scores.slice([1], [2]) as tf.Tensor1D;
      return infoNCELoss(pos, neg, temperature);
    });
    const singleGrad = g1[params.W_k[0].name]?.arraySync() as number[][];

    // 3x same example
    const { grads: g3, value: v3 } = tf.variableGrads(() => {
      let loss = tf.scalar(0);
      for (let i = 0; i < 3; i++) {
        const scores = forwardScoring(intent, nodes, params, SMALL_CONFIG);
        const pos = scores.slice([0], [1]).squeeze() as tf.Scalar;
        const neg = scores.slice([1], [2]) as tf.Tensor1D;
        loss = loss.add(infoNCELoss(pos, neg, temperature));
      }
      return loss as tf.Scalar;
    });
    const tripleGrad = g3[params.W_k[0].name]?.arraySync() as number[][];

    let maxRatioErr = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        if (Math.abs(singleGrad[i][j]) < 1e-10) continue;
        const ratio = tripleGrad[i][j] / singleGrad[i][j];
        const err = Math.abs(ratio - 3.0) / 3.0;
        if (err > maxRatioErr) maxRatioErr = err;
      }
    }

    console.log(`  Scale ratio error: ${(maxRatioErr * 100).toFixed(2)}%`);
    assertLess(maxRatioErr, 0.01, "Batch gradient ≠ 3x single gradient");

    Object.values(g1).forEach((g) => g.dispose());
    Object.values(g3).forEach((g) => g.dispose());
    (v1 as tf.Tensor).dispose();
    (v3 as tf.Tensor).dispose();
    intent.dispose();
    nodes.dispose();
  } finally {
    disposeParams(params);
  }
});
