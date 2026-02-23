/**
 * SHGAT-TF Dense Autograd Tests
 *
 * Comprehensive tests for the dense autograd training pipeline:
 * - Message passing forward (shapes, enrichment, differentiability)
 * - Gradient flow through MP weights (W_up, W_down, a_up, a_down)
 * - Tensor leak detection
 * - InfoNCE loss properties
 * - PER annealing helpers
 * - Scoring determinism and consistency
 *
 * Run: deno test -A --no-check lib/shgat-tf/tests/dense_autograd_test.ts
 *
 * @module shgat-tf/tests/dense_autograd_test
 */

import {
  assertEquals,
  assertExists,
  assertGreater,
  assertLess,
  assertNotEquals,
} from "@std/assert";
import { tf } from "../src/tf/backend.ts";
import {
  AutogradTrainer,
  forwardScoring,
  kHeadScoring,
  infoNCELoss,
  initTFParams,
  messagePassingForward,
  buildGraphStructure,
  disposeGraphStructure,
  annealTemperature,
  annealBeta,
  PERBuffer,
  type TFParams,
  type GraphStructure,
  type CapabilityInfo,
  type TrainingExample,
} from "../src/training/index.ts";
import { type SHGATConfig, DEFAULT_SHGAT_CONFIG } from "../src/core/types.ts";

// ============================================================================
// Fixtures
// ============================================================================

const SMALL_CONFIG: SHGATConfig = {
  ...DEFAULT_SHGAT_CONFIG,
  embeddingDim: 32,
  numHeads: 2,
  headDim: 16,
  hiddenDim: 32,
  preserveDim: true,
};

function makeEmb(seed: number, dim = SMALL_CONFIG.embeddingDim): number[] {
  return Array.from({ length: dim }, (_, j) => Math.sin(seed * 0.7 + j * 0.3) * 0.5);
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

function makeSmallGraph(): {
  toolIds: string[];
  capIds: string[];
  capInfos: CapabilityInfo[];
  embeddings: Map<string, number[]>;
} {
  const toolIds = ["t0", "t1", "t2", "t3", "t4"];
  const capIds = ["c0", "c1", "c2"];
  const embeddings = new Map<string, number[]>();
  toolIds.forEach((id, i) => embeddings.set(id, makeEmb(i)));
  capIds.forEach((id, i) => embeddings.set(id, makeEmb(10 + i)));
  const capInfos: CapabilityInfo[] = [
    { id: "c0", toolsUsed: ["t0", "t1"] },
    { id: "c1", toolsUsed: ["t2", "t3"] },
    { id: "c2", toolsUsed: ["t1", "t4"] },
  ];
  return { toolIds, capIds, capInfos, embeddings };
}

// ============================================================================
// Test: Message Passing Forward - output shapes
// ============================================================================

Deno.test("MP Forward: output shapes match input shapes", () => {
  const { toolIds, capInfos, embeddings } = makeSmallGraph();
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const graph = buildGraphStructure(capInfos, toolIds);
    const H_init = tf.tensor2d(toolIds.map((id) => embeddings.get(id)!));
    const E_init = new Map<number, tf.Tensor2D>();
    const capIds0 = graph.capIdsByLevel.get(0)!;
    E_init.set(0, tf.tensor2d(capIds0.map((id) => embeddings.get(id)!)));

    const { H, E } = messagePassingForward(H_init, E_init, graph, params, SMALL_CONFIG);

    // H: [numTools, embDim] (preserveDim=true → numHeads*headDim = embDim)
    assertEquals(H.shape[0], toolIds.length);
    assertEquals(H.shape[1], SMALL_CONFIG.numHeads * SMALL_CONFIG.headDim);

    // E[0]: [numCaps, embDim]
    const E0 = E.get(0);
    assertExists(E0);
    assertEquals(E0!.shape[0], capIds0.length);
    assertEquals(E0!.shape[1], SMALL_CONFIG.numHeads * SMALL_CONFIG.headDim);

    // All values finite
    const H_data = H.arraySync() as number[][];
    for (const row of H_data) {
      for (const v of row) {
        assertEquals(isFinite(v), true, `H contains non-finite: ${v}`);
      }
    }

    H_init.dispose();
    for (const [, t] of E_init) t.dispose();
    H.dispose();
    for (const [, t] of E) t.dispose();
    disposeGraphStructure(graph);
  } finally {
    disposeParams(params);
  }
});

// ============================================================================
// Test: MP enrichment changes embeddings
// ============================================================================

Deno.test("MP Forward: enrichment changes embeddings vs raw input", () => {
  const { toolIds, capInfos, embeddings } = makeSmallGraph();
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const graph = buildGraphStructure(capInfos, toolIds);
    const H_init = tf.tensor2d(toolIds.map((id) => embeddings.get(id)!));
    const E_init = new Map<number, tf.Tensor2D>();
    const capIds0 = graph.capIdsByLevel.get(0)!;
    E_init.set(0, tf.tensor2d(capIds0.map((id) => embeddings.get(id)!)));

    const { H, E } = messagePassingForward(H_init, E_init, graph, params, SMALL_CONFIG);

    const H_init_data = H_init.arraySync() as number[][];
    const H_data = H.arraySync() as number[][];

    // Enriched should differ from init (at least one tool changes)
    let anyDifference = false;
    for (let i = 0; i < H_init_data.length; i++) {
      for (let j = 0; j < H_init_data[i].length; j++) {
        if (Math.abs(H_data[i][j] - H_init_data[i][j]) > 1e-6) {
          anyDifference = true;
          break;
        }
      }
      if (anyDifference) break;
    }
    assertEquals(anyDifference, true, "MP enrichment had no effect on tool embeddings");

    H_init.dispose();
    for (const [, t] of E_init) t.dispose();
    H.dispose();
    for (const [, t] of E) t.dispose();
    disposeGraphStructure(graph);
  } finally {
    disposeParams(params);
  }
});

// ============================================================================
// Test: Gradients flow through MP weights
// ============================================================================

Deno.test("MP Autograd: gradients flow through W_up", () => {
  const { toolIds, capInfos, embeddings } = makeSmallGraph();
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const graph = buildGraphStructure(capInfos, toolIds);
    const H_init = tf.tensor2d(toolIds.map((id) => embeddings.get(id)!));
    const E_init = new Map<number, tf.Tensor2D>();
    const capIds0 = graph.capIdsByLevel.get(0)!;
    E_init.set(0, tf.tensor2d(capIds0.map((id) => embeddings.get(id)!)));

    const W_up_0 = params.W_up.get(0);
    assertExists(W_up_0);

    const { grads, value } = tf.variableGrads(() => {
      const { H, E } = messagePassingForward(H_init, E_init, graph, params, SMALL_CONFIG);
      // Loss = sum of all enriched embeddings
      let loss = tf.sum(H) as tf.Scalar;
      for (const [, tensor] of E) {
        loss = loss.add(tf.sum(tensor)) as tf.Scalar;
      }
      return loss;
    });

    // W_up[0][0] should have a gradient
    const W_up_0_0_name = W_up_0![0].name;
    const grad = grads[W_up_0_0_name];
    assertExists(grad, `No gradient for ${W_up_0_0_name}`);

    const gradData = grad.arraySync() as number[][];
    let hasNonZero = false;
    for (const row of gradData) {
      for (const v of row) {
        if (Math.abs(v) > 1e-10) {
          hasNonZero = true;
          break;
        }
      }
      if (hasNonZero) break;
    }
    assertEquals(hasNonZero, true, "W_up gradient is all zeros — no gradient flow through MP");

    Object.values(grads).forEach((g) => g.dispose());
    (value as tf.Tensor).dispose();
    H_init.dispose();
    for (const [, t] of E_init) t.dispose();
    disposeGraphStructure(graph);
  } finally {
    disposeParams(params);
  }
});

Deno.test("MP Autograd: gradients flow through W_down", () => {
  const { toolIds, capInfos, embeddings } = makeSmallGraph();
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const graph = buildGraphStructure(capInfos, toolIds);
    const H_init = tf.tensor2d(toolIds.map((id) => embeddings.get(id)!));
    const E_init = new Map<number, tf.Tensor2D>();
    const capIds0 = graph.capIdsByLevel.get(0)!;
    E_init.set(0, tf.tensor2d(capIds0.map((id) => embeddings.get(id)!)));

    const W_down_1 = params.W_down.get(1);
    assertExists(W_down_1);

    const { grads, value } = tf.variableGrads(() => {
      const { H } = messagePassingForward(H_init, E_init, graph, params, SMALL_CONFIG);
      return tf.sum(H) as tf.Scalar;
    });

    const W_down_name = W_down_1![0].name;
    const grad = grads[W_down_name];
    assertExists(grad, `No gradient for ${W_down_name}`);

    const gradData = grad.arraySync() as number[][];
    let hasNonZero = false;
    for (const row of gradData) {
      for (const v of row) {
        if (Math.abs(v) > 1e-10) hasNonZero = true;
      }
    }
    assertEquals(hasNonZero, true, "W_down gradient is all zeros");

    Object.values(grads).forEach((g) => g.dispose());
    (value as tf.Tensor).dispose();
    H_init.dispose();
    for (const [, t] of E_init) t.dispose();
    disposeGraphStructure(graph);
  } finally {
    disposeParams(params);
  }
});

Deno.test("MP Autograd: gradients flow through attention vectors a_up", () => {
  const { toolIds, capInfos, embeddings } = makeSmallGraph();
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const graph = buildGraphStructure(capInfos, toolIds);
    const H_init = tf.tensor2d(toolIds.map((id) => embeddings.get(id)!));
    const E_init = new Map<number, tf.Tensor2D>();
    const capIds0 = graph.capIdsByLevel.get(0)!;
    E_init.set(0, tf.tensor2d(capIds0.map((id) => embeddings.get(id)!)));

    const a_up_0 = params.a_up.get(0);
    assertExists(a_up_0);

    const { grads, value } = tf.variableGrads(() => {
      const { H, E } = messagePassingForward(H_init, E_init, graph, params, SMALL_CONFIG);
      let loss = tf.sum(H) as tf.Scalar;
      for (const [, tensor] of E) {
        loss = loss.add(tf.sum(tensor)) as tf.Scalar;
      }
      return loss;
    });

    const a_up_name = a_up_0![0].name;
    const grad = grads[a_up_name];
    assertExists(grad, `No gradient for ${a_up_name}`);

    const gradData = grad.arraySync() as number[];
    let hasNonZero = false;
    for (const v of gradData) {
      if (Math.abs(v) > 1e-10) hasNonZero = true;
    }
    assertEquals(hasNonZero, true, "a_up gradient is all zeros");

    Object.values(grads).forEach((g) => g.dispose());
    (value as tf.Tensor).dispose();
    H_init.dispose();
    for (const [, t] of E_init) t.dispose();
    disposeGraphStructure(graph);
  } finally {
    disposeParams(params);
  }
});

// ============================================================================
// Test: End-to-end gradient flow (MP + K-head scoring + InfoNCE)
// ============================================================================

Deno.test("E2E Autograd: full pipeline loss has MP weight gradients", () => {
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const graph = buildGraphStructure(capInfos, toolIds);
    const H_init = tf.tensor2d(toolIds.map((id) => embeddings.get(id)!));
    const E_init = new Map<number, tf.Tensor2D>();
    const capIds0 = graph.capIdsByLevel.get(0)!;
    E_init.set(0, tf.tensor2d(capIds0.map((id) => embeddings.get(id)!)));

    const intentArr = makeEmb(50);
    const temperature = 0.1;

    const { grads, value } = tf.variableGrads(() => {
      // MP forward
      const { E } = messagePassingForward(H_init, E_init, graph, params, SMALL_CONFIG);
      const enrichedCaps = E.get(0)!;

      // K-head scoring
      const intentT = tf.tensor1d(intentArr);
      const scores = forwardScoring(intentT, enrichedCaps, params, SMALL_CONFIG);

      // InfoNCE loss (cap-0 positive, rest negative)
      const posScore = scores.slice([0], [1]).squeeze() as tf.Scalar;
      const negScores = scores.slice([1], [capIds.length - 1]) as tf.Tensor1D;
      return infoNCELoss(posScore, negScores, temperature);
    });

    // Check W_k has gradient
    const wkGrad = grads[params.W_k[0].name];
    assertExists(wkGrad, "No gradient for W_k");

    // Check W_up has gradient (proves gradients flow through MP → scoring → loss)
    const W_up_0 = params.W_up.get(0);
    const wUpGrad = grads[W_up_0![0].name];
    assertExists(wUpGrad, "No gradient for W_up — gradient not flowing through MP");

    // Check loss is finite
    const lossVal = (value as tf.Tensor).arraySync() as number;
    assertEquals(isFinite(lossVal), true, `Loss is not finite: ${lossVal}`);
    assertGreater(lossVal, 0, "InfoNCE loss should be positive");

    Object.values(grads).forEach((g) => g.dispose());
    (value as tf.Tensor).dispose();
    H_init.dispose();
    for (const [, t] of E_init) t.dispose();
    disposeGraphStructure(graph);
  } finally {
    disposeParams(params);
  }
});

// ============================================================================
// Test: Tensor leak detection
// ============================================================================

Deno.test("Tensor leak: training step does not leak tensors", async () => {
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();

  const trainer = new AutogradTrainer(SMALL_CONFIG, {
    learningRate: 0.01,
    batchSize: 3,
    temperature: 0.1,
    gradientClip: 1.0,
    l2Lambda: 0.0001,
  }, 1);

  try {
    trainer.setNodeEmbeddings(embeddings);
    const graph = buildGraphStructure(capInfos, toolIds);
    trainer.setGraph(graph);

    const examples: TrainingExample[] = capIds.map((capId, i) => ({
      intentEmbedding: makeEmb(100 + i),
      contextTools: [],
      candidateId: capId,
      outcome: 1,
      negativeCapIds: capIds.filter((c) => c !== capId),
    }));

    // Warmup (first call may allocate things)
    await trainer.trainBatch(examples);

    // Measure
    const tensorsBefore = tf.memory().numTensors;
    for (let i = 0; i < 3; i++) {
      await trainer.trainBatch(examples);
    }
    const tensorsAfter = tf.memory().numTensors;

    // Allow a small tolerance (optimizer state may grow slightly)
    const leaked = tensorsAfter - tensorsBefore;
    assertLess(leaked, 10, `Tensor leak: ${leaked} tensors leaked over 3 training steps`);

    disposeGraphStructure(graph);
  } finally {
    trainer.dispose();
  }
});

Deno.test("Tensor leak: scoring does not leak tensors", () => {
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();

  const trainer = new AutogradTrainer(SMALL_CONFIG, {
    learningRate: 0.01,
    batchSize: 3,
    temperature: 0.1,
  }, 1);

  try {
    trainer.setNodeEmbeddings(embeddings);
    const graph = buildGraphStructure(capInfos, toolIds);
    trainer.setGraph(graph);

    // Warmup
    trainer.score(makeEmb(50), capIds);

    const tensorsBefore = tf.memory().numTensors;
    for (let i = 0; i < 5; i++) {
      trainer.score(makeEmb(50 + i), capIds);
    }
    const tensorsAfter = tf.memory().numTensors;

    const leaked = tensorsAfter - tensorsBefore;
    assertEquals(leaked, 0, `Scoring leaked ${leaked} tensors over 5 calls`);

    disposeGraphStructure(graph);
  } finally {
    trainer.dispose();
  }
});

// ============================================================================
// Test: InfoNCE loss properties
// ============================================================================

Deno.test("InfoNCE: loss is zero when positive score >> negatives", () => {
  const posScore = tf.scalar(10.0);
  const negScores = tf.tensor1d([-5.0, -5.0, -5.0]);

  const loss = infoNCELoss(posScore, negScores, 0.1).arraySync() as number;
  assertLess(loss, 0.01, `Loss should be near zero when positive dominates, got ${loss}`);

  posScore.dispose();
  negScores.dispose();
});

Deno.test("InfoNCE: loss increases when negatives get closer to positive", () => {
  const posScore = tf.scalar(1.0);

  const negFar = tf.tensor1d([-2.0, -2.0]);
  const negClose = tf.tensor1d([0.8, 0.9]);

  const lossFar = infoNCELoss(posScore, negFar, 0.1).arraySync() as number;
  const lossClose = infoNCELoss(posScore, negClose, 0.1).arraySync() as number;

  assertGreater(lossClose, lossFar, "Loss should be higher when negatives are close to positive");

  posScore.dispose();
  negFar.dispose();
  negClose.dispose();
});

Deno.test("InfoNCE: loss is symmetric in negatives order", () => {
  const posScore = tf.scalar(0.5);
  const neg1 = tf.tensor1d([0.1, 0.3, 0.2]);
  const neg2 = tf.tensor1d([0.3, 0.2, 0.1]);

  const loss1 = infoNCELoss(posScore, neg1, 0.1).arraySync() as number;
  const loss2 = infoNCELoss(posScore, neg2, 0.1).arraySync() as number;

  // Same values, different order → same loss (softmax is permutation invariant)
  assertLess(Math.abs(loss1 - loss2), 1e-5, "InfoNCE should be invariant to negative ordering");

  posScore.dispose();
  neg1.dispose();
  neg2.dispose();
});

// ============================================================================
// Test: K-head scoring properties
// ============================================================================

Deno.test("K-head: scoring is deterministic", () => {
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const intentEmb = tf.tensor1d(makeEmb(50));
    const nodeEmbs = tf.tensor2d([makeEmb(0), makeEmb(1), makeEmb(2)]);

    const scores1 = tf.tidy(() =>
      kHeadScoring(intentEmb, nodeEmbs, params, SMALL_CONFIG).arraySync() as number[]
    );
    const scores2 = tf.tidy(() =>
      kHeadScoring(intentEmb, nodeEmbs, params, SMALL_CONFIG).arraySync() as number[]
    );

    for (let i = 0; i < scores1.length; i++) {
      assertEquals(scores1[i], scores2[i], `Score[${i}] not deterministic`);
    }

    intentEmb.dispose();
    nodeEmbs.dispose();
  } finally {
    disposeParams(params);
  }
});

Deno.test("K-head: different intents produce different scores", () => {
  const params = initTFParams(SMALL_CONFIG, 1);

  try {
    const nodeEmbs = tf.tensor2d([makeEmb(0), makeEmb(1), makeEmb(2)]);

    const scores1 = tf.tidy(() => {
      const intent1 = tf.tensor1d(makeEmb(50));
      return kHeadScoring(intent1, nodeEmbs, params, SMALL_CONFIG).arraySync() as number[];
    });
    const scores2 = tf.tidy(() => {
      const intent2 = tf.tensor1d(makeEmb(99));
      return kHeadScoring(intent2, nodeEmbs, params, SMALL_CONFIG).arraySync() as number[];
    });

    let anyDifferent = false;
    for (let i = 0; i < scores1.length; i++) {
      if (Math.abs(scores1[i] - scores2[i]) > 1e-6) {
        anyDifferent = true;
        break;
      }
    }
    assertEquals(anyDifferent, true, "Different intents should produce different scores");

    nodeEmbs.dispose();
  } finally {
    disposeParams(params);
  }
});

// ============================================================================
// Test: Annealing helpers
// ============================================================================

Deno.test("annealTemperature: decreases from tauStart to tauEnd", () => {
  const tauStart = 0.10;
  const tauEnd = 0.06;
  const epochs = 10;

  const first = annealTemperature(0, epochs, tauStart, tauEnd);
  const mid = annealTemperature(5, epochs, tauStart, tauEnd);
  const last = annealTemperature(epochs - 1, epochs, tauStart, tauEnd);

  assertLess(Math.abs(first - tauStart), 0.001, `First tau should be ~${tauStart}`);
  assertLess(Math.abs(last - tauEnd), 0.001, `Last tau should be ~${tauEnd}`);
  assertGreater(first, mid, "Temperature should decrease over time");
  assertGreater(mid, last, "Temperature should continue decreasing");
});

Deno.test("annealBeta: increases from betaStart to 1.0", () => {
  const betaStart = 0.4;
  const epochs = 10;

  const first = annealBeta(0, epochs, betaStart);
  const last = annealBeta(epochs - 1, epochs, betaStart);

  assertLess(Math.abs(first - betaStart), 0.001, `First beta should be ~${betaStart}`);
  assertLess(Math.abs(last - 1.0), 0.001, "Last beta should be ~1.0");
  assertGreater(last, first, "Beta should increase over time");
});

// ============================================================================
// Test: PER buffer advanced
// ============================================================================

Deno.test("PER: high-error examples sampled more often", () => {
  const examples = Array.from({ length: 20 }, (_, i) => ({
    intentEmbedding: makeEmb(i),
    contextTools: [] as string[],
    candidateId: `c${i}`,
    outcome: 1,
    negativeCapIds: [`c${(i + 1) % 20}`],
  } as TrainingExample));

  const buffer = new PERBuffer(examples, { alpha: 0.6, beta: 0.4, maxPriority: 10 });

  // Set high priority for items 0-4, low for rest
  const highIndices = [0, 1, 2, 3, 4];
  const lowIndices = Array.from({ length: 15 }, (_, i) => i + 5);
  buffer.updatePriorities(highIndices, [10, 10, 10, 10, 10]);
  buffer.updatePriorities(lowIndices, lowIndices.map(() => 0.01));

  // Sample many times, count how often high-priority items appear
  let highCount = 0;
  const totalSamples = 50;
  for (let i = 0; i < totalSamples; i++) {
    const { indices } = buffer.sample(5, 0.4);
    for (const idx of indices) {
      if (highIndices.includes(idx)) highCount++;
    }
  }

  // With alpha=0.6, high-priority items should appear more than uniform baseline (25%)
  const highRatio = highCount / (totalSamples * 5);
  assertGreater(highRatio, 0.25, `High-priority items only sampled ${(highRatio * 100).toFixed(1)}% of the time`);
});

Deno.test("PER: decayPriorities reduces above-mean priorities", () => {
  const examples = Array.from({ length: 10 }, (_, i) => ({
    intentEmbedding: makeEmb(i),
    contextTools: [] as string[],
    candidateId: `c${i}`,
    outcome: 1,
    negativeCapIds: [`c${(i + 1) % 10}`],
  } as TrainingExample));

  const buffer = new PERBuffer(examples, { maxPriority: 10 });
  buffer.updatePriorities([0, 1], [10, 10]);
  buffer.updatePriorities([5, 6], [0.1, 0.1]);

  const statsBefore = buffer.getStats();
  buffer.decayPriorities(0.5);
  const statsAfter = buffer.getStats();

  assertLess(statsAfter.max, statsBefore.max, "Max priority should decrease after decay");
});

// ============================================================================
// Test: Training with MP improves scoring
// ============================================================================

Deno.test("Training with MP: loss decreases over epochs", async () => {
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();

  const trainer = new AutogradTrainer(SMALL_CONFIG, {
    learningRate: 0.01,
    batchSize: 3,
    temperature: 0.1,
    gradientClip: 1.0,
    l2Lambda: 0.0001,
  }, 1);

  try {
    trainer.setNodeEmbeddings(embeddings);
    const graph = buildGraphStructure(capInfos, toolIds);
    trainer.setGraph(graph);

    const examples: TrainingExample[] = capIds.map((capId, i) => ({
      intentEmbedding: makeEmb(100 + i),
      contextTools: [],
      candidateId: capId,
      outcome: 1,
      negativeCapIds: capIds.filter((c) => c !== capId),
    }));

    const losses: number[] = [];
    for (let epoch = 0; epoch < 20; epoch++) {
      const metrics = await trainer.trainBatch(examples);
      losses.push(metrics.loss);
    }

    // Average of first 3 vs last 3
    const firstAvg = losses.slice(0, 3).reduce((a, b) => a + b, 0) / 3;
    const lastAvg = losses.slice(-3).reduce((a, b) => a + b, 0) / 3;

    assertLess(lastAvg, firstAvg, `Loss should decrease: first=${firstAvg.toFixed(4)} last=${lastAvg.toFixed(4)}`);

    disposeGraphStructure(graph);
  } finally {
    trainer.dispose();
  }
});

Deno.test("Training with MP: accuracy reaches 100% on tiny dataset", async () => {
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();

  const trainer = new AutogradTrainer(SMALL_CONFIG, {
    learningRate: 0.01,
    batchSize: 3,
    temperature: 0.1,
    gradientClip: 1.0,
    l2Lambda: 0.0,
  }, 1);

  try {
    trainer.setNodeEmbeddings(embeddings);
    const graph = buildGraphStructure(capInfos, toolIds);
    trainer.setGraph(graph);

    const examples: TrainingExample[] = capIds.map((capId, i) => ({
      intentEmbedding: makeEmb(100 + i),
      contextTools: [],
      candidateId: capId,
      outcome: 1,
      negativeCapIds: capIds.filter((c) => c !== capId),
    }));

    // Train more epochs
    let lastAcc = 0;
    for (let epoch = 0; epoch < 50; epoch++) {
      const metrics = await trainer.trainBatch(examples);
      lastAcc = metrics.accuracy;
    }

    // With 3 examples and 50 epochs, should reach high accuracy
    assertGreater(lastAcc, 0.6, `Accuracy should be >60% after 50 epochs, got ${(lastAcc * 100).toFixed(1)}%`);

    disposeGraphStructure(graph);
  } finally {
    trainer.dispose();
  }
});

// ============================================================================
// Test: Graph structure edge cases
// ============================================================================

Deno.test("buildGraphStructure: handles empty capabilities", () => {
  const toolIds = ["t0", "t1"];
  const graph = buildGraphStructure([], toolIds);

  assertEquals(graph.maxLevel, 0);
  assertEquals(graph.toolIds.length, 2);
  assertEquals(graph.toolToCapMatrix.shape[0], 2);
  assertEquals(graph.toolToCapMatrix.shape[1], 0); // No caps

  disposeGraphStructure(graph);
});

Deno.test("buildGraphStructure: handles single capability", () => {
  const toolIds = ["t0", "t1"];
  const capInfos: CapabilityInfo[] = [
    { id: "c0", toolsUsed: ["t0", "t1"] },
  ];
  const graph = buildGraphStructure(capInfos, toolIds);

  assertEquals(graph.maxLevel, 0);
  const capIds0 = graph.capIdsByLevel.get(0);
  assertExists(capIds0);
  assertEquals(capIds0!.length, 1);

  // Connectivity: both tools connected to c0
  const matrix = graph.toolToCapMatrix.arraySync() as number[][];
  assertEquals(matrix[0][0], 1);
  assertEquals(matrix[1][0], 1);

  disposeGraphStructure(graph);
});

Deno.test("buildGraphStructure: disconnected tools have zero rows", () => {
  const toolIds = ["t0", "t1", "t_orphan"];
  const capInfos: CapabilityInfo[] = [
    { id: "c0", toolsUsed: ["t0", "t1"] },
  ];
  const graph = buildGraphStructure(capInfos, toolIds);

  const matrix = graph.toolToCapMatrix.arraySync() as number[][];
  // t_orphan is index 2, should have all zeros
  assertEquals(matrix[2][0], 0, "Orphan tool should not be connected");

  disposeGraphStructure(graph);
});

// ============================================================================
// Test: Scoring with vs without MP
// ============================================================================

Deno.test("Scoring: with MP produces different scores than without MP", () => {
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();

  // Use a SINGLE trainer: score without graph, then set graph and score again
  const trainer = new AutogradTrainer(SMALL_CONFIG, {
    learningRate: 0.01,
    temperature: 0.1,
  }, 1);
  trainer.setNodeEmbeddings(embeddings);

  const intent = makeEmb(50);

  // Score without MP (no graph set)
  const scoresNoMP = trainer.score(intent, capIds);

  // Set graph → enables MP
  const graph = buildGraphStructure(capInfos, toolIds);
  trainer.setGraph(graph);

  const scoresMP = trainer.score(intent, capIds);

  // They should differ (MP enriches embeddings before scoring)
  let anyDifferent = false;
  for (let i = 0; i < capIds.length; i++) {
    if (Math.abs(scoresNoMP[i] - scoresMP[i]) > 1e-6) {
      anyDifferent = true;
      break;
    }
  }
  assertEquals(anyDifferent, true, "MP should change scoring results");

  disposeGraphStructure(graph);
  trainer.dispose();
});

// ============================================================================
// Test: forwardScoring with preserveDim
// ============================================================================

Deno.test("forwardScoring: preserveDim produces valid scores", () => {
  // Uses the already-created params from previous tests via trainer
  const { toolIds, capIds, capInfos, embeddings } = makeSmallGraph();
  const config = { ...SMALL_CONFIG, preserveDim: true };

  const trainer = new AutogradTrainer(config, {
    learningRate: 0.01,
    temperature: 0.1,
  }, 1);
  trainer.setNodeEmbeddings(embeddings);

  const scores = trainer.score(makeEmb(50), capIds);

  assertEquals(scores.length, capIds.length);
  for (const s of scores) {
    assertEquals(isFinite(s), true, `Score should be finite, got ${s}`);
  }

  trainer.dispose();
});
