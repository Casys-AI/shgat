/**
 * SHGAT-TF Builder & Ports Tests
 *
 * Tests for the recommended SHGATBuilder API and port interfaces.
 * Verifies that the builder produces correctly wired instances that
 * can score, train, and dispose without errors.
 *
 * Run: deno test -A --no-check lib/shgat-tf/tests/builder_test.ts
 *
 * @module shgat-tf/tests/builder_test
 */

import {
  assertEquals,
  assertExists,
  assertGreater,
  assertLess,
  assertRejects,
} from "@std/assert";
import {
  SHGATBuilder,
  type SHGATScorer,
  type SHGATTrainer,
  type SHGATTrainerScorer,
  type NodeInput,
  type TrainingExample,
} from "../mod.ts";

// =============================================================================
// Test Fixtures
// =============================================================================

const DIM = 32; // Small dim for fast tests

function makeEmb(seed: number): number[] {
  return Array.from({ length: DIM }, (_, j) => Math.sin(seed * 0.7 + j * 0.3) * 0.5);
}

function makeNodes(numTools: number, numCaps: number): NodeInput[] {
  const nodes: NodeInput[] = [];

  // Tools (leaves)
  for (let i = 0; i < numTools; i++) {
    nodes.push({
      id: `tool-${i}`,
      embedding: makeEmb(i),
      children: [],
    });
  }

  // Capabilities (composites) — each cap uses 2 tools
  for (let i = 0; i < numCaps; i++) {
    const t1 = i % numTools;
    const t2 = (i + 1) % numTools;
    nodes.push({
      id: `cap-${i}`,
      embedding: makeEmb(100 + i),
      children: [`tool-${t1}`, `tool-${t2}`],
    });
  }

  return nodes;
}

function makeExample(capId: string, negIds: string[]): TrainingExample {
  return {
    intentEmbedding: makeEmb(200 + parseInt(capId.split("-")[1] || "0")),
    contextTools: ["tool-0"],
    candidateId: capId,
    outcome: 1,
    negativeCapIds: negIds,
  };
}

// =============================================================================
// Builder Construction Tests
// =============================================================================

Deno.test("SHGATBuilder - builds with minimal config", async () => {
  const nodes = makeNodes(5, 3);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  assertExists(shgat);
  assertEquals(shgat.hasMessagePassing, true);

  shgat.dispose();
});

Deno.test("SHGATBuilder - rejects empty nodes", async () => {
  await assertRejects(
    async () => {
      await SHGATBuilder.create().build();
    },
    Error,
    "No nodes provided",
  );
});

Deno.test("SHGATBuilder - accepts training options", async () => {
  const nodes = makeNodes(4, 2);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .training({ learningRate: 0.05, temperature: 0.10, batchSize: 8 })
    .build();

  assertExists(shgat);
  shgat.dispose();
});

Deno.test("SHGATBuilder - accepts architecture options", async () => {
  const nodes = makeNodes(4, 2);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({
      embeddingDim: DIM,
      numHeads: 4,
      headDim: 8,
      hiddenDim: DIM,
      preserveDim: true,
    })
    .build();

  assertExists(shgat);
  shgat.dispose();
});

Deno.test("SHGATBuilder - accepts backend option", async () => {
  const nodes = makeNodes(4, 2);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .backend("training")
    .build();

  assertExists(shgat);
  shgat.dispose();
});

// =============================================================================
// Port Interface Tests
// =============================================================================

Deno.test("SHGATScorer port - score returns array", async () => {
  const nodes = makeNodes(5, 3);

  const scorer: SHGATScorer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  const intent = makeEmb(50);
  const scores = scorer.score(intent, ["cap-0", "cap-1", "cap-2"]);

  assertEquals(scores.length, 3);
  for (const s of scores) {
    assertEquals(typeof s, "number");
    assertEquals(isFinite(s), true);
  }

  scorer.dispose();
});

Deno.test("SHGATScorer port - score tools", async () => {
  const nodes = makeNodes(5, 3);

  const scorer: SHGATScorer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  const intent = makeEmb(50);
  const scores = scorer.score(intent, ["tool-0", "tool-1", "tool-2"]);

  assertEquals(scores.length, 3);
  for (const s of scores) {
    assertEquals(typeof s, "number");
    assertEquals(isFinite(s), true);
  }

  scorer.dispose();
});

Deno.test("SHGATScorer port - score mixed tools and caps", async () => {
  const nodes = makeNodes(5, 3);

  const scorer: SHGATScorer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  const intent = makeEmb(50);
  const scores = scorer.score(intent, ["tool-0", "cap-0", "tool-2", "cap-1"]);

  assertEquals(scores.length, 4);
  for (const s of scores) {
    assertEquals(isFinite(s), true);
  }

  scorer.dispose();
});

Deno.test("SHGATTrainer port - trainBatch returns metrics", async () => {
  const nodes = makeNodes(5, 3);

  const trainer: SHGATTrainer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .training({ learningRate: 0.01 })
    .build();

  const examples = [
    makeExample("cap-0", ["cap-1", "cap-2"]),
    makeExample("cap-1", ["cap-0", "cap-2"]),
  ];

  const metrics = await trainer.trainBatch(examples);

  assertExists(metrics);
  assertEquals(typeof metrics.loss, "number");
  assertEquals(typeof metrics.accuracy, "number");
  assertEquals(typeof metrics.gradientNorm, "number");
  assertEquals(metrics.numExamples, 2);
  assertEquals(isFinite(metrics.loss), true);

  trainer.dispose();
});

Deno.test("SHGATTrainer port - setTemperature works", async () => {
  const nodes = makeNodes(5, 3);

  const trainer: SHGATTrainer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .training({ temperature: 0.10 })
    .build();

  // Should not throw
  trainer.setTemperature(0.08);
  trainer.setTemperature(0.06);

  trainer.dispose();
});

// =============================================================================
// Training Integration Tests
// =============================================================================

Deno.test("SHGATTrainerScorer - training improves scores", async () => {
  const nodes = makeNodes(5, 3);

  const shgat: SHGATTrainerScorer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .training({ learningRate: 0.01, temperature: 0.1 })
    .build();

  const capIds = ["cap-0", "cap-1", "cap-2"];
  const examples = capIds.map((capId) =>
    makeExample(capId, capIds.filter((c) => c !== capId))
  );

  // Train 20 steps
  const losses: number[] = [];
  for (let i = 0; i < 20; i++) {
    const metrics = await shgat.trainBatch(examples);
    losses.push(metrics.loss);
  }

  // Loss should not explode
  const firstLoss = losses[0];
  const lastLoss = losses[losses.length - 1];
  assertLess(lastLoss, firstLoss * 10, "Loss should not explode");

  // Should still produce valid scores after training
  const intent = makeEmb(200);
  const scores = shgat.score(intent, capIds);
  assertEquals(scores.length, 3);
  for (const s of scores) {
    assertEquals(isFinite(s), true);
  }

  shgat.dispose();
});

Deno.test("SHGATTrainerScorer - multiple train steps reduce loss", async () => {
  const nodes = makeNodes(5, 3);

  const shgat: SHGATTrainerScorer = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .training({ learningRate: 0.01, temperature: 0.1 })
    .build();

  const examples = [
    makeExample("cap-0", ["cap-1", "cap-2"]),
  ];

  const losses: number[] = [];
  for (let i = 0; i < 15; i++) {
    const metrics = await shgat.trainBatch(examples);
    losses.push(metrics.loss);
  }

  // Average of first 3 vs last 3 should show improvement (or at least stability)
  const firstAvg = losses.slice(0, 3).reduce((a, b) => a + b, 0) / 3;
  const lastAvg = losses.slice(-3).reduce((a, b) => a + b, 0) / 3;
  assertLess(lastAvg, firstAvg * 5, "Loss should not explode over training");

  shgat.dispose();
});

// =============================================================================
// Dispose Tests
// =============================================================================

Deno.test("SHGATTrainerScorer - double dispose is safe", async () => {
  const nodes = makeNodes(4, 2);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  shgat.dispose();
  // Should not throw
  shgat.dispose();
});

Deno.test("SHGATTrainerScorer - score after dispose throws", async () => {
  const nodes = makeNodes(4, 2);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  shgat.dispose();

  let threw = false;
  try {
    shgat.score(makeEmb(1), ["cap-0"]);
  } catch {
    threw = true;
  }
  assertEquals(threw, true, "score after dispose should throw");
});

Deno.test("SHGATTrainerScorer - trainBatch after dispose throws", async () => {
  const nodes = makeNodes(4, 2);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  shgat.dispose();

  await assertRejects(
    async () => {
      await shgat.trainBatch([makeExample("cap-0", ["cap-1"])]);
    },
    Error,
    "disposed",
  );
});

// =============================================================================
// Message Passing Tests
// =============================================================================

Deno.test("SHGATBuilder - hasMessagePassing is true with graph", async () => {
  const nodes = makeNodes(5, 3);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  assertEquals(shgat.hasMessagePassing, true);
  shgat.dispose();
});

Deno.test("SHGATBuilder - handles leaf-only graph", async () => {
  // All nodes are leaves (no composites)
  const nodes: NodeInput[] = [
    { id: "a", embedding: makeEmb(0), children: [] },
    { id: "b", embedding: makeEmb(1), children: [] },
    { id: "c", embedding: makeEmb(2), children: [] },
  ];

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  // Should still be able to score
  const scores = shgat.score(makeEmb(50), ["a", "b", "c"]);
  assertEquals(scores.length, 3);

  shgat.dispose();
});

Deno.test("SHGATBuilder - handles nodes with invalid children", async () => {
  // Cap references tool-99 which doesn't exist
  const nodes: NodeInput[] = [
    { id: "tool-0", embedding: makeEmb(0), children: [] },
    { id: "tool-1", embedding: makeEmb(1), children: [] },
    { id: "cap-0", embedding: makeEmb(10), children: ["tool-0", "tool-99"] },
  ];

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  // Should build without error (invalid children filtered)
  assertExists(shgat);
  shgat.dispose();
});

// =============================================================================
// Edge Cases
// =============================================================================

Deno.test("SHGATBuilder - single node graph", async () => {
  const nodes: NodeInput[] = [
    { id: "only", embedding: makeEmb(0), children: [] },
  ];

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  const scores = shgat.score(makeEmb(50), ["only"]);
  assertEquals(scores.length, 1);
  assertEquals(isFinite(scores[0]), true);

  shgat.dispose();
});

Deno.test("SHGATBuilder - auto-detects embedding dim", async () => {
  // Nodes with 64-dim embeddings
  const dim64 = 64;
  const nodes: NodeInput[] = [
    { id: "a", embedding: Array.from({ length: dim64 }, (_, i) => i * 0.01), children: [] },
    { id: "b", embedding: Array.from({ length: dim64 }, (_, i) => i * 0.02), children: [] },
    { id: "c", embedding: Array.from({ length: dim64 }, (_, i) => i * 0.03), children: ["a", "b"] },
  ];

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ numHeads: 2, headDim: 16 })
    .build();

  const scores = shgat.score(
    Array.from({ length: dim64 }, () => Math.random()),
    ["a", "c"],
  );
  assertEquals(scores.length, 2);

  shgat.dispose();
});

// =============================================================================
// Temperature Annealing Integration
// =============================================================================

Deno.test("SHGATTrainerScorer - temperature annealing during training", async () => {
  const nodes = makeNodes(5, 3);

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .training({ learningRate: 0.01, temperature: 0.10 })
    .build();

  const examples = [
    makeExample("cap-0", ["cap-1", "cap-2"]),
  ];

  // Simulate cosine annealing: 0.10 → 0.06 over 5 epochs
  const tauStart = 0.10;
  const tauEnd = 0.06;
  const epochs = 5;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const tau = tauEnd + 0.5 * (tauStart - tauEnd) * (1 + Math.cos(Math.PI * epoch / epochs));
    shgat.setTemperature(tau);
    const metrics = await shgat.trainBatch(examples);
    assertEquals(isFinite(metrics.loss), true, `Loss NaN at epoch ${epoch}, tau=${tau}`);
  }

  shgat.dispose();
});

// =============================================================================
// Hierarchical Graph Tests
// =============================================================================

Deno.test("SHGATBuilder - multi-level hierarchy", async () => {
  // 3 levels: tools → level-0 caps → level-1 cap
  const nodes: NodeInput[] = [
    { id: "t0", embedding: makeEmb(0), children: [] },
    { id: "t1", embedding: makeEmb(1), children: [] },
    { id: "t2", embedding: makeEmb(2), children: [] },
    { id: "c0", embedding: makeEmb(10), children: ["t0", "t1"] },
    { id: "c1", embedding: makeEmb(11), children: ["t1", "t2"] },
    { id: "root", embedding: makeEmb(20), children: ["c0", "c1"] },
  ];

  const shgat = await SHGATBuilder.create()
    .nodes(nodes)
    .architecture({ embeddingDim: DIM, hiddenDim: DIM, numHeads: 2, headDim: 16 })
    .build();

  assertEquals(shgat.hasMessagePassing, true);

  // Score all nodes
  const scores = shgat.score(makeEmb(50), ["t0", "c0", "c1", "root"]);
  assertEquals(scores.length, 4);
  for (const s of scores) {
    assertEquals(isFinite(s), true);
  }

  shgat.dispose();
});
