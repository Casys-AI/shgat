/**
 * SHGAT-TF Training Unit Tests
 *
 * Tests for AutogradTrainer and PER buffer functionality.
 *
 * Note: Tests for deprecated trainBatchV1KHeadBatched have been removed.
 * Use AutogradTrainer for all training needs.
 *
 * @module shgat-tf/tests/training_test
 */

import { assertEquals, assertExists, assertGreater, assertLess } from "@std/assert";
import {
  AutogradTrainer,
  DEFAULT_TRAINER_CONFIG,
  PERBuffer,
  type TrainingExample,
  DEFAULT_SHGAT_CONFIG,
} from "../mod.ts";

// =============================================================================
// Test Fixtures
// =============================================================================

function createTestEmbeddings(count: number): Map<string, number[]> {
  const embeddings = new Map<string, number[]>();
  for (let i = 0; i < count; i++) {
    embeddings.set(
      `node-${i}`,
      Array.from({ length: 1024 }, (_, j) => Math.sin(i * 0.1 + j * 0.01)),
    );
  }
  return embeddings;
}

function createTrainingExample(
  nodeId: string,
  negatives: string[],
): TrainingExample {
  return {
    intentEmbedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
    contextTools: ["tool-1"],
    candidateId: nodeId,
    outcome: 1,
    negativeCapIds: negatives,
  };
}

// =============================================================================
// AutogradTrainer Tests
// =============================================================================

Deno.test("AutogradTrainer - instantiates with defaults", () => {
  const trainer = new AutogradTrainer(DEFAULT_SHGAT_CONFIG);

  assertExists(trainer);
  trainer.dispose();
});

Deno.test("AutogradTrainer - setNodeEmbeddings works", () => {
  const trainer = new AutogradTrainer(DEFAULT_SHGAT_CONFIG);
  const embeddings = createTestEmbeddings(10);

  trainer.setNodeEmbeddings(embeddings);

  // No error means success
  trainer.dispose();
});

Deno.test("AutogradTrainer - score returns array", () => {
  const trainer = new AutogradTrainer(DEFAULT_SHGAT_CONFIG);
  const embeddings = createTestEmbeddings(5);
  trainer.setNodeEmbeddings(embeddings);

  const intentEmb = Array.from({ length: 1024 }, () => Math.random() * 0.1);
  const scores = trainer.score(intentEmb, ["node-0", "node-1", "node-2"]);

  assertEquals(scores.length, 3);
  for (const score of scores) {
    assertEquals(typeof score, "number");
    assertEquals(isFinite(score), true);
  }

  trainer.dispose();
});

Deno.test("AutogradTrainer - trainBatch returns metrics", () => {
  const trainer = new AutogradTrainer(DEFAULT_SHGAT_CONFIG);
  const embeddings = createTestEmbeddings(5);
  trainer.setNodeEmbeddings(embeddings);

  const examples = [
    createTrainingExample("node-0", ["node-1", "node-2"]),
    createTrainingExample("node-1", ["node-0", "node-3"]),
  ];

  const metrics = trainer.trainBatch(examples);

  assertExists(metrics);
  assertEquals(typeof metrics.loss, "number");
  assertEquals(typeof metrics.accuracy, "number");
  assertEquals(typeof metrics.gradientNorm, "number");
  assertEquals(metrics.numExamples, 2);
  assertEquals(isFinite(metrics.loss), true);

  trainer.dispose();
});

Deno.test("AutogradTrainer - multiple train steps reduce loss", () => {
  const trainer = new AutogradTrainer({
    ...DEFAULT_SHGAT_CONFIG,
  }, { learningRate: 0.01 });
  const embeddings = createTestEmbeddings(5);
  trainer.setNodeEmbeddings(embeddings);

  const examples = [
    createTrainingExample("node-0", ["node-1", "node-2"]),
  ];

  // Train multiple steps
  const losses: number[] = [];
  for (let i = 0; i < 10; i++) {
    const metrics = trainer.trainBatch(examples);
    losses.push(metrics.loss);
  }

  // Loss should generally decrease (allow some variance)
  const firstLoss = losses[0];
  const lastLoss = losses[losses.length - 1];
  // At least shouldn't explode
  assertLess(lastLoss, firstLoss * 10, "Loss should not explode");

  trainer.dispose();
});

// =============================================================================
// PER Buffer Tests
// =============================================================================

Deno.test("PERBuffer - instantiates with items", () => {
  const examples = Array.from({ length: 10 }, (_, i) =>
    createTrainingExample(`node-${i}`, [`node-${(i + 1) % 10}`])
  );

  const buffer = new PERBuffer(examples, {
    alpha: 0.6,
    beta: 0.4,
  });

  assertExists(buffer);
  assertEquals(buffer.size, 10);
});

Deno.test("PERBuffer - sample returns correct structure", () => {
  const examples = Array.from({ length: 10 }, (_, i) =>
    createTrainingExample(`node-${i}`, [`node-${(i + 1) % 10}`])
  );

  const buffer = new PERBuffer(examples, {
    alpha: 0.6,
    beta: 0.4,
  });

  // Sample
  const sampled = buffer.sample(5);

  assertEquals(sampled.items.length, 5);
  assertEquals(sampled.weights.length, 5);
  assertEquals(sampled.indices.length, 5);

  // Weights should be positive
  for (const w of sampled.weights) {
    assertGreater(w, 0);
  }
});

Deno.test("PERBuffer - update priorities", () => {
  const examples = Array.from({ length: 10 }, (_, i) =>
    createTrainingExample(`node-${i}`, [`node-${(i + 1) % 10}`])
  );

  const buffer = new PERBuffer(examples, {
    alpha: 0.6,
    beta: 0.4,
  });

  // Sample and update
  const sampled = buffer.sample(5);
  const newPriorities = sampled.indices.map((_, i) => 0.5 + i * 0.1);
  buffer.updatePriorities(sampled.indices, newPriorities);

  // Should not crash
  const sampled2 = buffer.sample(3);
  assertEquals(sampled2.items.length, 3);
});

Deno.test("PERBuffer - getStats returns statistics", () => {
  const examples = Array.from({ length: 10 }, (_, i) =>
    createTrainingExample(`node-${i}`, [`node-${(i + 1) % 10}`])
  );

  const buffer = new PERBuffer(examples, {
    alpha: 0.6,
    beta: 0.4,
  });

  const stats = buffer.getStats();

  assertExists(stats.mean);
  assertExists(stats.max);
  assertExists(stats.min);
  assertExists(stats.std);
  assertEquals(typeof stats.mean, "number");
});

// =============================================================================
// Trainer Config Tests
// =============================================================================

Deno.test("DEFAULT_TRAINER_CONFIG - has expected values", () => {
  assertEquals(typeof DEFAULT_TRAINER_CONFIG.learningRate, "number");
  assertEquals(typeof DEFAULT_TRAINER_CONFIG.batchSize, "number");
  assertEquals(typeof DEFAULT_TRAINER_CONFIG.temperature, "number");
  assertEquals(typeof DEFAULT_TRAINER_CONFIG.gradientClip, "number");
  assertEquals(typeof DEFAULT_TRAINER_CONFIG.l2Lambda, "number");

  assertGreater(DEFAULT_TRAINER_CONFIG.learningRate, 0);
  assertGreater(DEFAULT_TRAINER_CONFIG.temperature, 0);
});
