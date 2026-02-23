/**
 * SHGAT score() enriched embeddings path test
 *
 * Verifies that score() uses pre-computed enriched embeddings when available,
 * instead of running full message passing for every call.
 *
 * Run with: npx tsx --test src/training/__tests__/score-enriched.test.ts
 *
 * @module shgat-tf/training/__tests__/score-enriched
 */

import { describe, it, before } from "node:test";
import { strict as assert } from "node:assert";
import { AutogradTrainer } from "../autograd-trainer.ts";
import type { SHGATConfig } from "../../core/types.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const EMB_DIM = 64; // small for fast tests
const NUM_HEADS = 2;
const HEAD_DIM = 32;

function randomEmb(dim: number): number[] {
  return Array.from({ length: dim }, () => Math.random() * 2 - 1);
}

function makeConfig(): SHGATConfig {
  return {
    numHeads: NUM_HEADS,
    headDim: HEAD_DIM,
    embeddingDim: EMB_DIM,
    hiddenDim: EMB_DIM,
    leakyReluSlope: 0.2,
    activation: "elu",
    useProjectionHead: false,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("AutogradTrainer.score() enriched path", () => {
  let trainer: AutogradTrainer;
  const toolIds = ["tool_a", "tool_b", "tool_c"];

  before(() => {
    const config = makeConfig();
    trainer = new AutogradTrainer(config, { learningRate: 0.001 }, 1, 42);

    // Set raw node embeddings
    const embs = new Map<string, number[]>();
    for (const id of toolIds) {
      embs.set(id, randomEmb(EMB_DIM));
    }
    trainer.setNodeEmbeddings(embs);
  });

  it("returns scores without enriched embeddings (raw path)", () => {
    const intent = randomEmb(EMB_DIM);
    const scores = trainer.score(intent, toolIds);

    assert.equal(scores.length, toolIds.length, "should return one score per tool");
    assert.ok(scores.every((s) => typeof s === "number" && isFinite(s)), "scores should be finite numbers");
  });

  it("uses enriched embeddings when precomputed", () => {
    // Manually set enriched embeddings (simulating precomputeEnrichedEmbeddings)
    const enriched = new Map<string, number[]>();
    for (const id of toolIds) {
      // Use a very different embedding to detect which path is used
      enriched.set(id, Array.from({ length: EMB_DIM }, () => 99.0));
    }
    // Access private field for test purposes
    // deno-lint-ignore no-explicit-any
    (trainer as any).enrichedNodeEmbeddings = enriched;

    const intent = randomEmb(EMB_DIM);
    const scoresEnriched = trainer.score(intent, toolIds);

    // Clear enriched to get raw scores
    // deno-lint-ignore no-explicit-any
    (trainer as any).enrichedNodeEmbeddings = null;
    const scoresRaw = trainer.score(intent, toolIds);

    // Scores should be different because embeddings are different
    let anyDifferent = false;
    for (let i = 0; i < scoresEnriched.length; i++) {
      if (Math.abs(scoresEnriched[i] - scoresRaw[i]) > 1e-4) {
        anyDifferent = true;
        break;
      }
    }
    assert.ok(anyDifferent, "enriched scores should differ from raw scores");
  });

  it("falls back to raw embeddings for unknown nodes in enriched map", () => {
    const enriched = new Map<string, number[]>();
    // Only enrich tool_a, leave tool_b and tool_c un-enriched
    enriched.set("tool_a", Array.from({ length: EMB_DIM }, () => 50.0));
    // deno-lint-ignore no-explicit-any
    (trainer as any).enrichedNodeEmbeddings = enriched;

    const intent = randomEmb(EMB_DIM);
    const scores = trainer.score(intent, toolIds);

    assert.equal(scores.length, 3, "should still return scores for all tools");
    assert.ok(scores.every((s) => isFinite(s)), "all scores should be finite");

    // Cleanup
    // deno-lint-ignore no-explicit-any
    (trainer as any).enrichedNodeEmbeddings = null;
  });
});
