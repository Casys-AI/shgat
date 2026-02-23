/**
 * SHGAT Persistence Unit Tests
 *
 * Tests for export/import params functionality.
 *
 * Note: Training tests have been removed as trainBatchV1KHeadBatched is deprecated.
 * Use AutogradTrainer from training/autograd-trainer.ts for training.
 *
 * @module shgat/tests/persistence_test
 */

import { assertEquals, assertExists } from "@std/assert";
import { createSHGAT, generateDefaultToolEmbedding, type Node } from "../mod.ts";

// =============================================================================
// Test Fixtures
// =============================================================================

function createTestNodes(count: number = 3): Node[] {
  const nodes: Node[] = [];
  for (let i = 0; i < count; i++) {
    // Tool node (leaf)
    nodes.push({
      id: `tool-${i + 1}`,
      embedding: generateDefaultToolEmbedding(`tool-${i + 1}`, 1024),
      children: [],
      level: 0,
    });
    // Capability node (composite)
    nodes.push({
      id: `cap-${i + 1}`,
      embedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
      children: [`tool-${i + 1}`],
      level: 0, // buildGraph will compute actual level
    });
  }
  return nodes;
}

// =============================================================================
// Export Tests
// =============================================================================

Deno.test("Persistence - exportParams returns object", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);

  const params = shgat.exportParams();

  assertExists(params);
  assertEquals(typeof params, "object");
});

Deno.test("Persistence - exportParams is JSON serializable", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);

  const params = shgat.exportParams();
  const json = JSON.stringify(params);

  assertExists(json);
  assertEquals(typeof json, "string");

  // Should parse back
  const parsed = JSON.parse(json);
  assertExists(parsed);
});

Deno.test("Persistence - exportParams includes required keys", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);

  const params = shgat.exportParams();

  // Check for expected keys
  assertExists(params.levelParams, "Should have levelParams");
  assertExists(params.W_intent, "Should have W_intent");
});

// =============================================================================
// Import Tests
// =============================================================================

Deno.test("Persistence - importParams restores weights", () => {
  const nodes = createTestNodes(3);

  // Create first SHGAT
  const shgat1 = createSHGAT(nodes);

  // Export params
  const params = shgat1.exportParams();

  // Create fresh SHGAT and import
  const shgat2 = createSHGAT(nodes);
  shgat2.importParams(params);

  // Export from second should match
  const params2 = shgat2.exportParams();

  assertEquals(
    JSON.stringify(params.levelParams),
    JSON.stringify(params2.levelParams),
    "levelParams should match after import",
  );
});

Deno.test("Persistence - imported model produces same scores", () => {
  const nodes = createTestNodes(3);
  // Use deterministic intent
  const intent = Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.01) * 0.1);

  // Create first SHGAT
  const shgat1 = createSHGAT(nodes);

  // Get scores from original model
  const scores1 = shgat1.scoreNodes(intent, 1); // composites only

  // Export and import to new model
  const params = shgat1.exportParams();
  const shgat2 = createSHGAT(nodes);
  shgat2.importParams(params);

  // Get scores from imported model
  const scores2 = shgat2.scoreNodes(intent, 1);

  // Scores should be same length
  assertEquals(scores1.length, scores2.length);

  // Scores should match (same params = same scores)
  for (let i = 0; i < scores1.length; i++) {
    assertEquals(
      scores1[i].score.toFixed(6),
      scores2[i].score.toFixed(6),
      `Score for ${scores1[i].nodeId} should match`,
    );
  }
});

// =============================================================================
// Round-trip Tests
// =============================================================================

Deno.test("Persistence - JSON round-trip preserves params", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);

  // Export -> JSON -> Parse -> Import
  const exported = shgat.exportParams();
  const json = JSON.stringify(exported);
  const parsed = JSON.parse(json);

  const shgat2 = createSHGAT(nodes);
  shgat2.importParams(parsed);

  // Verify
  const exported2 = shgat2.exportParams();
  assertEquals(
    JSON.stringify(exported.levelParams),
    JSON.stringify(exported2.levelParams),
    "Params should survive JSON round-trip",
  );
});

Deno.test("Persistence - multiple round-trips preserve params", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);

  let params = shgat.exportParams();
  const originalJson = JSON.stringify(params);

  // Do multiple round-trips
  for (let i = 0; i < 3; i++) {
    const json = JSON.stringify(params);
    params = JSON.parse(json);

    const newShgat = createSHGAT(nodes);
    newShgat.importParams(params);
    params = newShgat.exportParams();
  }

  const finalJson = JSON.stringify(params);

  // levelParams should survive all round-trips
  const original = JSON.parse(originalJson);
  const final = JSON.parse(finalJson);
  assertEquals(
    JSON.stringify(original.levelParams),
    JSON.stringify(final.levelParams),
    "levelParams should survive multiple round-trips",
  );
});

// =============================================================================
// Simulated Storage Tests
// =============================================================================

Deno.test("Persistence - simulated file storage", () => {
  const nodes = createTestNodes(3);

  // Create and get params
  const shgat1 = createSHGAT(nodes);
  const params = shgat1.exportParams();

  // Simulate file save/load
  const fileContent = JSON.stringify(params);
  const loaded = JSON.parse(fileContent);

  // Restore to same instance
  const shgat2 = createSHGAT(nodes);
  shgat2.importParams(loaded);

  // Verify params were imported
  const exported2 = shgat2.exportParams();
  assertEquals(
    JSON.stringify((params as Record<string, unknown>).levelParams),
    JSON.stringify((exported2 as Record<string, unknown>).levelParams),
    "levelParams should survive file storage simulation",
  );

  // Verify model is usable
  const intent = Array.from({ length: 1024 }, (_, i) => Math.cos(i * 0.02) * 0.1);
  const scores = shgat2.scoreNodes(intent, 1);
  assertEquals(scores.length, 3, "Should return scores for all composites");
});

Deno.test("Persistence - simulated database storage", () => {
  const nodes = createTestNodes(3);

  // Create and get params
  const shgat1 = createSHGAT(nodes);
  const params = shgat1.exportParams();

  // Simulate database storage (often stores as string)
  const dbRecord = {
    id: "shgat-model-1",
    params: JSON.stringify(params),
    createdAt: new Date().toISOString(),
  };

  // Restore from "database"
  const loadedParams = JSON.parse(dbRecord.params);
  const shgat2 = createSHGAT(nodes);
  shgat2.importParams(loadedParams);

  // Verify params were imported
  const exported2 = shgat2.exportParams();
  assertEquals(
    JSON.stringify((params as Record<string, unknown>).levelParams),
    JSON.stringify((exported2 as Record<string, unknown>).levelParams),
    "levelParams should survive database storage simulation",
  );

  // Verify model is usable
  const intent = Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.03) * 0.1);
  const scores = shgat2.scoreNodes(intent, 1);
  assertEquals(scores.length, 3, "Should return scores for all composites");
});
