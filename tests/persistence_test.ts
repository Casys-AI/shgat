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
import { createSHGATFromCapabilities } from "../mod.ts";

// =============================================================================
// Test Fixtures
// =============================================================================

function createTestCapabilities(count: number = 3) {
  return Array.from({ length: count }, (_, i) => ({
    id: `cap-${i + 1}`,
    embedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
    toolsUsed: [`tool-${i + 1}`],
    successRate: 0.8,
  }));
}

// =============================================================================
// Export Tests
// =============================================================================

Deno.test("Persistence - exportParams returns object", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  const params = shgat.exportParams();

  assertExists(params);
  assertEquals(typeof params, "object");
});

Deno.test("Persistence - exportParams is JSON serializable", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  const params = shgat.exportParams();
  const json = JSON.stringify(params);

  assertExists(json);
  assertEquals(typeof json, "string");

  // Should parse back
  const parsed = JSON.parse(json);
  assertExists(parsed);
});

Deno.test("Persistence - exportParams includes required keys", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  const params = shgat.exportParams();

  // Check for expected keys
  assertExists(params.levelParams, "Should have levelParams");
  assertExists(params.W_intent, "Should have W_intent");
});

// =============================================================================
// Import Tests
// =============================================================================

Deno.test("Persistence - importParams restores weights", () => {
  const caps = createTestCapabilities(3);

  // Create first SHGAT
  const shgat1 = createSHGATFromCapabilities(caps);

  // Export params
  const params = shgat1.exportParams();

  // Create fresh SHGAT and import
  const shgat2 = createSHGATFromCapabilities(caps);
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
  const caps = createTestCapabilities(3);
  // Use deterministic intent
  const intent = Array.from({ length: 1024 }, (_, i) => Math.sin(i * 0.01) * 0.1);

  // Create first SHGAT
  const shgat1 = createSHGATFromCapabilities(caps);

  // Get scores from original model
  const scores1 = shgat1.scoreAllCapabilities(intent, []);

  // Export and import to new model
  const params = shgat1.exportParams();
  const shgat2 = createSHGATFromCapabilities(caps);
  shgat2.importParams(params);

  // Get scores from imported model
  const scores2 = shgat2.scoreAllCapabilities(intent, []);

  // Scores should be same length
  assertEquals(scores1.length, scores2.length);

  // Scores should match (same params = same scores)
  for (let i = 0; i < scores1.length; i++) {
    assertEquals(
      scores1[i].score.toFixed(6),
      scores2[i].score.toFixed(6),
      `Score for ${scores1[i].capabilityId} should match`,
    );
  }
});

// =============================================================================
// Round-trip Tests
// =============================================================================

Deno.test("Persistence - JSON round-trip preserves params", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  // Export → JSON → Parse → Import
  const exported = shgat.exportParams();
  const json = JSON.stringify(exported);
  const parsed = JSON.parse(json);

  const shgat2 = createSHGATFromCapabilities(caps);
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
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  let params = shgat.exportParams();
  const originalJson = JSON.stringify(params);

  // Do multiple round-trips
  for (let i = 0; i < 3; i++) {
    const json = JSON.stringify(params);
    params = JSON.parse(json);

    const newShgat = createSHGATFromCapabilities(caps);
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
  const caps = createTestCapabilities(3);

  // Create and get params
  const shgat1 = createSHGATFromCapabilities(caps);
  const params = shgat1.exportParams();

  // Simulate file save/load
  const fileContent = JSON.stringify(params);
  const loaded = JSON.parse(fileContent);

  // Restore to same instance
  const shgat2 = createSHGATFromCapabilities(caps);
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
  const scores = shgat2.scoreAllCapabilities(intent, []);
  assertEquals(scores.length, caps.length, "Should return scores for all capabilities");
});

Deno.test("Persistence - simulated database storage", () => {
  const caps = createTestCapabilities(3);

  // Create and get params
  const shgat1 = createSHGATFromCapabilities(caps);
  const params = shgat1.exportParams();

  // Simulate database storage (often stores as string)
  const dbRecord = {
    id: "shgat-model-1",
    params: JSON.stringify(params),
    createdAt: new Date().toISOString(),
  };

  // Restore from "database"
  const loadedParams = JSON.parse(dbRecord.params);
  const shgat2 = createSHGATFromCapabilities(caps);
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
  const scores = shgat2.scoreAllCapabilities(intent, []);
  assertEquals(scores.length, caps.length, "Should return scores for all capabilities");
});
