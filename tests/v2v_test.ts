/**
 * V→V Co-occurrence Phase Tests
 *
 * Tests for:
 * - buildCooccurrenceFromWorkflows: sparse matrix construction
 * - v2vEnrich: one-shot enrichment API
 * - VertexToVertexPhase: class-based forward/backward
 * - Chunked attention equivalence with dense attention
 *
 * Run: deno test -A --no-check lib/shgat-tf/tests/v2v_test.ts
 *
 * @module shgat-tf/tests/v2v_test
 */

import { assertEquals, assertGreater, assertLess, assertNotEquals } from "@std/assert";
import {
  buildCooccurrenceFromWorkflows,
  buildCooccurrenceMatrix,
  type CooccurrenceEntry,
  v2vEnrich,
  VertexToVertexPhase,
} from "../src/message-passing/vertex-to-vertex-phase.ts";

// ============================================================================
// Fixtures
// ============================================================================

/** Create a simple embedding: deterministic, normalized-ish */
function makeEmb(seed: number, dim = 32): number[] {
  const emb = Array.from({ length: dim }, (_, j) => Math.sin(seed * 1.7 + j * 0.3));
  const norm = Math.sqrt(emb.reduce((s, v) => s + v * v, 0));
  return emb.map((v) => v / norm);
}

/** 5 tools, 3 workflows */
function makeFixture() {
  const toolIds = ["toolA", "toolB", "toolC", "toolD", "toolE"];
  const toolIdToIdx = new Map(toolIds.map((id, i) => [id, i]));
  const H = toolIds.map((_, i) => makeEmb(i));

  // Workflow 1: A, B, C co-occur
  // Workflow 2: B, D co-occur
  // Workflow 3: A, B, D co-occur
  const workflows = [
    ["toolA", "toolB", "toolC"],
    ["toolB", "toolD"],
    ["toolA", "toolB", "toolD"],
  ];

  return { toolIds, toolIdToIdx, H, workflows };
}

// ============================================================================
// buildCooccurrenceFromWorkflows
// ============================================================================

Deno.test("buildCooccurrenceFromWorkflows: returns bidirectional edges", () => {
  const { toolIdToIdx, workflows } = makeFixture();
  const entries = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);

  // Each unique pair produces 2 entries (forward + backward)
  // Pairs: (A,B), (A,C), (B,C), (B,D), (A,D) = 5 pairs × 2 = 10 entries
  assertEquals(entries.length, 10);

  // Check bidirectionality: for every (a,b), there's a (b,a)
  const edgeSet = new Set(entries.map((e) => `${e.from}:${e.to}`));
  for (const entry of entries) {
    const reverse = `${entry.to}:${entry.from}`;
    assertEquals(edgeSet.has(reverse), true, `Missing reverse edge for ${entry.from}→${entry.to}`);
  }
});

Deno.test("buildCooccurrenceFromWorkflows: weights reflect co-occurrence count", () => {
  const { toolIdToIdx, workflows } = makeFixture();
  const entries = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);

  // A-B co-occur in workflows 1 and 3 → count=2 → weight=log2(3)≈1.585
  // A-C co-occur in workflow 1 only → count=1 → weight=log2(2)=1.0
  const idxA = toolIdToIdx.get("toolA")!;
  const idxB = toolIdToIdx.get("toolB")!;
  const idxC = toolIdToIdx.get("toolC")!;

  const abEntry = entries.find((e) => e.from === idxA && e.to === idxB);
  const acEntry = entries.find((e) => e.from === idxA && e.to === idxC);

  assertGreater(
    abEntry!.weight,
    acEntry!.weight,
    "A-B (2 workflows) should have higher weight than A-C (1 workflow)",
  );
  assertLess(
    Math.abs(acEntry!.weight - 1.0),
    0.01,
    "Single co-occurrence should have weight ≈ log2(2) = 1.0",
  );
});

Deno.test("buildCooccurrenceFromWorkflows: ignores unknown tools", () => {
  const { toolIdToIdx } = makeFixture();
  const workflows = [["toolA", "unknown_tool", "toolB"]];
  const entries = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);

  // Only A-B pair, "unknown_tool" is skipped
  assertEquals(entries.length, 2); // forward + backward
});

Deno.test("buildCooccurrenceFromWorkflows: empty workflows produce no edges", () => {
  const { toolIdToIdx } = makeFixture();
  const entries = buildCooccurrenceFromWorkflows([], toolIdToIdx);
  assertEquals(entries.length, 0);
});

Deno.test("buildCooccurrenceFromWorkflows: single-tool workflows produce no edges", () => {
  const { toolIdToIdx } = makeFixture();
  const entries = buildCooccurrenceFromWorkflows([["toolA"], ["toolB"]], toolIdToIdx);
  assertEquals(entries.length, 0);
});

// ============================================================================
// v2vEnrich
// ============================================================================

Deno.test("v2vEnrich: output shape matches input", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);
  const enriched = v2vEnrich(H, cooc);

  assertEquals(enriched.length, H.length);
  assertEquals(enriched[0].length, H[0].length);
});

Deno.test("v2vEnrich: connected tools change, isolated tools don't", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);
  const enriched = v2vEnrich(H, cooc);

  // toolE (index 4) is not in any workflow → should be unchanged
  const idxE = toolIdToIdx.get("toolE")!;
  let eDelta = 0;
  for (let d = 0; d < H[idxE].length; d++) {
    eDelta += Math.abs(enriched[idxE][d] - H[idxE][d]);
  }
  assertLess(eDelta, 1e-6, "Isolated tool should not change");

  // toolA (index 0) is in 2 workflows → should change
  const idxA = toolIdToIdx.get("toolA")!;
  let aDelta = 0;
  for (let d = 0; d < H[idxA].length; d++) {
    aDelta += Math.abs(enriched[idxA][d] - H[idxA][d]);
  }
  assertGreater(aDelta, 1e-4, "Connected tool should change after V→V enrichment");
});

Deno.test("v2vEnrich: output embeddings are normalized (unit sphere)", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);
  const enriched = v2vEnrich(H, cooc);

  for (let i = 0; i < enriched.length; i++) {
    const norm = Math.sqrt(enriched[i].reduce((s, v) => s + v * v, 0));
    assertLess(Math.abs(norm - 1.0), 0.01, `Tool ${i} embedding norm should be ≈1.0, got ${norm}`);
  }
});

Deno.test("v2vEnrich: residualWeight=0 means no change", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);
  const enriched = v2vEnrich(H, cooc, { residualWeight: 0 });

  // With residualWeight=0, enriched = H + 0*aggregated = H (before normalization)
  // Since original H is already normalized, they should be identical
  for (let i = 0; i < H.length; i++) {
    for (let d = 0; d < H[i].length; d++) {
      assertLess(
        Math.abs(enriched[i][d] - H[i][d]),
        1e-5,
        `residualWeight=0 should not change embeddings`,
      );
    }
  }
});

Deno.test("v2vEnrich: empty co-occurrence returns original embeddings", () => {
  const { H } = makeFixture();
  const enriched = v2vEnrich(H, []);

  for (let i = 0; i < H.length; i++) {
    for (let d = 0; d < H[i].length; d++) {
      assertLess(Math.abs(enriched[i][d] - H[i][d]), 1e-6);
    }
  }
});

Deno.test("v2vEnrich: deterministic — same input gives same output", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);
  const enriched1 = v2vEnrich(H, cooc);
  const enriched2 = v2vEnrich(H, cooc);

  for (let i = 0; i < enriched1.length; i++) {
    for (let d = 0; d < enriched1[i].length; d++) {
      assertLess(Math.abs(enriched1[i][d] - enriched2[i][d]), 1e-10, "V→V should be deterministic");
    }
  }
});

Deno.test("v2vEnrich: higher residualWeight produces larger changes", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);

  const low = v2vEnrich(H, cooc, { residualWeight: 0.1 });
  const high = v2vEnrich(H, cooc, { residualWeight: 0.9 });

  let deltaLow = 0, deltaHigh = 0;
  for (let i = 0; i < H.length; i++) {
    for (let d = 0; d < H[i].length; d++) {
      deltaLow += Math.abs(low[i][d] - H[i][d]);
      deltaHigh += Math.abs(high[i][d] - H[i][d]);
    }
  }

  assertGreater(
    deltaHigh,
    deltaLow,
    "Higher residualWeight should produce larger embedding changes",
  );
});

// ============================================================================
// buildCooccurrenceMatrix (legacy PriorPattern API)
// ============================================================================

Deno.test("buildCooccurrenceMatrix: converts PriorPatterns to sparse entries", () => {
  const toolIndex = new Map([["t0", 0], ["t1", 1], ["t2", 2]]);
  const patterns = [
    { from: "t0", to: "t1", weight: 1.0, frequency: 5 },
    { from: "t1", to: "t2", weight: 2.0, frequency: 3 },
  ];

  const entries = buildCooccurrenceMatrix(patterns, toolIndex);

  // 2 patterns × 2 (bidirectional) = 4 entries
  assertEquals(entries.length, 4);

  // Lower weight → higher co-occurrence
  const t0t1 = entries.find((e) => e.from === 0 && e.to === 1);
  const t1t2 = entries.find((e) => e.from === 1 && e.to === 2);
  assertGreater(
    t0t1!.weight,
    t1t2!.weight,
    "Lower PriorPattern weight should give higher coocWeight",
  );
});

// ============================================================================
// VertexToVertexPhase class API
// ============================================================================

Deno.test("VertexToVertexPhase: forward returns attention weights for debugging", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);

  const phase = new VertexToVertexPhase({ residualWeight: 0.3, useAttention: true });
  const result = phase.forward(H, cooc);

  assertGreater(result.attentionWeights.length, 0, "Should have attention weights");
  assertEquals(result.embeddings.length, H.length);

  // Attention weights should sum to ~1 per source tool
  const perTool = new Map<number, number>();
  for (const w of result.attentionWeights) {
    perTool.set(w.from, (perTool.get(w.from) ?? 0) + w.weight);
  }
  for (const [toolIdx, total] of perTool) {
    assertLess(
      Math.abs(total - 1.0),
      0.01,
      `Attention weights for tool ${toolIdx} should sum to ~1.0, got ${total}`,
    );
  }
});

Deno.test("VertexToVertexPhase: simple weighted sum mode (useAttention=false)", () => {
  const { toolIdToIdx, H, workflows } = makeFixture();
  const cooc = buildCooccurrenceFromWorkflows(workflows, toolIdToIdx);

  const phase = new VertexToVertexPhase({ residualWeight: 0.3, useAttention: false });
  const result = phase.forward(H, cooc);

  assertEquals(result.embeddings.length, H.length);
  // No attention weights in simple mode
  assertEquals(result.attentionWeights.length, 0);
});
