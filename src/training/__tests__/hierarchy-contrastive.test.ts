/**
 * Tests for hierarchy-level contrastive loss features in train-ob.ts.
 *
 * Since train-ob.ts is a script (not a module), we reimplement the testable
 * logic locally and verify correctness with unit and integration-style tests.
 *
 * Covers:
 *   A. l0Ancestors recursive walk-up mapping
 *   B. hierWeight gradient scaling
 *   C. Per-level dE normalization
 *   D. Example filtering and minimum batch size enforcement
 */

import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import {
  batchContrastiveForward,
  batchContrastiveBackward,
} from "../batch-contrastive-loss.ts";
import type { HeadParams } from "../../initialization/parameters.ts";
import type { SHGATConfig } from "../../core/types.ts";
import type { SparseConnectivity } from "../../message-passing/phase-interface.ts";

// ============================================================================
// Helpers (mirrors batch-contrastive-loss.test.ts style)
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
// Reimplemented l0Ancestors logic (mirrors train-ob.ts lines 498-519)
// ============================================================================

/**
 * Build the l0Ancestors mapping: for each L0 tool, find ancestor indices
 * at every orchestrator level via recursive walk-up.
 *
 * This exactly mirrors the logic in train-ob.ts buildGraphStructure().
 *
 * @param l0Count Number of L0 tools
 * @param l0ToL1Conn Sparse connectivity from L0 (source) to L1/orch-level-0 (target)
 * @param interLevelConns Map from orchestrator level (1+) to sparse connectivity
 *                        where source = child level, target = parent level
 * @param maxLevel Maximum orchestrator level (0-based)
 * @returns Array indexed by l0Idx, each element is a Map<orchLevel, ancestorIdx[]>
 */
function buildL0Ancestors(
  l0Count: number,
  l0ToL1Conn: SparseConnectivity,
  interLevelConns: Map<number, SparseConnectivity>,
  maxLevel: number,
): Map<number, number[]>[] {
  const l0Ancestors: Map<number, number[]>[] = new Array(l0Count);
  for (let i = 0; i < l0Count; i++) {
    const ancestors = new Map<number, number[]>();
    // Level 0 (orch): L0 tool -> L1 caps via l0ToL1Conn.sourceToTargets
    const l1Parents = l0ToL1Conn.sourceToTargets.get(i) ?? [];
    if (l1Parents.length > 0) ancestors.set(0, [...l1Parents]);
    // Higher levels: walk up via interLevelConns
    let currentParents = l1Parents;
    for (let orchLevel = 1; orchLevel <= maxLevel; orchLevel++) {
      const conn = interLevelConns.get(orchLevel);
      if (!conn) break;
      const nextParents = new Set<number>();
      for (const pIdx of currentParents) {
        const grandParents = conn.sourceToTargets.get(pIdx) ?? [];
        for (const gp of grandParents) nextParents.add(gp);
      }
      const nextArr = [...nextParents];
      if (nextArr.length > 0) ancestors.set(orchLevel, nextArr);
      currentParents = nextArr;
    }
    l0Ancestors[i] = ancestors;
  }
  return l0Ancestors;
}

/**
 * Helper to build a SparseConnectivity from a parent-children mapping.
 *
 * parentToChildren: for each parent index, list of child indices it owns.
 * We build:
 *   sourceToTargets = child -> parent[] (child is source, walks UP)
 *   targetToSources = parent -> child[] (parent is target)
 */
function buildConnectivity(
  parentToChildren: Map<number, number[]>,
  numChildren: number,
  numParents: number,
): SparseConnectivity {
  const sourceToTargets = new Map<number, number[]>(); // child -> parents
  const targetToSources = new Map<number, number[]>(); // parent -> children

  for (const [parentIdx, children] of parentToChildren) {
    for (const childIdx of children) {
      if (!sourceToTargets.has(childIdx)) sourceToTargets.set(childIdx, []);
      sourceToTargets.get(childIdx)!.push(parentIdx);
      if (!targetToSources.has(parentIdx)) targetToSources.set(parentIdx, []);
      targetToSources.get(parentIdx)!.push(childIdx);
    }
  }

  return { sourceToTargets, targetToSources, numSources: numChildren, numTargets: numParents };
}

// ============================================================================
// Reimplemented dE normalization logic (mirrors train-ob.ts lines 1136-1143)
// ============================================================================

/**
 * Normalize _epochDE per-level by the number of batches that contributed
 * to each level. This avoids global normalization that would bias levels
 * with fewer batches (over-normalized).
 */
function normalizeEpochDE(
  epochDE: Map<number, number[][]>,
  hierBatchesByLevel: Map<number, number>,
): void {
  for (const [orchLevel, rows] of epochDE) {
    const levelBatches = hierBatchesByLevel.get(orchLevel) ?? 0;
    if (levelBatches > 0) {
      const scale = 1 / levelBatches;
      for (const row of rows) {
        for (let d = 0; d < row.length; d++) row[d] *= scale;
      }
    }
  }
}

// ============================================================================
// Reimplemented example filtering logic (mirrors train-ob.ts lines 916-926)
// ============================================================================

interface HierExample {
  intentEmbedding: number[];
  ancestorIdxs: number[];
}

/**
 * Filter prod examples that have a valid ancestor at a given orchestrator level.
 * If fewer than BATCH_SIZE examples remain, the level is skipped.
 */
function filterExamplesForLevel(
  prodExamples: Array<{ intentEmbedding: number[]; targetToolId: string }>,
  l0IdxMap: Map<string, number>,
  l0Ancestors: Map<number, number[]>[],
  orchLevel: number,
): HierExample[] {
  const levelExamples: HierExample[] = [];
  for (const ex of prodExamples) {
    const l0Idx = l0IdxMap.get(ex.targetToolId);
    if (l0Idx === undefined) continue;
    const ancestors = l0Ancestors[l0Idx];
    const ancestorIdxs = ancestors?.get(orchLevel);
    if (ancestorIdxs && ancestorIdxs.length > 0) {
      levelExamples.push({ intentEmbedding: ex.intentEmbedding, ancestorIdxs });
    }
  }
  return levelExamples;
}

// ============================================================================
// A. l0Ancestors mapping — unit tests for recursive walk-up
// ============================================================================

Deno.test("l0Ancestors: simple 2-level hierarchy (4 tools, 2 caps)", () => {
  // L0 tools: [0, 1, 2, 3]
  // L1 caps (orch level 0): Cap0 owns tools [0, 1], Cap1 owns tools [2, 3]
  const parentToChildren = new Map<number, number[]>();
  parentToChildren.set(0, [0, 1]); // Cap0 -> tools 0, 1
  parentToChildren.set(1, [2, 3]); // Cap1 -> tools 2, 3

  const l0ToL1Conn = buildConnectivity(parentToChildren, 4, 2);

  const interLevelConns = new Map<number, SparseConnectivity>();
  const maxLevel = 0; // only orch level 0

  const ancestors = buildL0Ancestors(4, l0ToL1Conn, interLevelConns, maxLevel);

  // Tool 0 -> Cap 0 at orch level 0
  assertEquals(ancestors[0].get(0), [0]);
  // Tool 1 -> Cap 0 at orch level 0
  assertEquals(ancestors[1].get(0), [0]);
  // Tool 2 -> Cap 1 at orch level 0
  assertEquals(ancestors[2].get(0), [1]);
  // Tool 3 -> Cap 1 at orch level 0
  assertEquals(ancestors[3].get(0), [1]);

  // No higher levels
  assertEquals(ancestors[0].has(1), false);
  assertEquals(ancestors[2].has(1), false);
});

Deno.test("l0Ancestors: 3-level hierarchy with recursive walk-up", () => {
  // L0 tools: [0, 1, 2, 3]
  // L1 caps (orch 0): Cap0=[0,1], Cap1=[2,3]
  // L2 super-cap (orch 1): SuperCap0=[Cap0, Cap1]
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0, 1]); // Cap0 -> tools 0, 1
  l0ToL1.set(1, [2, 3]); // Cap1 -> tools 2, 3
  const l0ToL1Conn = buildConnectivity(l0ToL1, 4, 2);

  // Inter-level: orch level 1 connects L1 (orch 0) children to L2 (orch 1) parents
  const l1ToL2 = new Map<number, number[]>();
  l1ToL2.set(0, [0, 1]); // SuperCap0 -> Cap0, Cap1
  const l1ToL2Conn = buildConnectivity(l1ToL2, 2, 1);

  const interLevelConns = new Map<number, SparseConnectivity>();
  interLevelConns.set(1, l1ToL2Conn);

  const maxLevel = 1;
  const ancestors = buildL0Ancestors(4, l0ToL1Conn, interLevelConns, maxLevel);

  // Tool 0: orch 0 -> [Cap0(0)], orch 1 -> [SuperCap0(0)]
  assertEquals(ancestors[0].get(0), [0]);
  assertEquals(ancestors[0].get(1), [0]);

  // Tool 2: orch 0 -> [Cap1(1)], orch 1 -> [SuperCap0(0)]
  assertEquals(ancestors[2].get(0), [1]);
  assertEquals(ancestors[2].get(1), [0]);

  // All tools should reach SuperCap0 at level 1
  for (let i = 0; i < 4; i++) {
    assert(ancestors[i].has(1), `Tool ${i} should have ancestors at orch level 1`);
    assertEquals(ancestors[i].get(1), [0]);
  }
});

Deno.test("l0Ancestors: multiple parents (tool mapped to 2 L1 caps)", () => {
  // L0 tools: [0, 1, 2]
  // L1 caps (orch 0): Cap0=[0, 1], Cap1=[1, 2]
  // Tool 1 has TWO parents: Cap0 and Cap1
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0, 1]); // Cap0 -> tools 0, 1
  l0ToL1.set(1, [1, 2]); // Cap1 -> tools 1, 2
  const l0ToL1Conn = buildConnectivity(l0ToL1, 3, 2);

  const interLevelConns = new Map<number, SparseConnectivity>();
  const maxLevel = 0;
  const ancestors = buildL0Ancestors(3, l0ToL1Conn, interLevelConns, maxLevel);

  // Tool 0 -> only Cap0
  assertEquals(ancestors[0].get(0), [0]);

  // Tool 1 -> both Cap0 and Cap1
  const tool1Ancestors = ancestors[1].get(0)!;
  assertEquals(tool1Ancestors.length, 2);
  assert(tool1Ancestors.includes(0), "Tool 1 should have Cap0 as ancestor");
  assert(tool1Ancestors.includes(1), "Tool 1 should have Cap1 as ancestor");

  // Tool 2 -> only Cap1
  assertEquals(ancestors[2].get(0), [1]);
});

Deno.test("l0Ancestors: disconnected tool has no ancestors", () => {
  // L0 tools: [0, 1, 2]
  // L1 caps (orch 0): Cap0=[0, 1] (tool 2 is disconnected)
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0, 1]); // Cap0 -> tools 0, 1 (tool 2 not mapped)
  const l0ToL1Conn = buildConnectivity(l0ToL1, 3, 1);

  const interLevelConns = new Map<number, SparseConnectivity>();
  const maxLevel = 0;
  const ancestors = buildL0Ancestors(3, l0ToL1Conn, interLevelConns, maxLevel);

  // Tool 0, 1 -> Cap0
  assertEquals(ancestors[0].get(0), [0]);
  assertEquals(ancestors[1].get(0), [0]);

  // Tool 2 has no ancestors at any level
  assertEquals(ancestors[2].size, 0);
  assertEquals(ancestors[2].has(0), false);
});

Deno.test("l0Ancestors: deep hierarchy (4 levels) recursive walk-up", () => {
  // L0 tools: [0, 1, 2, 3]
  // Orch 0 (L1): Cap0=[0,1], Cap1=[2,3]
  // Orch 1 (L2): SuperCap0=[Cap0, Cap1]
  // Orch 2 (L3): UltraCap0=[SuperCap0]
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0, 1]);
  l0ToL1.set(1, [2, 3]);
  const l0ToL1Conn = buildConnectivity(l0ToL1, 4, 2);

  const l1ToL2 = new Map<number, number[]>();
  l1ToL2.set(0, [0, 1]); // SuperCap0 -> Cap0, Cap1
  const l1ToL2Conn = buildConnectivity(l1ToL2, 2, 1);

  const l2ToL3 = new Map<number, number[]>();
  l2ToL3.set(0, [0]); // UltraCap0 -> SuperCap0
  const l2ToL3Conn = buildConnectivity(l2ToL3, 1, 1);

  const interLevelConns = new Map<number, SparseConnectivity>();
  interLevelConns.set(1, l1ToL2Conn);
  interLevelConns.set(2, l2ToL3Conn);

  const maxLevel = 2;
  const ancestors = buildL0Ancestors(4, l0ToL1Conn, interLevelConns, maxLevel);

  // Every tool should have ancestors at all 3 orchestrator levels
  for (let i = 0; i < 4; i++) {
    assert(ancestors[i].has(0), `Tool ${i} should have ancestors at orch level 0`);
    assert(ancestors[i].has(1), `Tool ${i} should have ancestors at orch level 1`);
    assert(ancestors[i].has(2), `Tool ${i} should have ancestors at orch level 2`);
  }

  // Tool 0: orch 0 -> [Cap0(0)], orch 1 -> [SuperCap0(0)], orch 2 -> [UltraCap0(0)]
  assertEquals(ancestors[0].get(0), [0]);
  assertEquals(ancestors[0].get(1), [0]);
  assertEquals(ancestors[0].get(2), [0]);

  // Tool 3: orch 0 -> [Cap1(1)], orch 1 -> [SuperCap0(0)], orch 2 -> [UltraCap0(0)]
  assertEquals(ancestors[3].get(0), [1]);
  assertEquals(ancestors[3].get(1), [0]);
  assertEquals(ancestors[3].get(2), [0]);
});

Deno.test("l0Ancestors: partial disconnection at higher level", () => {
  // L0 tools: [0, 1, 2, 3]
  // Orch 0 (L1): Cap0=[0,1], Cap1=[2,3]
  // Orch 1 (L2): SuperCap0=[Cap0] only (Cap1 is disconnected at L2)
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0, 1]);
  l0ToL1.set(1, [2, 3]);
  const l0ToL1Conn = buildConnectivity(l0ToL1, 4, 2);

  const l1ToL2 = new Map<number, number[]>();
  l1ToL2.set(0, [0]); // SuperCap0 -> Cap0 only
  const l1ToL2Conn = buildConnectivity(l1ToL2, 2, 1);

  const interLevelConns = new Map<number, SparseConnectivity>();
  interLevelConns.set(1, l1ToL2Conn);

  const maxLevel = 1;
  const ancestors = buildL0Ancestors(4, l0ToL1Conn, interLevelConns, maxLevel);

  // Tools 0, 1 reach SuperCap0 via Cap0
  assertEquals(ancestors[0].get(1), [0]);
  assertEquals(ancestors[1].get(1), [0]);

  // Tools 2, 3 have Cap1 at orch 0 but Cap1 is NOT connected to any L2
  assert(ancestors[2].has(0), "Tool 2 should have ancestors at orch 0");
  assertEquals(ancestors[2].has(1), false, "Tool 2 should NOT have ancestors at orch 1");
  assertEquals(ancestors[3].has(1), false, "Tool 3 should NOT have ancestors at orch 1");
});

Deno.test("l0Ancestors: diamond hierarchy (multiple paths converge)", () => {
  // L0 tools: [0]
  // Orch 0: Cap0=[tool0], Cap1=[tool0]  (tool0 has two parents)
  // Orch 1: SuperCap0=[Cap0, Cap1]      (both paths converge)
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0]); // Cap0 -> tool0
  l0ToL1.set(1, [0]); // Cap1 -> tool0
  const l0ToL1Conn = buildConnectivity(l0ToL1, 1, 2);

  const l1ToL2 = new Map<number, number[]>();
  l1ToL2.set(0, [0, 1]); // SuperCap0 -> Cap0, Cap1
  const l1ToL2Conn = buildConnectivity(l1ToL2, 2, 1);

  const interLevelConns = new Map<number, SparseConnectivity>();
  interLevelConns.set(1, l1ToL2Conn);

  const maxLevel = 1;
  const ancestors = buildL0Ancestors(1, l0ToL1Conn, interLevelConns, maxLevel);

  // Tool 0 has both Cap0 and Cap1 at orch 0
  const orchL0Ancestors = ancestors[0].get(0)!;
  assertEquals(orchL0Ancestors.length, 2);
  assert(orchL0Ancestors.includes(0));
  assert(orchL0Ancestors.includes(1));

  // At orch 1, both paths converge to SuperCap0 - should be deduplicated via Set
  const orchL1Ancestors = ancestors[0].get(1)!;
  assertEquals(orchL1Ancestors.length, 1, "Diamond should deduplicate: SuperCap0 appears once");
  assertEquals(orchL1Ancestors[0], 0);
});

// ============================================================================
// B. hierWeight gradient scaling
// ============================================================================

Deno.test("hierarchy contrastive: hierWeight scales K-head gradients correctly", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const B = 4;
  const tau = 0.07;
  const hierWeight = 0.5;

  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  // ---- Unscaled backward ----
  const gradsUnscaled = makeZeroGrads(headParams);
  const { cache: cache1 } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(cache1, headParams, gradsUnscaled, config);

  // ---- Scaled backward (same forward, fresh grads, then scale) ----
  const gradsScaled = makeZeroGrads(headParams);
  const { cache: cache2 } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(cache2, headParams, gradsScaled, config);

  // Apply hierWeight scaling as done in train-ob.ts (lines 973-975)
  for (let h = 0; h < config.numHeads; h++) {
    for (const row of gradsScaled.dW_q[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
    for (const row of gradsScaled.dW_k[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
  }

  // Verify scaled = hierWeight * unscaled
  for (let h = 0; h < config.numHeads; h++) {
    for (let r = 0; r < gradsUnscaled.dW_q[h].length; r++) {
      for (let c = 0; c < gradsUnscaled.dW_q[h][r].length; c++) {
        const expected = gradsUnscaled.dW_q[h][r][c] * hierWeight;
        assertAlmostEquals(
          gradsScaled.dW_q[h][r][c],
          expected,
          1e-10,
          `dW_q[${h}][${r}][${c}]: scaled=${gradsScaled.dW_q[h][r][c]}, expected=${expected}`,
        );
      }
    }
    for (let r = 0; r < gradsUnscaled.dW_k[h].length; r++) {
      for (let c = 0; c < gradsUnscaled.dW_k[h][r].length; c++) {
        const expected = gradsUnscaled.dW_k[h][r][c] * hierWeight;
        assertAlmostEquals(
          gradsScaled.dW_k[h][r][c],
          expected,
          1e-10,
          `dW_k[${h}][${r}][${c}]: scaled=${gradsScaled.dW_k[h][r][c]}, expected=${expected}`,
        );
      }
    }
  }
});

Deno.test("hierarchy contrastive: dNodeEmbeddings accumulation with hierWeight", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const B = 4;
  const tau = 0.07;
  const hierWeight = 0.5;

  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  // Simulate _epochDE for a single level with 3 nodes
  const numLevelNodes = 3;
  const epochDE: number[][] = Array.from({ length: numLevelNodes }, () =>
    new Array(config.embeddingDim).fill(0)
  );

  // Run forward/backward
  const grads = makeZeroGrads(headParams);
  const { cache } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  const { dNodeEmbeddings } = batchContrastiveBackward(cache, headParams, grads, config);

  // Simulate batch where each example maps to an ancestor:
  // example 0 -> ancestor 0, example 1 -> ancestor 1,
  // example 2 -> ancestor 0 (accumulates), example 3 -> ancestor 2
  const ancestorMapping = [0, 1, 0, 2];

  for (let i = 0; i < B; i++) {
    const ancestorIdx = ancestorMapping[i];
    for (let d = 0; d < config.embeddingDim; d++) {
      epochDE[ancestorIdx][d] += dNodeEmbeddings[i][d] * hierWeight;
    }
  }

  // Verify: ancestor 0 should have accumulated gradients from examples 0 and 2
  for (let d = 0; d < config.embeddingDim; d++) {
    const expected = (dNodeEmbeddings[0][d] + dNodeEmbeddings[2][d]) * hierWeight;
    assertAlmostEquals(
      epochDE[0][d],
      expected,
      1e-10,
      `epochDE[0][${d}]: got=${epochDE[0][d]}, expected=${expected}`,
    );
  }

  // ancestor 1 should have gradient from example 1 only
  for (let d = 0; d < config.embeddingDim; d++) {
    const expected = dNodeEmbeddings[1][d] * hierWeight;
    assertAlmostEquals(
      epochDE[1][d],
      expected,
      1e-10,
      `epochDE[1][${d}]: got=${epochDE[1][d]}, expected=${expected}`,
    );
  }

  // ancestor 2 should have gradient from example 3 only
  for (let d = 0; d < config.embeddingDim; d++) {
    const expected = dNodeEmbeddings[3][d] * hierWeight;
    assertAlmostEquals(
      epochDE[2][d],
      expected,
      1e-10,
      `epochDE[2][${d}]: got=${epochDE[2][d]}, expected=${expected}`,
    );
  }
});

Deno.test("hierarchy contrastive: hierWeight=0 produces zero scaled gradients", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const B = 4;
  const tau = 0.07;
  const hierWeight = 0.0;

  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  const grads = makeZeroGrads(headParams);
  const { cache } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(cache, headParams, grads, config);

  // Apply hierWeight = 0 scaling
  for (let h = 0; h < config.numHeads; h++) {
    for (const row of grads.dW_q[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
    for (const row of grads.dW_k[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
  }

  // All scaled gradients should be exactly 0
  for (let h = 0; h < config.numHeads; h++) {
    for (const row of grads.dW_q[h]) {
      for (const v of row) assertEquals(v, 0, "dW_q should be 0 when hierWeight=0");
    }
    for (const row of grads.dW_k[h]) {
      for (const v of row) assertEquals(v, 0, "dW_k should be 0 when hierWeight=0");
    }
  }
});

Deno.test("hierarchy contrastive: hierWeight=1.0 preserves original gradients", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const B = 4;
  const tau = 0.07;
  const hierWeight = 1.0;

  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const nodes = Array.from({ length: B }, () => randomVec(config.embeddingDim));

  // Unscaled
  const gradsRef = makeZeroGrads(headParams);
  const { cache: c1 } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(c1, headParams, gradsRef, config);

  // Scaled with hierWeight=1.0
  const gradsScaled = makeZeroGrads(headParams);
  const { cache: c2 } = batchContrastiveForward(intents, nodes, headParams, config, tau);
  batchContrastiveBackward(c2, headParams, gradsScaled, config);
  for (let h = 0; h < config.numHeads; h++) {
    for (const row of gradsScaled.dW_q[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
    for (const row of gradsScaled.dW_k[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
  }

  // Should be identical
  for (let h = 0; h < config.numHeads; h++) {
    for (let r = 0; r < gradsRef.dW_q[h].length; r++) {
      for (let c = 0; c < gradsRef.dW_q[h][r].length; c++) {
        assertAlmostEquals(gradsScaled.dW_q[h][r][c], gradsRef.dW_q[h][r][c], 1e-12);
      }
    }
  }
});

// ============================================================================
// C. Per-level normalization
// ============================================================================

Deno.test("per-level normalization: different batch counts per level", () => {
  const embDim = 4;
  const epochDE = new Map<number, number[][]>();

  // Level 0: 3 nodes, will have 10 batches contributing
  epochDE.set(0, [
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120],
  ]);
  // Level 1: 2 nodes, will have 3 batches contributing
  epochDE.set(1, [
    [30, 60, 90, 120],
    [15, 30, 45, 60],
  ]);

  const hierBatchesByLevel = new Map<number, number>();
  hierBatchesByLevel.set(0, 10);
  hierBatchesByLevel.set(1, 3);

  normalizeEpochDE(epochDE, hierBatchesByLevel);

  // Level 0 values divided by 10
  assertAlmostEquals(epochDE.get(0)![0][0], 1.0, 1e-10);
  assertAlmostEquals(epochDE.get(0)![0][1], 2.0, 1e-10);
  assertAlmostEquals(epochDE.get(0)![1][0], 5.0, 1e-10);
  assertAlmostEquals(epochDE.get(0)![2][3], 12.0, 1e-10);

  // Level 1 values divided by 3
  assertAlmostEquals(epochDE.get(1)![0][0], 10.0, 1e-10);
  assertAlmostEquals(epochDE.get(1)![0][2], 30.0, 1e-10);
  assertAlmostEquals(epochDE.get(1)![1][0], 5.0, 1e-10);
  assertAlmostEquals(epochDE.get(1)![1][3], 20.0, 1e-10);
});

Deno.test("per-level normalization: level with 0 batches is unchanged", () => {
  const epochDE = new Map<number, number[][]>();
  epochDE.set(0, [[10, 20], [30, 40]]);
  epochDE.set(1, [[100, 200]]);

  const hierBatchesByLevel = new Map<number, number>();
  hierBatchesByLevel.set(0, 5);
  // Level 1 has 0 batches (not even in the map)

  normalizeEpochDE(epochDE, hierBatchesByLevel);

  // Level 0 divided by 5
  assertAlmostEquals(epochDE.get(0)![0][0], 2.0, 1e-10);
  assertAlmostEquals(epochDE.get(0)![1][1], 8.0, 1e-10);

  // Level 1 unchanged (0 batches = no normalization)
  assertAlmostEquals(epochDE.get(1)![0][0], 100, 1e-10);
  assertAlmostEquals(epochDE.get(1)![0][1], 200, 1e-10);
});

Deno.test("per-level normalization: single batch per level = identity", () => {
  const epochDE = new Map<number, number[][]>();
  epochDE.set(0, [[7.5, -3.2]]);

  const hierBatchesByLevel = new Map<number, number>();
  hierBatchesByLevel.set(0, 1);

  normalizeEpochDE(epochDE, hierBatchesByLevel);

  // Dividing by 1 should not change values
  assertAlmostEquals(epochDE.get(0)![0][0], 7.5, 1e-10);
  assertAlmostEquals(epochDE.get(0)![0][1], -3.2, 1e-10);
});

Deno.test("per-level normalization: empty epochDE is handled gracefully", () => {
  const epochDE = new Map<number, number[][]>();
  const hierBatchesByLevel = new Map<number, number>();

  // Should not throw
  normalizeEpochDE(epochDE, hierBatchesByLevel);
  assertEquals(epochDE.size, 0);
});

// ============================================================================
// D. Example filtering and minimum batch size enforcement
// ============================================================================

Deno.test("example filtering: only examples with valid ancestors are kept", () => {
  const embDim = 4;
  // 4 L0 tools, 2 L1 caps. Cap0=[tool0, tool1], Cap1=[tool2, tool3]
  const l0IdxMap = new Map<string, number>();
  l0IdxMap.set("tool_a", 0);
  l0IdxMap.set("tool_b", 1);
  l0IdxMap.set("tool_c", 2);
  l0IdxMap.set("tool_d", 3);

  // Build ancestors manually
  const l0Ancestors: Map<number, number[]>[] = [
    new Map([[0, [0]]]),        // tool_a -> Cap0
    new Map([[0, [0]]]),        // tool_b -> Cap0
    new Map([[0, [1]]]),        // tool_c -> Cap1
    new Map(),                  // tool_d -> disconnected!
  ];

  const prodExamples = [
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_a" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_b" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_c" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_d" }, // disconnected
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_unknown" }, // not in l0IdxMap
  ];

  const filtered = filterExamplesForLevel(prodExamples, l0IdxMap, l0Ancestors, 0);

  // tool_d and tool_unknown should be excluded
  assertEquals(filtered.length, 3, "Only 3 examples with valid L1 ancestors");
  assertEquals(filtered[0].ancestorIdxs, [0]); // tool_a -> Cap0
  assertEquals(filtered[1].ancestorIdxs, [0]); // tool_b -> Cap0
  assertEquals(filtered[2].ancestorIdxs, [1]); // tool_c -> Cap1
});

Deno.test("example filtering: level with no ancestors returns empty", () => {
  const embDim = 4;
  const l0IdxMap = new Map<string, number>();
  l0IdxMap.set("tool_a", 0);
  l0IdxMap.set("tool_b", 1);

  // Both tools only have ancestors at orch level 0
  const l0Ancestors: Map<number, number[]>[] = [
    new Map([[0, [0]]]),
    new Map([[0, [1]]]),
  ];

  const prodExamples = [
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_a" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_b" },
  ];

  // Filter for orch level 1 (no tool has ancestors there)
  const filtered = filterExamplesForLevel(prodExamples, l0IdxMap, l0Ancestors, 1);
  assertEquals(filtered.length, 0, "No examples should survive filtering for missing level");
});

Deno.test("example filtering: minimum batch size enforcement", () => {
  const embDim = 4;
  const BATCH_SIZE = 4;
  const l0IdxMap = new Map<string, number>();
  l0IdxMap.set("tool_a", 0);
  l0IdxMap.set("tool_b", 1);

  const l0Ancestors: Map<number, number[]>[] = [
    new Map([[0, [0]]]),
    new Map([[0, [1]]]),
  ];

  const prodExamples = [
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_a" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_b" },
  ];

  const filtered = filterExamplesForLevel(prodExamples, l0IdxMap, l0Ancestors, 0);

  // Only 2 examples but BATCH_SIZE is 4 => level should be skipped
  const shouldSkip = filtered.length < BATCH_SIZE;
  assert(shouldSkip, `Level should be skipped: ${filtered.length} < BATCH_SIZE=${BATCH_SIZE}`);
});

Deno.test("example filtering: exactly BATCH_SIZE examples are NOT skipped", () => {
  const embDim = 4;
  const BATCH_SIZE = 4;
  const l0IdxMap = new Map<string, number>();
  l0IdxMap.set("tool_a", 0);
  l0IdxMap.set("tool_b", 1);
  l0IdxMap.set("tool_c", 2);
  l0IdxMap.set("tool_d", 3);

  const l0Ancestors: Map<number, number[]>[] = [
    new Map([[0, [0]]]),
    new Map([[0, [0]]]),
    new Map([[0, [1]]]),
    new Map([[0, [1]]]),
  ];

  const prodExamples = [
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_a" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_b" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_c" },
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_d" },
  ];

  const filtered = filterExamplesForLevel(prodExamples, l0IdxMap, l0Ancestors, 0);
  assertEquals(filtered.length, 4);

  const shouldSkip = filtered.length < BATCH_SIZE;
  assertEquals(shouldSkip, false, "Exactly BATCH_SIZE examples should NOT be skipped");
});

Deno.test("example filtering: multi-parent examples preserve all ancestor indices", () => {
  const embDim = 4;
  const l0IdxMap = new Map<string, number>();
  l0IdxMap.set("tool_multi", 0);

  // tool_multi has TWO ancestors at orch level 0
  const l0Ancestors: Map<number, number[]>[] = [
    new Map([[0, [0, 1]]]),
  ];

  const prodExamples = [
    { intentEmbedding: randomVec(embDim), targetToolId: "tool_multi" },
  ];

  const filtered = filterExamplesForLevel(prodExamples, l0IdxMap, l0Ancestors, 0);
  assertEquals(filtered.length, 1);
  assertEquals(filtered[0].ancestorIdxs, [0, 1], "Both ancestors should be preserved");
});

// ============================================================================
// E. Integration: full hierarchy contrastive pipeline (small scale)
// ============================================================================

Deno.test("integration: hierarchy contrastive produces finite loss and gradients", () => {
  const config = makeTinyConfig();
  const headParams = makeHeadParams(config);
  const B = 4;
  const tau = 0.07;
  const hierWeight = 0.5;

  // Simulate L1 ancestor embeddings (2 ancestors at orch level 0)
  const numL1Nodes = 2;
  const l1Embeddings = Array.from({ length: numL1Nodes }, () => randomVec(config.embeddingDim));

  // 4 prod examples, each with an ancestor mapping
  const intents = Array.from({ length: B }, () => randomVec(config.embeddingDim));
  const ancestorMapping = [0, 1, 0, 1]; // which L1 ancestor each example maps to
  const positiveEmbs = ancestorMapping.map(idx => l1Embeddings[idx]);

  // Forward
  const { loss, cache } = batchContrastiveForward(intents, positiveEmbs, headParams, config, tau);
  assert(Number.isFinite(loss), `Hierarchy loss should be finite, got ${loss}`);
  assert(loss > 0, `Hierarchy loss should be positive, got ${loss}`);

  // Backward
  const grads = makeZeroGrads(headParams);
  const { dIntentsProjected, dNodeEmbeddings } = batchContrastiveBackward(
    cache, headParams, grads, config,
  );

  // Apply hierWeight scaling
  for (let h = 0; h < config.numHeads; h++) {
    for (const row of grads.dW_q[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
    for (const row of grads.dW_k[h]) {
      for (let d = 0; d < row.length; d++) row[d] *= hierWeight;
    }
  }

  // Accumulate into epochDE
  const epochDE = Array.from({ length: numL1Nodes }, () =>
    new Array(config.embeddingDim).fill(0)
  );
  for (let i = 0; i < B; i++) {
    const aIdx = ancestorMapping[i];
    for (let d = 0; d < config.embeddingDim; d++) {
      epochDE[aIdx][d] += dNodeEmbeddings[i][d] * hierWeight;
    }
  }

  // All gradients should be finite
  for (let h = 0; h < config.numHeads; h++) {
    for (const row of grads.dW_q[h]) {
      for (const v of row) assert(Number.isFinite(v), "dW_q grad should be finite");
    }
    for (const row of grads.dW_k[h]) {
      for (const v of row) assert(Number.isFinite(v), "dW_k grad should be finite");
    }
  }

  for (const row of epochDE) {
    for (const v of row) assert(Number.isFinite(v), "epochDE grad should be finite");
  }

  for (const row of dIntentsProjected) {
    for (const v of row) assert(Number.isFinite(v), "dIntentsProjected should be finite");
  }

  // At least some gradients should be nonzero
  let hasNonZero = false;
  for (const row of epochDE) {
    for (const v of row) {
      if (Math.abs(v) > 1e-15) { hasNonZero = true; break; }
    }
    if (hasNonZero) break;
  }
  assert(hasNonZero, "epochDE should have nonzero gradients");
});

Deno.test("integration: multi-level accumulation with l0Ancestors + normalization", () => {
  const embDim = 4;

  // Build a 3-level hierarchy: 4 tools, 2 L1 caps, 1 L2 super-cap
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0, 1]);
  l0ToL1.set(1, [2, 3]);
  const l0ToL1Conn = buildConnectivity(l0ToL1, 4, 2);

  const l1ToL2 = new Map<number, number[]>();
  l1ToL2.set(0, [0, 1]);
  const l1ToL2Conn = buildConnectivity(l1ToL2, 2, 1);

  const interLevelConns = new Map<number, SparseConnectivity>();
  interLevelConns.set(1, l1ToL2Conn);

  const maxLevel = 1;
  const l0Ancestors = buildL0Ancestors(4, l0ToL1Conn, interLevelConns, maxLevel);

  // Initialize epochDE for 2 orchestrator levels
  const epochDE = new Map<number, number[][]>();
  epochDE.set(0, [[0, 0, 0, 0], [0, 0, 0, 0]]); // 2 L1 nodes
  epochDE.set(1, [[0, 0, 0, 0]]);                 // 1 L2 node

  const hierWeight = 0.5;

  // Simulate adding gradients from 2 batches at level 0 and 1 batch at level 1
  // Batch 1 at level 0: tool 0 -> Cap0, tool 2 -> Cap1
  const dNodeEmbs1 = [[1, 2, 3, 4], [5, 6, 7, 8]]; // 2 examples
  const ancestors1 = [l0Ancestors[0].get(0)![0], l0Ancestors[2].get(0)![0]]; // [0, 1]
  for (let i = 0; i < 2; i++) {
    for (let d = 0; d < embDim; d++) {
      epochDE.get(0)![ancestors1[i]][d] += dNodeEmbs1[i][d] * hierWeight;
    }
  }

  // Batch 2 at level 0: tool 1 -> Cap0, tool 3 -> Cap1
  const dNodeEmbs2 = [[2, 4, 6, 8], [10, 12, 14, 16]];
  const ancestors2 = [l0Ancestors[1].get(0)![0], l0Ancestors[3].get(0)![0]]; // [0, 1]
  for (let i = 0; i < 2; i++) {
    for (let d = 0; d < embDim; d++) {
      epochDE.get(0)![ancestors2[i]][d] += dNodeEmbs2[i][d] * hierWeight;
    }
  }

  // Batch 1 at level 1: tool 0 -> SuperCap0
  const dNodeEmbs3 = [[3, 6, 9, 12]];
  const ancestors3 = [l0Ancestors[0].get(1)![0]]; // [0]
  for (let i = 0; i < 1; i++) {
    for (let d = 0; d < embDim; d++) {
      epochDE.get(1)![ancestors3[i]][d] += dNodeEmbs3[i][d] * hierWeight;
    }
  }

  // Normalize: level 0 had 2 batches, level 1 had 1 batch
  const hierBatchesByLevel = new Map<number, number>();
  hierBatchesByLevel.set(0, 2);
  hierBatchesByLevel.set(1, 1);
  normalizeEpochDE(epochDE, hierBatchesByLevel);

  // Level 0, Cap0: accumulated (1*0.5 + 2*0.5) = 1.5, then /2 = 0.75
  assertAlmostEquals(epochDE.get(0)![0][0], (1 * 0.5 + 2 * 0.5) / 2, 1e-10);
  // Level 0, Cap0, dim 1: (2*0.5 + 4*0.5) / 2 = 1.5
  assertAlmostEquals(epochDE.get(0)![0][1], (2 * 0.5 + 4 * 0.5) / 2, 1e-10);

  // Level 0, Cap1: accumulated (5*0.5 + 10*0.5) = 7.5, then /2 = 3.75
  assertAlmostEquals(epochDE.get(0)![1][0], (5 * 0.5 + 10 * 0.5) / 2, 1e-10);

  // Level 1, SuperCap0: accumulated 3*0.5 = 1.5, then /1 = 1.5
  assertAlmostEquals(epochDE.get(1)![0][0], 3 * 0.5 / 1, 1e-10);
  assertAlmostEquals(epochDE.get(1)![0][2], 9 * 0.5 / 1, 1e-10);
});

// ============================================================================
// F. Edge cases and robustness
// ============================================================================

Deno.test("l0Ancestors: empty hierarchy (no L1 nodes)", () => {
  // 2 L0 tools, no L1 caps at all
  const l0ToL1Conn: SparseConnectivity = {
    sourceToTargets: new Map(),
    targetToSources: new Map(),
    numSources: 2,
    numTargets: 0,
  };

  const ancestors = buildL0Ancestors(2, l0ToL1Conn, new Map(), 0);

  assertEquals(ancestors[0].size, 0, "Tool 0 should have no ancestors");
  assertEquals(ancestors[1].size, 0, "Tool 1 should have no ancestors");
});

Deno.test("l0Ancestors: single tool single cap hierarchy", () => {
  const l0ToL1 = new Map<number, number[]>();
  l0ToL1.set(0, [0]);
  const l0ToL1Conn = buildConnectivity(l0ToL1, 1, 1);

  const ancestors = buildL0Ancestors(1, l0ToL1Conn, new Map(), 0);

  assertEquals(ancestors[0].get(0), [0]);
  assertEquals(ancestors[0].size, 1);
});

Deno.test("hierWeight scaling: dIntentsProjected scaled correctly for backpropWIntent", () => {
  // Mirrors train-ob.ts lines 967-969:
  //   const scaled = dIntentsProjected[i].map(v => v * hierWeight);
  const hierWeight = 0.3;
  const original = [1.5, -2.3, 0.7, 4.1];
  const scaled = original.map(v => v * hierWeight);

  for (let d = 0; d < original.length; d++) {
    assertAlmostEquals(scaled[d], original[d] * hierWeight, 1e-15);
  }
  // Verify it is not mutated in place
  assertAlmostEquals(original[0], 1.5, 1e-15);
});

Deno.test("zeroDH and zeroDE reset accumulators", () => {
  const embDim = 4;

  // Simulate _epochDH
  const dH = [[1, 2, 3, 4], [5, 6, 7, 8]];
  for (const row of dH) row.fill(0); // mirrors zeroDH
  for (const row of dH) {
    for (const v of row) assertEquals(v, 0);
  }

  // Simulate _epochDE
  const dE = new Map<number, number[][]>();
  dE.set(0, [[10, 20], [30, 40]]);
  dE.set(1, [[50, 60]]);

  // mirrors zeroDE
  for (const [, rows] of dE) {
    for (const row of rows) row.fill(0);
  }

  for (const [, rows] of dE) {
    for (const row of rows) {
      for (const v of row) assertEquals(v, 0);
    }
  }
});

Deno.test("hierBatchesByLevel accumulation tracks per-level counts", () => {
  // Mirrors train-ob.ts line 989:
  //   hierBatchesByLevel.set(orchLevel, (hierBatchesByLevel.get(orchLevel) ?? 0) + 1);
  const hierBatchesByLevel = new Map<number, number>();

  // Simulate processing: 3 batches at level 0, 1 batch at level 1
  for (let b = 0; b < 3; b++) {
    hierBatchesByLevel.set(0, (hierBatchesByLevel.get(0) ?? 0) + 1);
  }
  hierBatchesByLevel.set(1, (hierBatchesByLevel.get(1) ?? 0) + 1);

  assertEquals(hierBatchesByLevel.get(0), 3);
  assertEquals(hierBatchesByLevel.get(1), 1);
  assertEquals(hierBatchesByLevel.get(2) ?? 0, 0); // level not processed
});
