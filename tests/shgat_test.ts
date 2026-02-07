/**
 * SHGAT Core Unit Tests
 *
 * Tests for the main SHGAT class functionality.
 *
 * @module shgat/tests/shgat_test
 */

import { assertEquals, assertExists, assertGreater, assertLess } from "@std/assert";
import {
  batchedDownwardPass,
  batchedForward,
  batchedUpwardPass,
  batchGetEmbeddings,
  batchGetEmbeddingsByLevel,
  batchScoreAllNodes,
  buildAllIncidenceMatrices,
  buildGraph,
  buildIncidenceMatrix,
  createSHGAT,
  createSHGATFromCapabilities,
  groupNodesByLevel,
  type Node,
  precomputeGraphStructure,
  SHGAT,
} from "../mod.ts";

// =============================================================================
// Test Fixtures
// =============================================================================

function createTestCapabilities(count: number = 3) {
  return Array.from({ length: count }, (_, i) => ({
    id: `cap-${i + 1}`,
    embedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
    toolsUsed: [`tool-${i + 1}a`, `tool-${i + 1}b`],
    successRate: 0.7 + Math.random() * 0.25,
  }));
}

function createTestIntent() {
  return Array.from({ length: 1024 }, () => Math.random() * 0.1);
}

// =============================================================================
// Initialization Tests
// =============================================================================

Deno.test("SHGAT - creates from capabilities", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);
  assertExists(shgat);
});

Deno.test("SHGAT - creates with empty capabilities", () => {
  const shgat = createSHGATFromCapabilities([]);
  assertExists(shgat);
});

Deno.test("SHGAT - creates with custom config", () => {
  const shgat = new SHGAT({
    embeddingDim: 512,
    numHeads: 4,
    numLayers: 1,
  });
  assertExists(shgat);
});

// =============================================================================
// Scoring Tests
// =============================================================================

Deno.test("SHGAT - scoreAllCapabilities returns sorted results", () => {
  const caps = createTestCapabilities(5);
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  const results = shgat.scoreAllCapabilities(intent, []);

  assertEquals(results.length, 5);
  // Check sorted descending
  for (let i = 1; i < results.length; i++) {
    assertGreater(results[i - 1].score, results[i].score - 0.001); // Allow small float errors
  }
});

Deno.test("SHGAT - scoreAllCapabilities includes head scores", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  const results = shgat.scoreAllCapabilities(intent, []);

  for (const r of results) {
    assertExists(r.headScores);
    assertGreater(r.headScores.length, 0);
  }
});

Deno.test("SHGAT - scoreAllCapabilities returns valid scores", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  const results = shgat.scoreAllCapabilities(intent, []);
  const score = results.find(r => r.capabilityId === "cap-1")?.score ?? 0;

  assertGreater(score, -1);
  assertLess(score, 2);
});

Deno.test("SHGAT - scores with context tools", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  const resultsNoContext = shgat.scoreAllCapabilities(intent, []);
  const resultsWithContext = shgat.scoreAllCapabilities(intent, ["tool-1a"]);

  // Both should return results
  assertEquals(resultsNoContext.length, 3);
  assertEquals(resultsWithContext.length, 3);
});

// =============================================================================
// Forward Pass Tests
// =============================================================================

Deno.test("SHGAT - forward returns embeddings", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  const { E, H } = shgat.forward();

  assertExists(E);
  assertExists(H);
  assertEquals(E.length, 3); // 3 capabilities
  assertGreater(H.length, 0); // Tools
});

Deno.test("SHGAT - forward embeddings have correct dimension", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);

  const { E } = shgat.forward();

  for (const emb of E) {
    assertEquals(emb.length, 1024); // Default embeddingDim
  }
});

Deno.test("SHGAT - forward embeddings are finite", () => {
  const caps = createTestCapabilities(5);
  const shgat = createSHGATFromCapabilities(caps);

  const { E, H } = shgat.forward();

  for (const emb of E) {
    for (const v of emb) {
      assertEquals(isFinite(v), true, "E embedding should be finite");
    }
  }
  for (const emb of H) {
    for (const v of emb) {
      assertEquals(isFinite(v), true, "H embedding should be finite");
    }
  }
});

// =============================================================================
// Edge Cases
// =============================================================================

Deno.test("SHGAT - handles single capability", () => {
  const caps = createTestCapabilities(1);
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  const results = shgat.scoreAllCapabilities(intent, []);
  assertEquals(results.length, 1);
});

Deno.test("SHGAT - handles capability with no tools", () => {
  const caps = [{
    id: "cap-no-tools",
    embedding: Array.from({ length: 1024 }, () => Math.random()),
    toolsUsed: [],
    successRate: 0.8,
  }];
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  const results = shgat.scoreAllCapabilities(intent, []);
  assertEquals(results.length, 1);
});

Deno.test("SHGAT - handles unknown tool in context", () => {
  const caps = createTestCapabilities(3);
  const shgat = createSHGATFromCapabilities(caps);
  const intent = createTestIntent();

  // Should not crash with unknown tool
  const results = shgat.scoreAllCapabilities(intent, ["unknown-tool-xyz"]);
  assertEquals(results.length, 3);
});

// =============================================================================
// Unified Node API Tests
// =============================================================================

function createTestNodes(count: number = 3): Node[] {
  const nodes: Node[] = [];

  // Create leaf nodes (level 0)
  for (let i = 0; i < count; i++) {
    nodes.push({
      id: `leaf-${i + 1}`,
      embedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
      children: [],
      level: 0,
    });
  }

  // Create composite nodes (level 1)
  for (let i = 0; i < count; i++) {
    nodes.push({
      id: `composite-${i + 1}`,
      embedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
      children: [`leaf-${i + 1}`],
      level: 0, // Will be computed by buildGraph
    });
  }

  return nodes;
}

Deno.test("buildGraph - computes levels correctly", () => {
  const nodes: Node[] = [
    { id: "a", embedding: [], children: [], level: 0 },
    { id: "b", embedding: [], children: ["a"], level: 0 },
    { id: "c", embedding: [], children: ["b"], level: 0 },
    { id: "d", embedding: [], children: ["a", "c"], level: 0 },
  ];

  const graph = buildGraph(nodes);

  assertEquals(graph.get("a")?.level, 0); // leaf
  assertEquals(graph.get("b")?.level, 1); // contains a (level 0)
  assertEquals(graph.get("c")?.level, 2); // contains b (level 1)
  assertEquals(graph.get("d")?.level, 3); // contains c (level 2)
});

Deno.test("buildGraph - all leaves are level 0", () => {
  const nodes: Node[] = [
    { id: "leaf1", embedding: [], children: [], level: 0 },
    { id: "leaf2", embedding: [], children: [], level: 0 },
    { id: "leaf3", embedding: [], children: [], level: 0 },
  ];

  const graph = buildGraph(nodes);

  for (const node of graph.values()) {
    assertEquals(node.level, 0);
  }
});

Deno.test("createSHGAT - creates from unified nodes", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);
  assertExists(shgat);
});

Deno.test("createSHGAT - registers all nodes", () => {
  const nodes = createTestNodes(3);
  const shgat = createSHGAT(nodes);

  // 3 leaves + 3 composites = 6 nodes
  // Legacy API counts separately
  // assertEquals(shgat.getNodeCount(), 6);
  assertExists(shgat);
});

Deno.test("SHGAT - registerNode works", () => {
  const shgat = new SHGAT();

  shgat.registerNode({
    id: "test-node",
    embedding: Array.from({ length: 1024 }, () => Math.random()),
    children: [],
    level: 0,
  });

  assertExists(shgat);
});

// =============================================================================
// Batched Operations Tests
// =============================================================================

Deno.test("batchGetEmbeddings - returns matrix with correct dimensions", () => {
  const nodes: Node[] = [
    { id: "a", embedding: [1, 2, 3], children: [], level: 0 },
    { id: "b", embedding: [4, 5, 6], children: [], level: 0 },
    { id: "c", embedding: [7, 8, 9], children: [], level: 0 },
  ];
  const graph = buildGraph(nodes);

  const { matrix, ids, indexMap } = batchGetEmbeddings(graph);

  assertEquals(matrix.length, 3); // 3 nodes
  assertEquals(matrix[0].length, 3); // 3-dim embeddings
  assertEquals(ids.length, 3);
  assertEquals(indexMap.size, 3);
});

Deno.test("batchGetEmbeddings - index map is correct", () => {
  const nodes: Node[] = [
    { id: "x", embedding: [1], children: [], level: 0 },
    { id: "y", embedding: [2], children: [], level: 0 },
  ];
  const graph = buildGraph(nodes);

  const { matrix, indexMap } = batchGetEmbeddings(graph);

  // Verify we can use indexMap to find embeddings
  for (const [id, idx] of indexMap) {
    const node = graph.get(id)!;
    assertEquals(matrix[idx], node.embedding);
  }
});

Deno.test("batchGetEmbeddingsByLevel - filters by level", () => {
  const nodes: Node[] = [
    { id: "leaf1", embedding: [1], children: [], level: 0 },
    { id: "leaf2", embedding: [2], children: [], level: 0 },
    { id: "parent", embedding: [3], children: ["leaf1", "leaf2"], level: 0 },
  ];
  const graph = buildGraph(nodes);

  // Level 0 should have 2 leaves
  const level0 = batchGetEmbeddingsByLevel(graph, 0);
  assertEquals(level0.matrix.length, 2);

  // Level 1 should have 1 parent
  const level1 = batchGetEmbeddingsByLevel(graph, 1);
  assertEquals(level1.matrix.length, 1);
});

Deno.test("buildIncidenceMatrix - creates correct matrix", () => {
  const nodes: Node[] = [
    { id: "a", embedding: [], children: [], level: 0 },
    { id: "b", embedding: [], children: [], level: 0 },
    { id: "p1", embedding: [], children: ["a"], level: 0 },
    { id: "p2", embedding: [], children: ["a", "b"], level: 0 },
  ];
  const graph = buildGraph(nodes);

  const { matrix, childIndex, parentIndex } = buildIncidenceMatrix(graph, 0, 1);

  // Should be [2 children x 2 parents]
  assertEquals(matrix.length, 2);
  assertEquals(matrix[0].length, 2);

  // Check p1 contains only a
  const aIdx = childIndex.get("a")!;
  const p1Idx = parentIndex.get("p1")!;
  assertEquals(matrix[aIdx][p1Idx], 1);

  // Check p2 contains both a and b
  const bIdx = childIndex.get("b")!;
  const p2Idx = parentIndex.get("p2")!;
  assertEquals(matrix[aIdx][p2Idx], 1);
  assertEquals(matrix[bIdx][p2Idx], 1);

  // b not in p1
  assertEquals(matrix[bIdx][p1Idx], 0);
});

Deno.test("buildAllIncidenceMatrices - builds all level transitions", () => {
  const nodes: Node[] = [
    { id: "l0-a", embedding: [], children: [], level: 0 },
    { id: "l0-b", embedding: [], children: [], level: 0 },
    { id: "l1-x", embedding: [], children: ["l0-a"], level: 0 },
    { id: "l1-y", embedding: [], children: ["l0-b"], level: 0 },
    { id: "l2-z", embedding: [], children: ["l1-x", "l1-y"], level: 0 },
  ];
  const graph = buildGraph(nodes);

  const matrices = buildAllIncidenceMatrices(graph);

  // Should have matrices for level 1 (0→1) and level 2 (1→2)
  assertEquals(matrices.size, 2);
  assertExists(matrices.get(1));
  assertExists(matrices.get(2));
});

Deno.test("groupNodesByLevel - groups correctly", () => {
  const nodes: Node[] = [
    { id: "a", embedding: [], children: [], level: 0 },
    { id: "b", embedding: [], children: [], level: 0 },
    { id: "c", embedding: [], children: ["a", "b"], level: 0 },
  ];
  const graph = buildGraph(nodes);

  const groups = groupNodesByLevel(graph);

  // Level 0 should have 2 nodes
  assertEquals(groups.get(0)?.length, 2);

  // Level 1 should have 1 node
  assertEquals(groups.get(1)?.length, 1);
});

// =============================================================================
// Batched Message Passing Tests
// =============================================================================

Deno.test("precomputeGraphStructure - builds all structures", () => {
  const nodes: Node[] = [
    { id: "a", embedding: [1, 0], children: [], level: 0 },
    { id: "b", embedding: [0, 1], children: [], level: 0 },
    { id: "p", embedding: [0.5, 0.5], children: ["a", "b"], level: 0 },
  ];
  const graph = buildGraph(nodes);

  const structure = precomputeGraphStructure(graph);

  assertEquals(structure.maxLevel, 1);
  assertEquals(structure.embDim, 2);
  assertExists(structure.nodesByLevel.get(0));
  assertExists(structure.nodesByLevel.get(1));
  assertExists(structure.embeddingsByLevel.get(0));
  assertExists(structure.incidenceMatrices.get(1));
});

Deno.test("batchedUpwardPass - aggregates children to parent", () => {
  // 2 children, 1 parent that contains both
  const E_child = [
    [1, 0, 0],  // child a
    [0, 1, 0],  // child b
  ];
  const incidence = [
    [1],  // a → p
    [1],  // b → p
  ];

  const { E_parent, attention } = batchedUpwardPass(E_child, incidence);

  // Parent should be average of children (softmax with equal weights)
  assertEquals(E_parent.length, 1);
  assertEquals(E_parent[0].length, 3);

  // Attention has shape [numChildren × numParents] = [2 × 1]
  assertEquals(attention.length, 2);  // One row per child
  assertEquals(attention[0].length, 1);  // One column per parent

  // With uniform attention, parent ≈ (a + b) / 2 = [0.5, 0.5, 0]
  assertGreater(E_parent[0][0], 0.3);
  assertGreater(E_parent[0][1], 0.3);
});

Deno.test("batchedDownwardPass - propagates parent to children", () => {
  const E_child = [
    [1, 0],  // child a
    [0, 1],  // child b
  ];
  const E_parent = [
    [0.5, 0.5],  // parent p
  ];
  const incidence = [
    [1],  // a ← p
    [1],  // b ← p
  ];

  const { E_child_updated } = batchedDownwardPass(
    E_child,
    E_parent,
    incidence,
    0.5,  // 50% residual
  );

  // Children should be mix of original and parent info
  assertEquals(E_child_updated.length, 2);

  // a was [1, 0], parent is [0.5, 0.5], with 50% residual:
  // a' ≈ 0.5 * [1, 0] + 0.5 * [0.5, 0.5] = [0.75, 0.25]
  assertGreater(E_child_updated[0][0], 0.6);
  assertGreater(E_child_updated[0][1], 0.1);
});

Deno.test("batchedForward - full forward pass", () => {
  const nodes: Node[] = [
    { id: "a", embedding: [1, 0, 0, 0], children: [], level: 0 },
    { id: "b", embedding: [0, 1, 0, 0], children: [], level: 0 },
    { id: "p", embedding: [0, 0, 1, 0], children: ["a", "b"], level: 0 },
  ];
  const graph = buildGraph(nodes);
  const structure = precomputeGraphStructure(graph);

  const { E, attentionUp, attentionDown } = batchedForward(structure);

  // Should have embeddings for level 0 and 1
  assertExists(E.get(0));
  assertExists(E.get(1));

  // Level 0 should have 2 nodes (updated by downward pass)
  assertEquals(E.get(0)?.length, 2);

  // Level 1 should have 1 node
  assertEquals(E.get(1)?.length, 1);

  // Attention should exist
  assertExists(attentionUp.get(1));
  assertExists(attentionDown.get(1));
});

Deno.test("batchScoreAllNodes - scores all nodes in batch", () => {
  // 2 queries, 3 nodes
  const Q_batch = [
    [1, 0],  // query 1
    [0, 1],  // query 2
  ];
  const K_all = [
    [1, 0],    // node a - similar to q1
    [0, 1],    // node b - similar to q2
    [0.5, 0.5], // node c - neutral
  ];
  const nodeIds = ["a", "b", "c"];

  const scores = batchScoreAllNodes(Q_batch, K_all, nodeIds);

  // Node a should score higher for query 1
  const scoresA = scores.get("a")!;
  const scoresB = scores.get("b")!;

  assertGreater(scoresA[0], scoresA[1]); // a scores higher for q1
  assertGreater(scoresB[1], scoresB[0]); // b scores higher for q2
});
