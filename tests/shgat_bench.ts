/**
 * SHGAT-TF Benchmark
 *
 * Benchmarks for the TensorFlow.js-backed SHGAT implementation.
 * Tests scoring, forward pass, and batched operations.
 *
 * Run: deno bench -A --no-check lib/shgat-tf/tests/shgat_bench.ts
 *
 * @module shgat-tf/tests/shgat_bench
 */

import {
  createSHGAT,
  type Node,
  SHGAT,
  generateDefaultToolEmbedding,
  batchedForward,
  batchedUpwardPass,
  batchScoreAllNodes,
  precomputeGraphStructure,
  buildGraph,
} from "../mod.ts";

// ============================================================================
// Fixtures
// ============================================================================

function createNodes(leafCount: number, compositeCount: number): Node[] {
  const nodes: Node[] = [];

  for (let i = 0; i < leafCount; i++) {
    nodes.push({
      id: `tool-${i}`,
      embedding: generateDefaultToolEmbedding(`tool-${i}`, 1024),
      children: [],
      level: 0,
    });
  }

  const leafsPerComposite = Math.ceil(leafCount / compositeCount);
  for (let c = 0; c < compositeCount; c++) {
    const start = c * leafsPerComposite;
    const end = Math.min(start + leafsPerComposite, leafCount);
    const children: string[] = [];
    for (let i = start; i < end; i++) {
      children.push(`tool-${i}`);
    }
    nodes.push({
      id: `cap-${c}`,
      embedding: Array.from({ length: 1024 }, () => Math.random() * 0.1),
      children,
      level: 0,
    });
  }

  return nodes;
}

function createIntent(): number[] {
  return Array.from({ length: 1024 }, () => Math.random() * 0.1);
}

// ============================================================================
// Small graph (6 tools, 3 composites) — typical dev scenario
// ============================================================================

const smallNodes = createNodes(6, 3);
const smallShgat = createSHGAT(smallNodes);
const intent = createIntent();

Deno.bench({
  name: "small (6+3): scoreNodes (all)",
  group: "scoreNodes",
  baseline: true,
  fn: () => {
    smallShgat.scoreNodes(intent);
  },
});

Deno.bench({
  name: "small (6+3): scoreNodes (composites)",
  group: "scoreNodes-composites",
  baseline: true,
  fn: () => {
    smallShgat.scoreNodes(intent, 1);
  },
});

Deno.bench({
  name: "small (6+3): scoreNodes (leaves)",
  group: "scoreNodes-leaves",
  baseline: true,
  fn: () => {
    smallShgat.scoreNodes(intent, 0);
  },
});

Deno.bench({
  name: "small (6+3): forward (array)",
  group: "forward",
  baseline: true,
  fn: () => {
    smallShgat.forward();
  },
});

// ============================================================================
// Medium graph (50 tools, 10 composites) — typical production scenario
// ============================================================================

const medNodes = createNodes(50, 10);
const medShgat = createSHGAT(medNodes);

Deno.bench({
  name: "medium (50+10): scoreNodes (all)",
  group: "scoreNodes",
  fn: () => {
    medShgat.scoreNodes(intent);
  },
});

Deno.bench({
  name: "medium (50+10): scoreNodes (composites)",
  group: "scoreNodes-composites",
  fn: () => {
    medShgat.scoreNodes(intent, 1);
  },
});

Deno.bench({
  name: "medium (50+10): scoreNodes (leaves)",
  group: "scoreNodes-leaves",
  fn: () => {
    medShgat.scoreNodes(intent, 0);
  },
});

Deno.bench({
  name: "medium (50+10): forward (array)",
  group: "forward",
  fn: () => {
    medShgat.forward();
  },
});

// ============================================================================
// Large graph (218 tools, 26 composites) — production catalog (from docs)
// ============================================================================

const largeNodes = createNodes(218, 26);
const largeShgat = createSHGAT(largeNodes);

Deno.bench({
  name: "large (218+26): scoreNodes (all)",
  group: "scoreNodes",
  fn: () => {
    largeShgat.scoreNodes(intent);
  },
});

Deno.bench({
  name: "large (218+26): scoreNodes (composites)",
  group: "scoreNodes-composites",
  fn: () => {
    largeShgat.scoreNodes(intent, 1);
  },
});

Deno.bench({
  name: "large (218+26): scoreNodes (leaves)",
  group: "scoreNodes-leaves",
  fn: () => {
    largeShgat.scoreNodes(intent, 0);
  },
});

Deno.bench({
  name: "large (218+26): forward (array)",
  group: "forward",
  fn: () => {
    largeShgat.forward();
  },
});

// ============================================================================
// Batched ops (pure array, no TF.js)
// ============================================================================

const batchGraph = buildGraph(createNodes(50, 10));
const batchStructure = precomputeGraphStructure(batchGraph);

Deno.bench({
  name: "batched: upward pass (50+10)",
  group: "batched-ops",
  baseline: true,
  fn: () => {
    const E0 = batchStructure.embeddingsByLevel.get(0)!;
    const inc = batchStructure.incidenceMatrices.get(1)!;
    batchedUpwardPass(E0.matrix, inc.matrix);
  },
});

Deno.bench({
  name: "batched: full forward (50+10)",
  group: "batched-ops",
  fn: () => {
    batchedForward(batchStructure);
  },
});

Deno.bench({
  name: "batched: score all nodes (50+10, 1 query)",
  group: "batched-ops",
  fn: () => {
    const Q = [intent];
    const allEmbs: number[][] = [];
    const allIds: string[] = [];
    for (const [, emb] of batchStructure.embeddingsByLevel) {
      for (let i = 0; i < emb.matrix.length; i++) {
        allEmbs.push(emb.matrix[i]);
        allIds.push(emb.ids[i]);
      }
    }
    batchScoreAllNodes(Q, allEmbs, allIds);
  },
});

// ============================================================================
// Creation benchmark
// ============================================================================

Deno.bench({
  name: "createSHGAT: 6+3 nodes",
  group: "create",
  baseline: true,
  fn: () => {
    createSHGAT(smallNodes);
  },
});

Deno.bench({
  name: "createSHGAT: 50+10 nodes",
  group: "create",
  fn: () => {
    createSHGAT(medNodes);
  },
});

Deno.bench({
  name: "createSHGAT: 218+26 nodes",
  group: "create",
  fn: () => {
    createSHGAT(largeNodes);
  },
});

// ============================================================================
// Serialization
// ============================================================================

Deno.bench({
  name: "exportParams (medium)",
  group: "serialization",
  baseline: true,
  fn: () => {
    medShgat.exportParams();
  },
});

const medParams = medShgat.exportParams();

Deno.bench({
  name: "importParams (medium)",
  group: "serialization",
  fn: () => {
    const fresh = createSHGAT(medNodes);
    fresh.importParams(medParams);
  },
});
