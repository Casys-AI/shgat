/**
 * Sparse Message Passing Equivalence Tests
 *
 * Verifies that the sparse (SparseConnectivity) implementations of
 * VE, EV, and EE phases produce correct results:
 *   - Forward pass embeddings have correct shapes
 *   - Backward pass gradients have correct shapes
 *   - Attention weights sum to 1 per softmax group
 *   - denseToSparse() and transposeSparse() utilities work correctly
 *   - Disconnected nodes produce zero embeddings
 *   - Forward pass is deterministic (same input => same output)
 *   - Backward gradients match finite differences (numerical gradient check)
 *
 * Run: deno test -A --no-check lib/shgat-tf/src/message-passing/__tests__/sparse-equivalence.test.ts
 *
 * @module shgat-tf/tests/sparse-equivalence
 */

import { assertEquals } from "@std/assert";
import { assert } from "@std/assert";
import {
  denseToSparse,
  transposeSparse,
  type PhaseParameters,
} from "../phase-interface.ts";
import { VertexToEdgePhase } from "../vertex-to-edge-phase.ts";
import { EdgeToVertexPhase } from "../edge-to-vertex-phase.ts";
import { EdgeToEdgePhase } from "../edge-to-edge-phase.ts";

// ============================================================================
// Seeded PRNG (mulberry32) for reproducibility
// ============================================================================

function mulberry32(seed: number): () => number {
  let s = seed | 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ============================================================================
// Test fixtures
// ============================================================================

const SEED = 42;
const NUM_L0 = 5;
const NUM_L1 = 3;
const EMB_DIM = 8;
const HEAD_DIM = 4;
const LEAKY_RELU_SLOPE = 0.2;
const CONFIG = { leakyReluSlope: LEAKY_RELU_SLOPE };

/**
 * Dense incidence matrix [numL0=5][numL1=3]:
 *
 *   L1_0  L1_1  L1_2
 * n0  1     0     1      L0_0 in L1_0, L1_2
 * n1  1     1     0      L0_1 in L1_0, L1_1
 * n2  0     1     0      L0_2 in L1_1
 * n3  0     0     1      L0_3 in L1_2
 * n4  1     0     0      L0_4 in L1_0
 */
const DENSE_INCIDENCE: number[][] = [
  [1, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [0, 0, 1],
  [1, 0, 0],
];

function makeSeededEmbeddings(rng: () => number, rows: number, dim: number): number[][] {
  return Array.from({ length: rows }, () =>
    Array.from({ length: dim }, () => (rng() - 0.5) * 2)
  );
}

function makeSeededParams(rng: () => number, headDim: number, embDim: number): PhaseParameters {
  return {
    W_source: Array.from({ length: headDim }, () =>
      Array.from({ length: embDim }, () => (rng() - 0.5) * 0.2)
    ),
    W_target: Array.from({ length: headDim }, () =>
      Array.from({ length: embDim }, () => (rng() - 0.5) * 0.2)
    ),
    a_attention: Array.from({ length: 2 * headDim }, () => (rng() - 0.5) * 0.2),
  };
}

function almostEqual(a: number, b: number, tol: number, msg?: string): void {
  const diff = Math.abs(a - b);
  assert(diff < tol, `${msg ?? ""} expected ${a} ≈ ${b} (diff=${diff}, tol=${tol})`);
}

/**
 * Dense incidence for EE phase [numChild=3][numParent=2]:
 *
 *       parent0  parent1
 * child0   1       0      child0 in parent0
 * child1   1       1      child1 in both
 * child2   0       1      child2 in parent1
 */
const DENSE_EE_INCIDENCE: number[][] = [
  [1, 0],
  [1, 1],
  [0, 1],
];

// ============================================================================
// 1. denseToSparse() and transposeSparse() utility tests
// ============================================================================

Deno.test("denseToSparse: converts dense incidence to correct sparse representation", () => {
  const sparse = denseToSparse(DENSE_INCIDENCE);

  assertEquals(sparse.numSources, 5, "numSources should be 5 (L0 nodes)");
  assertEquals(sparse.numTargets, 3, "numTargets should be 3 (L1 nodes)");

  // L0_0 -> [L1_0, L1_2]
  assertEquals(sparse.sourceToTargets.get(0)!, [0, 2]);
  // L0_1 -> [L1_0, L1_1]
  assertEquals(sparse.sourceToTargets.get(1)!, [0, 1]);
  // L0_2 -> [L1_1]
  assertEquals(sparse.sourceToTargets.get(2)!, [1]);
  // L0_3 -> [L1_2]
  assertEquals(sparse.sourceToTargets.get(3)!, [2]);
  // L0_4 -> [L1_0]
  assertEquals(sparse.sourceToTargets.get(4)!, [0]);

  // Reverse: L1_0 <- [L0_0, L0_1, L0_4]
  assertEquals(sparse.targetToSources.get(0)!, [0, 1, 4]);
  // L1_1 <- [L0_1, L0_2]
  assertEquals(sparse.targetToSources.get(1)!, [1, 2]);
  // L1_2 <- [L0_0, L0_3]
  assertEquals(sparse.targetToSources.get(2)!, [0, 3]);
});

Deno.test("denseToSparse: empty matrix produces empty maps", () => {
  const sparse = denseToSparse([
    [0, 0],
    [0, 0],
  ]);

  assertEquals(sparse.numSources, 2);
  assertEquals(sparse.numTargets, 2);
  assertEquals(sparse.sourceToTargets.size, 0);
  assertEquals(sparse.targetToSources.size, 0);
});

Deno.test("denseToSparse: fully connected matrix", () => {
  const dense = [
    [1, 1],
    [1, 1],
    [1, 1],
  ];
  const sparse = denseToSparse(dense);

  assertEquals(sparse.numSources, 3);
  assertEquals(sparse.numTargets, 2);

  // Every source connects to every target
  for (let s = 0; s < 3; s++) {
    assertEquals(sparse.sourceToTargets.get(s)!, [0, 1]);
  }
  for (let t = 0; t < 2; t++) {
    assertEquals(sparse.targetToSources.get(t)!, [0, 1, 2]);
  }
});

Deno.test("transposeSparse: swaps source/target roles correctly", () => {
  const sparse = denseToSparse(DENSE_INCIDENCE);
  const transposed = transposeSparse(sparse);

  assertEquals(transposed.numSources, 3, "transposed numSources should be original numTargets");
  assertEquals(transposed.numTargets, 5, "transposed numTargets should be original numSources");

  // transposed.sourceToTargets was sparse.targetToSources
  // L1_0 -> [L0_0, L0_1, L0_4]
  assertEquals(transposed.sourceToTargets.get(0)!, [0, 1, 4]);
  // transposed.targetToSources was sparse.sourceToTargets
  // L0_0 -> [L1_0, L1_2]
  assertEquals(transposed.targetToSources.get(0)!, [0, 2]);
});

Deno.test("transposeSparse: double transpose is identity", () => {
  const sparse = denseToSparse(DENSE_INCIDENCE);
  const doubleTransposed = transposeSparse(transposeSparse(sparse));

  assertEquals(doubleTransposed.numSources, sparse.numSources);
  assertEquals(doubleTransposed.numTargets, sparse.numTargets);

  // Verify sourceToTargets are the same references
  for (const [key, val] of sparse.sourceToTargets) {
    assertEquals(doubleTransposed.sourceToTargets.get(key)!, val);
  }
  for (const [key, val] of sparse.targetToSources) {
    assertEquals(doubleTransposed.targetToSources.get(key)!, val);
  }
});

// ============================================================================
// 2. Vertex → Edge (VE) Phase Tests
// ============================================================================

Deno.test("VE forward: output embeddings have correct shape [numL1][headDim]", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();
  const result = phase.forwardWithCache(H, E, conn, params, CONFIG);

  assertEquals(result.embeddings.length, NUM_L1, "VE output rows = numL1");
  assertEquals(result.embeddings[0].length, HEAD_DIM, "VE output cols = headDim");
});

Deno.test("VE forward: attention matrix has correct shape [numL0][numL1]", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();
  const result = phase.forwardWithCache(H, E, conn, params, CONFIG);

  assertEquals(result.attention.length, NUM_L0, "VE attention rows = numL0");
  assertEquals(result.attention[0].length, NUM_L1, "VE attention cols = numL1");
});

Deno.test("VE forward: attention weights sum to 1 per L1 node (softmax group)", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();
  const result = phase.forwardWithCache(H, E, conn, params, CONFIG);

  // In VE, softmax is per L1 node (column). Sum attention over L0 nodes for each L1.
  for (let c = 0; c < NUM_L1; c++) {
    let sum = 0;
    for (let t = 0; t < NUM_L0; t++) {
      sum += result.attention[t][c];
    }
    // Only L1 nodes with connected L0 nodes should sum to 1
    const connectedL0 = conn.targetToSources.get(c);
    if (connectedL0 && connectedL0.length > 0) {
      almostEqual(sum, 1.0, 1e-6, `VE: attention sum for L1 node ${c}`);
    }
  }
});

Deno.test("VE forward: attention is zero for non-connected (L0, L1) pairs", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();
  const result = phase.forwardWithCache(H, E, conn, params, CONFIG);

  // Check zeros where dense incidence is 0
  for (let t = 0; t < NUM_L0; t++) {
    for (let c = 0; c < NUM_L1; c++) {
      if (DENSE_INCIDENCE[t][c] === 0) {
        assertEquals(result.attention[t][c], 0, `VE: attention[${t}][${c}] should be 0 (not connected)`);
      }
    }
  }
});

Deno.test("VE forward: deterministic — same seed produces identical output", () => {
  const run = () => {
    const rng = mulberry32(SEED);
    const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
    const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
    const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
    const conn = denseToSparse(DENSE_INCIDENCE);

    const phase = new VertexToEdgePhase();
    return phase.forwardWithCache(H, E, conn, params, CONFIG);
  };

  const r1 = run();
  const r2 = run();

  for (let c = 0; c < NUM_L1; c++) {
    for (let d = 0; d < HEAD_DIM; d++) {
      assertEquals(
        r1.embeddings[c][d],
        r2.embeddings[c][d],
        `VE: deterministic output mismatch at [${c}][${d}]`,
      );
    }
  }
});

Deno.test("VE backward: gradient shapes are correct", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();
  const { cache } = phase.forwardWithCache(H, E, conn, params, CONFIG);

  // Upstream gradient: ones [numL1][headDim]
  const dE_new = Array.from({ length: NUM_L1 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_new, cache, params);

  assertEquals(grads.dW_source.length, HEAD_DIM, "VE dW_source rows = headDim");
  assertEquals(grads.dW_source[0].length, EMB_DIM, "VE dW_source cols = embDim");
  assertEquals(grads.dW_target.length, HEAD_DIM, "VE dW_target rows = headDim");
  assertEquals(grads.dW_target[0].length, EMB_DIM, "VE dW_target cols = embDim");
  assertEquals(grads.da_attention.length, 2 * HEAD_DIM, "VE da_attention length = 2*headDim");
  assertEquals(grads.dH.length, NUM_L0, "VE dH rows = numL0");
  assertEquals(grads.dH[0].length, EMB_DIM, "VE dH cols = embDim");
  assertEquals(grads.dE.length, NUM_L1, "VE dE rows = numL1");
  assertEquals(grads.dE[0].length, EMB_DIM, "VE dE cols = embDim");
});

Deno.test("VE backward: gradients are non-zero for connected nodes", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();
  const { cache } = phase.forwardWithCache(H, E, conn, params, CONFIG);

  const dE_new = Array.from({ length: NUM_L1 }, () =>
    Array.from({ length: HEAD_DIM }, () => 1.0)
  );
  const grads = phase.backward(dE_new, cache, params);

  // dW_source should be non-zero (L0 nodes contribute to L1 nodes)
  const dW_source_norm = grads.dW_source.flat().reduce((s, v) => s + v * v, 0);
  assert(dW_source_norm > 1e-10, "VE dW_source should be non-zero");

  // dH for connected L0 nodes should be non-zero
  // L0_0 is connected to L1_0 and L1_2
  const dH0_norm = grads.dH[0].reduce((s, v) => s + v * v, 0);
  assert(dH0_norm > 1e-10, "VE dH[0] should be non-zero (L0_0 is connected)");
});

Deno.test("VE backward: dW_source is finite and deterministic", () => {
  // dW_source gradient: matmulTranspose bug fixed (now uses matmul for dX_proj^T @ X).
  // Verify the sparse implementation produces finite, non-zero, deterministic grads.
  const run = () => {
    const rng = mulberry32(SEED);
    const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
    const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
    const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
    const conn = denseToSparse(DENSE_INCIDENCE);

    const phase = new VertexToEdgePhase();
    const { cache } = phase.forwardWithCache(H, E, conn, params, CONFIG);
    const dE_new = Array.from({ length: NUM_L1 }, () => Array(HEAD_DIM).fill(1.0));
    return phase.backward(dE_new, cache, params);
  };

  const g1 = run();
  const g2 = run();

  // Verify finite and non-zero
  for (let r = 0; r < HEAD_DIM; r++) {
    for (let c = 0; c < EMB_DIM; c++) {
      assert(Number.isFinite(g1.dW_source[r][c]), `VE dW_source[${r}][${c}] should be finite`);
    }
  }
  const norm = g1.dW_source.flat().reduce((s, v) => s + v * v, 0);
  assert(norm > 1e-10, "VE dW_source should be non-zero");

  // Verify deterministic
  for (let r = 0; r < HEAD_DIM; r++) {
    for (let c = 0; c < EMB_DIM; c++) {
      assertEquals(g1.dW_source[r][c], g2.dW_source[r][c], `VE dW_source[${r}][${c}] deterministic`);
    }
  }
});

Deno.test("VE backward: dH matches finite differences", () => {
  const rng = mulberry32(SEED);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(H, E, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(H, E, conn, params, CONFIG);
  const dE_new = Array.from({ length: NUM_L1 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_new, cache, params);

  const eps = 1e-5;
  // Check L0_0 (connected to L1_0, L1_2)
  for (let d = 0; d < Math.min(3, EMB_DIM); d++) {
    const orig = H[0][d];

    H[0][d] = orig + eps;
    const lossPlus = computeLoss();

    H[0][d] = orig - eps;
    const lossMinus = computeLoss();

    H[0][d] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    almostEqual(
      grads.dH[0][d],
      numerical,
      1e-3,
      `VE dH[0][${d}]: analytical=${grads.dH[0][d]}, numerical=${numerical}`,
    );
  }
});

// ============================================================================
// 3. Edge → Vertex (EV) Phase Tests
// ============================================================================

Deno.test("EV forward: output embeddings have correct shape [numL0][headDim]", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  // EV uses the same incidence but connectivity is [L0][L1]: L0 nodes are sources
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const result = phase.forwardWithCache(E, H, conn, params, CONFIG);

  assertEquals(result.embeddings.length, NUM_L0, "EV output rows = numL0");
  assertEquals(result.embeddings[0].length, HEAD_DIM, "EV output cols = headDim");
});

Deno.test("EV forward: attention matrix has correct shape [numL1][numL0]", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const result = phase.forwardWithCache(E, H, conn, params, CONFIG);

  assertEquals(result.attention.length, NUM_L1, "EV attention rows = numL1");
  assertEquals(result.attention[0].length, NUM_L0, "EV attention cols = numL0");
});

Deno.test("EV forward: attention weights sum to 1 per L0 node (softmax group)", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const result = phase.forwardWithCache(E, H, conn, params, CONFIG);

  // In EV, softmax is per L0 node. Sum attention over L1 nodes for each L0 node.
  for (let t = 0; t < NUM_L0; t++) {
    let sum = 0;
    for (let c = 0; c < NUM_L1; c++) {
      sum += result.attention[c][t];
    }
    const connectedL1 = conn.sourceToTargets.get(t);
    if (connectedL1 && connectedL1.length > 0) {
      almostEqual(sum, 1.0, 1e-6, `EV: attention sum for L0 node ${t}`);
    }
  }
});

Deno.test("EV forward: attention is zero for non-connected (L1, L0) pairs", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const result = phase.forwardWithCache(E, H, conn, params, CONFIG);

  // attention[c][t] should be 0 where DENSE_INCIDENCE[t][c] = 0
  for (let t = 0; t < NUM_L0; t++) {
    for (let c = 0; c < NUM_L1; c++) {
      if (DENSE_INCIDENCE[t][c] === 0) {
        assertEquals(result.attention[c][t], 0, `EV: attention[${c}][${t}] should be 0 (not connected)`);
      }
    }
  }
});

Deno.test("EV forward: deterministic — same seed produces identical output", () => {
  const run = () => {
    const rng = mulberry32(SEED + 100);
    const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
    const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
    const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
    const conn = denseToSparse(DENSE_INCIDENCE);

    const phase = new EdgeToVertexPhase();
    return phase.forwardWithCache(E, H, conn, params, CONFIG);
  };

  const r1 = run();
  const r2 = run();

  for (let t = 0; t < NUM_L0; t++) {
    for (let d = 0; d < HEAD_DIM; d++) {
      assertEquals(
        r1.embeddings[t][d],
        r2.embeddings[t][d],
        `EV: deterministic output mismatch at [${t}][${d}]`,
      );
    }
  }
});

Deno.test("EV backward: gradient shapes are correct", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const { cache } = phase.forwardWithCache(E, H, conn, params, CONFIG);

  const dH_new = Array.from({ length: NUM_L0 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dH_new, cache, params);

  assertEquals(grads.dW_source.length, HEAD_DIM, "EV dW_source rows = headDim");
  assertEquals(grads.dW_source[0].length, EMB_DIM, "EV dW_source cols = embDim");
  assertEquals(grads.dW_target.length, HEAD_DIM, "EV dW_target rows = headDim");
  assertEquals(grads.dW_target[0].length, EMB_DIM, "EV dW_target cols = embDim");
  assertEquals(grads.da_attention.length, 2 * HEAD_DIM, "EV da_attention length = 2*headDim");
  assertEquals(grads.dE.length, NUM_L1, "EV dE rows = numL1");
  assertEquals(grads.dE[0].length, EMB_DIM, "EV dE cols = embDim");
  assertEquals(grads.dH.length, NUM_L0, "EV dH rows = numL0");
  assertEquals(grads.dH[0].length, EMB_DIM, "EV dH cols = embDim");
});

Deno.test("EV backward: gradients are non-zero for connected nodes", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const { cache } = phase.forwardWithCache(E, H, conn, params, CONFIG);

  const dH_new = Array.from({ length: NUM_L0 }, () =>
    Array.from({ length: HEAD_DIM }, () => 1.0)
  );
  const grads = phase.backward(dH_new, cache, params);

  const dW_source_norm = grads.dW_source.flat().reduce((s, v) => s + v * v, 0);
  assert(dW_source_norm > 1e-10, "EV dW_source should be non-zero");

  // dE for connected L1 nodes should be non-zero
  // L1_0 is connected to L0_0, L0_1, L0_4
  const dE0_norm = grads.dE[0].reduce((s, v) => s + v * v, 0);
  assert(dE0_norm > 1e-10, "EV dE[0] should be non-zero (L1_0 is connected)");
});

Deno.test("EV backward: dW_source is finite and deterministic", () => {
  // dW_source gradient: matmulTranspose bug fixed (now uses matmul for dX_proj^T @ X).
  const run = () => {
    const rng = mulberry32(SEED + 100);
    const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
    const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
    const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
    const conn = denseToSparse(DENSE_INCIDENCE);

    const phase = new EdgeToVertexPhase();
    const { cache } = phase.forwardWithCache(E, H, conn, params, CONFIG);
    const dH_new = Array.from({ length: NUM_L0 }, () => Array(HEAD_DIM).fill(1.0));
    return phase.backward(dH_new, cache, params);
  };

  const g1 = run();
  const g2 = run();

  for (let r = 0; r < HEAD_DIM; r++) {
    for (let c = 0; c < EMB_DIM; c++) {
      assert(Number.isFinite(g1.dW_source[r][c]), `EV dW_source[${r}][${c}] should be finite`);
    }
  }
  const norm = g1.dW_source.flat().reduce((s, v) => s + v * v, 0);
  assert(norm > 1e-10, "EV dW_source should be non-zero");

  for (let r = 0; r < HEAD_DIM; r++) {
    for (let c = 0; c < EMB_DIM; c++) {
      assertEquals(g1.dW_source[r][c], g2.dW_source[r][c], `EV dW_source[${r}][${c}] deterministic`);
    }
  }
});

Deno.test("EV backward: dE matches finite differences", () => {
  const rng = mulberry32(SEED + 100);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(E, H, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(E, H, conn, params, CONFIG);
  const dH_new = Array.from({ length: NUM_L0 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dH_new, cache, params);

  const eps = 1e-5;
  // Check L1_0 (connected to L0_0, L0_1, L0_4)
  for (let d = 0; d < Math.min(3, EMB_DIM); d++) {
    const orig = E[0][d];

    E[0][d] = orig + eps;
    const lossPlus = computeLoss();

    E[0][d] = orig - eps;
    const lossMinus = computeLoss();

    E[0][d] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    almostEqual(
      grads.dE[0][d],
      numerical,
      1e-3,
      `EV dE[0][${d}]: analytical=${grads.dE[0][d]}, numerical=${numerical}`,
    );
  }
});

// ============================================================================
// 4. Edge → Edge (EE) Phase Tests
// ============================================================================

const NUM_CHILD_NODES = 3;
const NUM_PARENT_NODES = 2;

Deno.test("EE forward: output embeddings have correct shape [numParent][headDim]", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);
  const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  assertEquals(result.embeddings.length, NUM_PARENT_NODES, "EE output rows = numParent");
  assertEquals(result.embeddings[0].length, HEAD_DIM, "EE output cols = headDim");
});

Deno.test("EE forward: attention matrix has correct shape [numChild][numParent]", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);
  const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  assertEquals(result.attention.length, NUM_CHILD_NODES, "EE attention rows = numChild");
  assertEquals(result.attention[0].length, NUM_PARENT_NODES, "EE attention cols = numParent");
});

Deno.test("EE forward: attention weights sum to 1 per parent (softmax group)", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);
  const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  // Softmax is per parent (column). Sum attention over children for each parent.
  for (let p = 0; p < NUM_PARENT_NODES; p++) {
    let sum = 0;
    for (let c = 0; c < NUM_CHILD_NODES; c++) {
      sum += result.attention[c][p];
    }
    const connectedChildren = conn.targetToSources.get(p);
    if (connectedChildren && connectedChildren.length > 0) {
      almostEqual(sum, 1.0, 1e-6, `EE: attention sum for parent ${p}`);
    }
  }
});

Deno.test("EE forward: attention is zero for non-connected (child, parent) pairs", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);
  const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  for (let c = 0; c < NUM_CHILD_NODES; c++) {
    for (let p = 0; p < NUM_PARENT_NODES; p++) {
      if (DENSE_EE_INCIDENCE[c][p] === 0) {
        assertEquals(result.attention[c][p], 0, `EE: attention[${c}][${p}] should be 0 (not connected)`);
      }
    }
  }
});

Deno.test("EE forward: deterministic — same seed produces identical output", () => {
  const run = () => {
    const rng = mulberry32(SEED + 200);
    const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
    const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
    const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
    const conn = denseToSparse(DENSE_EE_INCIDENCE);

    const phase = new EdgeToEdgePhase(0, 1);
    return phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);
  };

  const r1 = run();
  const r2 = run();

  for (let p = 0; p < NUM_PARENT_NODES; p++) {
    for (let d = 0; d < HEAD_DIM; d++) {
      assertEquals(
        r1.embeddings[p][d],
        r2.embeddings[p][d],
        `EE: deterministic output mismatch at [${p}][${d}]`,
      );
    }
  }
});

Deno.test("EE backward: gradient shapes are correct", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);
  const { cache } = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  const dE_kPlus1_new = Array.from({ length: NUM_PARENT_NODES }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_kPlus1_new, cache, params);

  assertEquals(grads.dW_source.length, HEAD_DIM, "EE dW_source rows = headDim");
  assertEquals(grads.dW_source[0].length, EMB_DIM, "EE dW_source cols = embDim");
  assertEquals(grads.dW_target.length, HEAD_DIM, "EE dW_target rows = headDim");
  assertEquals(grads.dW_target[0].length, EMB_DIM, "EE dW_target cols = embDim");
  assertEquals(grads.da_attention.length, 2 * HEAD_DIM, "EE da_attention length = 2*headDim");
  assertEquals(grads.dE_k.length, NUM_CHILD_NODES, "EE dE_k rows = numChild");
  assertEquals(grads.dE_k[0].length, EMB_DIM, "EE dE_k cols = embDim");
  assertEquals(grads.dE_kPlus1.length, NUM_PARENT_NODES, "EE dE_kPlus1 rows = numParent");
  assertEquals(grads.dE_kPlus1[0].length, EMB_DIM, "EE dE_kPlus1 cols = embDim");
});

Deno.test("EE backward: gradients are non-zero for connected nodes", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);
  const { cache } = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  const dE_kPlus1_new = Array.from({ length: NUM_PARENT_NODES }, () =>
    Array.from({ length: HEAD_DIM }, () => 1.0)
  );
  const grads = phase.backward(dE_kPlus1_new, cache, params);

  const dW_source_norm = grads.dW_source.flat().reduce((s, v) => s + v * v, 0);
  assert(dW_source_norm > 1e-10, "EE dW_source should be non-zero");

  // dE_k for child1 (connected to both parents) should be non-zero
  const dE_k1_norm = grads.dE_k[1].reduce((s, v) => s + v * v, 0);
  assert(dE_k1_norm > 1e-10, "EE dE_k[1] should be non-zero (child1 connected to both parents)");
});

Deno.test("EE backward: dW_source is finite and deterministic", () => {
  // dW_source gradient: matmulTranspose bug fixed (now uses matmul for dX_proj^T @ X).
  const run = () => {
    const rng = mulberry32(SEED + 200);
    const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
    const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
    const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
    const conn = denseToSparse(DENSE_EE_INCIDENCE);

    const phase = new EdgeToEdgePhase(0, 1);
    const { cache } = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);
    const dE_kPlus1_new = Array.from({ length: NUM_PARENT_NODES }, () => Array(HEAD_DIM).fill(1.0));
    return phase.backward(dE_kPlus1_new, cache, params);
  };

  const g1 = run();
  const g2 = run();

  for (let r = 0; r < HEAD_DIM; r++) {
    for (let c = 0; c < EMB_DIM; c++) {
      assert(Number.isFinite(g1.dW_source[r][c]), `EE dW_source[${r}][${c}] should be finite`);
    }
  }
  const norm = g1.dW_source.flat().reduce((s, v) => s + v * v, 0);
  assert(norm > 1e-10, "EE dW_source should be non-zero");

  for (let r = 0; r < HEAD_DIM; r++) {
    for (let c = 0; c < EMB_DIM; c++) {
      assertEquals(g1.dW_source[r][c], g2.dW_source[r][c], `EE dW_source[${r}][${c}] deterministic`);
    }
  }
});

Deno.test("EE backward: dE_k matches finite differences", () => {
  const rng = mulberry32(SEED + 200);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);
  const dE_kPlus1_new = Array.from({ length: NUM_PARENT_NODES }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_kPlus1_new, cache, params);

  const eps = 1e-5;
  // Check child1 (connected to both parents)
  for (let d = 0; d < Math.min(3, EMB_DIM); d++) {
    const orig = E_k[1][d];

    E_k[1][d] = orig + eps;
    const lossPlus = computeLoss();

    E_k[1][d] = orig - eps;
    const lossMinus = computeLoss();

    E_k[1][d] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    almostEqual(
      grads.dE_k[1][d],
      numerical,
      1e-3,
      `EE dE_k[1][${d}]: analytical=${grads.dE_k[1][d]}, numerical=${numerical}`,
    );
  }
});

// ============================================================================
// 5. Edge cases: disconnected nodes, single-edge graphs
// ============================================================================

Deno.test("VE forward: disconnected L1 node gets zero embedding", () => {
  // A graph where L1_2 has NO connected L0 nodes
  const dense: number[][] = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
  ];
  const rng = mulberry32(SEED + 300);
  const H = makeSeededEmbeddings(rng, 3, EMB_DIM);
  const E = makeSeededEmbeddings(rng, 3, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(dense);

  const phase = new VertexToEdgePhase();
  const result = phase.forwardWithCache(H, E, conn, params, CONFIG);

  // L1_2 has no connected L0 nodes -> aggregated = zeros -> ELU(0) = 0
  for (let d = 0; d < HEAD_DIM; d++) {
    assertEquals(result.embeddings[2][d], 0, `Disconnected L1_2 embedding[${d}] should be 0`);
  }
});

Deno.test("EV forward: L0 node connected to single L1 node gets all attention on that L1 node", () => {
  // L0_2 is only connected to L1_1
  const rng = mulberry32(SEED + 400);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();
  const result = phase.forwardWithCache(E, H, conn, params, CONFIG);

  // L0_2 is only connected to L1_1 -> softmax over 1 item = 1.0
  almostEqual(result.attention[1][2], 1.0, 1e-6, "EV: L0_2 should have attention=1.0 on L1_1");
  // All other L1 nodes should have 0 attention for L0_2
  assertEquals(result.attention[0][2], 0, "EV: attention[L1_0][L0_2] should be 0");
  assertEquals(result.attention[2][2], 0, "EV: attention[L1_2][L0_2] should be 0");
});

Deno.test("EE forward: single-child parent gets attention=1.0 on that child", () => {
  // parent0 connected to child0, child1
  // parent1 connected to child1, child2
  // Modify: parent with a single child
  const dense: number[][] = [
    [1, 0],  // child0 only in parent0
    [0, 1],  // child1 only in parent1
  ];
  const rng = mulberry32(SEED + 500);
  const E_k = makeSeededEmbeddings(rng, 2, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, 2, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(dense);

  const phase = new EdgeToEdgePhase(0, 1);
  const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);

  // parent0 has only child0 -> attention[child0][parent0] = 1.0
  almostEqual(result.attention[0][0], 1.0, 1e-6, "EE: single-child parent gets attention=1.0");
  // parent1 has only child1 -> attention[child1][parent1] = 1.0
  almostEqual(result.attention[1][1], 1.0, 1e-6, "EE: single-child parent gets attention=1.0");
});

// ============================================================================
// 6. Cross-phase consistency: VE->EV roundtrip shape check
// ============================================================================

Deno.test("VE+EV roundtrip: shapes are consistent through two phases", () => {
  const rng = mulberry32(SEED + 600);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const veParams = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  // EV params project from headDim back, but source/target are headDim
  const evParams: PhaseParameters = {
    W_source: Array.from({ length: HEAD_DIM }, () =>
      Array.from({ length: HEAD_DIM }, () => (rng() - 0.5) * 0.2)
    ),
    W_target: Array.from({ length: HEAD_DIM }, () =>
      Array.from({ length: HEAD_DIM }, () => (rng() - 0.5) * 0.2)
    ),
    a_attention: Array.from({ length: 2 * HEAD_DIM }, () => (rng() - 0.5) * 0.2),
  };

  const conn = denseToSparse(DENSE_INCIDENCE);

  // Phase 1: V -> E
  const vePhase = new VertexToEdgePhase();
  const veResult = vePhase.forwardWithCache(H, E, conn, veParams, CONFIG);

  assertEquals(veResult.embeddings.length, NUM_L1);
  assertEquals(veResult.embeddings[0].length, HEAD_DIM);

  // Phase 2: E -> V (using VE output as source, projected L0 embeddings as target)
  const evPhase = new EdgeToVertexPhase();
  // For EV, source=L1 nodes, target=L0 nodes. The conn orientation stays the same
  // (conn.sourceToTargets = L0->L1) which is what EV expects.
  const H_proj_for_ev = H.map((row) => row.slice(0, HEAD_DIM)); // truncate for dim compat
  const evResult = evPhase.forwardWithCache(veResult.embeddings, H_proj_for_ev, conn, evParams, CONFIG);

  assertEquals(evResult.embeddings.length, NUM_L0, "EV output should have numL0 rows");
  assertEquals(evResult.embeddings[0].length, HEAD_DIM, "EV output should have headDim cols");
});

// ============================================================================
// 7. Larger graph stress test — verify no crashes and attention invariants
// ============================================================================

Deno.test("VE forward: larger graph (20 L0, 10 L1) — attention sums and shapes", () => {
  const nL0 = 20;
  const nL1 = 10;
  const rng = mulberry32(SEED + 700);

  // Generate a random sparse incidence (each L0 node connected to 1-3 L1 nodes)
  const dense: number[][] = Array.from({ length: nL0 }, () => {
    const row = Array(nL1).fill(0);
    const numConns = 1 + Math.floor(rng() * 3);
    for (let i = 0; i < numConns; i++) {
      row[Math.floor(rng() * nL1)] = 1;
    }
    return row;
  });

  const H = makeSeededEmbeddings(rng, nL0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, nL1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(dense);

  const phase = new VertexToEdgePhase();
  const result = phase.forwardWithCache(H, E, conn, params, CONFIG);

  assertEquals(result.embeddings.length, nL1);
  assertEquals(result.embeddings[0].length, HEAD_DIM);

  // Attention sums per L1 node
  for (let c = 0; c < nL1; c++) {
    let sum = 0;
    for (let t = 0; t < nL0; t++) {
      sum += result.attention[t][c];
    }
    const connectedL0 = conn.targetToSources.get(c);
    if (connectedL0 && connectedL0.length > 0) {
      almostEqual(sum, 1.0, 1e-5, `Large VE: attention sum for L1 node ${c}`);
    } else {
      almostEqual(sum, 0.0, 1e-10, `Large VE: disconnected L1 node ${c} attention should be 0`);
    }
  }
});

Deno.test("EV forward: larger graph (20 L0, 10 L1) — attention sums and shapes", () => {
  const nL0 = 20;
  const nL1 = 10;
  const rng = mulberry32(SEED + 800);

  const dense: number[][] = Array.from({ length: nL0 }, () => {
    const row = Array(nL1).fill(0);
    const numConns = 1 + Math.floor(rng() * 3);
    for (let i = 0; i < numConns; i++) {
      row[Math.floor(rng() * nL1)] = 1;
    }
    return row;
  });

  const E = makeSeededEmbeddings(rng, nL1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, nL0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(dense);

  const phase = new EdgeToVertexPhase();
  const result = phase.forwardWithCache(E, H, conn, params, CONFIG);

  assertEquals(result.embeddings.length, nL0);
  assertEquals(result.embeddings[0].length, HEAD_DIM);

  // Attention sums per L0 node
  for (let t = 0; t < nL0; t++) {
    let sum = 0;
    for (let c = 0; c < nL1; c++) {
      sum += result.attention[c][t];
    }
    const connectedL1 = conn.sourceToTargets.get(t);
    if (connectedL1 && connectedL1.length > 0) {
      almostEqual(sum, 1.0, 1e-5, `Large EV: attention sum for L0 node ${t}`);
    } else {
      almostEqual(sum, 0.0, 1e-10, `Large EV: disconnected L0 node ${t} attention should be 0`);
    }
  }
});

// ============================================================================
// 8. Finite difference checks for a_attention and W_target
// ============================================================================

Deno.test("VE backward: da_attention matches finite differences", () => {
  const rng = mulberry32(SEED + 900);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(H, E, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(H, E, conn, params, CONFIG);
  const dE_new = Array.from({ length: NUM_L1 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_new, cache, params);

  const eps = 1e-5;
  for (let i = 0; i < Math.min(4, 2 * HEAD_DIM); i++) {
    const orig = params.a_attention[i];

    params.a_attention[i] = orig + eps;
    const lossPlus = computeLoss();

    params.a_attention[i] = orig - eps;
    const lossMinus = computeLoss();

    params.a_attention[i] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    almostEqual(
      grads.da_attention[i],
      numerical,
      1e-3,
      `VE da_attention[${i}]: analytical=${grads.da_attention[i]}, numerical=${numerical}`,
    );
  }
});

Deno.test("EV backward: da_attention matches finite differences", () => {
  const rng = mulberry32(SEED + 1000);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new EdgeToVertexPhase();

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(E, H, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(E, H, conn, params, CONFIG);
  const dH_new = Array.from({ length: NUM_L0 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dH_new, cache, params);

  const eps = 1e-5;
  for (let i = 0; i < Math.min(4, 2 * HEAD_DIM); i++) {
    const orig = params.a_attention[i];

    params.a_attention[i] = orig + eps;
    const lossPlus = computeLoss();

    params.a_attention[i] = orig - eps;
    const lossMinus = computeLoss();

    params.a_attention[i] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    almostEqual(
      grads.da_attention[i],
      numerical,
      1e-3,
      `EV da_attention[${i}]: analytical=${grads.da_attention[i]}, numerical=${numerical}`,
    );
  }
});

Deno.test("EE backward: da_attention matches finite differences", () => {
  const rng = mulberry32(SEED + 1100);
  const E_k = makeSeededEmbeddings(rng, NUM_CHILD_NODES, EMB_DIM);
  const E_kPlus1 = makeSeededEmbeddings(rng, NUM_PARENT_NODES, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_EE_INCIDENCE);

  const phase = new EdgeToEdgePhase(0, 1);

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(E_k, E_kPlus1, conn, params, CONFIG);
  const dE_kPlus1_new = Array.from({ length: NUM_PARENT_NODES }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_kPlus1_new, cache, params);

  const eps = 1e-5;
  for (let i = 0; i < Math.min(4, 2 * HEAD_DIM); i++) {
    const orig = params.a_attention[i];

    params.a_attention[i] = orig + eps;
    const lossPlus = computeLoss();

    params.a_attention[i] = orig - eps;
    const lossMinus = computeLoss();

    params.a_attention[i] = orig;

    const numerical = (lossPlus - lossMinus) / (2 * eps);
    almostEqual(
      grads.da_attention[i],
      numerical,
      1e-3,
      `EE da_attention[${i}]: analytical=${grads.da_attention[i]}, numerical=${numerical}`,
    );
  }
});

Deno.test("VE backward: dW_target matches finite differences", () => {
  const rng = mulberry32(SEED + 1200);
  const H = makeSeededEmbeddings(rng, NUM_L0, EMB_DIM);
  const E = makeSeededEmbeddings(rng, NUM_L1, EMB_DIM);
  const params = makeSeededParams(rng, HEAD_DIM, EMB_DIM);
  const conn = denseToSparse(DENSE_INCIDENCE);

  const phase = new VertexToEdgePhase();

  const computeLoss = (): number => {
    const result = phase.forwardWithCache(H, E, conn, params, CONFIG);
    return result.embeddings.flat().reduce((s, v) => s + v, 0);
  };

  const { cache } = phase.forwardWithCache(H, E, conn, params, CONFIG);
  const dE_new = Array.from({ length: NUM_L1 }, () => Array(HEAD_DIM).fill(1.0));
  const grads = phase.backward(dE_new, cache, params);

  const eps = 1e-5;
  for (let r = 0; r < Math.min(2, HEAD_DIM); r++) {
    for (let c = 0; c < Math.min(2, EMB_DIM); c++) {
      const orig = params.W_target[r][c];

      params.W_target[r][c] = orig + eps;
      const lossPlus = computeLoss();

      params.W_target[r][c] = orig - eps;
      const lossMinus = computeLoss();

      params.W_target[r][c] = orig;

      const numerical = (lossPlus - lossMinus) / (2 * eps);
      almostEqual(
        grads.dW_target[r][c],
        numerical,
        1e-3,
        `VE dW_target[${r}][${c}]: analytical=${grads.dW_target[r][c]}, numerical=${numerical}`,
      );
    }
  }
});
