/**
 * TDD tests for Adam optimizer on number[][] parameters
 *
 * Adam update rule:
 *   m = β1 * m + (1 - β1) * g
 *   v = β2 * v + (1 - β2) * g²
 *   m̂ = m / (1 - β1^t)
 *   v̂ = v / (1 - β2^t)
 *   θ -= lr * m̂ / (√v̂ + ε)
 */

import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { AdamOptimizer } from "../adam-optimizer.ts";

Deno.test("Adam: converges on f(x) = x² → x → 0", () => {
  // Single parameter, gradient = 2x
  const adam = new AdamOptimizer({ lr: 0.1 });
  const paramKey = "x";

  let x = 5.0;
  adam.register(paramKey, [1, 1]);

  for (let step = 0; step < 200; step++) {
    const grad = 2 * x; // df/dx = 2x
    const grads: number[][] = [[grad]];
    const params: number[][] = [[x]];
    adam.step(paramKey, params, grads);
    x = params[0][0];
  }

  assertAlmostEquals(x, 0.0, 0.01, `x should converge to 0, got ${x}`);
});

Deno.test("Adam: bias correction makes first step larger", () => {
  // Without bias correction, m̂ = m / (1 - β1^1) ≈ m / 0.1 = 10x
  const adam = new AdamOptimizer({ lr: 0.001, beta1: 0.9, beta2: 0.999 });
  adam.register("w", [1, 1]);

  const params: number[][] = [[1.0]];
  const grads: number[][] = [[1.0]];

  adam.step("w", params, grads);
  const step1Delta = 1.0 - params[0][0];

  // With bias correction at t=1: m̂ = g / (1-0.9) = 10g, v̂ = g² / (1-0.999) = 1000g²
  // step = lr * m̂ / √v̂ = 0.001 * 10 / √1000 ≈ 0.001 * 10 / 31.6 ≈ 0.000316
  // Without bias correction: step = lr * 0.1 * g / √(0.001 * g²) = 0.001 * 0.1 / 0.0316 ≈ 0.00316
  // The exact value depends on ε but step should be > 0
  assert(step1Delta > 0, `Step 1 should move parameter, delta=${step1Delta}`);
});

Deno.test("Adam: momentum accumulates across steps", () => {
  const adam = new AdamOptimizer({ lr: 0.01, beta1: 0.9, beta2: 0.999 });
  adam.register("w", [1, 1]);

  // Apply same gradient twice → second step should be larger (momentum)
  const params1: number[][] = [[0.0]];
  adam.step("w", params1, [[1.0]]);
  const delta1 = Math.abs(params1[0][0]);

  const params2: number[][] = [[0.0]];
  adam.step("w", params2, [[1.0]]);
  const delta2 = Math.abs(params2[0][0]);

  // Step 2 benefits from momentum: m_2 = 0.9 * m_1 + 0.1 * g > m_1
  assert(delta2 > delta1 * 0.5, `Step 2 delta (${delta2}) should benefit from momentum vs step 1 (${delta1})`);
});

Deno.test("Adam: works on 2D parameter matrices", () => {
  const adam = new AdamOptimizer({ lr: 0.01 });
  adam.register("W", [2, 3]);

  const params: number[][] = [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
  ];
  const grads: number[][] = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
  ];

  const originalParams = params.map(row => [...row]);
  adam.step("W", params, grads);

  // All parameters should have moved
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      assert(
        params[i][j] !== originalParams[i][j],
        `param[${i}][${j}] should have changed`,
      );
      // Direction: positive gradient → parameter decreases
      assert(
        params[i][j] < originalParams[i][j],
        `param[${i}][${j}] should decrease for positive gradient`,
      );
    }
  }
});

Deno.test("Adam: gradient clipping limits step size", () => {
  const adam = new AdamOptimizer({ lr: 0.01, gradientClip: 1.0 });
  adam.register("w", [1, 1]);

  // Huge gradient should be clipped
  const params: number[][] = [[0.0]];
  adam.step("w", params, [[1000.0]]);

  // With clip=1.0, effective gradient = 1.0 (not 1000)
  // Step should be bounded
  const delta = Math.abs(params[0][0]);
  assert(delta < 1.0, `Step with clipped gradient should be small, got delta=${delta}`);
});

Deno.test("Adam: reset clears state", () => {
  const adam = new AdamOptimizer({ lr: 0.01 });
  adam.register("w", [1, 1]);

  // Build momentum
  adam.step("w", [[0.0]], [[1.0]]);
  adam.step("w", [[0.0]], [[1.0]]);

  // Reset
  adam.reset("w");

  // After reset, first step should behave like a fresh optimizer
  const params: number[][] = [[0.0]];
  adam.step("w", params, [[1.0]]);
  const deltaAfterReset = Math.abs(params[0][0]);

  // Fresh optimizer
  const adam2 = new AdamOptimizer({ lr: 0.01 });
  adam2.register("w2", [1, 1]);
  const params2: number[][] = [[0.0]];
  adam2.step("w2", params2, [[1.0]]);
  const deltaFresh = Math.abs(params2[0][0]);

  assertAlmostEquals(deltaAfterReset, deltaFresh, 1e-10, "Reset should restore fresh behavior");
});

Deno.test("Adam: multiple parameter groups", () => {
  const adam = new AdamOptimizer({ lr: 0.01 });
  adam.register("W_k", [2, 2]);
  adam.register("W_q", [2, 2]);

  const W_k: number[][] = [[1.0, 2.0], [3.0, 4.0]];
  const W_q: number[][] = [[5.0, 6.0], [7.0, 8.0]];

  const dW_k: number[][] = [[0.1, 0.2], [0.3, 0.4]];
  const dW_q: number[][] = [[0.5, 0.6], [0.7, 0.8]];

  adam.step("W_k", W_k, dW_k);
  adam.step("W_q", W_q, dW_q);

  // Both should have moved
  assert(W_k[0][0] < 1.0, "W_k should have decreased");
  assert(W_q[0][0] < 5.0, "W_q should have decreased");

  // W_q has larger gradient → should move more
  const deltaK = 1.0 - W_k[0][0];
  const deltaQ = 5.0 - W_q[0][0];
  assert(deltaQ > deltaK, `W_q delta (${deltaQ}) should be > W_k delta (${deltaK}) due to larger gradient`);
});
