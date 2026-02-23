/**
 * TDD tests for InfoNCE loss + analytical gradient
 *
 * InfoNCE loss:
 *   L = -log(exp(s_pos/τ) / Σ_j exp(s_j/τ))
 *     = -s_pos/τ + logsumexp(s/τ)
 *
 * Analytical gradient:
 *   ∂L/∂s_j = softmax(s/τ)[j] - (j === posIdx ? 1 : 0)) / τ
 *           = (p_j - 1_{j=pos}) / τ
 *
 * where p_j = exp(s_j/τ) / Σ_k exp(s_k/τ)
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/assert_equals.ts";
import { assertAlmostEquals } from "https://deno.land/std@0.224.0/assert/assert_almost_equals.ts";
import { assert } from "https://deno.land/std@0.224.0/assert/assert.ts";
import { infoNCELoss, infoNCEGradient, infoNCELossAndGradient } from "../infonce-loss.ts";

// ============================================================================
// Loss tests
// ============================================================================

Deno.test("infoNCELoss: perfect positive → loss ≈ 0", () => {
  // Positive score much higher than negatives → loss approaches 0
  const scores = [10.0, -5.0, -5.0, -5.0, -5.0];
  const posIdx = 0;
  const tau = 0.07;

  const loss = infoNCELoss(scores, posIdx, tau);
  // exp(10/0.07) >> exp(-5/0.07), so softmax(pos) ≈ 1, loss ≈ 0
  assertAlmostEquals(loss, 0, 1e-3);
});

Deno.test("infoNCELoss: uniform scores → loss = log(N)", () => {
  // All scores equal → softmax = 1/N → loss = log(N)
  const N = 5;
  const scores = Array(N).fill(1.0);
  const posIdx = 0;
  const tau = 1.0; // τ=1 so scores/τ = scores

  const loss = infoNCELoss(scores, posIdx, tau);
  assertAlmostEquals(loss, Math.log(N), 1e-6);
});

Deno.test("infoNCELoss: loss is non-negative", () => {
  const scores = [2.0, 1.0, 3.0, 0.5];
  const loss = infoNCELoss(scores, 0, 0.1);
  assert(loss >= 0, `Loss should be >= 0, got ${loss}`);
});

Deno.test("infoNCELoss: lower τ → sharper distinction", () => {
  const scores = [2.0, 1.5, 1.0, 0.5];
  const posIdx = 0;

  const lossHighTau = infoNCELoss(scores, posIdx, 1.0);
  const lossLowTau = infoNCELoss(scores, posIdx, 0.1);

  // Lower τ amplifies the gap → lower loss when positive is highest
  assert(lossLowTau < lossHighTau, `τ=0.1 loss (${lossLowTau}) should be < τ=1.0 loss (${lossHighTau})`);
});

Deno.test("infoNCELoss: positive not at index 0", () => {
  const scores = [1.0, 5.0, 1.0, 1.0];
  const posIdx = 1;
  const tau = 0.1;

  const loss = infoNCELoss(scores, posIdx, tau);
  // Positive has highest score → loss should be small
  assert(loss < 0.1, `Loss should be small when positive has highest score, got ${loss}`);
});

// ============================================================================
// Gradient tests
// ============================================================================

Deno.test("infoNCEGradient: uniform scores → gradient is symmetric", () => {
  const N = 4;
  const scores = Array(N).fill(1.0);
  const posIdx = 0;
  const tau = 1.0;

  const grad = infoNCEGradient(scores, posIdx, tau);

  assertEquals(grad.length, N);
  // For positive: (1/N - 1) / τ = (1/4 - 1) / 1 = -3/4
  assertAlmostEquals(grad[posIdx], (1 / N - 1) / tau, 1e-6);
  // For negatives: (1/N - 0) / τ = 1/4
  for (let i = 0; i < N; i++) {
    if (i !== posIdx) {
      assertAlmostEquals(grad[i], (1 / N) / tau, 1e-6);
    }
  }
});

Deno.test("infoNCEGradient: gradient sums to 0 (for any τ)", () => {
  // ∂L/∂s_j = (p_j - 1_{pos}) / τ
  // Σ_j ∂L/∂s_j = (Σ p_j - 1) / τ = (1 - 1) / τ = 0
  const scores = [3.0, 1.0, 2.0, 0.5, 4.0];
  const posIdx = 2;
  const tau = 0.07;

  const grad = infoNCEGradient(scores, posIdx, tau);
  const sum = grad.reduce((a, b) => a + b, 0);
  assertAlmostEquals(sum, 0, 1e-6);
});

Deno.test("infoNCEGradient: matches finite differences", () => {
  const scores = [2.0, 1.5, 0.8, 3.0, 1.2];
  const posIdx = 0;
  const tau = 0.5;
  const eps = 1e-5;

  const analyticalGrad = infoNCEGradient(scores, posIdx, tau);

  // Numerical gradient via central finite differences
  for (let i = 0; i < scores.length; i++) {
    const scoresPlus = [...scores];
    const scoresMinus = [...scores];
    scoresPlus[i] += eps;
    scoresMinus[i] -= eps;

    const lossPlus = infoNCELoss(scoresPlus, posIdx, tau);
    const lossMinus = infoNCELoss(scoresMinus, posIdx, tau);
    const numericalGrad = (lossPlus - lossMinus) / (2 * eps);

    assertAlmostEquals(
      analyticalGrad[i],
      numericalGrad,
      1e-4,
      `Gradient mismatch at index ${i}: analytical=${analyticalGrad[i]}, numerical=${numericalGrad}`,
    );
  }
});

Deno.test("infoNCEGradient: perfect positive → gradient ≈ 0", () => {
  // When positive score is overwhelmingly high, softmax(pos) ≈ 1
  // → grad[pos] = (1 - 1)/τ ≈ 0, grad[neg] = (0 - 0)/τ ≈ 0
  const scores = [100.0, -100.0, -100.0];
  const posIdx = 0;
  const tau = 1.0;

  const grad = infoNCEGradient(scores, posIdx, tau);
  for (const g of grad) {
    assertAlmostEquals(g, 0, 1e-6);
  }
});

// ============================================================================
// Combined loss + gradient
// ============================================================================

Deno.test("infoNCELossAndGradient: returns consistent loss and gradient", () => {
  const scores = [2.0, 1.0, 3.0, 0.5];
  const posIdx = 2;
  const tau = 0.3;

  const { loss, gradient } = infoNCELossAndGradient(scores, posIdx, tau);

  // Should match individual calls
  const expectedLoss = infoNCELoss(scores, posIdx, tau);
  const expectedGrad = infoNCEGradient(scores, posIdx, tau);

  assertAlmostEquals(loss, expectedLoss, 1e-10);
  for (let i = 0; i < gradient.length; i++) {
    assertAlmostEquals(gradient[i], expectedGrad[i], 1e-10);
  }
});

Deno.test("infoNCELossAndGradient: batch of examples", () => {
  // Test that we can compute loss/gradient for multiple examples
  const examples = [
    { scores: [3.0, 1.0, 0.5], posIdx: 0 },
    { scores: [1.0, 3.0, 0.5], posIdx: 1 },
    { scores: [0.5, 1.0, 3.0], posIdx: 2 },
  ];
  const tau = 0.5;

  let totalLoss = 0;
  const avgGrad = [0, 0, 0];

  for (const ex of examples) {
    const { loss, gradient } = infoNCELossAndGradient(ex.scores, ex.posIdx, tau);
    totalLoss += loss;
    for (let i = 0; i < gradient.length; i++) {
      avgGrad[i] += gradient[i] / examples.length;
    }
  }

  // All examples have the same structure (highest score = positive)
  // → each loss should be small (positive always highest)
  assert(totalLoss / examples.length < 0.5, `Average loss should be small, got ${totalLoss / examples.length}`);

  // Average gradient across symmetric examples should be ≈ uniform
  // (each position is positive once)
  const gradRange = Math.max(...avgGrad) - Math.min(...avgGrad);
  assert(gradRange < 0.5, `Average gradient should be roughly uniform, range=${gradRange}`);
});
