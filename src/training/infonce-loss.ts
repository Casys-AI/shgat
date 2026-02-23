/**
 * InfoNCE Loss and Analytical Gradient
 *
 * For contrastive learning: rank the positive higher than K negatives.
 *
 * Loss:
 *   L = -log(exp(s_pos/τ) / Σ_j exp(s_j/τ))
 *     = -s_pos/τ + logsumexp(s/τ)
 *
 * Gradient (∂L/∂s_j):
 *   ∂L/∂s_j = (softmax(s/τ)[j] - 1_{j=posIdx}) / τ
 *
 * Uses numerically stable logsumexp to avoid overflow.
 *
 * @module shgat-tf/training/infonce-loss
 */

/**
 * Compute InfoNCE loss for a single example.
 *
 * @param scores - Raw logit scores for all candidates [N]
 * @param posIdx - Index of the positive candidate
 * @param tau - Temperature (lower = sharper)
 * @returns Scalar loss value (non-negative)
 */
export function infoNCELoss(scores: number[], posIdx: number, tau: number): number {
  const N = scores.length;
  const invTau = 1 / tau;

  // Numerically stable logsumexp: log(Σ exp(x_i)) = max + log(Σ exp(x_i - max))
  const scaled = new Array(N);
  for (let i = 0; i < N; i++) {
    scaled[i] = scores[i] * invTau;
  }

  let max = scaled[0];
  for (let i = 1; i < N; i++) {
    if (scaled[i] > max) max = scaled[i];
  }

  let sumExp = 0;
  for (let i = 0; i < N; i++) {
    sumExp += Math.exp(scaled[i] - max);
  }

  const logsumexp = max + Math.log(sumExp);
  return -scaled[posIdx] + logsumexp;
}

/**
 * Compute InfoNCE gradient with respect to scores.
 *
 * ∂L/∂s_j = (softmax(s/τ)[j] - 1_{j=posIdx}) / τ
 *
 * @param scores - Raw logit scores for all candidates [N]
 * @param posIdx - Index of the positive candidate
 * @param tau - Temperature
 * @returns Gradient array [N], same length as scores
 */
export function infoNCEGradient(scores: number[], posIdx: number, tau: number): number[] {
  const N = scores.length;
  const invTau = 1 / tau;

  // Compute softmax(s/τ) with numerical stability
  const scaled = new Array(N);
  for (let i = 0; i < N; i++) {
    scaled[i] = scores[i] * invTau;
  }

  let max = scaled[0];
  for (let i = 1; i < N; i++) {
    if (scaled[i] > max) max = scaled[i];
  }

  const exps = new Array(N);
  let sumExp = 0;
  for (let i = 0; i < N; i++) {
    exps[i] = Math.exp(scaled[i] - max);
    sumExp += exps[i];
  }

  // gradient[j] = (p_j - 1_{j=posIdx}) / τ
  const gradient = new Array(N);
  for (let i = 0; i < N; i++) {
    const p = exps[i] / sumExp;
    const indicator = i === posIdx ? 1 : 0;
    gradient[i] = (p - indicator) * invTau;
  }

  return gradient;
}

/**
 * Compute both InfoNCE loss and gradient in a single pass (shared softmax computation).
 *
 * @param scores - Raw logit scores for all candidates [N]
 * @param posIdx - Index of the positive candidate
 * @param tau - Temperature
 * @returns { loss, gradient }
 */
export function infoNCELossAndGradient(
  scores: number[],
  posIdx: number,
  tau: number,
): { loss: number; gradient: number[] } {
  const N = scores.length;
  const invTau = 1 / tau;

  // Scaled scores
  const scaled = new Array(N);
  for (let i = 0; i < N; i++) {
    scaled[i] = scores[i] * invTau;
  }

  // Stable logsumexp + softmax
  let max = scaled[0];
  for (let i = 1; i < N; i++) {
    if (scaled[i] > max) max = scaled[i];
  }

  const exps = new Array(N);
  let sumExp = 0;
  for (let i = 0; i < N; i++) {
    exps[i] = Math.exp(scaled[i] - max);
    sumExp += exps[i];
  }

  // Loss = -s_pos/τ + logsumexp
  const logsumexp = max + Math.log(sumExp);
  const loss = -scaled[posIdx] + logsumexp;

  // Gradient
  const gradient = new Array(N);
  for (let i = 0; i < N; i++) {
    const p = exps[i] / sumExp;
    const indicator = i === posIdx ? 1 : 0;
    gradient[i] = (p - indicator) * invTau;
  }

  return { loss, gradient };
}
