/**
 * Prioritized Experience Replay (PER) Buffer
 *
 * Samples training examples proportionally to their TD error magnitude.
 * Higher TD error = more "surprising" example = sampled more often.
 *
 * Based on: "Prioritized Experience Replay" (Schaul et al., 2015)
 *
 * Key concepts:
 * - Priority p_i = |TD_error_i| + ε (small constant to ensure non-zero)
 * - Sampling probability P(i) = p_i^α / Σ p_j^α
 * - Importance sampling weight w_i = (N * P(i))^(-β) for bias correction
 *
 * @module graphrag/algorithms/shgat/training/per-buffer
 */

import { random } from "../initialization/parameters.ts";

/**
 * PER hyperparameters
 */
export interface PERConfig {
  /** Priority exponent (0 = uniform, 1 = full prioritization). Default: 0.6 */
  alpha: number;
  /** IS weight exponent (0 = no correction, 1 = full correction). Default: 0.4, annealed to 1 */
  beta: number;
  /** Small constant to ensure non-zero priorities. Default: 1e-6 */
  epsilon: number;
  /** Maximum priority for new examples. Default: 1.0 */
  maxPriority: number;
}

const DEFAULT_PER_CONFIG: PERConfig = {
  alpha: 0.6,
  beta: 0.4,
  epsilon: 0.01, // Minimum priority floor (prevents starvation of "easy" examples)
  maxPriority: 1.0,
};

/**
 * PER Buffer for prioritized sampling
 */
export class PERBuffer<T> {
  private items: T[];
  private priorities: number[];
  private config: PERConfig;

  constructor(items: T[], config: Partial<PERConfig> = {}) {
    this.items = items;
    this.config = { ...DEFAULT_PER_CONFIG, ...config };
    // Initialize all priorities to max (ensures new examples are sampled)
    this.priorities = new Array(items.length).fill(this.config.maxPriority);
  }

  /**
   * Get buffer size
   */
  get size(): number {
    return this.items.length;
  }

  /**
   * Sample a batch using prioritized sampling
   *
   * @param batchSize Number of examples to sample
   * @param beta Current beta value for IS weights (can be annealed)
   * @returns Sampled items, their indices, and importance sampling weights
   */
  sample(
    batchSize: number,
    beta?: number,
  ): { items: T[]; indices: number[]; weights: number[] } {
    const { alpha, epsilon } = this.config;
    const currentBeta = beta ?? this.config.beta;
    const n = this.items.length;

    if (batchSize >= n) {
      // Return all items with uniform weights
      return {
        items: [...this.items],
        indices: this.items.map((_, i) => i),
        weights: new Array(n).fill(1.0),
      };
    }

    // Compute sampling probabilities: P(i) = p_i^α / Σ p_j^α
    const scaledPriorities = this.priorities.map((p) => Math.pow(p + epsilon, alpha));
    const sumPriorities = scaledPriorities.reduce((a, b) => a + b, 0);
    const probs = scaledPriorities.map((p) => p / sumPriorities);

    // Sample indices according to probabilities (without replacement)
    const sampledIndices = this.sampleWithoutReplacement(probs, batchSize);

    // Compute importance sampling weights: w_i = (N * P(i))^(-β)
    // Normalized by max weight for stability
    const rawWeights = sampledIndices.map((i) => Math.pow(n * probs[i], -currentBeta));
    const maxWeight = Math.max(...rawWeights);
    const weights = rawWeights.map((w) => w / maxWeight);

    return {
      items: sampledIndices.map((i) => this.items[i]),
      indices: sampledIndices,
      weights,
    };
  }

  /**
   * Update priorities for sampled examples based on TD errors
   *
   * @param indices Indices of examples that were trained on
   * @param tdErrors Absolute TD errors for each example
   */
  updatePriorities(indices: number[], tdErrors: number[]): void {
    const { epsilon, maxPriority } = this.config;

    for (let i = 0; i < indices.length; i++) {
      const idx = indices[i];
      const tdError = Math.abs(tdErrors[i]);
      const newPriority = Math.min(tdError + epsilon, maxPriority);

      if (newPriority > this.priorities[idx]) {
        // Hard example → direct growth (allows high end to increase)
        this.priorities[idx] = newPriority;
      } else {
        // Easy example → gradual decay (prevents brutal drops)
        this.priorities[idx] = 0.7 * this.priorities[idx] + 0.3 * newPriority;
      }
    }

    // Update max priority for future new examples
    const currentMax = Math.max(...this.priorities);
    if (currentMax > this.config.maxPriority) {
      this.config.maxPriority = currentMax;
    }
  }

  /**
   * Get current priority statistics
   */
  getStats(): { mean: number; max: number; min: number; std: number } {
    const n = this.priorities.length;
    const mean = this.priorities.reduce((a, b) => a + b, 0) / n;
    const max = Math.max(...this.priorities);
    const min = Math.min(...this.priorities);
    const variance = this.priorities.reduce((sum, p) => sum + (p - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    return { mean, max, min, std };
  }

  /**
   * Decay priorities toward mean to prevent starvation.
   * Only decays priorities ABOVE mean (high priorities can't stay forever).
   * Priorities below mean stay low (easy examples remain easy).
   *
   * @param decay Decay factor (0.9 = slow decay, 0.5 = fast decay). Default: 0.9
   */
  decayPriorities(decay: number = 0.9): void {
    const mean = this.priorities.reduce((a, b) => a + b, 0) / this.priorities.length;
    for (let i = 0; i < this.priorities.length; i++) {
      if (this.priorities[i] > mean) {
        // Decay only above-mean (prevents high priorities from staying forever)
        this.priorities[i] = this.priorities[i] * decay + mean * (1 - decay);
      }
      // Below mean: no decay (easy examples stay low, maintaining range)
    }
  }

  /**
   * Sample indices without replacement according to probabilities
   */
  private sampleWithoutReplacement(probs: number[], k: number): number[] {
    const selected: number[] = [];
    const remaining = [...probs];
    const indices = probs.map((_, i) => i);

    for (let i = 0; i < k; i++) {
      // Normalize remaining probabilities
      const sum = remaining.reduce((a, b) => a + b, 0);
      if (sum <= 0) break;

      // Sample one index
      const r = random() * sum;
      let cumSum = 0;
      let selectedIdx = 0;

      for (let j = 0; j < remaining.length; j++) {
        cumSum += remaining[j];
        if (r <= cumSum) {
          selectedIdx = j;
          break;
        }
      }

      // Add to selected and remove from remaining
      selected.push(indices[selectedIdx]);
      remaining.splice(selectedIdx, 1);
      indices.splice(selectedIdx, 1);
    }

    return selected;
  }
}

/**
 * Compute annealed beta value
 *
 * Beta starts at beta_start and linearly anneals to 1.0 over training.
 *
 * @param epoch Current epoch (0-indexed)
 * @param totalEpochs Total number of epochs
 * @param betaStart Starting beta value
 * @returns Annealed beta value
 */
export function annealBeta(epoch: number, totalEpochs: number, betaStart: number = 0.4): number {
  const progress = Math.min(epoch / Math.max(totalEpochs - 1, 1), 1.0);
  return betaStart + progress * (1.0 - betaStart);
}

/**
 * Cosine annealing for temperature τ (InfoNCE contrastive learning).
 * Starts high (soft probabilities, exploration) → ends low (sharp, focus on hard distinctions).
 *
 * τ(t) = τ_end + (τ_start - τ_end) * 0.5 * (1 + cos(π * t / T))
 *
 * @param epoch Current epoch (0-indexed)
 * @param totalEpochs Total number of epochs
 * @param tauStart Starting temperature (default 0.10 - soft, exploratory)
 * @param tauEnd Ending temperature (default 0.06 - sharp, discriminative)
 * @returns Annealed temperature value
 */
export function annealTemperature(
  epoch: number,
  totalEpochs: number,
  tauStart: number = 0.10,
  tauEnd: number = 0.06
): number {
  const progress = Math.min(epoch / Math.max(totalEpochs - 1, 1), 1.0);
  // Cosine annealing: smooth descent, slows down toward the end
  return tauEnd + (tauStart - tauEnd) * 0.5 * (1 + Math.cos(Math.PI * progress));
}
