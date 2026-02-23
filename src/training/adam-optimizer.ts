/**
 * Adam Optimizer for number[][] parameters
 *
 * Operates on plain JS arrays (not TF.js tensors) for use with
 * the manual backward pass training pipeline (shgat-ob approach).
 *
 * Adam update rule:
 *   m = β1 * m + (1 - β1) * g
 *   v = β2 * v + (1 - β2) * g²
 *   m̂ = m / (1 - β1^t)
 *   v̂ = v / (1 - β2^t)
 *   θ -= lr * m̂ / (√v̂ + ε)
 *
 * @module shgat-tf/training/adam-optimizer
 */

export interface AdamConfig {
  /** Learning rate (default 0.001) */
  lr: number;
  /** First moment decay (default 0.9) */
  beta1?: number;
  /** Second moment decay (default 0.999) */
  beta2?: number;
  /** Numerical stability (default 1e-8) */
  epsilon?: number;
  /** Gradient clipping by norm (0 = disabled) */
  gradientClip?: number;
}

interface ParamState {
  /** First moment (moving average of gradient) */
  m: number[][];
  /** Second moment (moving average of gradient²) */
  v: number[][];
  /** Timestep counter */
  t: number;
  /** Shape [rows, cols] */
  shape: [number, number];
}

export class AdamOptimizer {
  private _lr: number;
  private readonly beta1: number;
  private readonly beta2: number;
  private readonly epsilon: number;
  private readonly gradientClip: number;
  private readonly states = new Map<string, ParamState>();

  constructor(config: AdamConfig) {
    this._lr = config.lr;
    this.beta1 = config.beta1 ?? 0.9;
    this.beta2 = config.beta2 ?? 0.999;
    this.epsilon = config.epsilon ?? 1e-8;
    this.gradientClip = config.gradientClip ?? 0;
  }

  /** Get current learning rate */
  get lr(): number { return this._lr; }

  /** Set learning rate (for per-epoch scheduling) */
  set lr(value: number) { this._lr = value; }

  /**
   * Register a parameter group with its shape.
   * Must be called before step().
   */
  register(key: string, shape: [number, number]): void {
    const [rows, cols] = shape;
    this.states.set(key, {
      m: zeros2D(rows, cols),
      v: zeros2D(rows, cols),
      t: 0,
      shape,
    });
  }

  /**
   * Apply one Adam step to parameters in-place.
   *
   * @param key - Parameter group key (must be registered)
   * @param params - Parameter matrix [rows][cols] (modified in-place)
   * @param grads - Gradient matrix [rows][cols]
   */
  step(key: string, params: number[][], grads: number[][]): void {
    const state = this.states.get(key);
    if (!state) {
      throw new Error(`[Adam] Parameter '${key}' not registered. Call register() first.`);
    }

    state.t += 1;
    const { m, v, t } = state;
    const { _lr: lr, beta1, beta2, epsilon } = this;

    // Bias correction factors
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);

    // Optional gradient clipping
    let clippedGrads = grads;
    if (this.gradientClip > 0) {
      clippedGrads = clipGradients(grads, this.gradientClip);
    }

    // Update each element
    const rows = params.length;
    for (let i = 0; i < rows; i++) {
      const cols = params[i].length;
      for (let j = 0; j < cols; j++) {
        const g = clippedGrads[i]?.[j] ?? 0;

        // Update moments
        m[i][j] = beta1 * m[i][j] + (1 - beta1) * g;
        v[i][j] = beta2 * v[i][j] + (1 - beta2) * g * g;

        // Bias-corrected moments
        const mHat = m[i][j] / bc1;
        const vHat = v[i][j] / bc2;

        // Parameter update
        params[i][j] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
      }
    }
  }

  /**
   * Reset optimizer state for a parameter group.
   */
  reset(key: string): void {
    const state = this.states.get(key);
    if (!state) return;
    const [rows, cols] = state.shape;
    state.m = zeros2D(rows, cols);
    state.v = zeros2D(rows, cols);
    state.t = 0;
  }

  /**
   * Reset all parameter states.
   */
  resetAll(): void {
    for (const key of this.states.keys()) {
      this.reset(key);
    }
  }
}

function zeros2D(rows: number, cols: number): number[][] {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

function clipGradients(grads: number[][], maxNorm: number): number[][] {
  let normSq = 0;
  for (const row of grads) {
    for (const val of row) {
      normSq += val * val;
    }
  }
  const norm = Math.sqrt(normSq);
  if (norm <= maxNorm) return grads;

  // In-place scaling to avoid allocating a new matrix every call.
  // With 33 Adam calls per KL batch × 63 batches = 2079 calls/epoch,
  // the old grads.map().map() pattern caused heavy GC pressure.
  const scale = maxNorm / norm;
  for (const row of grads) {
    for (let j = 0; j < row.length; j++) {
      row[j] *= scale;
    }
  }
  return grads;
}
