/**
 * Mathematical utilities for SHGAT-TF
 *
 * Wrapper around TensorFlow.js ops with JS array interface.
 * Provides backward compatibility while using TF.js under the hood.
 *
 * For new code, prefer using tf/ops.ts directly with tensors.
 *
 * @module shgat-tf/utils/math
 */

import { tf, tidy } from "../tf/backend.ts";
import * as ops from "../tf/ops.ts";
import { random } from "../initialization/parameters.ts";

// ============================================================================
// Matrix Operations (TF.js backed)
// ============================================================================

/**
 * Matrix multiplication with transpose: A · B^T
 */
export function matmulTranspose(A: number[][], B: number[][]): number[][] {
  return tidy(() => {
    const tA = ops.toTensor(A);
    const tB = ops.toTensor(B);
    const result = ops.matmulTranspose(tA, tB);
    return result.arraySync() as number[][];
  });
}

/**
 * Standard matrix multiplication: A · B
 */
export function matmul(A: number[][], B: number[][]): number[][] {
  return tidy(() => {
    const tA = ops.toTensor(A);
    const tB = ops.toTensor(B);
    const result = ops.matmul(tA, tB);
    return result.arraySync() as number[][];
  });
}

/**
 * Matrix-vector multiplication: M · v
 */
export function matVec(M: number[][], v: number[]): number[] {
  return tidy(() => {
    const tM = ops.toTensor(M);
    const tV = ops.toTensor(v);
    const result = ops.matVec(tM, tV);
    return result.arraySync() as number[];
  });
}

/**
 * Transpose a matrix
 */
export function transpose(M: number[][]): number[][] {
  return tidy(() => {
    const tM = ops.toTensor(M);
    const result = ops.transpose(tM);
    return result.arraySync() as number[][];
  });
}

// ============================================================================
// Activation Functions (TF.js backed)
// ============================================================================

/**
 * Leaky ReLU activation
 */
export function leakyRelu(x: number, slope: number = 0.2): number {
  return x > 0 ? x : slope * x;
}

/**
 * Leaky ReLU for arrays
 */
export function leakyReluArray(x: number[], slope: number = 0.2): number[] {
  return tidy(() => {
    const tensor = ops.toTensor(x);
    const result = ops.leakyRelu(tensor, slope);
    return result.arraySync() as number[];
  });
}

/**
 * ELU activation
 */
export function elu(x: number, alpha: number = 1.0): number {
  return x >= 0 ? x : alpha * (Math.exp(x) - 1);
}

/**
 * Sigmoid activation
 */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Sigmoid for arrays
 */
export function sigmoidArray(x: number[]): number[] {
  return tidy(() => {
    const tensor = ops.toTensor(x);
    const result = ops.sigmoid(tensor);
    return result.arraySync() as number[];
  });
}

/**
 * Softmax function (TF.js backed, numerically stable)
 */
export function softmax(values: number[]): number[] {
  if (values.length === 0) return [];
  return tidy(() => {
    const tensor = ops.toTensor(values);
    const result = ops.softmax(tensor);
    return result.arraySync() as number[];
  });
}

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Dot product of two vectors
 */
export function dot(a: number[], b: number[]): number {
  return tidy(() => {
    const tA = ops.toTensor(a);
    const tB = ops.toTensor(b);
    return ops.dot(tA, tB).arraySync() as number;
  });
}

/**
 * Cosine similarity between two vectors
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  return tidy(() => {
    const tA = ops.toTensor(a);
    const tB = ops.toTensor(b);
    return ops.cosineSimilarity(tA, tB).arraySync() as number;
  });
}

/**
 * Normalize vector to unit length
 */
export function normalize(vector: number[]): number[] {
  return tidy(() => {
    const tensor = ops.toTensor(vector);
    const result = ops.l2Normalize(tensor);
    return result.arraySync() as number[];
  });
}

// ============================================================================
// Loss Functions
// ============================================================================

/**
 * Binary cross-entropy loss
 */
export function binaryCrossEntropy(pred: number, label: number): number {
  const eps = 1e-7;
  const p = Math.max(eps, Math.min(1 - eps, pred));
  return -label * Math.log(p) - (1 - label) * Math.log(1 - p);
}

// ============================================================================
// Pooling Operations
// ============================================================================

/**
 * Mean pooling of embeddings
 */
export function meanPool(embeddings: number[][], dim: number): number[] {
  if (embeddings.length === 0) {
    return new Array(dim).fill(0);
  }

  return tidy(() => {
    const tensor = ops.toTensor(embeddings);
    const result = ops.meanPool(tensor, 0);
    return result.arraySync() as number[];
  });
}

/**
 * Concatenate multi-head outputs
 */
export function concatHeads(heads: number[][][]): number[][] {
  if (heads.length === 0 || heads[0].length === 0) {
    return [];
  }

  const numNodes = heads[0].length;
  return Array.from({ length: numNodes }, (_, i) =>
    heads.flatMap((head) => head[i])
  );
}

// ============================================================================
// Regularization
// ============================================================================

/**
 * Apply dropout to matrix (for training)
 */
export function applyDropout(matrix: number[][], dropoutRate: number): number[][] {
  if (dropoutRate === 0) return matrix;

  const keepProb = 1 - dropoutRate;
  return matrix.map((row) =>
    row.map((x) => (random() < keepProb ? x / keepProb : 0))
  );
}

// ============================================================================
// BLAS Compatibility (no-op, TF.js handles optimization)
// ============================================================================

/**
 * @deprecated TF.js handles acceleration automatically
 */
export async function initBlasAcceleration(): Promise<boolean> {
  // TF.js uses WebGPU/WebGL automatically
  return true;
}

/**
 * Matrix-vector with optional BLAS (now just calls TF.js)
 */
export function matVecBlas(A: number[][], x: number[]): number[] {
  return matVec(A, x);
}

/**
 * Matrix-vector transpose with optional BLAS
 */
export function matVecTransposeBlas(A: number[][], x: number[]): number[] {
  return tidy(() => {
    const tA = ops.toTensor(A);
    const tX = ops.toTensor(x);
    const tAT = ops.transpose(tA);
    const result = ops.matVec(tAT, tX);
    return result.arraySync() as number[];
  });
}

/**
 * Outer product add: A = A + alpha * x @ y^T
 */
export function outerProductAdd(
  A: number[][],
  x: number[],
  y: number[],
  alpha: number = 1.0
): number[][] {
  return tidy(() => {
    const tA = ops.toTensor(A);
    const tX = ops.toTensor(x);
    const tY = ops.toTensor(y);

    // outer = x.reshape([n, 1]) @ y.reshape([1, m])
    const outer = tf.outerProduct(tX, tY);
    const scaled = outer.mul(alpha);
    const result = tA.add(scaled);

    return result.arraySync() as number[][];
  });
}

// ============================================================================
// Pure JS fallbacks (kept for reference, not used)
// ============================================================================

/**
 * Pure JS matrix multiplication (for reference only)
 * @deprecated Use matmul() which uses TF.js
 */
export function matmulJS(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const k = A[0]?.length || 0;
  const n = B[0]?.length || 0;

  const result: number[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    result[i] = new Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += A[i][p] * B[p][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}
