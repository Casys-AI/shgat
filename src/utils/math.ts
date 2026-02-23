/**
 * Mathematical utilities for SHGAT
 *
 * Pure functions extracted from SHGAT class for better testability and reusability.
 *
 * @module graphrag/algorithms/shgat/utils/math
 */

// Note: uses Math.random() for dropout (non-seeded) to avoid importing parameters.ts
// which transitively pulls in TF.js. For seeded RNG, use parameters.ts directly.

/** Any numeric row type — number[] or Float32Array */
export type NumericArray = number[] | Float32Array;

/** Any row-based numeric matrix */
export type NumericMatrix = NumericArray[];

/**
 * Matrix multiplication with transpose: A · B^T
 * Uses BLAS acceleration for larger matrices (~10x speedup).
 *
 * @param A - Matrix A [m][k]
 * @param B - Matrix B [n][k] (will be transposed)
 * @returns Result matrix [m][n]
 */
export function matmulTranspose(A: NumericMatrix, B: NumericMatrix): number[][] {
  // Use BLAS for larger matrices (message passing projections: ~105×1024 × 64×1024^T)
  if (isBlasReady() && A.length >= 10 && (A[0]?.length || 0) >= 64) {
    return blasModule!.blasMatmulTranspose(A, B);
  }
  // JS fallback
  const m = A.length;
  const n = B.length;
  const k = A[0]?.length || 0;
  const result: number[][] = new Array(m);
  for (let i = 0; i < m; i++) {
    result[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) {
        sum += A[i][p] * B[j][p];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * Matrix multiplication with transpose returning Float32Array rows: A · B^T
 * Saves ~50% RAM vs number[][] for cache storage.
 * Requires BLAS (no JS fallback). Call initBlasAcceleration() first.
 *
 * @param A - Matrix A [m][k]
 * @param B - Matrix B [n][k] (will be transposed)
 * @returns Result as Float32Array[] [m][n]
 */
export function matmulTransposeF32(A: NumericMatrix, B: NumericMatrix): Float32Array[] {
  if (!isBlasReady()) {
    throw new Error(
      "[math] matmulTransposeF32 requires BLAS acceleration. " +
      "Call initBlasAcceleration() or ensureBLAS() at startup.",
    );
  }
  return blasModule!.blasMatmulTransposeF32(A, B);
}

/**
 * Leaky ReLU activation
 *
 * f(x) = x if x > 0, else slope * x
 *
 * @param x - Input value
 * @param slope - Negative slope (default 0.2)
 * @returns Activated value
 */
export function leakyRelu(x: number, slope: number = 0.2): number {
  return x > 0 ? x : slope * x;
}

/**
 * Exponential Linear Unit (ELU) activation
 *
 * f(x) = x if x ≥ 0, else α(e^x - 1)
 *
 * @param x - Input value
 * @param alpha - Scale parameter (default 1.0)
 * @returns Activated value
 */
export function elu(x: number, alpha: number = 1.0): number {
  return x >= 0 ? x : alpha * (Math.exp(x) - 1);
}

/**
 * Sigmoid activation
 *
 * f(x) = 1 / (1 + e^(-x))
 *
 * @param x - Input value
 * @returns Value in range (0, 1)
 */
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Softmax function (numerically stable)
 *
 * Subtracts max value before exp to prevent overflow.
 *
 * @param values - Input values
 * @returns Normalized probabilities summing to 1
 */
export function softmax(values: number[]): number[] {
  if (values.length === 0) return [];

  // Use loop instead of Math.max(...) to avoid stack overflow with large arrays
  let maxVal = -Infinity;
  for (const v of values) {
    if (v > maxVal) maxVal = v;
  }
  const exps = values.map((v) => Math.exp(v - maxVal));
  const sum = exps.reduce((a, b) => a + b, 0);

  return sum > 0 ? exps.map((e) => e / sum) : new Array(values.length).fill(1 / values.length);
}

/**
 * Dot product of two vectors
 * Accepts number[] or Float32Array.
 *
 * @param a - Vector a
 * @param b - Vector b
 * @returns Scalar dot product
 */
export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Cosine similarity between two vectors
 *
 * sim(a, b) = (a · b) / (||a|| × ||b||)
 *
 * @param a - Vector a
 * @param b - Vector b
 * @returns Similarity in range [-1, 1]
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const dotProduct = dot(a, b);
  const normA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
  const normB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));

  return normA * normB > 0 ? dotProduct / (normA * normB) : 0;
}

/**
 * Binary cross-entropy loss
 *
 * BCE(p, y) = -y log(p) - (1-y) log(1-p)
 *
 * @param pred - Predicted probability [0, 1]
 * @param label - True label (0 or 1)
 * @returns Loss value
 */
export function binaryCrossEntropy(pred: number, label: number): number {
  const eps = 1e-7;
  const p = Math.max(eps, Math.min(1 - eps, pred));
  return -label * Math.log(p) - (1 - label) * Math.log(1 - p);
}

/**
 * Mean pooling of embeddings
 *
 * Averages embeddings element-wise. Returns zero vector if input is empty.
 *
 * @param embeddings - Array of embeddings
 * @param dim - Target dimension
 * @returns Mean-pooled embedding
 */
export function meanPool(embeddings: number[][], dim: number): number[] {
  if (embeddings.length === 0) {
    return new Array(dim).fill(0);
  }

  const result = new Array(dim).fill(0);
  for (const emb of embeddings) {
    for (let i = 0; i < Math.min(dim, emb.length); i++) {
      result[i] += emb[i];
    }
  }

  for (let i = 0; i < dim; i++) {
    result[i] /= embeddings.length;
  }

  return result;
}

/**
 * Concatenate multi-head outputs
 *
 * @param heads - Embeddings per head [numHeads][numNodes][headDim]
 * @returns Concatenated embeddings [numNodes][numHeads * headDim]
 */
export function concatHeads(heads: number[][][]): number[][] {
  if (heads.length === 0 || heads[0].length === 0) {
    return [];
  }

  const numNodes = heads[0].length;
  return Array.from({ length: numNodes }, (_, i) => heads.flatMap((head) => head[i]));
}

/**
 * Apply dropout to matrix (for training)
 *
 * Randomly zero out elements with probability `dropoutRate`.
 * Scales remaining elements by 1/(1-dropoutRate) to maintain expected value.
 *
 * @param matrix - Input matrix
 * @param dropoutRate - Dropout probability [0, 1]
 * @returns Matrix with dropout applied
 */
export function applyDropout(matrix: number[][], dropoutRate: number): number[][] {
  if (dropoutRate === 0) return matrix;

  const keepProb = 1 - dropoutRate;
  return matrix.map((row) => row.map((x) => (Math.random() < keepProb ? x / keepProb : 0)));
}

/**
 * Normalize vector to unit length
 *
 * @param vector - Input vector
 * @returns Normalized vector (or zero vector if input norm is 0)
 */
export function normalize(vector: number[]): number[] {
  const norm = Math.sqrt(vector.reduce((s, x) => s + x * x, 0));
  return norm > 0 ? vector.map((x) => x / norm) : new Array(vector.length).fill(0);
}

// ============================================================================
// Batch Matrix Operations (for K-head scoring optimization)
// ============================================================================

// BLAS acceleration (lazy-loaded)
let blasModule: typeof import("./blas-ffi.ts") | null = null;
let blasLoadAttempted = false;

async function tryLoadBlas(): Promise<boolean> {
  if (blasLoadAttempted) return blasModule !== null;
  blasLoadAttempted = true;

  try {
    blasModule = await import("./blas-ffi.ts");
    return blasModule.isBlasAvailable();
  } catch {
    return false;
  }
}

// Synchronous check (after first async load)
function isBlasReady(): boolean {
  return blasModule !== null && blasModule.isBlasAvailable();
}

/**
 * Initialize BLAS acceleration (call once at startup for best performance)
 */
export async function initBlasAcceleration(): Promise<boolean> {
  return await tryLoadBlas();
}

/**
 * Standard matrix multiplication: A · B
 * Uses BLAS if available for ~10x speedup on large matrices.
 * Accepts number[][] or Float32Array[] inputs.
 *
 * @param A - Matrix A [m][k]
 * @param B - Matrix B [k][n]
 * @returns Result matrix [m][n]
 */
export function matmul(A: NumericMatrix, B: NumericMatrix): number[][] {
  // Use BLAS for larger matrices (overhead not worth it for small)
  if (isBlasReady() && A.length >= 10 && (A[0]?.length || 0) >= 64) {
    return blasModule!.blasMatmul(A, B);
  }
  return matmulJS(A, B);
}

/**
 * Pure JS matrix multiplication (fallback)
 */
export function matmulJS(A: NumericMatrix, B: NumericMatrix): number[][] {
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

/**
 * Batch matrix-vector multiplication: M · v for each row
 *
 * Computes M @ v where M is [m][k] and v is [k], returning [m] scores.
 *
 * @param M - Matrix [m][k]
 * @param v - Vector [k]
 * @returns Vector of dot products [m]
 */
export function matVec(M: NumericMatrix, v: ArrayLike<number>): number[] {
  const m = M.length;
  const result = new Array(m);
  for (let i = 0; i < m; i++) {
    let sum = 0;
    const row = M[i];
    const len = Math.min(row.length, v.length);
    for (let j = 0; j < len; j++) {
      sum += row[j] * v[j];
    }
    result[i] = sum;
  }
  return result;
}

/**
 * Transpose a matrix
 * Accepts number[][] or Float32Array[].
 *
 * @param M - Matrix [m][n]
 * @returns Transposed matrix [n][m]
 */
export function transpose(M: NumericMatrix): number[][] {
  if (M.length === 0) return [];
  const m = M.length;
  const n = M[0].length;
  const result: number[][] = new Array(n);
  for (let j = 0; j < n; j++) {
    result[j] = new Array(m);
    for (let i = 0; i < m; i++) {
      result[j][i] = M[i][j];
    }
  }
  return result;
}

// ============================================================================
// BLAS-accelerated Training Operations
// ============================================================================

/**
 * Matrix-vector multiplication with optional BLAS acceleration: y = A @ x
 * Uses BLAS for larger matrices (threshold: 64 rows).
 *
 * @param A - Matrix [m][n]
 * @param x - Vector [n]
 * @returns Vector [m]
 */
export function matVecBlas(A: NumericMatrix, x: ArrayLike<number>): number[] {
  // Use BLAS for larger matrices (high threshold to avoid FFI overhead)
  if (isBlasReady() && A.length >= 256) {
    return blasModule!.blasMatVec(A, x);
  }
  return matVec(A, x);
}

/**
 * Matrix-vector multiplication with transpose: y = A^T @ x
 * Uses BLAS for larger matrices (threshold: 64 rows).
 *
 * @param A - Matrix [m][n]
 * @param x - Vector [m]
 * @returns Vector [n]
 */
export function matVecTransposeBlas(A: NumericMatrix, x: ArrayLike<number>): number[] {
  // Use BLAS for larger matrices (high threshold to avoid FFI overhead)
  if (isBlasReady() && A.length >= 256) {
    return blasModule!.blasMatVecTranspose(A, x);
  }
  // JS fallback: transpose then multiply
  const result = new Array(A[0]?.length || 0).fill(0);
  for (let i = 0; i < A.length; i++) {
    const xi = x[i] || 0;
    for (let j = 0; j < A[i].length; j++) {
      result[j] += A[i][j] * xi;
    }
  }
  return result;
}

/**
 * Outer product (rank-1 update): A = A + alpha * x @ y^T
 * Used for gradient accumulation in training.
 * Uses BLAS for larger matrices (threshold: 64 dimensions).
 *
 * @param A - Matrix [m][n] (modified in-place)
 * @param x - Vector [m]
 * @param y - Vector [n]
 * @param alpha - Scalar multiplier (default 1.0)
 * @returns Modified matrix A
 */
export function outerProductAdd(A: number[][], x: ArrayLike<number>, y: ArrayLike<number>, alpha: number = 1.0): number[][] {
  // Use BLAS for larger dimensions (high threshold to avoid FFI overhead for small ops)
  if (isBlasReady() && x.length >= 256 && y.length >= 256) {
    return blasModule!.blasOuterProduct(A, x, y, alpha);
  }
  // JS fallback
  for (let i = 0; i < x.length; i++) {
    if (!A[i]) A[i] = new Array(y.length).fill(0);
    const xi = x[i] * alpha;
    for (let j = 0; j < y.length; j++) {
      A[i][j] += xi * y[j];
    }
  }
  return A;
}

// ============================================================================
// Zero matrix helpers (moved from parameters.ts to avoid TF.js dependency)
// ============================================================================

/**
 * Create zeros matrix with same shape as input
 */
export function zerosLike2D(matrix: number[][]): number[][] {
  return matrix.map((row) => row.map(() => 0));
}

/**
 * Create zeros tensor with same shape as input
 */
export function zerosLike3D(tensor: number[][][]): number[][][] {
  return tensor.map((m) => m.map((r) => r.map(() => 0)));
}

// ============================================================================
// Float32Array utilities (for cache RAM optimization)
// ============================================================================

/**
 * Convert number[][] to Float32Array[] for cache storage.
 * Halves RAM usage (64-bit → 32-bit per element).
 *
 * @param m - Matrix as number[][]
 * @returns Matrix as Float32Array[]
 */
export function toFloat32Rows(m: number[][]): Float32Array[] {
  const result = new Array<Float32Array>(m.length);
  for (let i = 0; i < m.length; i++) {
    result[i] = Float32Array.from(m[i]);
  }
  return result;
}
