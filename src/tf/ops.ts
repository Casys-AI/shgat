/**
 * TensorFlow.js Operations for SHGAT-TF
 *
 * All mathematical operations using TensorFlow.js tensors.
 * This replaces the manual math.ts and blas-ffi.ts implementations.
 *
 * @module shgat-tf/tf/ops
 */

import { tf } from "./backend.ts";

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * Matrix multiplication: A @ B
 *
 * Handles empty matrices gracefully by returning appropriately sized zeros.
 *
 * @param A - Tensor [m, k]
 * @param B - Tensor [k, n]
 * @returns Result [m, n]
 */
export function matmul(A: tf.Tensor2D, B: tf.Tensor2D): tf.Tensor2D {
  // Handle empty matrices
  const [m, k1] = A.shape;
  const [k2, n] = B.shape;
  if (m === 0 || k1 === 0 || k2 === 0 || n === 0) {
    return tf.zeros([m, n]) as tf.Tensor2D;
  }
  return tf.matMul(A, B);
}

/**
 * Matrix multiplication with transpose: A @ B^T
 *
 * Handles empty matrices gracefully by returning appropriately sized zeros.
 *
 * @param A - Tensor [m, k]
 * @param B - Tensor [n, k]
 * @returns Result [m, n]
 */
export function matmulTranspose(A: tf.Tensor2D, B: tf.Tensor2D): tf.Tensor2D {
  // Handle empty matrices: A[m,k] @ B[n,k]^T => [m, n]
  const [m, k1] = A.shape;
  const [n, k2] = B.shape;
  if (m === 0 || k1 === 0 || n === 0 || k2 === 0) {
    return tf.zeros([m, n]) as tf.Tensor2D;
  }
  return tf.matMul(A, B, false, true);
}

/**
 * Matrix-vector multiplication: M @ v
 *
 * Handles empty matrices/vectors gracefully.
 *
 * @param M - Matrix [m, n]
 * @param v - Vector [n]
 * @returns Vector [m]
 */
export function matVec(M: tf.Tensor2D, v: tf.Tensor1D): tf.Tensor1D {
  const [m, n] = M.shape;
  if (m === 0 || n === 0 || v.shape[0] === 0) {
    return tf.zeros([m]) as tf.Tensor1D;
  }
  return tf.squeeze(tf.matMul(M, v.expandDims(1))) as tf.Tensor1D;
}

/**
 * Transpose a matrix
 */
export function transpose(M: tf.Tensor2D): tf.Tensor2D {
  return tf.transpose(M) as tf.Tensor2D;
}

/**
 * Batch matrix multiplication: batched A @ B
 *
 * @param A - Tensor [batch, m, k]
 * @param B - Tensor [batch, k, n] or [k, n] (broadcast)
 * @returns Result [batch, m, n]
 */
export function batchMatmul(
  A: tf.Tensor3D,
  B: tf.Tensor3D | tf.Tensor2D,
): tf.Tensor3D {
  if (B.rank === 2) {
    // Broadcast B to batch dimension
    const batch = A.shape[0];
    B = tf.tile(B.expandDims(0), [batch, 1, 1]) as tf.Tensor3D;
  }
  return tf.matMul(A, B as tf.Tensor3D);
}

// ============================================================================
// Vector Operations
// ============================================================================

/**
 * Dot product of two vectors
 */
export function dot(a: tf.Tensor1D, b: tf.Tensor1D): tf.Scalar {
  return tf.sum(tf.mul(a, b));
}

/**
 * Cosine similarity between two vectors
 */
export function cosineSimilarity(a: tf.Tensor1D, b: tf.Tensor1D): tf.Scalar {
  const dotProduct = dot(a, b);
  const normA = tf.norm(a);
  const normB = tf.norm(b);
  return tf.div(dotProduct, tf.mul(normA, normB).add(1e-8)) as tf.Scalar;
}

/**
 * L2 normalize a vector or batch of vectors
 */
export function l2Normalize(x: tf.Tensor, axis = -1): tf.Tensor {
  return tf.div(x, tf.norm(x, 2, axis, true).add(1e-8));
}

/**
 * Concatenate tensors along axis
 */
export function concat<T extends tf.Tensor>(
  tensors: T[],
  axis = 0,
): T {
  return tf.concat(tensors, axis) as T;
}

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * Softmax activation (numerically stable)
 */
export function softmax(x: tf.Tensor, axis = -1): tf.Tensor {
  return tf.softmax(x, axis);
}

/**
 * Sigmoid activation
 */
export function sigmoid(x: tf.Tensor): tf.Tensor {
  return tf.sigmoid(x);
}

/**
 * ReLU activation
 */
export function relu(x: tf.Tensor): tf.Tensor {
  return tf.relu(x);
}

/**
 * Leaky ReLU activation
 *
 * @param x - Input tensor
 * @param alpha - Negative slope (default 0.2)
 */
export function leakyRelu(x: tf.Tensor, alpha = 0.2): tf.Tensor {
  return tf.leakyRelu(x, alpha);
}

/**
 * ELU activation
 *
 * Note: TF.js elu uses fixed alpha=1.0
 */
export function elu(x: tf.Tensor): tf.Tensor {
  return tf.elu(x);
}

/**
 * Tanh activation
 */
export function tanh(x: tf.Tensor): tf.Tensor {
  return tf.tanh(x);
}

// ============================================================================
// Loss Functions
// ============================================================================

/**
 * Binary cross-entropy loss
 *
 * @param pred - Predicted probabilities [0, 1]
 * @param label - True labels (0 or 1)
 */
export function binaryCrossEntropy(
  pred: tf.Tensor,
  label: tf.Tensor,
): tf.Scalar {
  const eps = 1e-7;
  const clipped = tf.clipByValue(pred, eps, 1 - eps);
  const loss = tf.neg(
    tf.add(
      tf.mul(label, tf.log(clipped)),
      tf.mul(tf.sub(1, label), tf.log(tf.sub(1, clipped))),
    ),
  );
  return tf.mean(loss) as tf.Scalar;
}

/**
 * Softmax cross-entropy loss
 *
 * @param logits - Raw scores [batch, numClasses]
 * @param labels - One-hot labels [batch, numClasses]
 */
export function softmaxCrossEntropy(
  logits: tf.Tensor2D,
  labels: tf.Tensor2D,
): tf.Scalar {
  return tf.losses.softmaxCrossEntropy(labels, logits) as tf.Scalar;
}

/**
 * InfoNCE / Contrastive loss (CLIP-style)
 *
 * @param anchorEmb - Anchor embeddings [batch, dim]
 * @param positiveEmb - Positive embeddings [batch, dim]
 * @param temperature - Temperature parameter (default 0.07)
 */
export function infoNCELoss(
  anchorEmb: tf.Tensor2D,
  positiveEmb: tf.Tensor2D,
  temperature = 0.07,
): tf.Scalar {
  // Normalize embeddings
  const anchor = l2Normalize(anchorEmb, 1) as tf.Tensor2D;
  const positive = l2Normalize(positiveEmb, 1) as tf.Tensor2D;

  // Compute similarity matrix [batch, batch]
  const similarity = tf.div(
    matmulTranspose(anchor, positive),
    temperature,
  );

  // Labels: diagonal is positive (i matches i)
  const batch = anchor.shape[0];
  const labels = tf.eye(batch);

  // Cross-entropy both directions
  const lossI2T = tf.losses.softmaxCrossEntropy(labels, similarity);
  const lossT2I = tf.losses.softmaxCrossEntropy(labels, tf.transpose(similarity));

  return tf.div(tf.add(lossI2T, lossT2I), 2) as tf.Scalar;
}

// ============================================================================
// Pooling Operations
// ============================================================================

/**
 * Mean pooling along axis
 */
export function meanPool(x: tf.Tensor, axis: number | number[]): tf.Tensor {
  return tf.mean(x, axis);
}

/**
 * Max pooling along axis
 */
export function maxPool(x: tf.Tensor, axis: number | number[]): tf.Tensor {
  return tf.max(x, axis);
}

/**
 * Sum pooling along axis
 */
export function sumPool(x: tf.Tensor, axis: number | number[]): tf.Tensor {
  return tf.sum(x, axis);
}

// ============================================================================
// Dropout & Regularization
// ============================================================================

/**
 * Apply dropout
 *
 * @param x - Input tensor
 * @param rate - Dropout rate (0 = no dropout)
 * @param training - Whether in training mode
 */
export function dropout(
  x: tf.Tensor,
  rate: number,
  training = true,
): tf.Tensor {
  if (!training || rate === 0) return x;
  return tf.dropout(x, rate);
}

// ============================================================================
// Initialization
// ============================================================================

/**
 * Xavier/Glorot uniform initialization
 *
 * @param shape - Tensor shape
 */
export function glorotUniform(shape: number[]): tf.Tensor {
  const fanIn = shape.length > 1 ? shape[shape.length - 2] : shape[0];
  const fanOut = shape[shape.length - 1];
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return tf.randomUniform(shape, -limit, limit);
}

/**
 * Xavier/Glorot normal initialization
 */
export function glorotNormal(shape: number[]): tf.Tensor {
  const fanIn = shape.length > 1 ? shape[shape.length - 2] : shape[0];
  const fanOut = shape[shape.length - 1];
  const stddev = Math.sqrt(2 / (fanIn + fanOut));
  return tf.randomNormal(shape, 0, stddev);
}

/**
 * He/Kaiming normal initialization (for ReLU)
 */
export function heNormal(shape: number[]): tf.Tensor {
  const fanIn = shape.length > 1 ? shape[shape.length - 2] : shape[0];
  const stddev = Math.sqrt(2 / fanIn);
  return tf.randomNormal(shape, 0, stddev);
}

/**
 * Create a trainable variable
 */
export function variable(
  initialValue: tf.Tensor,
  trainable = true,
  name?: string,
): tf.Variable {
  return tf.variable(initialValue, trainable, name);
}

/**
 * Create zeros tensor
 */
export function zeros(shape: number[]): tf.Tensor {
  return tf.zeros(shape);
}

/**
 * Create ones tensor
 */
export function ones(shape: number[]): tf.Tensor {
  return tf.ones(shape);
}

// ============================================================================
// Array <-> Tensor Conversion
// ============================================================================

/**
 * Convert JS array to tensor
 */
export function toTensor(data: number[]): tf.Tensor1D;
export function toTensor(data: number[][]): tf.Tensor2D;
export function toTensor(data: number[][][]): tf.Tensor3D;
export function toTensor(
  data: number[] | number[][] | number[][][],
): tf.Tensor {
  if (data.length === 0) {
    return tf.tensor([]);
  }
  if (!Array.isArray(data[0])) {
    return tf.tensor1d(data as number[]);
  }
  if (!Array.isArray((data as number[][])[0][0])) {
    return tf.tensor2d(data as number[][]);
  }
  return tf.tensor3d(data as number[][][]);
}

/**
 * Convert tensor to JS array (sync - avoid in production)
 */
export function toArray(tensor: tf.Tensor): number[] | number[][] | number[][][] {
  return tensor.arraySync() as number[] | number[][] | number[][][];
}

/**
 * Convert tensor to JS array (async - preferred)
 */
export async function toArrayAsync(
  tensor: tf.Tensor,
): Promise<number[] | number[][] | number[][][]> {
  return (await tensor.array()) as number[] | number[][] | number[][][];
}

// ============================================================================
// Sparse Operations (for graph adjacency)
// ============================================================================

/**
 * Sparse-dense matrix multiplication using gather
 *
 * For adjacency matrices, more efficient than full matmul.
 *
 * @param indices - Non-zero indices [nnz, 2] (row, col)
 * @param values - Non-zero values [nnz]
 * @param dense - Dense matrix [n, d]
 * @param outputRows - Number of output rows
 */
export function sparseMatmul(
  indices: tf.Tensor2D,
  values: tf.Tensor1D,
  dense: tf.Tensor2D,
  outputRows: number,
): tf.Tensor2D {
  // This is a simplified version - for production, use tf.sparseToDense
  // or implement proper sparse ops
  return tf.tidy(() => {
    const rows = indices.slice([0, 0], [-1, 1]).squeeze() as tf.Tensor1D;
    const cols = indices.slice([0, 1], [-1, 1]).squeeze() as tf.Tensor1D;

    // Gather dense rows at col indices
    const gathered = tf.gather(dense, cols);

    // Scale by values
    const scaled = tf.mul(gathered, values.expandDims(1));

    // Segment sum by row indices
    const result = tf.unsortedSegmentSum(scaled, rows, outputRows);

    return result as tf.Tensor2D;
  });
}
