/**
 * TensorFlow FFI Module for SHGAT-TF
 *
 * Uses libtensorflow via FFI for operations that need proper autograd support.
 * Provides UnsortedSegmentSum for proper tf.gather gradients!
 *
 * @module shgat-tf/tf
 */

// Core FFI bindings
export * as tff from "./tf-ffi.ts";
export { isAvailable, version, close } from "./tf-ffi.ts";
export type { TFTensor } from "./tf-ffi.ts";

// Re-export tensor operations from tf-ffi.ts
export {
  // Tensor creation
  tensor,
  tensorInt32,
  zeros,
  ones,
  fill,
  eye,
  randomUniform,
  randomNormal,
  oneHot,
  // Math operations
  matmul as matMul,
  batchMatmul as batchMatMul,
  add,
  sub,
  mul,
  div,
  transpose,
  reshape,
  gather,
  unsortedSegmentSum,
  concat,
  slice,
  expandDims,
  squeeze,
  clipByValue,
  // Activations
  softmax,
  relu,
  leakyRelu,
  elu,
  sigmoid,
  tanh,
  // Reductions
  sum,
  mean,
  max,
  square,
  sqrt,
  exp,
  log,
  neg,
} from "./tf-ffi.ts";

// ============================================================================
// TF.js-compatible helpers (minimal wrappers)
// ============================================================================

import { isAvailable } from "./tf-ffi.ts";
import type { TFTensor } from "./tf-ffi.ts";

let _initialized = false;

/**
 * Initialize TensorFlow FFI
 */
export async function initTensorFlow(): Promise<string> {
  if (!isAvailable()) {
    throw new Error("[TF-FFI] libtensorflow not available");
  }
  _initialized = true;
  return "ffi";
}

/**
 * Get current backend name
 */
export function getBackend(): string {
  return isAvailable() ? "ffi" : "none";
}

/**
 * Check if initialized
 */
export function isInitialized(): boolean {
  return _initialized && isAvailable();
}

/**
 * Execute function and dispose intermediate tensors (no-op for FFI)
 * Note: FFI tensors must be manually disposed
 */
export function tidy<T>(fn: () => T): T {
  return fn();
}

/**
 * Dispose a tensor
 */
export function dispose(t: TFTensor | TFTensor[]): void {
  if (Array.isArray(t)) {
    for (const tensor of t) {
      tensor.dispose();
    }
  } else {
    t.dispose();
  }
}

/**
 * Simple Variable wrapper for FFI tensors
 */
export class Variable {
  private tensor: TFTensor;
  readonly name: string;

  constructor(initialValue: TFTensor, name = "variable") {
    this.tensor = initialValue;
    this.name = name;
  }

  get shape(): number[] {
    return this.tensor.shape;
  }

  read(): TFTensor {
    return this.tensor;
  }

  assign(newValue: TFTensor): void {
    if (this.tensor !== newValue) {
      this.tensor.dispose();
    }
    this.tensor = newValue;
  }

  dispose(): void {
    this.tensor.dispose();
  }
}

/**
 * Create a variable from tensor
 */
export function variable(tensor: TFTensor, name?: string): Variable {
  return new Variable(tensor, name);
}

/**
 * Memory info (placeholder)
 */
export function memory(): { numTensors: number; numBytes: number } {
  return { numTensors: 0, numBytes: 0 };
}

/**
 * Log memory usage
 */
export function logMemory(): void {
  console.log("[TF-FFI] Memory tracking not implemented for FFI backend");
}

// Alias for compatibility
export { initTensorFlow as init };
