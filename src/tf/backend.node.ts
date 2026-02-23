/**
 * TensorFlow.js Backend for SHGAT-TF — Node.js version
 *
 * Uses @tensorflow/tfjs-node (C++ TensorFlow binding) for maximum performance.
 * All kernels available, full autograd support, native CPU/GPU acceleration.
 *
 * This file is the Node.js equivalent of backend.ts (Deno version).
 * For Node distribution, replace the import of backend.ts with backend.node.ts
 * or use the build script to swap them automatically.
 *
 * @module shgat-tf/tf/backend.node
 */

// @tensorflow/tfjs-node registers the 'tensorflow' backend and re-exports tfjs core
// deno-lint-ignore-file no-explicit-any
import * as tf from "@tensorflow/tfjs-node";

// Re-export tf for use throughout the codebase (same interface as backend.ts)
export { tf };

// Backend state
let initialized = false;
let currentBackend: string = "tensorflow";

/** Backend mode determines kernel requirements */
export type BackendMode = "training" | "inference";

/**
 * Ensure TensorFlow.js is initialized before use.
 */
export async function ensureInitialized(): Promise<void> {
  if (initialized) return;
  await initTensorFlow();
}

/**
 * Initialize TensorFlow.js with tfjs-node backend
 *
 * On Node.js, tfjs-node always uses the 'tensorflow' C++ backend.
 * All kernels are available (UnsortedSegmentSum, full autograd, etc.)
 * so there's no need for backend priority lists.
 *
 * @param _preferredBackendOrMode - Ignored on Node.js (always uses 'tensorflow' backend).
 *   Accepted for API compatibility with the Deno version.
 * @returns The backend name ('tensorflow')
 */
export async function initTensorFlow(
  _preferredBackendOrMode?: "ffi" | "webgpu" | "wasm" | "cpu" | BackendMode,
): Promise<string> {
  if (initialized) {
    return currentBackend;
  }

  await tf.ready();
  currentBackend = tf.getBackend() || "tensorflow";
  initialized = true;

  console.error(`[TF] Using tfjs-node backend: ${currentBackend}`);
  return currentBackend;
}

/**
 * Switch backend at runtime — no-op on Node.js
 *
 * tfjs-node only has the 'tensorflow' backend, so switching is not needed.
 * Returns current backend for API compatibility.
 */
export async function switchBackend(_mode: BackendMode): Promise<string> {
  return currentBackend;
}

/**
 * Check if current backend supports full autograd (all kernels)
 * Always true on Node.js — tfjs-node has all kernels
 */
export function supportsAutograd(): boolean {
  return true;
}

/**
 * Get current backend name
 */
export function getBackend(): string {
  return currentBackend;
}

/**
 * Check if TensorFlow.js is initialized
 */
export function isInitialized(): boolean {
  return initialized;
}

/**
 * Get memory info (useful for debugging leaks)
 */
export function getMemoryInfo(): tf.MemoryInfo {
  return tf.memory();
}

/**
 * Log memory stats to console
 */
export function logMemory(prefix = ""): void {
  const mem = tf.memory();
  console.log(
    `${prefix}[TF Memory] tensors: ${mem.numTensors}, bytes: ${(mem.numBytes / 1024 / 1024).toFixed(2)}MB`,
  );
}

/**
 * Dispose all tensors (use with caution - only for cleanup)
 */
export function disposeAll(): void {
  tf.disposeVariables();
}

/**
 * Run a function within tf.tidy() for automatic cleanup
 */
export function tidy<T extends tf.TensorContainer>(fn: () => T): T {
  return tf.tidy(fn);
}

/**
 * Dispose tensors safely
 */
export function dispose(tensors: tf.Tensor | tf.Tensor[] | null | undefined): void {
  if (!tensors) return;
  if (Array.isArray(tensors)) {
    tensors.forEach((t: any) => t?.dispose());
  } else {
    tensors.dispose();
  }
}

// Auto-initialize on module load
await initTensorFlow();
