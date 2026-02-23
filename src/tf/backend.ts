/**
 * TensorFlow.js Backend for SHGAT-TF
 *
 * Smart backend selection based on usage mode:
 *
 * TRAINING mode (requires full autograd / all kernels):
 *   WebGPU > CPU (never WASM - missing UnsortedSegmentSum, tf.gather grad)
 *
 * INFERENCE mode (forward-only, speed priority):
 *   WebGPU > WASM > CPU
 *
 * The same model code runs on all backends - TF.js abstracts the difference.
 * Users in the browser automatically get WebGPU if available.
 *
 * For Node.js, use backend.node.ts instead (tfjs-node C++ binding).
 *
 * @module shgat-tf/tf/backend
 */

import * as tf from "@tensorflow/tfjs";
// Import WASM backend to register it
import "@tensorflow/tfjs-backend-wasm";

// Re-export tf for use throughout the codebase
export { tf };

// Backend state
let initialized = false;
let currentBackend: string = "cpu";
let initPromise: Promise<string> | null = null;

/** Backend mode determines kernel requirements */
export type BackendMode = "training" | "inference";

/**
 * Backend priority by mode.
 *
 * Training needs full autograd (all kernels) so WASM is excluded.
 * Inference only needs forward ops so WASM is fine and faster than CPU.
 */
const BACKEND_PRIORITY: Record<BackendMode, string[]> = {
  training: ["webgpu", "cpu"], // Never WASM: missing UnsortedSegmentSum
  inference: ["webgpu", "wasm", "cpu"], // WASM is fast for forward-only
};

/**
 * Ensure TensorFlow.js is initialized before use.
 * This is called automatically when needed.
 */
export async function ensureInitialized(): Promise<void> {
  if (initialized) return;
  if (!initPromise) {
    initPromise = initTensorFlow();
  }
  await initPromise;
}

/**
 * Initialize TensorFlow.js with optimal backend
 *
 * Call once at application startup for best performance.
 *
 * @param preferredBackendOrMode - Backend name or mode:
 *   - 'training': Auto-select best backend with full autograd (WebGPU > CPU)
 *   - 'inference': Auto-select fastest backend (WebGPU > WASM > CPU)
 *   - 'webgpu': TF.js WebGPU backend
 *   - 'wasm': TF.js WASM backend (fast, but missing some grad kernels)
 *   - 'cpu': TF.js CPU backend (slowest, but all kernels available)
 * @returns The backend that was selected
 *
 * @example
 * ```typescript
 * // For training (full autograd support):
 * await initTensorFlow("training");
 *
 * // For inference (max speed):
 * await initTensorFlow("inference");
 *
 * // Force specific backend:
 * await initTensorFlow("cpu");
 * ```
 */
export async function initTensorFlow(
  preferredBackendOrMode?: "webgpu" | "wasm" | "cpu" | BackendMode,
): Promise<string> {
  if (initialized) {
    return currentBackend;
  }

  await tf.ready();

  // Resolve backend priority list
  let backends: string[];
  if (preferredBackendOrMode === "training" || preferredBackendOrMode === "inference") {
    backends = BACKEND_PRIORITY[preferredBackendOrMode];
  } else if (preferredBackendOrMode) {
    backends = [preferredBackendOrMode];
  } else {
    // Default: inference mode (backward-compatible, WASM preferred for speed)
    backends = BACKEND_PRIORITY.inference;
  }

  for (const backend of backends) {
    try {
      await tf.setBackend(backend);
      currentBackend = tf.getBackend();
      if (currentBackend === backend) {
        break;
      }
    } catch {
      // Backend not available, try next
    }
  }

  initialized = true;
  currentBackend = tf.getBackend();

  return currentBackend;
}

/**
 * Switch backend at runtime (e.g., from inference to training mode)
 *
 * Use this to switch between WASM (fast inference) and CPU (full autograd training)
 * within the same session.
 *
 * @param mode - 'training' or 'inference'
 * @returns The backend that was selected
 */
export async function switchBackend(mode: BackendMode): Promise<string> {
  const backends = BACKEND_PRIORITY[mode];

  for (const backend of backends) {
    try {
      await tf.setBackend(backend);
      currentBackend = tf.getBackend();
      if (currentBackend === backend) {
        break;
      }
    } catch {
      // Backend not available, try next
    }
  }

  currentBackend = tf.getBackend();
  return currentBackend;
}

/**
 * Check if current backend supports full autograd (all kernels)
 */
export function supportsAutograd(): boolean {
  // WASM is the only TF.js backend missing grad kernels
  return getBackend() !== "wasm";
}

/**
 * Get current backend name
 */
export function getBackend(): string {
  return tf.getBackend();
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
    `${prefix}[TF Memory] tensors: ${mem.numTensors}, bytes: ${
      (mem.numBytes / 1024 / 1024).toFixed(2)
    }MB`,
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
 *
 * @param fn Function to run
 * @returns Result of the function
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
    tensors.forEach((t) => t?.dispose());
  } else {
    tensors.dispose();
  }
}

// Auto-initialize TF.js backend on module load.
// This ensures tf.ready() + tf.setBackend() are called before any tensor operations.
await initTensorFlow();
