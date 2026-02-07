/**
 * TensorFlow.js Backend for SHGAT-TF
 *
 * Initializes TensorFlow.js with optimal backend selection:
 * - FFI backend (libtensorflow via Deno FFI - has UnsortedSegmentSum!)
 * - WASM backend (2-10x faster than CPU, works in Deno)
 * - CPU as fallback
 *
 * Note: tfjs-node is not compatible with Deno due to Node.js internals.
 * WASM backend provides good acceleration for Deno environments.
 *
 * @module shgat-tf/tf/backend
 */

import * as tf from "npm:@tensorflow/tfjs@4.22.0";
// Import WASM backend to register it
import "npm:@tensorflow/tfjs-backend-wasm@4.22.0";

// FFI backend (requires libtensorflow)
import * as tff from "./tf-ffi.ts";

// Re-export tf for use throughout the codebase
export { tf };

// Backend state
let initialized = false;
let currentBackend: string = "cpu";
let initPromise: Promise<string> | null = null;
let usingFFI = false;

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
 * @param preferredBackend - Optional preferred backend:
 *   - 'ffi': Use libtensorflow via FFI (has UnsortedSegmentSum for full autograd!)
 *   - 'wasm': TF.js WASM backend (fast, but missing some kernels)
 *   - 'cpu': TF.js CPU backend (slowest, but complete)
 *   - 'webgpu': TF.js WebGPU backend (if available)
 * @returns The backend that was selected
 *
 * @example
 * ```typescript
 * import { initTensorFlow } from "./tf/backend.ts";
 * const backend = await initTensorFlow("ffi");  // Use libtensorflow FFI
 * console.log(`Using backend: ${backend}`);
 * ```
 */
export async function initTensorFlow(
  preferredBackend?: "ffi" | "webgpu" | "wasm" | "cpu",
): Promise<string> {
  if (initialized) {
    return currentBackend;
  }

  // Handle FFI backend specially
  if (preferredBackend === "ffi") {
    if (!tff.isAvailable()) {
      throw new Error(
        "[TF-FFI] libtensorflow not found. Install with:\n" +
        "  ./lib/shgat-tf/scripts/install-libtensorflow.sh\n" +
        "Or use a different backend: initTensorFlow('wasm')"
      );
    }
    initialized = true;
    currentBackend = "ffi";
    usingFFI = true;
    console.error(`[TF] Using FFI backend (libtensorflow ${tff.version()})`);
    return currentBackend;
  }

  await tf.ready();

  // Try backends in order of preference
  // WASM is preferred as it works reliably in Deno and is 2-10x faster than CPU
  const backends = preferredBackend
    ? [preferredBackend]
    : ["wasm", "webgpu", "cpu"];

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
 * Get current backend name
 */
export function getBackend(): string {
  if (usingFFI) return "ffi";
  return tf.getBackend();
}

/**
 * Check if TensorFlow.js is initialized
 */
export function isInitialized(): boolean {
  return initialized;
}

/**
 * Check if using FFI backend (libtensorflow)
 */
export function isUsingFFI(): boolean {
  return usingFFI;
}

/**
 * Get FFI module for direct access to libtensorflow operations
 * Returns null if FFI is not available
 */
export function getFFI(): typeof tff | null {
  return tff.isAvailable() ? tff : null;
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
