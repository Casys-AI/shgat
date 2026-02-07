/**
 * Custom Kernel: UnsortedSegmentSum via FFI
 *
 * Registers a custom kernel for UnsortedSegmentSum that uses libtensorflow FFI.
 * This enables proper tf.gather gradients on WASM backend!
 *
 * @module shgat-tf/tf/kernels/unsorted-segment-sum
 */

import * as tf from "npm:@tensorflow/tfjs@4.22.0";
import * as tff from "../tf-ffi.ts";

let registered = false;

/**
 * Register the custom UnsortedSegmentSum kernel
 *
 * Must be called after TF.js is initialized but before using gather gradients.
 */
export function registerUnsortedSegmentSumKernel(): void {
  if (registered) return;

  // Check if FFI is available
  if (!tff.isAvailable()) {
    console.warn(
      "[UnsortedSegmentSum] FFI not available - gather gradients may fail on WASM"
    );
    return;
  }

  // Get the WASM backend
  const backend = tf.backend();
  if (!backend) {
    console.warn("[UnsortedSegmentSum] No backend available yet");
    return;
  }

  const backendName = tf.getBackend();

  // Register custom kernel
  tf.registerKernel({
    kernelName: "UnsortedSegmentSum",
    backendName: backendName,
    kernelFunc: ({ inputs, attrs, backend: _backend }) => {
      const { x, segmentIds } = inputs as {
        x: tf.TensorInfo;
        segmentIds: tf.TensorInfo;
      };
      const { numSegments } = attrs as { numSegments: number };

      // Read data from TF.js tensors
      const xData = tf.backend().readSync(x.dataId) as Float32Array;
      const segData = tf.backend().readSync(segmentIds.dataId) as Int32Array;

      // Create FFI tensors
      const xTensor = tff.tensor(xData, x.shape as number[]);
      const segTensor = tff.tensorInt32(segData, segmentIds.shape as number[]);

      // Call FFI implementation
      const result = tff.unsortedSegmentSum(xTensor, segTensor, numSegments);

      // Get result data
      const resultData = result.dataSync();

      // Compute output shape
      const outShape = [numSegments, ...x.shape.slice(1)];

      // Cleanup FFI tensors
      xTensor.dispose();
      segTensor.dispose();
      result.dispose();

      // Create output tensor info
      const outId = tf.backend().write(
        resultData,
        outShape,
        x.dtype as tf.DataType
      );

      return { dataId: outId, shape: outShape, dtype: x.dtype };
    },
  });

  registered = true;
  console.error(`[UnsortedSegmentSum] Custom FFI kernel registered for ${backendName}`);
}

/**
 * Check if the custom kernel is registered
 */
export function isRegistered(): boolean {
  return registered;
}
