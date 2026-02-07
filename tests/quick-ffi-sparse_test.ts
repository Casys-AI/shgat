/**
 * Quick test: FFI backend with sparse operations
 *
 * Demonstrates that UnsortedSegmentSum now works, enabling
 * proper autograd for sparse matrix multiplication (tf.gather gradients).
 *
 * Run: deno test --allow-all --unstable-ffi lib/shgat-tf/tests/quick-ffi-sparse_test.ts
 */

import { assertEquals } from "jsr:@std/assert";
import * as tff from "../src/tf/tf-ffi.ts";

Deno.test({
  name: "FFI sparse matmul (gather + unsortedSegmentSum)",
  sanitizeResources: false,  // Library stays loaded - that's OK
  sanitizeOps: false,
  fn: async (t) => {
  if (!tff.isAvailable()) {
    console.log("Skipping - libtensorflow not available");
    return;
  }

  console.log(`[TF-FFI] Version: ${tff.version()}`);

  await t.step("sparseMatmul via gather + unsortedSegmentSum", () => {
    // Simulate sparse matrix multiplication: A_sparse @ B_dense
    // where A_sparse is represented by (row_indices, col_indices, values)
    //
    // A_sparse (3x4):
    //   [1, 0, 2, 0]    -> row 0: cols [0, 2], values [1, 2]
    //   [0, 3, 0, 0]    -> row 1: cols [1], values [3]
    //   [0, 0, 0, 4]    -> row 2: cols [3], values [4]
    //
    // B_dense (4x2):
    //   [[1, 2],
    //    [3, 4],
    //    [5, 6],
    //    [7, 8]]
    //
    // Result (3x2):
    //   row 0: 1*[1,2] + 2*[5,6] = [11, 14]
    //   row 1: 3*[3,4] = [9, 12]
    //   row 2: 4*[7,8] = [28, 32]

    const B = tff.tensor(
      [
        [1, 2],
        [3, 4],
        [5, 6],
        [7, 8],
      ],
      [4, 2]
    );

    // Sparse A representation
    const rowIndices = new Int32Array([0, 0, 1, 2]); // which output row
    const colIndices = new Int32Array([0, 2, 1, 3]); // which B row to gather
    const values = tff.tensor([1, 2, 3, 4], [4]); // sparse values

    // Step 1: Gather B rows at colIndices
    // gathered[i] = B[colIndices[i]] -> shape [4, 2]
    const colIdxTensor = tff.tensorInt32(colIndices, [4]);
    const gathered = tff.gather(B, colIdxTensor);
    assertEquals(gathered.shape, [4, 2]);

    // Step 2: Scale by values
    // scaled[i] = values[i] * gathered[i] -> shape [4, 2]
    const valuesExpanded = tff.expandDims(values, 1); // [4, 1]
    const scaled = tff.mul(gathered, valuesExpanded);

    // Step 3: UnsortedSegmentSum to aggregate by row
    // result[r] = sum of scaled[i] where rowIndices[i] == r
    const rowIdxTensor = tff.tensorInt32(rowIndices, [4]);
    const result = tff.unsortedSegmentSum(scaled, rowIdxTensor, 3);

    assertEquals(result.shape, [3, 2]);
    const data = result.arraySync() as number[][];

    // Verify results
    assertEquals(data[0][0], 11); // 1*1 + 2*5
    assertEquals(data[0][1], 14); // 1*2 + 2*6
    assertEquals(data[1][0], 9); // 3*3
    assertEquals(data[1][1], 12); // 3*4
    assertEquals(data[2][0], 28); // 4*7
    assertEquals(data[2][1], 32); // 4*8

    console.log("Sparse matmul result:", data);
    console.log("SUCCESS: gather + unsortedSegmentSum works for sparse ops!");

    // Cleanup
    B.dispose();
    values.dispose();
    colIdxTensor.dispose();
    gathered.dispose();
    valuesExpanded.dispose();
    scaled.dispose();
    rowIdxTensor.dispose();
    result.dispose();
  });
  },
});
