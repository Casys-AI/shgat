/**
 * Tests for TensorFlow FFI Backend
 *
 * Run: deno test --allow-ffi --allow-read lib/shgat-tf/tests/tf-ffi_test.ts
 */

import { assertEquals } from "jsr:@std/assert";
import * as tff from "../src/tf/tf-ffi.ts";

Deno.test("TF-FFI: isAvailable returns boolean", () => {
  const available = tff.isAvailable();
  assertEquals(typeof available, "boolean");
  console.log(`[TF-FFI] Available: ${available}`);
  if (available) {
    console.log(`[TF-FFI] Version: ${tff.version()}`);
  }
});

// Skip remaining tests if libtensorflow not available
const runIfAvailable = tff.isAvailable() ? Deno.test : Deno.test.ignore;

runIfAvailable("TF-FFI: tensor creation", () => {
  const t = tff.tensor([1, 2, 3, 4], [2, 2]);
  assertEquals(t.shape, [2, 2]);
  assertEquals(t.rank, 2);

  const data = t.dataSync();
  assertEquals(data.length, 4);
  assertEquals(data[0], 1);
  assertEquals(data[3], 4);

  t.dispose();
});

runIfAvailable("TF-FFI: tensor2d creation", () => {
  const t = tff.tensor([[1, 2], [3, 4]]);
  assertEquals(t.shape, [2, 2]);

  const arr = t.arraySync() as number[][];
  assertEquals(arr[0][0], 1);
  assertEquals(arr[1][1], 4);

  t.dispose();
});

runIfAvailable("TF-FFI: zeros and ones", () => {
  const z = tff.zeros([3, 3]);
  assertEquals(z.shape, [3, 3]);
  assertEquals(z.dataSync()[0], 0);
  z.dispose();

  const o = tff.ones([2, 4]);
  assertEquals(o.shape, [2, 4]);
  assertEquals(o.dataSync()[0], 1);
  o.dispose();
});

runIfAvailable("TF-FFI: matmul", () => {
  // [2, 3] @ [3, 2] = [2, 2]
  const a = tff.tensor([[1, 2, 3], [4, 5, 6]], [2, 3]);
  const b = tff.tensor([[1, 2], [3, 4], [5, 6]], [3, 2]);

  const c = tff.matmul(a, b);
  assertEquals(c.shape, [2, 2]);

  const data = c.dataSync();
  // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
  // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
  assertEquals(data[0], 22);
  assertEquals(data[1], 28);
  assertEquals(data[2], 49);
  assertEquals(data[3], 64);

  a.dispose();
  b.dispose();
  c.dispose();
});

runIfAvailable("TF-FFI: add/sub/mul/div", () => {
  const a = tff.tensor([1, 2, 3, 4], [4]);
  const b = tff.tensor([2, 2, 2, 2], [4]);

  const sum = tff.add(a, b);
  assertEquals(Array.from(sum.dataSync()), [3, 4, 5, 6]);

  const diff = tff.sub(a, b);
  assertEquals(Array.from(diff.dataSync()), [-1, 0, 1, 2]);

  const prod = tff.mul(a, b);
  assertEquals(Array.from(prod.dataSync()), [2, 4, 6, 8]);

  const quot = tff.div(a, b);
  assertEquals(Array.from(quot.dataSync()), [0.5, 1, 1.5, 2]);

  a.dispose();
  b.dispose();
  sum.dispose();
  diff.dispose();
  prod.dispose();
  quot.dispose();
});

runIfAvailable("TF-FFI: softmax", () => {
  const x = tff.tensor([1, 2, 3], [3]);
  const s = tff.softmax(x);

  const data = s.dataSync();
  // Softmax should sum to 1
  const total = data[0] + data[1] + data[2];
  assertEquals(Math.abs(total - 1) < 0.001, true);

  // Higher values should have higher probabilities
  assertEquals(data[2] > data[1], true);
  assertEquals(data[1] > data[0], true);

  x.dispose();
  s.dispose();
});

runIfAvailable("TF-FFI: leakyRelu", () => {
  const x = tff.tensor([-2, -1, 0, 1, 2], [5]);
  const y = tff.leakyRelu(x, 0.2);

  const data = y.dataSync();
  // Use approximate comparison for floating point
  assertEquals(Math.abs(data[0] - (-0.4)) < 0.001, true); // -2 * 0.2
  assertEquals(Math.abs(data[1] - (-0.2)) < 0.001, true); // -1 * 0.2
  assertEquals(data[2], 0);
  assertEquals(data[3], 1);
  assertEquals(data[4], 2);

  x.dispose();
  y.dispose();
});

runIfAvailable("TF-FFI: gather", () => {
  const params = tff.tensor([[1, 2], [3, 4], [5, 6]], [3, 2]);
  const indices = tff.tensorInt32(new Int32Array([0, 2]), [2]);

  const gathered = tff.gather(params, indices);
  assertEquals(gathered.shape, [2, 2]);

  const data = gathered.arraySync() as number[][];
  assertEquals(data[0], [1, 2]);
  assertEquals(data[1], [5, 6]);

  params.dispose();
  indices.dispose();
  gathered.dispose();
});

runIfAvailable("TF-FFI: unsortedSegmentSum - THE KEY TEST", () => {
  // This is the operation that was missing in TF.js WASM!
  // data = [[1, 2], [3, 4], [5, 6], [7, 8]]
  // segment_ids = [0, 1, 0, 1]
  // num_segments = 2
  // result[0] = data[0] + data[2] = [1+5, 2+6] = [6, 8]
  // result[1] = data[1] + data[3] = [3+7, 4+8] = [10, 12]

  const data = tff.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], [4, 2]);
  const segmentIds = tff.tensorInt32(new Int32Array([0, 1, 0, 1]), [4]);

  const result = tff.unsortedSegmentSum(data, segmentIds, 2);
  assertEquals(result.shape, [2, 2]);

  const arr = result.arraySync() as number[][];
  assertEquals(arr[0], [6, 8]);
  assertEquals(arr[1], [10, 12]);

  console.log("[TF-FFI] UnsortedSegmentSum works! This enables proper autograd for gather.");

  data.dispose();
  segmentIds.dispose();
  result.dispose();
});

runIfAvailable("TF-FFI: concat", () => {
  const a = tff.tensor([[1, 2], [3, 4]], [2, 2]);
  const b = tff.tensor([[5, 6], [7, 8]], [2, 2]);

  const c = tff.concat([a, b], 0);
  assertEquals(c.shape, [4, 2]);
  const cData = c.arraySync() as number[][];
  assertEquals(cData[0], [1, 2]);
  assertEquals(cData[3], [7, 8]);

  const d = tff.concat([a, b], 1);
  assertEquals(d.shape, [2, 4]);
  const dData = d.arraySync() as number[][];
  assertEquals(dData[0], [1, 2, 5, 6]);
  assertEquals(dData[1], [3, 4, 7, 8]);

  a.dispose();
  b.dispose();
  c.dispose();
  d.dispose();
});

runIfAvailable("TF-FFI: transpose", () => {
  const a = tff.tensor([[1, 2, 3], [4, 5, 6]], [2, 3]);
  const t = tff.transpose(a);

  assertEquals(t.shape, [3, 2]);
  const data = t.arraySync() as number[][];
  assertEquals(data[0], [1, 4]);
  assertEquals(data[1], [2, 5]);
  assertEquals(data[2], [3, 6]);

  a.dispose();
  t.dispose();
});

runIfAvailable("TF-FFI: sum/mean/max with axis", () => {
  const x = tff.tensor([[1, 2, 3], [4, 5, 6]], [2, 3]);

  // Sum along axis 1 (rows)
  const s = tff.sum(x, 1);
  const sData = s.dataSync();
  assertEquals(sData[0], 6);  // 1+2+3
  assertEquals(sData[1], 15); // 4+5+6

  // Mean along axis 0 (columns)
  const m = tff.mean(x, 0);
  const mData = m.dataSync();
  assertEquals(mData[0], 2.5); // (1+4)/2
  assertEquals(mData[1], 3.5); // (2+5)/2
  assertEquals(mData[2], 4.5); // (3+6)/2

  // Max along axis 1
  const mx = tff.max(x, 1);
  const mxData = mx.dataSync();
  assertEquals(mxData[0], 3);
  assertEquals(mxData[1], 6);

  x.dispose();
  s.dispose();
  m.dispose();
  mx.dispose();
});

// Note: Don't close the library in tests - causes leak detection issues
// The library will be closed when the process exits
