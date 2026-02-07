/**
 * POC Test: TensorFlow.js + TransitionModel
 *
 * Validates that TF.js works correctly in Deno and that the
 * TransitionModel can be instantiated and run.
 */

import { assertEquals, assertExists } from "jsr:@std/assert";
import {
  initTensorFlow,
  tf,
  tidy,
  dispose,
  ops,
  TransitionModel,
} from "../mod.ts";

Deno.test("TF.js initializes correctly", async () => {
  const backend = await initTensorFlow();
  assertExists(backend);
  console.log(`Backend: ${backend}`);
});

Deno.test("TF.js basic ops work", async () => {
  await initTensorFlow();

  const result = tidy(() => {
    const a = tf.tensor2d([[1, 2], [3, 4]]);
    const b = tf.tensor2d([[5, 6], [7, 8]]);
    const c = tf.matMul(a, b);
    return c.arraySync();
  });

  assertEquals(result, [[19, 22], [43, 50]]);
});

Deno.test("ops wrapper works", async () => {
  await initTensorFlow();

  const result = tidy(() => {
    const a = ops.toTensor([[1, 2], [3, 4]]);
    const b = ops.toTensor([[5, 6], [7, 8]]);
    const c = ops.matmul(a, b);
    return c.arraySync();
  });

  assertEquals(result, [[19, 22], [43, 50]]);
});

Deno.test("softmax works", async () => {
  await initTensorFlow();

  const result = tidy(() => {
    const x = ops.toTensor([1, 2, 3]);
    const s = ops.softmax(x);
    return s.arraySync() as number[];
  });

  // Verify it sums to 1
  const sum = (result as number[]).reduce((a, b) => a + b, 0);
  assertEquals(Math.abs(sum - 1) < 0.001, true);
});

Deno.test("GRU layer works", async () => {
  await initTensorFlow();

  const result = tidy(() => {
    const gru = tf.layers.gru({
      units: 64,
      returnState: true,
    });

    // [batch=1, seq=3, features=128]
    const input = tf.randomNormal([1, 3, 128]);
    const output = gru.apply(input) as tf.Tensor[];

    return {
      outputShape: output[0].shape,
      stateShape: output[1].shape,
    };
  });

  assertEquals(result.outputShape, [1, 64]); // [batch, units]
  assertEquals(result.stateShape, [1, 64]); // [batch, units]
});

Deno.test("TransitionModel instantiates", async () => {
  await initTensorFlow();

  const model = new TransitionModel({
    embeddingDim: 128,
    hiddenDim: 64,
    numTools: 10,
  });

  assertExists(model);
});

Deno.test("TransitionModel forward pass", async () => {
  await initTensorFlow();

  const model = new TransitionModel({
    embeddingDim: 128,
    hiddenDim: 64,
    numTools: 10,
  });

  // Setup tool vocabulary
  const tools = new Map<string, number[]>();
  for (let i = 0; i < 10; i++) {
    tools.set(`tool:${i}`, Array(128).fill(0).map(() => Math.random()));
  }
  model.setToolVocabulary(tools);

  // Forward pass
  const intentEmb = Array(128).fill(0).map(() => Math.random());
  const result = await model.predictNext(intentEmb, ["tool:0", "tool:1"]);

  assertExists(result.toolId);
  assertEquals(typeof result.shouldTerminate, "boolean");
  assertEquals(typeof result.confidence, "number");

  console.log(`Predicted: ${result.toolId}, terminate: ${result.shouldTerminate}, conf: ${result.confidence.toFixed(3)}`);
});

Deno.test("TransitionModel buildPath", async () => {
  await initTensorFlow();

  const model = new TransitionModel({
    embeddingDim: 128,
    hiddenDim: 64,
    numTools: 10,
    terminationThreshold: 0.99, // Very high threshold to see path building
    maxPathLength: 5,
  });

  // Setup tool vocabulary
  const tools = new Map<string, number[]>();
  for (let i = 0; i < 10; i++) {
    tools.set(`tool:${i}`, Array(128).fill(0).map(() => Math.random()));
  }
  model.setToolVocabulary(tools);

  // Build path
  const intentEmb = Array(128).fill(0).map(() => Math.random());
  const path = await model.buildPath(intentEmb, "tool:0");

  assertExists(path);
  assertEquals(path[0], "tool:0");
  // Path should have at least 1 element (the starting tool)
  assertEquals(path.length >= 1, true, `Path length is ${path.length}`);
  assertEquals(path.length <= 5, true, `Path length ${path.length} exceeds max`);

  console.log(`Path: ${path.join(" -> ")} (length: ${path.length})`);
});

Deno.test("Memory management", async () => {
  await initTensorFlow();

  const before = tf.memory().numTensors;

  // Do some operations - dispose returned tensors
  for (let i = 0; i < 10; i++) {
    const result = tidy(() => {
      const a = tf.randomNormal([100, 100]);
      const b = tf.randomNormal([100, 100]);
      return tf.matMul(a, b);
    });
    // Dispose the returned tensor
    result.dispose();
  }

  const after = tf.memory().numTensors;

  // Should not leak tensors
  assertEquals(after - before, 0, `Leaked ${after - before} tensors`);
});

Deno.test("InfoNCE loss", async () => {
  await initTensorFlow();

  const loss = tidy(() => {
    const anchor = tf.randomNormal([8, 64]) as tf.Tensor2D;
    const positive = tf.randomNormal([8, 64]) as tf.Tensor2D;
    return ops.infoNCELoss(anchor, positive, 0.07);
  });

  const lossValue = loss.arraySync() as number;
  assertEquals(typeof lossValue, "number");
  assertEquals(lossValue > 0, true);

  console.log(`InfoNCE loss: ${lossValue.toFixed(4)}`);
  dispose(loss);
});
