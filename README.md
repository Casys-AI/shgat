# SHGAT-TF

SuperHyperGraph Attention Networks with TensorFlow FFI for Deno.

Multi-level message passing on hypergraphs with K-head attention scoring,
designed for tool/capability selection in agentic systems.

## Features

- **Multi-level message passing**: V→E→...→V across hierarchy levels
- **K-head attention**: 4-16 adaptive heads with InfoNCE contrastive loss
- **Sparse message passing**: ~10x faster training on large graphs
- **PER training**: Prioritized Experience Replay for sample efficiency
- **Curriculum learning**: Easy→hard negative sampling with temperature annealing
- **libtensorflow FFI**: Native C performance via `Deno.dlopen` (no WASM overhead)

## Requirements

- Deno 2.x+
- libtensorflow 2.x (see [installation](#tensorflow-installation))

## Installation

```bash
deno add jsr:@casys/shgat
```

### TensorFlow Installation

SHGAT-TF uses libtensorflow via Deno FFI. Install the shared library:

```bash
# Linux (x86_64)
curl -L https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.16.1.tar.gz | \
  sudo tar -xz -C /usr/local
sudo ldconfig

# macOS (arm64)
curl -L https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-arm64-2.16.1.tar.gz | \
  sudo tar -xz -C /usr/local
```

## Quick Start

```typescript
import {
  createSHGATFromCapabilities,
  type TrainingExample,
} from "@casys/shgat";

// Create model from capabilities
const capabilities = [
  {
    id: "cap-1",
    embedding: new Array(1024).fill(0).map(() => Math.random()),
    toolsUsed: ["tool-a", "tool-b"],
    successRate: 0.85,
  },
];

const shgat = createSHGATFromCapabilities(capabilities);

// Score capabilities for an intent
const intentEmbedding = new Array(1024).fill(0).map(() => Math.random());
const scores = shgat.scoreAllCapabilities(intentEmbedding, ["tool-a"]);
console.log(scores[0]); // { capabilityId: "cap-1", score: 0.73, ... }
```

## Training

```typescript
const examples: TrainingExample[] = [
  {
    intentEmbedding: new Array(1024).fill(0),
    contextTools: ["tool-a"],
    candidateId: "cap-1",
    outcome: 1,
    negativeCapIds: ["cap-2", "cap-3"],
  },
];

// Train with K-head attention + InfoNCE loss
const result = shgat.trainBatchV1KHeadBatched(
  examples,
  examples.map(() => 1.0), // IS weights (for PER)
  false,                   // evaluateOnly
  0.08,                    // temperature
);

console.log(`Loss: ${result.loss}, Accuracy: ${result.accuracy}`);
```

### Temperature Annealing

Start warm (0.10), cool down (0.06) for sharper predictions:

```typescript
for (let epoch = 0; epoch < 25; epoch++) {
  const temp = 0.10 - (0.10 - 0.06) * (epoch / 24);
  shgat.trainBatchV1KHeadBatched(examples, weights, false, temp);
}
```

### Prioritized Experience Replay (PER)

```typescript
import { PERBuffer } from "@casys/shgat";

const buffer = new PERBuffer(examples, { alpha: 0.6, beta: 0.4 });
const { items, weights, indices } = buffer.sample(batchSize, beta);

const result = shgat.trainBatchV1KHeadBatched(items, weights);
buffer.updatePriorities(indices, result.tdErrors);
```

## Persistence

SHGAT params are plain objects:

```typescript
// Export
const params = shgat.exportParams();
await Deno.writeTextFile("model.json", JSON.stringify(params));

// Import
const loaded = JSON.parse(await Deno.readTextFile("model.json"));
shgat.importParams(loaded);
```

## Configuration

```typescript
import { SHGAT, DEFAULT_SHGAT_CONFIG } from "@casys/shgat";

const shgat = new SHGAT({
  ...DEFAULT_SHGAT_CONFIG,
  embeddingDim: 1024,    // BGE-M3 embeddings
  numHeads: 16,          // K-head attention
  headDim: 64,           // Per-head dimension
  numLayers: 2,          // Message passing layers
  dropout: 0.1,          // Dropout rate
  learningRate: 0.05,    // SGD learning rate
});
```

## Architecture

```
Intent embedding (1024-dim)
        |
        v
  K-head Attention Scoring (16 heads x 64D)
        |
        v
  Multi-level Message Passing
     UPWARD:   Tools(H) → E^0 → E^1 → ... → E^L
     DOWNWARD: E^L → ... → E^1 → E^0 → Tools(H_enriched)
        |
        v
  InfoNCE Contrastive Loss (with temperature annealing)
        |
        v
  Ranked capability/tool scores
```

## API Reference

### Core

| Export | Description |
|--------|------------|
| `SHGAT` | Main class with scoring, training, persistence |
| `createSHGATFromCapabilities()` | Factory from capability definitions |
| `createSHGAT()` | Low-level factory |
| `DEFAULT_SHGAT_CONFIG` | Default configuration |

### Training

| Export | Description |
|--------|------------|
| `AutogradTrainer` | TF autograd-based trainer |
| `sparseMPForward()` | Sparse message passing forward |
| `sparseMPBackward()` | Sparse message passing backward |
| `PERBuffer` | Prioritized Experience Replay |
| `annealTemperature()` | Temperature scheduling |

### Graph

| Export | Description |
|--------|------------|
| `GraphBuilder` | Hypergraph construction |
| `computeHierarchyLevels()` | Hierarchy level computation |
| `buildMultiLevelIncidence()` | Incidence matrix construction |

### TensorFlow FFI

| Export | Description |
|--------|------------|
| `initTensorFlow()` | Initialize libtensorflow backend |
| `tff.*` | Low-level FFI tensor operations |
| `tensor()`, `matMul()`, `softmax()` | High-level tensor ops |

## License

MIT
