# SHGAT-TF

SuperHyperGraph Attention Networks with TensorFlow.js.

Multi-level message passing on hypergraphs with K-head attention scoring,
designed for tool/capability selection in agentic systems.

## Features

- **Multi-level message passing**: V→E→...→V across hierarchy levels
- **K-head attention**: 4-16 adaptive heads with InfoNCE contrastive loss
- **Dense TF.js autograd**: Automatic differentiation for training
- **PER training**: Prioritized Experience Replay for sample efficiency
- **Curriculum learning**: Easy→hard negative sampling with temperature annealing
- **Dual runtime**: Deno (WebGPU/WASM/CPU) + Node.js (tfjs-node C++ binding)

## Requirements

- Deno 2.x+ or Node.js 20+

## Quick Start

```typescript
import { SHGATBuilder } from "@casys/shgat-tf";

const nodes = [
  { id: "tool-a", embedding: toolAEmb, children: [] },
  { id: "tool-b", embedding: toolBEmb, children: [] },
  { id: "cap-1",  embedding: capEmb,   children: ["tool-a", "tool-b"] },
];

const shgat = await SHGATBuilder.create()
  .nodes(nodes)
  .training({ learningRate: 0.05, temperature: 0.10 })
  .build();

// Score nodes
const scores = shgat.score(intentEmbedding, ["cap-1"]);

// Train
const metrics = await shgat.trainBatch(examples);

// Cleanup
shgat.dispose();
```

## Training

```typescript
import { AutogradTrainer, type TrainingExample } from "@casys/shgat-tf";

const trainer = new AutogradTrainer({
  numHeads: 16,
  embeddingDim: 1024,
  learningRate: 0.05,
});

const examples: TrainingExample[] = [
  {
    intentEmbedding: new Array(1024).fill(0),
    contextTools: ["tool-a"],
    candidateId: "cap-1",
    outcome: 1,
    negativeCapIds: ["cap-2", "cap-3"],
  },
];

const metrics = trainer.trainBatch(examples);
console.log(`Loss: ${metrics.loss}, Accuracy: ${metrics.accuracy}`);
```

### Prioritized Experience Replay (PER)

```typescript
import { PERBuffer } from "@casys/shgat-tf";

const buffer = new PERBuffer(examples, { alpha: 0.6, beta: 0.4 });
const { items, weights, indices } = buffer.sample(batchSize, beta);
```

## Training Pipeline (end-to-end)

The full pipeline from raw data to trained model has 3 steps.
All scripts assume `DATABASE_URL` is set (for prod examples from PostgreSQL).

### Step 1: Build n8n soft targets

Maps n8n workflow nodes to MCP tools using embedding similarity + service matching.
Generates soft target distributions (probability over 1884 tools per n8n node).

```bash
cd lib/gru
export $(grep DATABASE_URL ../../.env | head -1)
npx tsx src/n8n/build-soft-targets.ts
```

**Inputs**: `data/n8n-workflows.json`, `data/n8n-node-embeddings.json`, DB (Smithery + PML tools)
**Outputs**: `data/n8n-training-examples.parquet` (~38K examples), `data/expanded-vocab.json`, `data/n8n-shgat-contrastive-pairs.json`

3-tier matching strategy:
- **Tier 1** (service match): restricts to same-service MCP tools (gmail→gmail, http→http)
- **Tier 2** (CRUD boost): schema Jaccard blend + verb matching on full vocab
- **Tier 3** (cosine fallback): pure embedding similarity

Prerequisite scripts (run once, data cached in `data/`):
```bash
npx tsx src/n8n/scrape-n8n.ts          # Scrape n8n workflow library
npx tsx src/n8n/embed-n8n-nodes.ts     # Embed n8n node types (BGE-M3)
npx tsx src/n8n/scrape-mcp-tools.ts    # Scrape Smithery MCP registry
npx tsx src/n8n/embed-mcp-tools.ts     # Embed Smithery tools (BGE-M3)
```

### Step 2: Export bench dataset

Combines n8n soft targets + prod training examples from DB into bench Parquet files
with proper train/test splits (seeded, per-trace).

```bash
cd lib/shgat-tf
export $(grep DATABASE_URL ../../.env | head -1)
npx tsx tools/export-dataset.ts --no-msgpack
```

**Inputs**: Step 1 outputs + DB prod examples
**Outputs**: `lib/gru/data/bench-*.parquet` (nodes, prod-train, prod-test, n8n-train, n8n-eval)

### Step 3: Train SHGAT

Manual backward pass training with OpenBLAS FFI, contrastive + KL loss.

```bash
cd lib/shgat-tf
deno run -A --max-old-space-size=10240 tools/train-ob.ts \
  --kl --kl-weight 0.2 --seed 42 --epochs 10 --lr 0.005 --eval-every 2
```

Key flags:
- `--kl` — enable KL divergence loss on n8n soft targets (default: on)
- `--kl-weight 0.2` — KL divergence weight
- `--kl-subsample 0` — use all n8n examples (default; set >0 to subsample)
- `--kl-batch-size 2048` — KL batch size (default 2048; higher = faster, more RAM)
- `--kl-isolate-khead` — prevent KL from updating W_q/W_k (default: off, KL updates K-head)
- `--epochs 10` — number of training epochs
- `--lr 0.005` — learning rate (0.001 for <500 examples, 0.005 for 30K+)
- `--eval-every 2` — evaluate on prod test set every N epochs
- `--seed 42` — reproducible splits and initialization

**Outputs**: `lib/gru/data/shgat-params-ob-best.json` (best checkpoint), training report JSON

### Optional: GRU E2E benchmark

Evaluates SHGAT embeddings as input features for the GRU sequence model.

```bash
cd lib/gru
node --max-old-space-size=4096 dist-node/benchmark-e2e.js
```

## Persistence

```typescript
const params = shgat.exportParams();
await Deno.writeTextFile("model.json", JSON.stringify(params));

const loaded = JSON.parse(await Deno.readTextFile("model.json"));
shgat.importParams(loaded);
```

## Node.js Support

For Node.js, use the build script to generate a distribution with `@tensorflow/tfjs-node`:

```bash
cd lib/shgat-tf && ./scripts/build-node.sh
cd dist-node && npm install && npm test
```

This swaps `backend.ts` (Deno: WebGPU/WASM/CPU) with `backend.node.ts` (tfjs-node C++ binding).

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

### Recommended: Builder + Ports

| Export | Description |
|--------|------------|
| `SHGATBuilder` | Fluent builder for SHGAT instances |
| `SHGATScorer` | Scoring-only port interface |
| `SHGATTrainer` | Training-only port interface |
| `SHGATTrainerScorer` | Combined training + scoring port |

### Core

| Export | Description |
|--------|------------|
| `SHGAT` | Main class with scoring, training, persistence |
| `createSHGAT()` | Factory from unified `Node[]` |
| `DEFAULT_SHGAT_CONFIG` | Default configuration |

### Training

| Export | Description |
|--------|------------|
| `AutogradTrainer` | TF.js autograd-based trainer |
| `PERBuffer` | Prioritized Experience Replay |
| `annealTemperature()` | Temperature scheduling |

### Backend

| Export | Description |
|--------|------------|
| `initTensorFlow()` | Initialize backend (auto on import) |
| `switchBackend()` | Switch training/inference mode |
| `supportsAutograd()` | Check backend kernel support |

## License

MIT
