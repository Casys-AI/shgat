# SHGAT-TF Training Pipeline

Technical reference for the SHGAT-TF training system: data loading, graph construction,
multi-level message passing, contrastive/KL losses, and gradient propagation.

**Training script:** `tools/train-ob.ts`
**Runtime:** Deno + OpenBLAS FFI (manual backward, no TF.js autograd)
**Last updated:** 2026-02-14

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Pipeline](#2-data-pipeline)
3. [Graph Structure](#3-graph-structure)
4. [Loss Functions and Gradient Paths](#4-loss-functions-and-gradient-paths)
5. [Training Dynamics](#5-training-dynamics)
6. [Hyperparameters Reference](#6-hyperparameters-reference)
7. [Memory and Performance](#7-memory-and-performance)

---

## 1. Architecture Overview

### 1.1 Full Pipeline

```
                        DATA LOADING                    GRAPH
                        ============                    =====

  bench-nodes.parquet ──┐
  bench-prod-*.parquet ──┤  loadFullDataset()  ──►  buildGraphStructure()
  bench-n8n-*.parquet  ──┤  (tools/load-parquet.ts)    (tools/train-ob.ts:379)
  bench-metadata.json  ──┘                              │
                                                        ▼
                    ┌───────────────────────────────────────────────────────┐
                    │                 TRAINING EPOCH                       │
                    │                                                       │
                    │  1. MP Forward (once)                                 │
                    │     orchestrator.forwardMultiLevelWithCache()         │
                    │     H_init, E_levels_init ──► H_final, E_final       │
                    │                                                       │
                    │  2. InfoNCE batches (prod)                            │
                    │     batchContrastiveForward/Backward ──► dW_q, dW_k  │
                    │     backpropWIntent ──► dW_intent                     │
                    │     Accumulate dNodeEmbedding ──► _epochDH            │
                    │     Adam step: K-head + W_intent (per batch)          │
                    │                                                       │
                    │  3. HIER contrastive batches (L1+, prod+n8n)          │
                    │     InfoNCE(intent, ancestor_embedding)               │
                    │     Accumulate dNodeEmbedding ──► _epochDE[level]     │
                    │     Adam step: K-head + W_intent (per batch)          │
                    │                                                       │
                    │  4. KL batches (n8n soft targets)                     │
                    │     KL(q || p) per sparse target set                  │
                    │     backpropMultiHeadKHeadLogit ──► dW_q, dW_k        │
                    │     Accumulate dNodeEmbedding ──► _epochDH            │
                    │     Adam step: K-head + W_intent (per batch)          │
                    │                                                       │
                    │  5. Epoch-level MP backward ("autoroute")             │
                    │     Normalize _epochDH by numBatches                  │
                    │     Normalize _epochDE by per-level batch count       │
                    │     orchestrator.backwardMultiLevel() ──► mpGrads     │
                    │     Adam step: MP params (once, with MP_LR_SCALE)     │
                    │                                                       │
                    │  6. Eval (every EVAL_EVERY epochs)                    │
                    │     R@1, R@3, R@5, MRR on prod test set               │
                    └───────────────────────────────────────────────────────┘
```

### 1.2 Three Gradient Paths

The training loop maintains three distinct gradient paths, each targeting
different parameter groups with different loss functions.

```
                    ┌─────────────────────────────────────┐
                    │              GRADIENT PATHS          │
                    └─────────────────────────────────────┘

  PATH 1: InfoNCE (prod, hard targets, L0)
  ─────────────────────────────────────────
  intentProjected = W_intent @ intentEmbedding
  logits[i][j] = mean_h(Q_h[i] . K_h[j]) / (sqrt(dim) * tau)
  loss = symmetric_CE(softmax_rows, softmax_cols)
                │
                ├──► dW_q, dW_k           ──► Adam per-batch
                ├──► dW_intent            ──► Adam per-batch
                └──► dNodeEmbedding[i]    ──► _epochDH (accumulated)
                                                │
                                                └──► MP backward (epoch-level)
                                                     └──► dW_child, dW_parent,
                                                          da_upward, da_downward

  PATH 2: KL divergence (n8n, soft targets, L0)
  ──────────────────────────────────────────────
  logits[j] = mean_h(Q_h . K_h[j]) / (sqrt(dim) * tau)
  q[j] = softmax(logits)
  loss = sum_j(p[j] * log(p[j] / q[j])) * klWeight
                │
                ├──► dW_q, dW_k           ──► Adam per-batch
                ├──► dW_intent            ──► Adam per-batch
                └──► dNodeEmbedding[j]    ──► _epochDH (accumulated)
                                                │
                                                └──► MP backward (epoch-level)

  PATH 3: HIER contrastive (prod+n8n, hard targets, L1+)
  ───────────────────────────────────────────────────────
  For each orchestrator level 0..maxLevel:
    positiveEmbs = E_level[ancestorIdxs[0]]  (from MP forward)
    loss = InfoNCE(intentProjected, positiveEmbs) * hierWeight
                │
                ├──► dW_q, dW_k           ──► Adam per-batch (scaled by hierWeight)
                ├──► dW_intent            ──► Adam per-batch (scaled by hierWeight)
                └──► dNodeEmbedding[i]    ──► _epochDE[level] (accumulated)
                                                │
                                                └──► MP backward (epoch-level)
                                                     DIRECT gradient to L1+ nodes
```

### 1.3 Epoch-Level Flow (the "Autoroute")

The key design decision: K-head and W_intent parameters receive Adam updates
every batch (~36 updates/epoch for prod, ~63 for KL). MP parameters receive
exactly ONE update at epoch end.

**Why:** MP backward traverses the full graph (35,356 edges x 16 heads).
Per-batch MP backward would cost ~29s x 100 batches = ~48 min/epoch. Epoch-level
accumulation reduces this to a single ~29s pass, yielding a ~17x speedup.

**Trade-off:** MP gradients are stale relative to K-head updates within the epoch.
This is standard in GNN training (GNNAutoScale, FreshGNN) and acceptable for
single-layer architectures where staleness stays below empirical thresholds.

```
  Batch 1 ──► dH[1] ──┐
  Batch 2 ──► dH[2] ──┤
  ...                  ├──► _epochDH (sum) ──► normalize ──► MP backward
  Batch N ──► dH[N] ──┘    _epochDE (sum)      1/numBatches    (once)
                                                1/levelBatches     │
                                                                   ▼
                                                            Adam step MP
                                                           (epochLR * MP_LR_SCALE)
```

---

## 2. Data Pipeline

### 2.1 Data Sources

| Source | Train | Test/Eval | Loss Type | Purpose |
|--------|------:|----------:|-----------|---------|
| Production traces | 1,155 | 107 | InfoNCE (hard targets) | Ground-truth tool selection sequences |
| n8n workflows | ~30,000 train | ~7,500 eval | KL (soft targets) + HIER | Dense coverage of tool co-occurrence patterns |

Production examples originate from 52 real traces, each oversampled 3x, then
split by trace (not by example) to prevent contamination.

n8n examples are derived from scraped n8n community workflows. Soft targets are
computed with cosine similarity (threshold >= 0.70, temperature T=0.005) between
BGE-M3 embeddings, producing a sparse probability distribution per example.

### 2.2 Storage Format

The dataset is stored as Parquet files, loaded by `tools/load-parquet.ts`.

| File | Content | Columns |
|------|---------|---------|
| `bench-nodes.parquet` | Graph nodes (tools + capabilities) | `id`, `embedding` (Binary/Float32), `children_json`, `level` |
| `bench-prod-train.parquet` | Production training examples | `intent_embedding` (Binary), `context_tool_ids_json`, `target_tool_id`, `is_terminal`, `trace_id` |
| `bench-prod-test.parquet` | Production test examples | Same as above |
| `bench-n8n-train.parquet` | n8n training examples | `intent_embedding`, `context_tool_ids_json`, `target_tool_id`, `is_terminal`, `soft_target_indices` (Binary/Int32), `soft_target_probs` (Binary/Float32) |
| `bench-n8n-eval.parquet` | n8n evaluation examples | Same as above |
| `bench-metadata.json` | Vocabulary and workflow data | `leafIds`, `embeddingDim`, `workflowToolLists` |

Embedding columns store raw Float32 bytes (1024 dims x 4 bytes = 4,096 bytes/row).
The loader (`load-parquet.ts:106`) reconstructs `number[]` from `Uint8Array` via
aligned `Float32Array` views.

Parquet loading is sequential-per-table: only one table is fully materialized in
memory at a time during the loading phase. This avoids the ~5GB peak of the
legacy monolithic msgpack.gz approach.

### 2.3 Subsample Strategy

To limit memory, n8n training examples are trimmed in-place before the
training loop begins:

```
if KL_SUBSAMPLE > 0 and n8nTrain.length > KL_SUBSAMPLE * 2:
    shuffle n8nTrain in-place
    truncate to KL_SUBSAMPLE * 2 (default: 4,000 examples)
```

The 2x factor preserves shuffle variance across epochs. Each epoch then takes
a fresh subsample of `KL_SUBSAMPLE` (default: 2,000) from this pool.

**Reference:** `tools/train-ob.ts:599-606`

### 2.4 Leaf Node Statistics

The vocabulary contains 1,932 L0 leaf nodes:

| Category | Count | Has L1+ Parent | Notes |
|----------|------:|:--------------:|-------|
| n8n-mapped tools | 801 | Yes | Grouped into capabilities via `workflowToolLists` |
| Prod tools (code:*, filesystem:*, std:*, fetch:*) | 1,131 | No (orphans) | All 64 prod tools used in training are orphans |

This asymmetry has a critical training consequence: HIER contrastive examples
(Path 3) come almost entirely from n8n tools. Production tools cannot contribute
to HIER loss because they have no L1+ ancestors in the hierarchy.

---

## 3. Graph Structure

### 3.1 Recursive N-Tier Hierarchy

The graph has a recursive N-tier hierarchy, built by `buildGraphStructure()` (`tools/train-ob.ts:379`).
`maxLevel` is inferred from the data (lines 412-420), inter-level connectivity is built per
parent level `1..maxLevel` (lines 458-486), and `l0Ancestors` walks up all levels (lines 498-519).
The current dataset has 3 tiers, but the code handles any depth automatically:

```
  L0 (leaves/tools)    L1 (capabilities)     L2 (super-capabilities)
  ==================   =================     =======================
  1,932 nodes          6,916 nodes           9 nodes
       │                    │                     │
       └── 35,356 edges ──►│                     │
            (L0 → L1)      └──── 9 edges ──────►│
                             (L1 → L2)
```

| Tier | Dataset Level | Orch Level | Nodes | Edges (to parent) | Fill Rate |
|------|:------------:|:----------:|------:|-------------------:|---------:|
| L0 (tools) | 0 | N/A | 1,932 | N/A | N/A |
| L1 (caps) | 1 | 0 | 6,916 | 35,356 | ~0.26% sparse |
| L2 (super-caps) | 2 | 1 | 9 | 9 | ~0.01% sparse |

**Level mapping convention:** Dataset level `k+1` maps to orchestrator level `k`.
L0 nodes (dataset level 0) are the source nodes, not an orchestrator level.

### 3.2 Sparse Connectivity

Connectivity is stored as `SparseConnectivity` (defined in `src/message-passing/phase-interface.ts`):

```typescript
interface SparseConnectivity {
  sourceToTargets: Map<number, number[]>;  // child → parent indices
  targetToSources: Map<number, number[]>;  // parent → child indices
  numSources: number;
  numTargets: number;
}
```

Two categories of connectivity structures exist:
- `l0ToL1Conn`: L0 tools to L1 capabilities (always present, orch level 0)
- `interLevelConns`: Map keyed by parent orch level — one entry per pair of adjacent levels. With current data, contains level 1 (L1→L2). Deeper hierarchies would add more entries automatically.

### 3.3 `l0Ancestors` Mapping

Each L0 tool maintains a recursive mapping to its ancestors at every hierarchy level:

```
l0Ancestors[l0Idx] = Map<orchLevel, ancestorIdx[]>
```

Built by walk-up traversal in `buildGraphStructure()` (`tools/train-ob.ts:498-519`):

```
for each L0 tool (l0Idx):
  ancestors[l0Idx] = new Map()
  // Level 0: direct L1 parents
  parents_0 = l0ToL1Conn.sourceToTargets.get(l0Idx)
  ancestors[l0Idx].set(0, parents_0)
  // Levels 1..maxLevel: recursive walk-up
  for orchLevel = 1..maxLevel:
    conn = interLevelConns.get(orchLevel)
    prev = ancestors[l0Idx].get(orchLevel - 1)
    nextParents = union(conn.sourceToTargets.get(p) for p in prev)
    ancestors[l0Idx].set(orchLevel, nextParents)
```

Coverage: 801 out of 1,932 L0 tools have at least one ancestor. The remaining
1,131 are orphans (primarily production MCP tools with no capability grouping).

### 3.4 Embedding Matrices

| Matrix | Shape | Source | Mutated During Training? |
|--------|-------|--------|:------------------------:|
| `H_init` | [1932][1024] | L0 node embeddings from dataset | No (frozen input) |
| `E_levels_init[0]` | [6916][1024] | L1 capability embeddings | No (frozen input) |
| `E_levels_init[1]` | [9][1024] | L2 super-capability embeddings | No (frozen input) |
| `H_final` | [1932][1024] | After MP forward (enriched) | Recomputed each epoch |
| `E_final[level]` | varies | After MP forward (enriched) | Recomputed each epoch |

Base embeddings (`H_init`, `E_levels_init`) are never modified. The MP forward
pass produces enriched embeddings (`H_final`, `E_final`) that are used for
scoring within the epoch. These are recomputed at the start of every epoch
with the latest MP parameters.

---

## 4. Loss Functions and Gradient Paths

### 4.1 InfoNCE Loss (L0, Production Examples)

**File:** `src/training/batch-contrastive-loss.ts`

Uses in-batch negatives with symmetric cross-entropy. For a batch of B examples:

```
sim[i][j] = (1/numHeads) * sum_h(dot(Q_h[i], K_h[j])) * (1/sqrt(headDim))
logits[i][j] = sim[i][j] / tau

loss = (CE_rows(softmax(logits), I) + CE_cols(softmax(logits^T), I)) / 2
```

Where `I` is the identity matrix (diagonal = positive pairs).

**Forward** (`batchContrastiveForward`, line 63):
1. Project each intent: `Q_h[i] = W_q[h] @ intentsProjected[i]`
2. Project each positive: `K_h[j] = W_k[h] @ nodeEmbeddings[j]`
3. Compute B x B similarity matrix, average over heads, scale by `1/(sqrt(headDim) * tau)`
4. Compute row-wise and column-wise softmax
5. Symmetric CE: `(-mean(log(diag(softmax_rows))) + -mean(log(diag(softmax_cols)))) / 2`

**Backward** (`batchContrastiveBackward`, line 167):
1. `dLogits[i][j] = ((softmax_rows[i][j] - delta) + (softmax_cols[j][i] - delta)) / (2*B)`
2. Chain through scaling: `dSim = dLogits * (1/numHeads) * scale / tau`
3. Per-head: `dQ_h[i] = sum_j(dSim[i][j] * K_h[j])`, `dK_h[j] = sum_i(dSim[i][j] * Q_h[i])`
4. Accumulate: `dW_q[h] += outer(dQ_h[i], intentsProjected[i])`, `dW_k[h] += outer(dK_h[i], nodeEmbeddings[i])`
5. Back to inputs: `dIntentsProjected[i] += W_q^T @ dQ_h[i]`, `dNodeEmbeddings[i] += W_k^T @ dK_h[i]`

**Gradient flow downstream:**
- `dW_q`, `dW_k` --> Adam step (per batch)
- `dIntentsProjected` --> `backpropWIntent()` --> `dW_intent` --> Adam step (per batch)
- `dNodeEmbeddings[i]` --> `_epochDH[l0Idx]` (accumulated across all batches)

### 4.2 KL Divergence (L0, n8n Soft Targets)

**File:** `tools/train-ob.ts:1031-1152` (inline, not factored into a module)

For each n8n example with a sparse soft target distribution `p`:

```
q[j] = softmax(logits[j] / tau)   over the sparse target set
KL = sum_j(p[j] * log(p[j] / q[j]))
dLogit[j] = (q[j] - p[j]) * klWeight / tau
```

The gradient `dLogit[j]` flows through `backpropMultiHeadKHeadLogit()` (from
`multi-level-trainer-khead.ts:181`), which:
1. Distributes `dLogit` across heads: `dHeadLogit = dLogit / numHeads`
2. Computes `dQ` and `dK` per head via the Q.K/sqrt(dim) derivative
3. Accumulates `dW_q[h]`, `dW_k[h]` via outer products
4. Produces `dIntentProjected` and `dNodeEmbedding` for upstream propagation

**Gradient flow:**
- `dW_q`, `dW_k` --> Adam step (per batch)
- `dIntentProjected` --> `backpropWIntent()` --> `dW_intent` --> Adam step (per batch)
- `dNodeEmbedding` --> `_epochDH[l0Idx]` (accumulated)

**Key difference from InfoNCE:** KL operates on a sparse subset of L0 nodes
(typically 10-20 per example, defined by `softTargetSparse`). This makes KL the
primary source of *dense* `dH` gradients -- each KL example touches 10-20 L0
entries in `_epochDH`, compared to exactly 1 for InfoNCE.

### 4.3 HIER Contrastive (L1+, Production + n8n Examples)

**File:** `tools/train-ob.ts:911-1029`

For each orchestrator level, computes InfoNCE between the projected intent and
the ancestor's enriched embedding from the MP forward pass:

```
positiveEmbs[i] = E_level[ancestorIdxs[0]]  (from MP result)
loss = InfoNCE(intentsProjected, positiveEmbs) * hierWeight
```

**Example collection** (the `collectFromExamples` helper, line 938):

For each example (prod or n8n):
1. Look up `l0Idx` for `targetToolId`
2. Check `l0Ancestors[l0Idx]` for ancestors at the current orch level
3. If ancestor exists, include the example with `ancestorIdxs`

Since production tools are orphans (no L1+ parent), HIER examples come almost
entirely from n8n tools. This was a bug in run 4 (only prod was iterated),
fixed in run 5.

**Gradient flow:**
- `dW_q`, `dW_k` --> scaled by `hierWeight` --> Adam step (per batch)
- `dIntentsProjected` --> scaled by `hierWeight` --> `backpropWIntent()` --> Adam step
- `dNodeEmbeddings[i]` --> scaled by `hierWeight` --> `_epochDE[orchLevel][ancestorIdx]` (accumulated)

**This is the DIRECT gradient to L1+ nodes.** Before HIER contrastive (runs 1-4),
`_epochDE` was always zero. The MP backward only received signal through `_epochDH`
(indirect, through the downward E->V path). HIER provides a direct supervision
signal: "this capability aggregation must match this intent."

### 4.4 Epoch-Level Normalization

Before calling MP backward, accumulated gradients are normalized:

```
_epochDH /= (infoBatches + klBatches)     // total batches that contributed dH
_epochDE[level] /= hierBatchesByLevel[level]  // per-level batch count
```

**Why per-level normalization for dE:** Using the global batch count would
under-weight levels that have fewer examples (L2 has far fewer ancestor-bearing
examples than L1). Per-level normalization ensures each level's gradient
magnitude reflects its own batch density.

**Reference:** `tools/train-ob.ts:1157-1176`

### 4.5 MP Backward

**File:** `src/message-passing/multi-level-orchestrator.ts:982`

The `backwardMultiLevel()` method reverses the forward pass:

```
1. E^0 --> V backward (evCaches)
   dH receives gradients from E^0 phase
   dE[0] receives gradients flowing back to E^0

2. Downward passes backward (eeDownwardCaches)
   For level 0..maxLevel-1:
     dE[k] --> backward through E^(k+1)->E^k phase
     Accumulates dW_parent, dW_child, da_downward

3. Upward passes backward (veCaches + eeUpwardCaches)
   For level maxLevel..0:
     If level 0: V->E^0 backward (veCaches)
       Accumulates dW_child, dW_parent, da_upward, dH
     Else: E^(k-1)->E^k backward (eeUpwardCaches)
       Accumulates dW_child, dW_parent, da_upward
       Propagates dE to child level
```

**Output:** `MultiLevelGradients` containing per-level `dW_child`, `dW_parent`,
`da_upward`, `da_downward`, plus `dH` and `dE` for input embedding gradients.

These are applied via a single Adam step with `epochLR * MP_LR_SCALE`:

```typescript
adam.lr = epochLR * MP_LR_SCALE;
for (const [level, lp] of levelParams) {
  const lg = mpGrads.levelGrads.get(level);
  for (let h = 0; h < NUM_HEADS; h++) {
    adam.step(`W_child_L${level}_H${h}`, lp.W_child[h], lg.dW_child[h]);
    adam.step(`W_parent_L${level}_H${h}`, lp.W_parent[h], lg.dW_parent[h]);
  }
  adam.step(`a_up_L${level}`, lp.a_upward, lg.da_upward);
  adam.step(`a_down_L${level}`, lp.a_downward, lg.da_downward);
}
```

---

## 5. Training Dynamics

### 5.1 The 320x Gradient Suppression Factor (Runs 1-2)

Three independent issues combined to suppress MP gradients to near-zero:

| Issue | Factor | Root Cause | Fix |
|-------|:------:|------------|-----|
| Normalization: `totalExamples` instead of `numBatches` | 32x | Loss is already averaged by batch_size in `batchContrastiveBackward`. Dividing by `totalExamples` introduced an extra `1/BATCH_SIZE` factor. | Normalize by `numBatches` (run 2) |
| KL binary warmup (0 for 3 epochs) | Infinity (3 epochs) | KL is the main source of dense dH gradients (10-20 L0 nodes/example vs 1 for InfoNCE). Zeroing KL = starving MP. | Soft ramp from `plateau * 0.1` at epoch 0 (run 3) |
| `MP_LR_SCALE = 0.1` | 10x | With epoch-level accumulation (1 update/epoch vs 36 for K-head), 0.1 scale = 360x less effective update. | `MP_LR_SCALE = 1.0` (run 3) |

**Combined suppression:** up to ~320x (32 x 10) after epoch 3, and **zero signal**
before epoch 3.

After applying all three fixes (run 3), MP gradient norms reached non-zero values
(`|dW_child| = 4.28e-2, |dW_parent| = 4.28e-2`) for the first time.

### 5.2 The Adam Sqrt Scaling Rule

When batch size increases by factor `kappa`, Adam's optimal LR scales by
`sqrt(kappa)` (Princeton 2024, NeurIPS 2024). MP's effective batch is the
entire epoch:

```
kappa = epoch_examples / BATCH_SIZE = 1155 / 32 = 36
sqrt(36) = 6

Old MP_LR_SCALE = 0.1  -->  effective ratio = 1/(36 * 0.1) = 0.28x optimal
New MP_LR_SCALE = 1.0  -->  Adam's internal moment adaptation handles the rest
```

Adam's second moment (`v`) naturally adapts to different gradient magnitudes,
making explicit scaling less critical than for SGD.

### 5.3 HIER Contrastive: The Orphan Problem (Runs 4-5)

**Run 4:** Implemented HIER contrastive loss, but iterated only production
examples for ancestor collection. All 64 production tools used in training are
orphans (no L1 parent) --> 0 HIER examples --> code was dead.

**Root cause:** 801/1,932 L0 tools have ancestors, but they are all n8n-mapped
tools. Production MCP tools (code:*, filesystem:*, std:*, fetch:*) are intentionally
ungrouped in the hierarchy.

**Run 5 fix:** `collectFromExamples()` now iterates both `prodShuffled` AND
`ds.n8nTrain`, then shuffles the combined set before batching. This yields ~800+
HIER examples per epoch.

### 5.4 KL Weight Schedule

The KL weight follows a soft ramp instead of a binary warmup:

```
epoch 0:              klWeight = plateau * 0.1 = 0.02
epoch 1..rampEnd:     linear interpolation 0.02 --> plateau
epoch rampEnd..end:   klWeight = plateau = 0.2
```

Where `rampEnd = KL_WARMUP * 2 = 6` (default).

**Reference:** `scheduleKLWeight()` at `tools/train-ob.ts:252`

**Rationale:** The old binary warmup (0 for 3 epochs, then ramp) created an
information delay in the feedback loop. KL is the main source of dense dH
gradients. Cutting it entirely during warmup starved the MP of gradient signal
during the critical early epochs when K-head parameters are still learning.
The soft ramp ensures a minimum directional signal from epoch 0.

### 5.5 Optimizer Strategy: Dual Frequency

| Parameter Group | Update Frequency | LR | Optimizer |
|-----------------|:----------------:|:--:|-----------|
| `W_q[h]`, `W_k[h]` (K-head) | Per batch (~36-100x/epoch) | `epochLR` | Adam (gradient clip 1.0) |
| `W_intent` | Per batch | `epochLR` | Adam |
| `W_child[level][h]`, `W_parent[level][h]` (MP) | Once per epoch | `epochLR * MP_LR_SCALE` | Adam |
| `a_upward[level]`, `a_downward[level]` (MP attention) | Once per epoch | `epochLR * MP_LR_SCALE` | Adam |

All parameter groups share a single Adam optimizer instance with per-group
moment tracking. The LR is temporarily overridden for MP params:

```typescript
const savedLr = adam.lr;
adam.lr = epochLR * MP_LR_SCALE;
// ... MP Adam steps ...
adam.lr = savedLr;
```

### 5.6 Run History Summary

| Run | Changes | Outcome |
|-----|---------|---------|
| Run 1 | Epoch-level MP backward ("autoroute") | ~17x speedup, but MP grad norms ~0 (normalization bug) |
| Run 2 | Fix normalization: numBatches not totalExamples | MP grad norms still ~0 (KL warmup + low MP_LR_SCALE) |
| Run 3 | KL soft ramp + MP_LR_SCALE=1.0 + numBatches fix | MP grad norms non-zero (4.28e-2). First epoch: loss=2.334, acc=43.5% |
| Run 4 | Add HIER contrastive loss + l0Ancestors | 0 HIER examples (prod tools are orphans), no impact |
| Run 5 | Fix HIER source: collectFromExamples(prod + n8n) | ~800+ HIER examples per epoch, MP receives direct dE gradient |

---

## 6. Hyperparameters Reference

### 6.1 CLI Arguments

| Argument | Default | Description | Justification |
|----------|:-------:|-------------|---------------|
| `--epochs` | 15 | Training epochs | Balanced: enough for convergence, not too long (~7min/epoch) |
| `--batch-size` | 32 | Batch size for all loss paths | Standard for contrastive learning with 1K-3K examples |
| `--lr` | 0.005 | Peak learning rate | Scaled up from 0.001 (282 examples) to account for 36K examples. LR=0.01 oscillated (run 1) |
| `--lr-warmup` | 3 | LR warmup epochs | Cosine schedule: ramp up for 3 epochs, then cosine decay |
| `--temperature` | 0.10 | InfoNCE temperature start | Cosine annealed to 0.06. Controls sharpness of contrastive distribution |
| `--seed` | 42 | Random seed (mulberry32 PRNG) | Reproducible shuffles, splits, and parameter initialization |
| `--kl / --no-kl` | ON | Enable KL divergence on n8n | Provides dense dH gradients from n8n soft targets |
| `--kl-warmup` | 3 | KL warmup epochs (soft ramp period = 2x) | Soft ramp over 6 epochs, minimum weight = plateau * 0.1 from epoch 0 |
| `--kl-weight` | 0.2 | KL loss weight at plateau | Balances KL contribution vs InfoNCE (too high = n8n dominates) |
| `--mp-lr-scale` | 1.0 | LR multiplier for MP params | Was 0.1, increased to 1.0 (run 3). Adam adapts internally |
| `--eval-every` | 2 | Full eval every N epochs | Batched eval is fast (~1s); frequent eval catches early stopping opportunities |
| `--kl-subsample` | 2000 | Max n8n examples per epoch (0=all) | Memory management: keeps 2x in pool for shuffle variance |
| `--msgpack` | OFF | Use msgpack.gz instead of Parquet | Legacy loader, higher peak memory (~5GB vs ~3GB) |
| `--data-path` | auto | Path to msgpack.gz (only with --msgpack) | Override default data directory |

### 6.2 Hardcoded Constants

| Constant | Value | Location | Notes |
|----------|:-----:|----------|-------|
| `TAU_END` | 0.06 | `train-ob.ts:152` | Temperature floor (cosine annealed from TAU_START) |
| `NUM_HEADS` | 16 | `train-ob.ts:614` | Head count for K-head scoring and MP attention |
| `HEAD_DIM` | 64 | `train-ob.ts:615` | `embeddingDim / NUM_HEADS` (1024 / 16) |
| `hierWeight` | 0.5 | `train-ob.ts:916` | Scale for HIER contrastive relative to InfoNCE |
| `preserveDim` | true | `train-ob.ts:631` | `embeddingDim = hiddenDim = 1024` (no projection) |
| `leakyReluSlope` | 0.2 | `train-ob.ts:629` | For attention coefficient computation in MP phases |
| `gradientClip` | 1.0 | `train-ob.ts:672` | Adam gradient clipping by L2 norm |

### 6.3 Schedule Functions

**Learning rate** (`scheduleLR`, `train-ob.ts:241`):
```
epoch < warmup:  lrMin + (lrPeak - lrMin) * (epoch+1)/warmup
                 where lrMin = lrPeak * 0.01
epoch >= warmup: cosine decay from lrPeak to lrMin
```

**Temperature** (`scheduleTemperature`, `train-ob.ts:263`):
```
tau = TAU_END + (TAU_START - TAU_END) * 0.5 * (1 + cos(pi * epoch/(EPOCHS-1)))
```
Cosine annealing from 0.10 to 0.06 over all epochs.

**KL weight** (`scheduleKLWeight`, `train-ob.ts:252`):
```
minWeight = plateau * 0.1
rampEnd = warmupEpochs * 2
epoch >= rampEnd:  plateau
epoch < rampEnd:   minWeight + (plateau - minWeight) * epoch/rampEnd
```

---

## 7. Memory and Performance

### 7.1 Memory Budget

Target: 8GB heap (`--max-old-space-size=8192` or Deno default).
Peak observed: ~6GB RSS.

| Component | Estimated Size | Notes |
|-----------|---------------:|-------|
| Node embeddings (H_init, 1932 x 1024) | ~15 MB | Float64 in JS |
| Capability embeddings (E_init, 6916 x 1024) | ~54 MB | Float64 |
| Enriched copies (H_final, E_final) | ~69 MB | Recomputed each epoch |
| MP backward cache (per-phase caches) | ~200-400 MB | Released after MP backward |
| n8n train examples (4000 x 1024D embedding + soft targets) | ~50 MB | After subsample |
| Prod train examples (1155 x 1024D) | ~9 MB | |
| Adam optimizer states (m + v per param) | ~118 MB | 2 x 7.35M x 8 bytes |
| Trainable parameters | ~59 MB | 7.35M params x 8 bytes (W_q+W_k, W_intent, MP) |
| `_epochDH` accumulator (1932 x 1024) | ~15 MB | Zeroed each epoch |
| `_epochDE` accumulators | ~54 MB | Zeroed each epoch |
| **Total estimated** | **~700 MB - 1.0 GB** | Well within 8GB budget |

**Memory optimization applied:**
- `ds.nodes` cleared after graph construction (line 597): saves ~144 MB
- n8n train trimmed to `2 * KL_SUBSAMPLE` (line 601): saves ~400 MB
- `mpBackwardCache` and `mpResult` set to `null` after MP backward (line 1204): releases caches for GC

### 7.2 Timing Breakdown (Per Epoch)

| Phase | Duration | Notes |
|-------|:--------:|-------|
| MP Forward | ~12s | `forwardMultiLevelWithCache`: full graph traversal with cache |
| KL Pre-compute | ~0.5s | `matmulTranspose(H_final, W_k[h])` x numHeads (run 6+) |
| InfoNCE batches (~36 batches) | ~60s | K-head scoring + Adam per batch |
| HIER contrastive batches | ~30s | InfoNCE at L1+ levels |
| KL batches (~63 batches) | ~120s* | Sparse KL computation per example |
| MP Backward | ~29s | Single backward pass through full graph |
| Eval | ~1s | Batched K-projection precomputation |
| **Total per epoch** | **~4-7 min** | Varies with KL_SUBSAMPLE and data size |

\* KL batch timing expected to decrease significantly with pre-computed key projections (run 6+). Benchmarks pending.

### 7.3 Autoroute Speedup

| Approach | MP Backward Cost | Total (15 epochs) |
|----------|:----------------:|:------------------:|
| Per-batch MP backward | ~29s x ~100 batches = ~48 min/epoch | ~12 hours |
| Epoch-level (autoroute) | ~29s x 1 = ~29s/epoch | ~1.5 hours |
| **Speedup** | | **~8x** |

The theoretical 17x speedup (from CHANGELOG) assumes the original 20-25 min/epoch
timing. Actual speedup depends on the ratio of MP backward cost to other phases.

### 7.4 Parameter Count

| Group | Formula | Count |
|-------|---------|------:|
| W_q (K-head) | NUM_HEADS x HEAD_DIM x embDim | 1,048,576 |
| W_k (K-head) | NUM_HEADS x HEAD_DIM x embDim | 1,048,576 |
| W_intent | embDim x embDim | 1,048,576 |
| W_child (MP, per level) | (maxLevel+1) levels x NUM_HEADS x HEAD_DIM x embDim | 2,097,152 * |
| W_parent (MP, per level) | (maxLevel+1) levels x NUM_HEADS x HEAD_DIM x embDim | 2,097,152 * |
| a_upward (MP, per level) | (maxLevel+1) levels x NUM_HEADS x (2 * HEAD_DIM) | 4,096 * |
| a_downward (MP, per level) | (maxLevel+1) levels x NUM_HEADS x (2 * HEAD_DIM) | 4,096 * |
| **Total** | | **~7.35M** |

\* Counts shown for current dataset with `maxLevel = 1` (2 levels). Deeper hierarchies scale linearly.

Note: W_v and attention vector `a` from HeadParams are initialized but only
W_q and W_k are used in training. W_v is a legacy artifact from inference mode.

### 7.5 Evaluation

Eval is batched for speed. The key optimization is precomputing K projections
for all L0 nodes once per eval:

```
K_all[h] = W_k[h] @ AllL0Embs^T    // [headDim x numL0], computed once
```

For each test example:
```
Q_h = W_q[h] @ W_intent @ intentEmbedding    // [headDim]
scores[i] = mean_h(Q_h . K_all[h][:,i]) / sqrt(headDim)
rank = count of scores > targetScore
```

Metrics: R@1, R@3, R@5, MRR. Best R@1 and MRR are tracked across epochs.

**Reference:** `tools/train-ob.ts:1239-1319`

### 7.6 KL Batch Optimization: Pre-Computed Key Projections

The KL divergence phase is the most expensive per epoch because it scores each intent
against ALL tools in the vocabulary (not just in-batch negatives like InfoNCE).

#### Problem

For each KL batch, the naive approach computes:
```
for each example in batch:
  for each sparse target tool:
    for each head h:
      K = matVecBlas(W_k[h], nodeEmbedding)     // O(embDim x headDim) FFI call
      Q = matVecBlas(W_k[h], intentProjected)    // O(embDim x headDim) FFI call
      score = cosine(Q, K)
```

With ~2000 sparse targets per example, 16 heads, and 63 KL batches, this produces
~15K-25K `matVecBlas` FFI calls per batch — each with marshaling overhead.

#### Solution: Pre-Compute K Projections (run 6+)

Since `H_final` is constant within an epoch (MP forward runs once), all key projections
can be computed in a single BLAS `gemm` call per head:

```
// Once per epoch, after MP forward:
for each head h:
  projectedKeysPerHead[h] = matmulTranspose(H_final, W_k[h])
  // [numNodes x embDim] @ [embDim x headDim]^T -> [numNodes x headDim]
  // Single cblas_sgemm call via OpenBLAS FFI
```

Then during KL scoring:
```
for each sparse target l0Idx:
  K = projectedKeysPerHead[h][l0Idx]    // O(1) array lookup
```

#### Additional: Integer Index Optimization

Sparse target resolution changed from string IDs to integer indices:
```
// Before: sparseL0Ids: string[] -> l0IdxMap.get(nodeId) per access
// After:  sparseL0Idxs: number[] resolved once, direct H_final[idx] access
```

This eliminates hash-map lookups in the hot loop (both forward scoring and backward
gradient accumulation into `_epochDH[tIdx]`).

#### Impact

| Metric | Before | After |
|--------|:------:|:-----:|
| Pre-compute cost | 0 | ~500ms/epoch |
| KL scoring per batch | ~15K-25K matVec FFI calls | ~15K-25K array lookups |
| KL backward per batch | String hash lookups in hot loop | Direct integer index |
| Expected speedup | — | 10-30x on KL phase |

**Reference:** `tools/train-ob.ts` (search for `projectedKeysPerHead`)

---

## Appendix A: Key Source Files

| File | Lines | Purpose |
|------|------:|---------|
| `tools/train-ob.ts` | ~1360 | Main training script: data loading, graph, training loop, eval |
| `tools/load-parquet.ts` | ~295 | Parquet loader with typed extraction |
| `src/training/batch-contrastive-loss.ts` | ~246 | InfoNCE forward/backward with in-batch negatives |
| `src/training/multi-level-trainer-khead.ts` | ~241 | K-head scoring forward/backward, W_intent backprop |
| `src/training/adam-optimizer.ts` | ~163 | Adam optimizer for plain JS arrays |
| `src/message-passing/multi-level-orchestrator.ts` | ~1253 | MP forward/backward orchestration across hierarchy levels |
| `src/message-passing/vertex-to-edge-phase.ts` | — | V->E aggregation (upward) |
| `src/message-passing/edge-to-vertex-phase.ts` | — | E->V propagation (downward final) |
| `src/message-passing/edge-to-edge-phase.ts` | — | E->E propagation (inter-level) |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **K-head** | Multi-head projection scoring: `score = mean_h(Q_h . K_h / sqrt(dim))` |
| **MP** | Message Passing -- information propagation through the hypergraph (VE, EE, EV phases) |
| **InfoNCE** | Contrastive loss: maximize positive pair similarity relative to in-batch negatives |
| **KL divergence** | Loss for soft target distributions from n8n workflow co-occurrence |
| **HIER contrastive** | InfoNCE applied at L1+ hierarchy levels (intent vs ancestor embedding) |
| **Autoroute** | Epoch-level gradient accumulation for MP backward (single pass per epoch) |
| **Orphan tool** | L0 node with no L1+ parent in the hierarchy (all prod tools are orphans) |
| **OpenBLAS FFI** | Foreign Function Interface to OpenBLAS for accelerated matrix operations |
| **Orch level** | Orchestrator level: dataset level minus 1 (L1 caps = orch level 0) |
| **BGE-M3** | Embedding model producing 1024D vectors for intents and tool descriptions |
| **Soft targets** | Sparse probability distributions from n8n tool co-occurrence (T=0.005) |
| **hierWeight** | Scale factor (0.5) balancing HIER contrastive against L0 InfoNCE |
