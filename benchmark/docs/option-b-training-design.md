# Option B: Contrastive Training Design for LiveMCPBench

## 1. Context & Baseline

**Option A results (orthogonal projection, no training):**

| Scorer | R@1 | R@3 | R@5 |
|--------|-----|-----|-----|
| Cosine baseline | 14.4% | — | — |
| SHGAT-Flat (16-head random) | 14.7% | — | — |
| SHGAT-Hier (3-level, no train) | 14.2% | — | — |

**Conclusion Option A:** Random projection preserves cosine similarity (JL lemma) but cannot improve it. Hierarchy without trained attention collapses to noise. **Training is the only path to significant improvement.**

**Dataset:**
- 95 queries, 525 tools, 69 MCP servers, 8 categories
- Each query has 1-8 ground truth tools (avg ~4)
- All queries and tools have BGE-M3 1024-dim embeddings
- Hierarchy: Category(8) -> Server(69) -> Tool(525)

---

## 2. Core Challenge: 95 Queries

95 queries is extremely small for contrastive learning. With ~380 (query, positive_tool) pairs, overfitting is the dominant risk. The design must prioritize:

1. **Maximal data utilization** (every query contributes to training)
2. **Strong regularization** (L2, dropout, early stopping, few epochs)
3. **Robust evaluation** (cross-validation over single split)
4. **Hard negative mining** (maximize information per gradient step)

---

## 3. Data Splitting Strategy

### Recommended: Stratified 5-Fold Cross-Validation

```
95 queries -> 5 folds of 19 queries each
Per fold: 76 train / 19 test
Repeat 5x, report mean +/- std across folds
```

**Stratification:** Split by category (8 categories) to ensure each fold has proportional representation. With 95 queries and 8 categories, each fold gets ~2-3 queries per category.

**Why 5-fold over alternatives:**
- **Simple 80/20 split:** Wastes test data for such a small N. A bad split can dominate results. Multiple seeds help but don't solve the systematic issue.
- **Leave-one-out (LOO):** 95 training runs is expensive (~95x). Each run uses 94/95 = 98.9% of data, so models are nearly identical. Variance estimate is biased upward. Not recommended.
- **Train-on-all / eval-on-all:** Useful as an **upper bound** on model capacity, but NOT a valid generalization metric. Include as a supplementary run but never as the primary result.

### Implementation

```typescript
interface FoldSpec {
  foldIndex: number;
  trainQueryIds: string[];
  testQueryIds: string[];
}

function createStratifiedFolds(
  queries: LiveMCPQuery[],
  numFolds: number = 5,
  seed: number = 42,
): FoldSpec[] {
  // Group queries by category
  const byCategory = new Map<string, LiveMCPQuery[]>();
  for (const q of queries) {
    if (!byCategory.has(q.category)) byCategory.set(q.category, []);
    byCategory.get(q.category)!.push(q);
  }

  // Shuffle within each category (seeded)
  for (const [, qs] of byCategory) {
    seededShuffle(qs, seed);
  }

  // Round-robin assign to folds
  const foldAssignments = new Map<string, number>(); // queryId -> foldIndex
  for (const [, qs] of byCategory) {
    for (let i = 0; i < qs.length; i++) {
      foldAssignments.set(qs[i].id, i % numFolds);
    }
  }

  // Build fold specs
  const folds: FoldSpec[] = [];
  for (let f = 0; f < numFolds; f++) {
    const testIds: string[] = [];
    const trainIds: string[] = [];
    for (const [qid, fold] of foldAssignments) {
      (fold === f ? testIds : trainIds).push(qid);
    }
    folds.push({ foldIndex: f, trainQueryIds: trainIds, testQueryIds: testIds });
  }
  return folds;
}
```

---

## 4. Training Example Construction

### 4.1 Positive Pairs

Each (query, ground_truth_tool) pair forms one positive example:
- 95 queries x ~4 ground truth tools = ~380 positive pairs
- Per fold: ~304 training pairs (76 queries x 4 avg)

### 4.2 Hard Negative Mining

**Strategy: Cosine-sorted negatives with curriculum tiers.**

For each positive pair (query, tool_positive), pre-compute similarity of query to ALL 525 tools. Sort by descending cosine similarity. The non-positive tools sorted by cosine form `allNegativesSorted`.

```typescript
function buildTrainingExamples(
  queries: LiveMCPQuery[],
  tools: LiveMCPTool[],
  queryIds: Set<string>,
  numNegatives: number = 16,  // Increased from 8
): TrainingExample[] {
  const examples: TrainingExample[] = [];
  const toolEmbMap = new Map(tools.map(t => [t.id, t.embedding]));

  for (const query of queries) {
    if (!queryIds.has(query.id)) continue;

    // Pre-compute all similarities
    const sims = tools.map(t => ({
      id: t.id,
      sim: cosineSim(query.embedding, t.embedding),
    }));
    sims.sort((a, b) => b.sim - a.sim);

    const gtSet = new Set(query.ground_truth_tools); // tool names
    const positiveIds = sims
      .filter(s => isGroundTruth(s.id, gtSet, tools))
      .map(s => s.id);
    const allNegsSorted = sims
      .filter(s => !isGroundTruth(s.id, gtSet, tools))
      .map(s => s.id);

    for (const posId of positiveIds) {
      examples.push({
        intentEmbedding: query.embedding,
        contextTools: [],
        candidateId: posId,
        outcome: 1,
        negativeCapIds: allNegsSorted.slice(0, numNegatives), // Start with hardest
        allNegativesSorted: allNegsSorted,
      });
    }
  }

  return examples;
}
```

### 4.3 Negative Count

**Recommendation: NUM_NEGATIVES = 16** (up from default 8).

Rationale: With only ~300 training pairs, we need maximum contrastive signal per example. 16 negatives provides:
- Better gradient signal (InfoNCE converges faster with more negatives)
- Harder discrimination task (forces sharper attention)
- Still fits in memory (16 * 1024 floats = trivial)

### 4.4 Curriculum Learning

Use the existing `allNegativesSorted` + accuracy-based tier selection:

| Training accuracy | Negative tier | Source |
|---|---|---|
| < 0.35 | Last 1/3 of sorted negatives | Easy (help model bootstrap) |
| 0.35 - 0.55 | Middle 1/3 | Medium |
| > 0.55 | First 1/3 | Hard (maximize discrimination) |

This is already supported by `TrainingExample.allNegativesSorted`. The training script samples from the appropriate tier based on running accuracy.

---

## 5. Training Configuration

### 5.1 Architecture Variants to Compare

Run training for both Flat and Hier to determine if hierarchy adds value with learned weights:

| Variant | Description | MP | Parameters |
|---|---|---|---|
| **B-Flat** | Train K-head scoring only | No | W_k(16), W_intent ~ 2M |
| **B-Hier** | Train K-head + message passing | Yes | W_k, W_intent, W_up, W_down, a_up, a_down ~ 5M |
| **B-Proj** | Train K-head + projection head | No | W_k, W_intent, proj_W1, proj_W2 ~ 3M |

### 5.2 Hyperparameters

```typescript
const TRAINING_CONFIG = {
  // Optimizer
  learningRate: 0.0005,       // Conservative for small data (NOT 0.05)
  gradientClip: 1.0,          // Prevent gradient explosions

  // InfoNCE
  temperature: 0.07,          // CLIP-style fixed temperature
  numNegatives: 16,           // Hard negatives per example

  // Regularization
  l2Lambda: 0.001,            // 10x default (strong regularization for small data)
  dropout: 0.1,               // Keep default (0.1)

  // Training loop
  numEpochs: 30,              // Max epochs
  earlyStoppingPatience: 5,   // Stop if val loss doesn't improve for 5 epochs
  batchSize: 16,              // Smaller batch = more gradient updates per epoch

  // Schedule
  temperatureAnnealing: false, // Fixed temperature (annealing is risky with small data)
  learningRateWarmup: 3,       // Linear warmup for 3 epochs

  // Hierarchy-specific
  downwardResidual: 0.85,      // From Option A sweep (best residual config)
  preserveDimResiduals: [0.95, 0.7, 0.5],
  mpLearningRateScale: 50,     // Compensate for vanishing gradients in MP
};
```

### 5.3 Justification for Key Choices

**Learning rate 0.0005 (not 0.05):**
The default `SHGATConfig.learningRate = 0.05` is tuned for production training with thousands of examples and online learning. With 300 training pairs and 30 epochs, 0.05 will overfit within 2-3 epochs. Adam with 0.0005 provides smoother convergence.

**L2 lambda 0.001 (10x default):**
Strong L2 keeps weight magnitudes small, preventing memorization of the 300 examples. With ~2-5M parameters and 300 examples, the model capacity far exceeds the data — regularization is critical.

**Batch size 16 (not 32):**
With ~300 examples per fold, batch_size=16 gives ~19 gradient updates per epoch. Batch_size=32 gives only ~9 updates. More updates = better convergence for small datasets.

**Early stopping patience 5:**
Monitor validation loss (on the held-out fold). Stop if no improvement for 5 consecutive epochs. This is the most important overfitting guard.

---

## 6. Overfitting Mitigation

### 6.1 Primary Controls

| Control | Mechanism | Priority |
|---|---|---|
| 5-fold CV | No single test set dominance | Critical |
| Early stopping (patience=5) | Stop before memorization | Critical |
| Strong L2 (0.001) | Weight magnitude penalty | High |
| Small LR (0.0005) | Slow convergence = more epochs before overfit | High |
| Gradient clipping (1.0) | Prevent loss spikes | Medium |

### 6.2 Data Augmentation (Optional, Phase 2)

If overfitting remains severe after primary controls:

**Embedding noise injection:**
```typescript
function augmentExample(ex: TrainingExample, noiseScale: number = 0.01): TrainingExample {
  const noise = ex.intentEmbedding.map(() => gaussianNoise(0, noiseScale));
  const augmented = ex.intentEmbedding.map((v, i) => v + noise[i]);
  // Re-normalize to unit sphere
  const norm = Math.sqrt(augmented.reduce((s, v) => s + v * v, 0));
  return {
    ...ex,
    intentEmbedding: augmented.map(v => v / norm),
  };
}
```

This creates synthetic query variants by perturbing the intent embedding. The label remains the same since the perturbation is small.

**Rationale:** With 1024-dim BGE-M3 embeddings, a noise scale of 0.01 shifts the vector by ~0.32 in L2 norm (sqrt(1024) * 0.01), which is small relative to the unit sphere but creates meaningful training diversity.

**Inter-query negatives (in-batch negatives):**
The existing `batchContrastiveLoss()` function already supports in-batch negatives. With batch_size=16, every other positive in the batch serves as a negative for each example, giving 15 additional negatives for free.

---

## 7. Training Loop Design

### 7.1 Per-Fold Training

```
For each fold f in [0..4]:
  1. Build training examples from fold.trainQueryIds
  2. Build validation examples from fold.testQueryIds
  3. Initialize fresh SHGAT parameters (seed = SEED + f)
  4. For epoch in [0..numEpochs-1]:
     a. Shuffle training examples
     b. For each mini-batch:
        - Sample negatives from curriculum tier (based on running accuracy)
        - Call shgat.trainBatch(batch)
        - Track loss, accuracy, gradient norm
     c. Evaluate on validation set (R@1, R@3, R@5, NDCG@5)
     d. Early stopping check
  5. Score ALL 525 tools for each test query using trained model
  6. Evaluate test metrics
  7. Store fold results
```

### 7.2 Evaluation During Training

After each epoch, compute validation metrics on the held-out fold:

```typescript
function evaluateFold(
  shgat: SHGATTrainerScorer,
  testQueries: LiveMCPQuery[],
  allToolIds: string[],
  groundTruth: Map<string, RelevanceLabel[]>,
): EvalResults {
  const rankings = batchScoreSHGAT(shgat, testQueries, allToolIds, "train-eval");
  return evaluate("val", rankings, groundTruth);
}
```

**Key metric for early stopping:** Validation R@1 (the target metric). Not loss, because loss can decrease while R@1 stalls.

### 7.3 Learning Rate Warmup

Linear warmup from `lr/10` to `lr` over 3 epochs. This prevents large initial gradients from distorting the random orthogonal initialization.

```typescript
function getWarmupLR(epoch: number, baseLR: number, warmupEpochs: number): number {
  if (epoch < warmupEpochs) {
    return baseLR * (0.1 + 0.9 * epoch / warmupEpochs);
  }
  return baseLR;
}
```

---

## 8. Evaluation Protocol

### 8.1 Metrics

| Metric | Description | Use |
|---|---|---|
| **R@1** | Recall at 1 — primary metric | Main comparison |
| R@3 | Recall at 3 | Secondary |
| R@5 | Recall at 5 | Secondary |
| NDCG@5 | Normalized DCG at 5 | Ranking quality |
| MAP@5 | Mean Average Precision at 5 | Ranking quality |

All metrics already implemented in `benchmark/src/metrics.ts`.

### 8.2 Results Table Format

```
Scorer           | R@1 mean (std) | R@3 mean (std) | R@5 mean (std) | NDCG@5 mean (std)
Cosine baseline  | 14.4 (-)       | ...             | ...             | ...
SHGAT-A Flat     | 14.7 (-)       | ...             | ...             | ...
SHGAT-A Hier     | 14.2 (-)       | ...             | ...             | ...
SHGAT-B Flat     | X.X (Y.Y)      | ...             | ...             | ...
SHGAT-B Hier     | X.X (Y.Y)      | ...             | ...             | ...
SHGAT-B Proj     | X.X (Y.Y)      | ...             | ...             | ...
```

### 8.3 Statistical Significance

With 5 folds, report:
- Mean and standard deviation across folds
- Paired t-test vs Cosine baseline (p < 0.05)
- Confidence intervals for R@1 improvement

### 8.4 Overfit Diagnostic

Include train/val gap report per fold:

```
Fold 0: Train R@1=85.0% | Val R@1=18.0% | Gap=67.0% (SEVERE OVERFIT)
Fold 1: Train R@1=25.0% | Val R@1=20.0% | Gap=5.0% (HEALTHY)
```

A gap > 30% indicates the model is memorizing training data and regularization should be increased.

---

## 9. Expected Outcomes & Targets

### 9.1 Realistic Targets

Given 95 queries and 525 tools, the theoretical maximum is limited by:
- Embedding quality (BGE-M3 captures semantic similarity, not tool function)
- Ground truth ambiguity (some tools are functionally equivalent)
- Small training data (300 examples to learn 2-5M parameters)

**Conservative target:** R@1 = 18-22% (25-50% improvement over 14.4% baseline)
**Optimistic target:** R@1 = 25-30% (75-100% improvement)
**Failure threshold:** R@1 < 16% (training did not help)

### 9.2 Hierarchy Value Hypothesis

Option A showed hierarchy degraded performance (14.2% < 14.7% flat). With training:
- **Hypothesis:** Trained attention weights will learn to propagate category-level signal downward, making `B-Hier > B-Flat`
- **Alternative:** The 3-level hierarchy is too coarse for 95 queries and B-Hier will still underperform B-Flat
- **Discriminator:** If B-Hier R@1 > B-Flat R@1 by more than 1 standard deviation, hierarchy has value

---

## 10. Implementation Plan

### 10.1 File Structure

```
lib/shgat-tf/benchmark/
  src/
    train-livemcp.ts          # NEW: Training script (entry point)
    training-utils.ts         # NEW: Fold construction, example building, evaluation helpers
    run-livemcp.ts            # EXISTING: Add --train flag
  docs/
    option-b-training-design.md  # THIS FILE
```

### 10.2 Script Interface

```bash
# Full 5-fold training + evaluation (all 3 variants: Flat, Hier, Proj)
tsx src/train-livemcp.ts

# Single variant
tsx src/train-livemcp.ts --variant flat
tsx src/train-livemcp.ts --variant hier
tsx src/train-livemcp.ts --variant proj

# Override hyperparameters
tsx src/train-livemcp.ts --lr 0.001 --epochs 50 --l2 0.01

# Quick test (1 fold only)
tsx src/train-livemcp.ts --folds 1

# Train on ALL data, eval on ALL (overfit diagnostic)
tsx src/train-livemcp.ts --overfit-check
```

### 10.3 Key Integration Points

The training script will use:
1. `SHGATBuilder.create().nodes(nodes).training({...}).build()` to create trainer
2. `shgat.trainBatch(examples)` for training steps
3. `shgat.score(intentEmb, toolIds)` for evaluation scoring
4. `evaluate()` from `metrics.ts` for computing R@k, NDCG@k
5. `buildGraphStructure()` from `autograd-trainer.ts` for hierarchy
6. `shgat.setTemperature()` for temperature annealing (if enabled)

### 10.4 Execution Order

| Step | Action | Est. Time |
|---|---|---|
| 1 | Build `training-utils.ts` with fold/example helpers | 30 min |
| 2 | Build `train-livemcp.ts` main loop | 45 min |
| 3 | Run B-Flat (5-fold, ~30 epochs, ~300 ex/fold) | ~10 min |
| 4 | Run B-Hier (5-fold, ~30 epochs, with MP) | ~30 min |
| 5 | Run B-Proj (5-fold, ~30 epochs, with proj head) | ~15 min |
| 6 | Analyze results, compare to Option A | 15 min |

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Severe overfitting (train-val gap > 50%) | High | Training useless | Early stopping, strong L2, few epochs |
| No improvement over cosine | Medium | Option B fails | If R@1 < 16% after tuning, conclude no value |
| Message passing slows training too much | Medium | Cannot iterate | Run B-Flat first (no MP), only try B-Hier if B-Flat works |
| Fold variance > signal | Medium | Cannot draw conclusions | Report std, add more seeds if needed |
| TF.js CPU OOM with 525 tools + MP | Low | Training crashes | MP only processes hierarchy (525+69+8=602 nodes), should fit |

---

## 12. Decision Log

| Decision | Choice | Rationale |
|---|---|---|
| Splitting strategy | 5-fold stratified CV | Best trade-off for 95 queries. LOO too expensive, single split too noisy. |
| Learning rate | 0.0005 | 100x lower than production default (0.05). Small data requires slow convergence. |
| L2 regularization | 0.001 | 10x higher than default. 2-5M params with 300 examples demands strong reg. |
| Batch size | 16 | ~19 updates/epoch vs 9 with batch_size=32. More updates for small data. |
| Negatives per example | 16 | More contrastive signal per step. 8 is default but 16 is better for small data. |
| Early stopping metric | Val R@1 | Direct optimization target. Loss can mislead when R@1 is the goal. |
| Temperature | Fixed 0.07 | Annealing adds complexity for little benefit at 300 examples. |
| Variants | Flat + Hier + Proj | Tests 3 independent hypotheses. Flat is fastest, Hier tests structure, Proj tests nonlinear discrimination. |
