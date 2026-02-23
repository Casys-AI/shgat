# Tech Spec: Learned Projection Head for SHGAT-TF

**Date**: 2026-02-06
**Status**: In Progress
**Module**: `lib/shgat-tf/src/core/projection-head.ts`

## Motivation

K-head attention scoring (cosine Q/K) captures broad relevance but lacks fine-grained
discrimination between semantically similar but functionally distinct tools. For example,
`code:filter` vs `code:map` have very close BGE-M3 embeddings (cosine ~0.95) but serve
different purposes.

A **Learned Projection Head** (SimCLR/CLIP pattern) maps enriched 1024D embeddings into
a compact 256D contrastive space where similar tools are separated.

## Architecture

```
enrichedEmb [N, 1024]
  -> Linear(1024, 256) + ReLU       (trainable, "bottleneck")
  -> Linear(256, 256) + L2 normalize (trainable, "contrastive space")
= projected [N, 256]

Scoring:
  proj_score = dot(proj_intent, proj_nodes) / temperature
  final = (1-alpha) * khead_score + alpha * proj_score
```

**Parameters**: ~328K additional (+15% over K-head)
- W1: 1024 x 256 = 262,144
- b1: 256
- W2: 256 x 256 = 65,536
- b2: 256

## Config

In `SHGATConfig`:
```typescript
useProjectionHead?: boolean;       // Default: false
projectionHiddenDim?: number;      // Default: 256
projectionOutputDim?: number;      // Default: 256
projectionBlendAlpha?: number;     // Default: 0.5, range [0, 1]
projectionTemperature?: number;    // Default: 0.07
```

## Implementation Checklist

### Core Projection Head
- [x] `core/projection-head.ts` - Forward pass, scoring, serialization helpers
- [x] `core/types.ts` - Config fields added to `SHGATConfig`

### Training Integration
- [x] `training/autograd-trainer.ts` - TFParams.projectionHead, initTFParams, forwardScoring blend
- [x] `training/autograd-trainer.ts` - L2 regularization (10x stronger for projection head)
- [ ] `training/autograd-trainer.ts` - Alpha annealing (0 -> target over warmup epochs)
- [ ] `training/autograd-trainer.ts` - Export trained projection head to SHGATParams format

### Inference Integration
- [x] `attention/khead-scorer.ts` - `scoreNodesTensorDirect` accepts optional projectionParams, blends scores
- [x] `core/shgat.ts` - Passes `tensorParams.projectionHead` to scoring

### Persistence
- [x] `initialization/parameters.ts` - `SHGATParams.projectionHead?` (array format)
- [x] `initialization/parameters.ts` - `TensorScoringParams.projectionHead?` (tensor format)
- [x] `initialization/parameters.ts` - `createTensorScoringParamsSync` converts projection head
- [x] `initialization/parameters.ts` - `disposeTensorScoringParams` disposes projection head
- [x] `initialization/parameters.ts` - `updateTensorScoringParams` updates projection head
- [x] `initialization/parameters.ts` - `exportParams` / `importParams` handle projection head
- [x] `core/serialization.ts` - Passes through from SHGATParams (already handled by base)

### Testing
- [ ] Unit test: projection forward pass shape and L2 normalization
- [ ] Unit test: projection scoring with known values
- [ ] Unit test: blend weight (alpha=0 -> pure K-head, alpha=1 -> pure projection)
- [ ] Integration test: train with useProjectionHead=true, verify loss decreases
- [ ] Benchmark: compare K-head only vs K-head+projection on production traces

### Training Workflow (autograd-trainer -> SHGATParams)
- [ ] After training, extract projection head from TFParams to ProjectionHeadArrayParams
- [ ] Store in SHGATParams.projectionHead for DB persistence
- [ ] Verify roundtrip: train -> export -> import -> inference produces same scores

## Overfitting Mitigation

With 328K params / 357 examples = 920:1 ratio, overfitting is a real risk.

Mitigations:
1. **10x stronger L2 regularization** on projection head (vs K-head params)
2. **Bottleneck architecture** (1024 -> 256 -> 256, not 1024 -> 1024 -> 256)
3. **L2 normalization** on output prevents magnitude-based shortcuts
4. **Alpha warmup** (0 -> 0.5 over first N epochs) prevents early domination
5. **Dropout** (inherited from config.dropout)

## Files Modified

| File | Changes |
|------|---------|
| `core/projection-head.ts` | NEW - Full implementation |
| `core/types.ts` | 5 new config fields |
| `training/autograd-trainer.ts` | TFParams, init, forward scoring blend, L2 reg |
| `attention/khead-scorer.ts` | projectionParams in scoreNodesTensorDirect |
| `core/shgat.ts` | Pass projection head to scoring |
| `initialization/parameters.ts` | SHGATParams, TensorScoringParams, create/dispose/update/export/import |
| `core/serialization.ts` | Pass-through (base handles it) |

## Next Steps

1. **Alpha annealing**: Start alpha=0 for N warmup epochs, ramp to target. Prevents
   untrained projection head from corrupting early K-head learning.
2. **Export from autograd-trainer**: After training, convert TF Variables -> arrays
   and store in SHGATParams for DB persistence.
3. **Benchmark**: Run the full GRU+SHGAT benchmark with `useProjectionHead: true`
   and compare next-tool accuracy and E2E path match.
4. **Hard negative mining**: The projection head is ideal for NT-Xent loss with
   hard negatives (tools with high cosine but different function).
