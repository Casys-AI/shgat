# Tech Spec: AutogradTrainer Scoring Pipeline Fixes

**Date:** 2026-02-07
**Status:** Action Required
**Context:** Production bench shows 0% Hit@1 (vs ~80% ancien SHGAT). 6 bugs identifiés par analyse différentielle.

## Fichiers principaux

| Fichier | Role |
|---------|------|
| `src/training/autograd-trainer.ts` | Trainer principal, scoring, init params |
| `src/training/sparse-mp.ts` | Message passing forward/backward sparse |
| `src/attention/khead-scorer.ts` | K-head scoring (reference, cosine sim) |
| `src/core/types.ts` | Config (preserveDim, etc.) |
| `tests/shgat_prod_bench.ts` | Benchmark production |

## Reference: ancien SHGAT scoring path

```
scoreNodes(intent):
  1. if preserveDim: Q_raw = intent (1024-dim, PAS de W_intent)
     else: Q_raw = W_intent @ intent
  2. forwardMultiLevelWithCache() → MP enrichit embeddings
  3. L2 normalize embeddings après residuel
  4. Per head h: Q_h = W_q[h] @ Q_raw   (W_q = W_k shared)
  5. Per head h: K_h = W_k[h] @ emb_i
  6. score_i = (1/H) * Σ_h dot(Q_h, K_h) / sqrt(headDim)
```

---

## FIX-1: Scaling sqrt(headDim) manquant [CRITIQUE]

**Localisation:** `autograd-trainer.ts` function `kHeadScoring()` ~ligne 456

**Bug:** `scores = matMul(K, Q.expandDims(1))` — pas de division par sqrt(headDim)

**Impact:** Scores 8x trop grands → softmax trop peaked → overfitting, gradient explosion

**Fix:**
```typescript
// AVANT
const scores = tf.squeeze(tf.matMul(K, Q.expandDims(1)));

// APRES
const raw = tf.squeeze(tf.matMul(K, Q.expandDims(1)));
const scores = raw.div(Math.sqrt(config.headDim));
```

**Tests:** Vérifier que gradient norms descendent ~8x. Training accuracy devrait baisser initialement mais test metrics monter.

---

## FIX-2: preserveDim bypass manquant [CRITIQUE]

**Localisation:** `autograd-trainer.ts` function `forwardScoring()` ~ligne 477-480

**Bug:** Projette TOUJOURS à travers W_intent, même quand `config.preserveDim = true` (le défaut)

**Impact:** Détruit la structure sémantique des embeddings BGE-M3 pré-normalisés

**Fix:**
```typescript
// AVANT
const intentProj = tf.squeeze(
  tf.matMul(intentEmb.expandDims(0), params.W_intent)
) as tf.Tensor1D;

// APRES
let intentProj: tf.Tensor1D;
if (config.preserveDim) {
  // Skip W_intent projection, use raw intent (matches old SHGAT behavior)
  intentProj = intentEmb;
} else {
  intentProj = tf.squeeze(
    tf.matMul(intentEmb.expandDims(0), params.W_intent)
  ) as tf.Tensor1D;
}
```

**Note:** Quand preserveDim=true, W_intent ne reçoit plus de gradient du scoring. Le garder pour d'autres usages ou le retirer de l'init si inutile. Aussi appliquer dans `trainStepSparse()` pour le forward path du training.

---

## FIX-3: W_q jamais utilisé dans le scoring [CRITIQUE]

**Localisation:** `autograd-trainer.ts` function `kHeadScoring()` ~ligne 451-453

**Bug:** `Q = intentProj.slice([h * headDim], [headDim])` — simple slice au lieu d'appliquer W_q

**Reference ancien code:** W_q = W_k (shared). Donc `Q_h = W_k[h] @ intent` et `K_h = W_k[h] @ emb`.

**Option A (recommandée):** Appliquer W_q per head comme l'ancien code:
```typescript
// Per head: Q_h = intentProj @ W_q[h]
const Q = tf.squeeze(tf.matMul(intentProj.expandDims(0), params.W_q[h]));
```

**Option B:** Partager W_k pour Q et K (comme l'ancien SHGAT):
```typescript
// Per head: Q_h = intentProj @ W_k[h], K_h = emb @ W_k[h]
const Q = tf.squeeze(tf.matMul(intentProj.expandDims(0), params.W_k[h]));
```

**Choix:** Option B est plus fidèle à l'ancien code (W_q = W_k shared). Si on choisit B, supprimer W_q de l'init et du L2.

**Impact sur W_intent:** Avec preserveDim=true + Option B, le scoring fait `Q_h = W_k[h] @ intent` et `K_h = W_k[h] @ emb`. W_intent n'est pas utilisé dans le scoring mais peut servir au MP.

---

## FIX-4: L2 normalization manquante après résiduel MP [HIGH]

**Localisation:** `sparse-mp.ts` function `sparseMPForward()` ~ligne 392-410

**Bug:** Après le mix résiduel `H = (1-α)*H_enriched + α*H_init`, pas de normalisation L2

**Impact:** Embeddings perdent leur norme unitaire → magnitude bias dans le scoring

**Fix:** Ajouter normalisation L2 après chaque résiduel:
```typescript
// Après le mix résiduel pour H
for (let i = 0; i < H.length; i++) {
  let norm = 0;
  for (let j = 0; j < H[i].length; j++) norm += H[i][j] * H[i][j];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let j = 0; j < H[i].length; j++) H[i][j] /= norm;
  }
}
// Idem pour E à chaque level
```

**Note:** La normalisation empêche le backward de propager le gradient du norm. Soit on fait la normalisation uniquement dans le forward d'inférence (score), soit on l'ajoute au backward aussi (plus complexe).

---

## FIX-5: Cosine similarity vs dot product [HIGH]

**Localisation:** `autograd-trainer.ts` function `kHeadScoring()` ~ligne 456

**Bug:** Utilise dot product brut. `khead-scorer.ts` utilise cosine similarity.

**Decision:** Avec FIX-4 (L2 norm), les embeddings sont unitaires → `cosine(a,b) = dot(a,b)`. Avec FIX-1 (sqrt scaling), on a `dot(Q,K)/sqrt(d)` qui est le scaled dot-product attention standard.

**Fix:** Si FIX-4 est appliqué, le dot product + sqrt scaling est OK (équivalent cosine pour vecteurs unitaires). Pas de changement supplémentaire nécessaire.

---

## FIX-6: Résiduel additif vs pondéré [MEDIUM]

**Localisation:** `sparse-mp.ts` function `sparseMPForward()` ~ligne 392-410

**Bug:** Nouveau code fait `H = 0.7*H_mp + 0.3*H_init` (interpolation). Ancien fait `H = H_mp + H_orig` (additif).

**Fix:** Changer pour résiduel additif:
```typescript
// AVANT
H[i][j] = (1 - alpha) * H[i][j] + alpha * H_init[i][j];

// APRES (additif comme l'ancien)
H[i][j] = H[i][j] + H_init[i][j];
```

**Note:** Si FIX-4 (L2 norm) est appliqué après, le résiduel additif est safe car la norme est rétablie. Le `residualWeights` learnable param peut être retiré ou repurposé.

---

## FIX-7 (déjà appliqué): MP param init level 0

**Localisation:** `autograd-trainer.ts` function `initTFParams()` ~ligne 134

**Bug:** `for level = 1` → avec maxLevel=0, aucun param MP créé

**Fix appliqué:** `for level = 0; level <= Math.max(1, maxLevel)`

---

## FIX-8 (déjà appliqué): Transposed dW in applySparseMPGradients

**Localisation:** `sparse-mp.ts` function `applySparseMPGradients()` ~ligne 1340

**Bug:** `dW_h[i]?.[j]` au lieu de `dW_h[j]?.[i]` (dW shape [headDim][embDim], W shape [embDim][headDim])

**Fix appliqué:** `W_data[i][j] -= scale * (dW_h[j]?.[i] ?? 0)`

---

## FIX-9 (déjà appliqué): Tensor leak gradient norm

**Localisation:** `autograd-trainer.ts` trainStep/trainStepSparse

**Bug:** `tf.sum(tf.square(g)).arraySync()` leak 2 tensors par gradient

**Fix appliqué:** Explicit dispose des intermédiaires

---

## FIX-10 (déjà appliqué): K-head chain rule dans dH_accum

**Localisation:** `autograd-trainer.ts` trainStepSparse

**Bug:** Scalar broadcast au lieu du gradient correct à travers le K-head scoring

**Fix appliqué:** `dEmb_i = dLogit_i * gradBase[j]` avec `gradBase = (1/H) * Σ_h (W_k[h] @ Q_h)`

---

## Ordre d'implémentation

1. FIX-1 (sqrt scaling) — 1 ligne, impact immédiat sur gradient norms
2. FIX-2 (preserveDim) — ~10 lignes, dans forwardScoring + trainStepSparse
3. FIX-3 (W_q → utiliser W_k shared) — ~5 lignes dans kHeadScoring, supprimer W_q de init/L2
4. FIX-4 (L2 norm après résiduel) — ~15 lignes dans sparseMPForward
5. FIX-6 (résiduel additif) — ~3 lignes dans sparseMPForward

FIX-5 est résolu par FIX-1 + FIX-4.

## Validation

Après tous les fix, relancer `shgat_prod_bench.ts` en tmux. Attendu:
- Gradient norms: ~1-10 (vs 3000+ actuel)
- Training accuracy: ~90%+
- Test MRR: >0.5
- Test Hit@1: >60%
- Held-out Hit@1: >50%
