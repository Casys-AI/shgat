# Tech Spec: Corrections Backward Pass shgat-tf

**Date:** 2026-02-03
**Status:** Action Items
**MRR Actuel:** 0.778
**MRR Cible:** 0.933

---

## Contexte

Après review adversariale du backward pass, **11 bugs** ont été identifiés dont **6 critiques** qui expliquent l'écart de performance avec la cible production.

Le backward pass actuel ne propage correctement les gradients que pour une partie des paramètres. Les matrices de projection W_target et les embeddings E ne reçoivent pas de gradients corrects.

---

## Bugs Critiques

### BUG #8: Scalar dLogit Ajouté Uniformément (CRITIQUE)

**Fichier:** `autograd-trainer.ts` lignes 1111-1137
**Impact:** Signal gradient complètement corrompu

**Code Actuel (FAUX):**
```typescript
// trainStepSparse() - Propagation vers embeddings
for (let idx = 0; idx < nodeIds.length; idx++) {
  const nodeId = nodeIds[idx];
  const dLogit = dLogits[idx];

  // FAUX: ajoute la même valeur scalaire à TOUTES les dimensions!
  for (let j = 0; j < dH_accum[toolIdx].length; j++) {
    dH_accum[toolIdx][j] += dLogit;
  }
}
```

**Code Correct:**
```typescript
// Le gradient doit passer par la chain rule du K-head scoring:
// score = mean_h(Q_h @ K_h^T) / sqrt(headDim)
// dEmbedding = dLogit * d(score)/d(embedding)

for (let idx = 0; idx < nodeIds.length; idx++) {
  const nodeId = nodeIds[idx];
  const dLogit = dLogits[idx];

  // Pour chaque head, calculer le gradient correct
  for (let h = 0; h < numHeads; h++) {
    // dK = dLogit * Q / (numHeads * sqrt(headDim))
    // dEmbedding = dK @ W_k^T
    const scale = numHeads * Math.sqrt(headDim);
    for (let d = 0; d < headDim; d++) {
      const dK_d = dLogit * Q_h[h][d] / scale;
      for (let e = 0; e < embDim; e++) {
        dEmbedding[e] += dK_d * W_k[h][d][e];
      }
    }
  }
}
```

---

### BUG #1: Missing dW_target Gradient (CRITIQUE)

**Fichier:** `sparse-mp.ts` - `backwardVertexToEdge()`, `backwardEdgeToEdge()`, `backwardEdgeToVertex()`
**Impact:** 50% des poids d'attention non entraînables

**Problème:**
Le forward calcule `concat = [source_proj, target_proj]` mais le backward ne calcule `dW` que pour source_proj.

**Code Actuel:**
```typescript
// backwardVertexToEdge() ligne 999-1009
// SEULEMENT dW pour source (tools)
for (let d = 0; d < headDim; d++) {
  const dH_proj_d = dConcat[d];
  for (let e = 0; e < cache.H_init[t].length; e++) {
    dW[h][d][e] += dH_proj_d * cache.H_init[t][e];
  }
}
// dW pour target (capabilities) MANQUANT!
```

**Fix Requis:**
```typescript
// Ajouter après le bloc existant:
// dW_target contribution: dE_proj @ E^T
for (let d = 0; d < headDim; d++) {
  const dE_proj_d = dConcat[headDim + d];  // 2ème moitié de concat
  const E_c = cache.E_init?.[c] ?? E_level0?.[c];
  if (E_c) {
    for (let e = 0; e < E_c.length; e++) {
      dW_target[h][d][e] += dE_proj_d * E_c[e];
    }
  }
}
```

**Note:** Nécessite d'ajouter `dW_target` comme paramètre aux fonctions backward.

---

### BUG #2: Missing E_proj Gradient Path in V→E (CRITIQUE)

**Fichier:** `sparse-mp.ts` ligne 997-1026
**Impact:** Target embeddings ne reçoivent pas de gradients

**Problème:**
La 2ème moitié de `dConcat` (indices `headDim` à `2*headDim-1`) n'est jamais utilisée.

**Fix Requis:**
```typescript
// Après le calcul de dConcat, ajouter:
// dE from attention score path
for (let d = 0; d < headDim; d++) {
  const dE_proj_d = dConcat[headDim + d];
  // dE += dE_proj @ W_target^T
  for (let e = 0; e < W_target_h[d].length; e++) {
    dE[c][e] += dE_proj_d * W_target_h[d][e];
  }
}
```

---

### BUG #9: dConcat Second Half Unused in E→E (CRITIQUE)

**Fichier:** `sparse-mp.ts` - `backwardEdgeToEdge()` lignes 1145-1176
**Impact:** Gradients projection target perdus

**Même problème que Bug #2** mais dans la phase E→E.

**Fix:** Identique - utiliser `dConcat[headDim:]` pour calculer `dE_target` et `dW_target`.

---

### BUG #4: Incomplete dE_source Accumulation (CRITIQUE)

**Fichier:** `sparse-mp.ts` - `backwardEdgeToEdge()` lignes 1167-1175
**Impact:** Propagation hiérarchique cassée

**Problème:**
Seuls les gradients du chemin d'agrégation sont ajoutés à `dE_source`, pas ceux du chemin attention score.

**Code Actuel:**
```typescript
// dE_source from aggregation path SEULEMENT
if (dE_source) {
  for (let e = 0; e < W_h[d].length; e++) {
    dE_source[src][e] += dSrc_proj_agg_d * W_h[d][e];
  }
}
```

**Fix Requis:**
```typescript
// AJOUTER: dE_source from attention score path
for (let d = 0; d < headDim; d++) {
  const dSrc_proj_attn_d = dConcat[d];
  for (let e = 0; e < W_h[d].length; e++) {
    dE_source[src][e] += dSrc_proj_attn_d * W_h[d][e];
  }
}
```

---

### BUG #6: dE Enriched Gradients Discarded (CRITIQUE)

**Fichier:** `sparse-mp.ts` - `sparseMPBackward()` lignes 896-903
**Impact:** Capability embeddings n'apprennent pas du chemin enrichi

**Problème:**
`dE_accum` contient les gradients du chemin enrichi mais ils ne sont pas ajoutés à `grads.dE`.

**Code Actuel:**
```typescript
// Commentaire incorrect dans le code:
// "So grads.dE should remain as just the residual contribution"
// FAUX - les gradients enrichis doivent aussi être ajoutés!
```

**Fix Requis:**
```typescript
// À la fin de sparseMPBackward(), AJOUTER:
for (const [level, dE_level] of dE_accum) {
  if (!grads.dE.has(level)) {
    grads.dE.set(level, dE_level.map(row => [...row]));
  } else {
    const existing = grads.dE.get(level)!;
    for (let i = 0; i < dE_level.length; i++) {
      for (let j = 0; j < dE_level[i].length; j++) {
        existing[i][j] += dE_level[i][j];
      }
    }
  }
}
```

---

## Bugs Majeurs (Non-Critiques)

### BUG #3: Potential W Matrix Transpose Issue

**Fichier:** `sparse-mp.ts` - toutes les fonctions backward
**Sévérité:** HIGH

Le forward fait `H_proj = H @ W` où W est `[embDim, headDim]`.
Le backward devrait faire `dW = H^T @ dH_proj` donnant `[embDim, headDim]`.
Mais le code accumule `dW[h][d][e]` qui est `[headDim, embDim]` - potentiellement transposé.

**Action:** Vérifier l'orientation de W dans `autograd-trainer.ts` et aligner le backward.

---

### BUG #7: Missing W_intent Gradient in Sparse Path

**Fichier:** `autograd-trainer.ts` - `trainStepSparse()`
**Sévérité:** HIGH

W_intent est utilisé pour projeter l'intention mais son gradient n'est pas explicitement calculé dans le chemin sparse.

**Note:** Peut être géré par TF.js autograd si W_intent est dans le graphe de calcul.

---

### BUG #11: Level 0 Parameter Fallback

**Fichier:** `autograd-trainer.ts` ligne 127, `sparse-mp.ts` ligne 291
**Sévérité:** MEDIUM

Level 0 n'a pas ses propres paramètres W_up, utilise fallback vers level 1.
Peut causer des problèmes d'accumulation de gradient.

**Action:** Soit initialiser level 0, soit documenter clairement le partage de paramètres.

---

## Action Items

### Priorité 1 (Bloquants)

- [ ] **FIX-8:** Corriger gradient dLogit dans `trainStepSparse()` pour utiliser chain rule via K-head
- [ ] **FIX-1/2/9:** Ajouter gradient pour 2ème moitié de concat dans les 3 fonctions backward
- [ ] **FIX-4:** Ajouter gradient attention score path à `dE_source` dans `backwardEdgeToEdge()`
- [ ] **FIX-6:** Ajouter `dE_accum` à `grads.dE` à la fin de `sparseMPBackward()`

### Priorité 2 (Importants)

- [ ] **FIX-3:** Vérifier orientation matrice W et corriger si transposé
- [ ] **FIX-7:** Vérifier que W_intent reçoit des gradients via TF.js autograd
- [ ] **FIX-11:** Documenter ou corriger le fallback level 0

### Priorité 3 (Tests)

- [ ] Ajouter test de gradient checking numérique
- [ ] Ajouter test unitaire pour chaque fonction backward
- [ ] Benchmark avant/après sur production-traces

---

## Estimation d'Impact

| Fix | MRR Estimé Après |
|-----|------------------|
| Actuel | 0.778 |
| + FIX-8 | 0.82-0.85 |
| + FIX-1/2/9 | 0.85-0.88 |
| + FIX-4/6 | 0.88-0.92 |
| + FIX-3 | 0.90-0.93 |

**Cible:** MRR ≥ 0.933

---

## Fichiers à Modifier

| Fichier | Lignes | Modifications |
|---------|--------|---------------|
| `autograd-trainer.ts` | 1111-1137 | FIX-8: Chain rule K-head gradient |
| `sparse-mp.ts` | 997-1026 | FIX-1/2: dW_target + dE dans V→E |
| `sparse-mp.ts` | 1145-1176 | FIX-9/4: dConcat 2ème moitié + dE_source dans E→E |
| `sparse-mp.ts` | 1230-1290 | FIX-9: dConcat 2ème moitié dans E→V |
| `sparse-mp.ts` | 896-903 | FIX-6: Ajouter dE_accum à grads.dE |

---

## Références

- Review adversariale: 2026-02-03
- Implémentation référence: `lib/shgat/src/message-passing/`
- Plan original: `~/.claude/plans/golden-singing-cook.md`
