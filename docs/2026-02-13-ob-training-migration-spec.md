# Tech Spec: Migration Training OB (OpenBLAS Manual Backward)

**Date:** 2026-02-13
**Objectif:** Remplacer le training TF.js autograd (11-15GB RAM) par le backward manuel OB + BLAS FFI (1-3GB RAM) tout en gardant TF.js pour l'inférence.

---

## Contexte

### Problème actuel
- `autograd-trainer.ts` utilise `tf.variableGrads()` qui retient TOUS les tensors intermédiaires (~5-8GB tape)
- Les matrices d'attention dense dans le tape ajoutent ~2-4GB
- RAM totale : 11-15GB pour un graph de 8984 nodes
- Malgré les fixes C3/C4, le tape autograd reste le bottleneck structural

### Solution : backward manuel + BLAS FFI
`lib/shgat-ob/` contient le backward complet écrit à la main :
- Forward pass avec cache minimal (projections + attention par couche)
- Backward pass couche par couche avec libération immédiate du cache
- OpenBLAS FFI (`cblas_sgemm`, `cblas_sgemv`, `cblas_sger`) pour accélération ~10x matmul
- RAM estimée : 1-3GB (cache ~200MB + gradients ~50MB + BLAS buffers ~50MB)

### Architecture training/inférence séparée
```
Training (Deno + OpenBLAS FFI)          Inférence (au choix)
  shgat-ob manual backward        →      TF.js WASM/WebGPU (browser)
  1-3GB RAM, BLAS accéléré               TF.js CPU (serveur)
  InfoNCE + Adam                          number[][] forward pur JS (universel)
  Exporte: params JSON/binary             Charge: params JSON/binary
```

---

## Fichiers copiés depuis shgat-ob

| Fichier | Lignes | Rôle |
|---------|--------|------|
| `src/utils/blas-ffi.ts` | 511 | FFI OpenBLAS : sgemm, sgemv, sger + fallbacks JS |
| `src/utils/math.ts` | 388→410 | Math utils avec dispatch BLAS + zerosLike2D/3D (déplacés depuis parameters.ts) |
| `src/training/multi-level-trainer.ts` | 837 | Forward cache multi-niveaux, backward V→E→E→V, gradient accumulators |
| `src/training/multi-level-trainer-khead.ts` | 670 | K-head scoring forward/backward, W_q/W_k/W_intent gradients |
| `src/training/batched-khead.ts` | 402 | Batched K-head avec BLAS matmul |

Les phases MP (`vertex-to-edge-phase.ts`, `edge-to-edge-phase.ts`, `edge-to-vertex-phase.ts`) sont **identiques** dans les deux libs — pas de copie nécessaire.

---

## Avancement

### Étape 1 : InfoNCE loss — FAIT ✅
- `src/training/infonce-loss.ts` : loss + gradient analytique (logsumexp-stable)
- `src/training/__tests__/infonce-loss.test.ts` : **11 tests** passent
- Exports : `infoNCELoss()`, `infoNCEGradient()`, `infoNCELossAndGradient()`

### Étape 2 : Adam optimizer — FAIT ✅
- `src/training/adam-optimizer.ts` : Adam sur `number[][]`, bias correction, gradient clipping
- `src/training/__tests__/adam-optimizer.test.ts` : **7 tests** passent
- Export : `AdamOptimizer` class (register/step/reset/resetAll)

### Étape 3 : Backward MP wiring — FAIT ✅
- `multi-level-trainer-khead.ts` : TODO ligne 607 **supprimé**, `dNodeEmbedding` câblé → `backwardMultiLevel()`
- `multi-level-trainer.ts` : `backwardMultiLevel()` signature nettoyée — `dNodeEmbedding` **obligatoire** (pas de fallback cosine silencieux)
- `src/training/__tests__/ob-trainer-integration.test.ts` : **5 tests** passent
  - InfoNCE + K-head gradient flow (W_q, W_k)
  - Logit gradient vs finite differences
  - backwardUpwardPhase → W_child/W_parent gradients non-zéro
  - Pipeline complète InfoNCE → K-head → MP backward
  - dNodeEmbedding non-zéro pour tous les candidats contrastifs

#### Nettoyage naming VocabNode unifié
- `capEmbedding` → `nodeEmbedding` dans multi-level-trainer-khead.ts, batched-khead.ts
- `dCapEmbedding` → `dNodeEmbedding` partout
- `targetCapId` → `targetNodeId` dans multi-level-trainer.ts, multi-level-trainer-khead.ts
- `capId` → `nodeId` dans batched-khead.ts
- `getCapabilityIndex` → `getNodeIndex`
- `math.ts` : supprimé import transitif de `parameters.ts` (cassait dépendance TF.js)

### Étape 4 : Training loop — FAIT ✅
- **Fichier:** `tools/train-ob.ts` (840 lignes, Deno)
- **Data loader:** Option B retenue — `tools/export-dataset.ts` (Node.js) exporte le dataset complet en `msgpack.gz`, chargé en Deno via `@msgpack/msgpack` + `pako`
- **Orchestrateur:** `MultiLevelOrchestrator(trainingMode=true)` — MP forward 1x/epoch, cache `ExtendedMultiLevelForwardCache` pour backward
- **Training loops:**
  - InfoNCE (prod) : negative sampling, K-head forward/backward, W_intent backward, MP backward pour capability candidates
  - KL divergence (n8n soft targets) : sparse soft targets, K-head + W_intent only (pas de MP backward pour les tools)
- **Adam:** LR scheduling cosine avec warmup, MP LR scale separee (defaut 0.1x)
- **Eval:** Hit@1/3/5, MRR, chunkee pour eviter OOM sur grands vocabulaires
- **Export:** params JSON + training report JSON dans `gru/data/`
- **Type-check:** `deno check tools/train-ob.ts` passe sans erreur

### Étape 5 : Forward parity test — A FAIRE
- Verifier OB forward ≈ TF.js forward (a epsilon pres)

---

## Total tests OB : 23/23 ✅

| Suite | Tests | Statut |
|-------|-------|--------|
| infonce-loss.test.ts | 11 | ✅ |
| adam-optimizer.test.ts | 7 | ✅ |
| ob-trainer-integration.test.ts | 5 | ✅ |

---

## Orchestrateur : modifications pour le training

### Deux methodes de backward — choix d'architecture

Il existe **deux** systemes de backward dans le codebase :

| Systeme | Methode | Cache requis | Gere dH (tools) ? |
|---------|---------|-------------|-------------------|
| **multi-level-trainer.ts** | `backwardMultiLevel()` | `ExtendedMultiLevelForwardCache` (LevelIntermediates) | NON — ne propage que dE |
| **multi-level-orchestrator.ts** | `orchestrator.backwardMultiLevel()` | `MultiLevelBackwardCache` (per-phase VE/EE/EV caches) | OUI — propage dH via E→V backward |

**Decision :** `train-ob.ts` utilise l'orchestrateur (2eme option) car les candidats sont majoritairement des **tools** dont les embeddings enrichies viennent du downward pass E→V. Le gradient dH DOIT remonter par E→V backward pour entrainer W_child, W_parent, a_upward, a_downward. Sans cela, le MP ne recevrait aucun gradient sur les batches InfoNCE/KL (100% tools).

### `MultiLevelOrchestrator` (`src/message-passing/multi-level-orchestrator.ts`)

- **`forwardMultiLevelWithCache()`** — retourne `MultiLevelBackwardCache` avec per-phase caches (veCaches, evCaches, eeUpwardCaches, eeDownwardCaches)
- **`backwardMultiLevel(dE_final, dH_final, cache, levelParams)`** — propage dH et dE a travers toutes les phases en sens inverse :
  1. dH → backward E→V (downward) → dW_parent, dW_child, da_downward + dE[0]
  2. dE → backward E→E downward (par niveau) → dW_parent, dW_child, da_downward + dE[k+1]
  3. dE → backward E→E upward (par niveau descendant) → dW_child, dW_parent, da_upward + dE[k-1]
  4. dE[0] → backward V→E → dW_child, dW_parent, da_upward + dH

### Flow dans `train-ob.ts`

```
Epoch start
  │
  ├─ MP Forward (1x/epoch)
  │   orchestrator.forwardMultiLevelWithCache(H_init, E_levels_init, ...)
  │   → { result: MultiLevelEmbeddings, cache: MultiLevelBackwardCache }
  │   → Build enrichedEmbs map (tools + capabilities)
  │
  ├─ InfoNCE batches (prod)
  │   Pour chaque batch :
  │     Accumulate batchDH[numTools][embDim] + batchDE[level][numCaps][embDim]
  │     Pour chaque exemple :
  │       W_intent @ intentEmbedding → intentProjected
  │       Sample negatives, compute K-head logits
  │       InfoNCE loss + gradient → dLogits
  │       backpropMultiHeadKHeadLogit → dW_q, dW_k, dIntentProjected, dNodeEmbedding
  │       Accumulate dNodeEmbedding → batchDH (tool) ou batchDE (capability)
  │       backpropWIntent → dW_intent
  │     orchestrator.backwardMultiLevel(batchDE, batchDH, cache, levelParams)
  │       → dW_child, dW_parent, da_upward, da_downward pour TOUS niveaux
  │     Adam step (K-head + W_intent + MP avec LR reduit)
  │
  ├─ KL batches (n8n soft targets)
  │   Meme flow avec MP backward complet :
  │     - Pas de negative sampling (softTargetSparse = distribution cible)
  │     - KL gradient = (q[j] - p[j]) * klWeight / tau
  │     - dNodeEmbedding accumule dans klBatchDH (tools uniquement)
  │     - orchestrator.backwardMultiLevel(emptyDE, klBatchDH, ...)
  │     - Adam step (K-head + W_intent + MP avec LR reduit)
  │
  └─ Eval (Hit@1/3/5, MRR) tous les N epochs
```

### Flow de gradient complet

```
InfoNCE / KL loss
  │
  ├─ dLogits[j] = (softmax(s/τ)[j] - 1_{j=pos}) / τ    [InfoNCE]
  │              = (q[j] - p[j]) * klWeight / τ           [KL]
  │
  ├─ backpropMultiHeadKHeadLogit(dLogit, cache, intentProj, nodeEmb, ...)
  │   │
  │   ├─ Per head h :
  │   │   Q_h = W_q[h] @ intentProj         (cache)
  │   │   K_h = W_k[h] @ nodeEmb            (cache)
  │   │   logit_h = σ(Q_h · K_h / √dim)
  │   │   dLogit_h = dLogit / numHeads
  │   │   dPre = dLogit_h * σ'(pre) / √dim
  │   │   dQ_h = dPre * K_h  →  dW_q[h] += outerProduct(dQ_h, intentProj)
  │   │   dK_h = dPre * Q_h  →  dW_k[h] += outerProduct(dK_h, nodeEmb)
  │   │   dIntentProj += W_q[h]ᵀ @ dQ_h
  │   │   dNodeEmb += W_k[h]ᵀ @ dK_h
  │   │
  │   └─ Returns: { dIntentProjected, dNodeEmbedding }
  │
  ├─ Accumulate dNodeEmbedding → batchDH[toolIdx] ou batchDE[level][capIdx]
  │
  ├─ backpropWIntent(totalDIntentProj, intentEmb, grads, config)
  │   └─ dW_intent += outerProduct(totalDIntentProj, intentEmb)
  │
  └─ orchestrator.backwardMultiLevel(batchDE, batchDH, cache, levelParams)
      │
      ├─ dH → backward E→V (downward) → dW_parent, dW_child, da_downward
      ├─ dE → backward E→E downward (per level) → dW_parent, dW_child, da_downward
      ├─ dE → backward E→E upward (per level) → dW_child, dW_parent, da_upward
      └─ dE[0] → backward V→E → dW_child, dW_parent, da_upward
```

---

## CLI

```bash
cd lib/shgat-tf

# Prerequis : exporter le dataset (Node.js, une seule fois)
DATABASE_URL=... npx tsx tools/export-dataset.ts --seed 42 --oversample 3

# Training OB (Deno)
deno run --unstable-ffi --allow-ffi --allow-read --allow-write --allow-env \
  tools/train-ob.ts --epochs 15 --lr 0.01 --kl --seed 42

# Options avancees
deno run --unstable-ffi --allow-ffi --allow-read --allow-write --allow-env \
  tools/train-ob.ts \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.005 \
    --lr-warmup 5 \
    --temperature 0.08 \
    --num-negatives 64 \
    --kl-weight 0.3 \
    --mp-lr-scale 0.05 \
    --eval-every 3 \
    --seed 42

# Aide
deno run tools/train-ob.ts --help
```

---

## Decisions architecturales

| # | Decision | Raison | Alternative rejetee |
|---|----------|--------|---------------------|
| D1 | **W_q/W_k SEPARES** pour le training (pas partages comme dans l'init inference) | Plus de capacite d'expressivite, les gradients K-head sont le signal principal | W_q = W_k partage (inference pattern) — trop contraint |
| D2 | **preserveDim=true** → embeddingDim = hiddenDim = 1024 | Evite le mismatch de dimension entre K-head backward (dNodeEmbedding ∈ R^1024) et MP backward (attend R^embDim) | preserveDim=false avec projection — ajoute complexite et gradients a propager |
| D3 | **MP backward SEULEMENT pour capability candidates** (pas tools) | Les tools sont H (vertices), pas E (edges). `backwardMultiLevel()` opere sur la hierarchie E^0→E^L, pas sur H directement. Les tools recoivent l'enrichissement via la boucle V→E mais ne sont pas dans l'arbre MP | Propager dNodeEmb des tools vers les capabilities parentes via la matrice toolToCap — ajouterait un path gradient indirect mais le signal est dilue |
| D4 | **KL path : K-head + W_intent SEULEMENT** (pas de MP backward) | Les soft targets n8n referencent des tools (feuilles), pas des capabilities. Le gradient KL ne touche que les logits K-head, pas l'enrichissement MP des embeddings. Entraîner le MP sur le signal KL serait du bruit | Propager le gradient KL a travers le MP — signal trop indirect et bruite |
| D5 | **MP LR scale = 0.1x** par defaut | Les gradients MP sont bruites (subgraph partiel, attention Jacobian). Un LR reduit stabilise le training. Le K-head/W_intent sont le signal principal, le MP est un bonus | LR identique pour tous — risque d'instabilite MP |
| D6 | **MP forward 1x par epoch** (pas par batch) | Le forward MP est couteux (O(nodes × heads × dim²)). Recalculer a chaque batch gaspille du compute. Les embeddings enrichis changent lentement (LR × scale = faible). Compromis acceptable entre fraîcheur des embeddings et cout compute | MP forward par batch — 30x plus lent, marginal improvement |
| D7 | **Adam unique** avec LR switching (au lieu de 2 optimizers) | Simplifie le code, les moments Adam sont par-parametre donc independants. Le switching `adam.lr = epochLR * MP_LR_SCALE` puis restauration est propre | 2 AdamOptimizer (khead + mp) — complexite inutile, double registre de moments |
| D8 | **Export dataset en msgpack.gz** (Node.js → Deno) | `bench-dataset.ts` utilise `postgres` et `node:fs` (Node.js only). L'approche 2-stages (export Node.js → load Deno) decouple les runtimes. Le format msgpack est ~3x plus compact que JSON et ~10x plus rapide a decoder | Adapter bench-dataset.ts pour Deno (deno-postgres) — effort disproportionne, fragile |
| D9 | **Param init inline** (mulberry32, orthogonal, identity-like dans train-ob.ts) | `parameters.ts` importe TF.js — impossible a importer en Deno. Les fonctions d'init sont pures et courtes (~100 lignes), les inliner est le choix le plus simple et robuste | Extraire les inits dans un fichier sans dep TF.js — refactoring structurel non necessaire pour un script de training |
| D10 | **BLAS fail-fast** (erreur si OpenBLAS pas disponible) | Per la consigne utilisateur : "si c'est pas en ffi avec open blas initialise lance une erreur ca sert a rien". Le fallback JS est 10x plus lent, inacceptable pour le training de production | Fallback silencieux vers JS — viole la politique no-silent-fallbacks |

---

## Fichiers modifies/crees

| Fichier | Lignes | Action | Role |
|---------|--------|--------|------|
| `tools/train-ob.ts` | 840 | **CREE** | Script training OB complet (Deno) |
| `tools/export-dataset.ts` | 97 | **CREE** | Export dataset msgpack.gz (Node.js) |
| `src/training/infonce-loss.ts` | 141 | **CREE** | InfoNCE loss + gradient analytique |
| `src/training/adam-optimizer.ts` | 163 | **CREE+MODIFIE** | Adam optimizer avec LR getter/setter |
| `src/training/__tests__/infonce-loss.test.ts` | — | **CREE** | 11 tests |
| `src/training/__tests__/adam-optimizer.test.ts` | — | **CREE** | 7 tests |
| `src/training/__tests__/ob-trainer-integration.test.ts` | — | **CREE** | 5 tests |
| `src/training/multi-level-trainer-khead.ts` | 679 | **MODIFIE** | Naming cleanup cap→node, dNodeEmbedding wire |
| `src/training/multi-level-trainer.ts` | 834 | **MODIFIE** | dNodeEmbedding obligatoire (plus de fallback) |
| `src/message-passing/multi-level-orchestrator.ts` | 1263 | **MODIFIE** | Training mode : forwardWithCache + LevelIntermediates |
| `src/utils/blas-ffi.ts` | 529 | **COPIE shgat-ob** | FFI OpenBLAS (sgemm, sgemv, sger) |
| `src/utils/math.ts` | 406 | **COPIE+MODIFIE** | Math utils + zerosLike2D/3D (decouple de TF.js) |
| `src/training/batched-khead.ts` | 402 | **COPIE+MODIFIE** | Batched K-head BLAS, naming cleanup |

**Total : ~4950 lignes de code OB training**

---

## Resultats

> A remplir apres le premier run complet.

| Metrique | TF.js autograd | OB + BLAS |
|----------|----------------|-----------|
| RAM peak | 11.5 GB | TBD |
| Hit@1 | TBD | TBD |
| Hit@3 | TBD | TBD |
| MRR | TBD | TBD |
| Temps/epoch | 276s | TBD |

---

## Cibles

| Métrique | TF.js autograd (actuel) | OB + BLAS (cible) |
|----------|------------------------|-------------------|
| RAM peak | 11.5 GB | **1-3 GB** |
| Hit@1 epoch 1 | 18.7% | ≥ 18% (parité) |
| Temps/epoch | 276s | **< 60s** |
| W_up/W_down gradients | InfoNCE only | InfoNCE + backward MP ✅ |
| Runtime | Node.js | **Deno** |
| Backward | TF.js tape (3-8GB) | Manuel couche par couche (~200MB cache) |

---

## Risques

1. **Précision float32 FFI** : le BLAS FFI convertit `number[]` → `Float32Array` → BLAS → `Float32Array` → `number[]`. La perte de précision f64→f32→f64 est acceptable pour le training (<1e-6 relative error).

2. **Performance JS fallback** : si OpenBLAS n'est pas disponible, le fallback JS est ~10x plus lent. Acceptable pour le développement mais pas pour le training de production.

3. **Divergence forward** : le forward OB (number[][]) et le forward TF.js (tensors float32) peuvent diverger légèrement à cause de la précision. Le test de parité (étape 5) vérifie ça.

4. **Subgraph sampling** : le trainer OB n'a pas de subgraph sampling. Pour le graph actuel (8984 nodes), le forward complet tient en ~200MB de cache. Le sampling est optionnel (à ajouter si le graph grossit > 50K).

5. **Data loader Node→Deno** : ~~bench-dataset.ts utilise des API Node.js~~ **RESOLU** — Option B retenue : `tools/export-dataset.ts` (Node.js) exporte en `msgpack.gz`, charge en Deno. Le format msgpack est ~3x plus compact que JSON et ~10x plus rapide a decoder.
