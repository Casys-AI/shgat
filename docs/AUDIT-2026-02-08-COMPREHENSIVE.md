# SHGAT-TF AutogradTrainer - Audit Complet (2026-02-08)

> Consolidation de 3 audits paralleles : ML Training Dynamics, Hypergraph MP Correctness, Old-vs-New Diff

## Statut Actuel

- **Baseline (random W_k + MP)** : 70.6% Hit@1, MRR 0.712
- **Apres training SGD 6 epochs** : loss 3.00->2.98, acc 62%->65%, grad stable ~1.5
- **Crash** : WASM `Aborted()` dans `batchMatMul` a l'epoch 6 (fuite memoire tenseur)
- **Ancien SHGAT de production** : ~80% Hit@1

---

## FINDINGS CRITIQUES (Priorite 1 - Bloquants)

### C1. Double Batch-Averaging (32x learning rate trop faible)

- [ ] **FIXE**
- **Source** : audit-ml Finding 9
- **Fichier** : `autograd-trainer.ts` ~ligne 1252 + ~ligne 1293
- **Notes** :

Le loss est divise par `examples.length` dans `tf.variableGrads()` :
```typescript
const avgLoss = loss.div(examples.length); // <- premiere division
```
Puis le learning rate est divise par `examples.length` dans le SGD manuel :
```typescript
const kheadLr = trainerConfig.learningRate / examples.length; // <- deuxieme division
```

**Effective lr** = 0.05 / 32^2 = **0.0000488** au lieu de 0.05 / 32 = 0.00156
**Impact** : 32x trop lent. Explique la convergence minuscule (62% -> 65% en 6 epochs).

**Fix** : Retirer `loss.div(examples.length)` dans le closure `variableGrads`, OU retirer `/ examples.length` du SGD manuel.

---

### C2. Backward MP : dE_proj / dH_proj / dTgt_proj droppes (3 fonctions)

- [ ] **FIXE**
- **Source** : audit-hypergraph F6, F7, F8 + audit-ml confirmation
- **Fichier** : `sparse-mp.ts`
- **Notes** :

Dans les 3 backward functions, le vecteur `dConcat` de taille `2 * headDim` est split en source/target, mais **seule la moitie source est utilisee**. La moitie target est silencieusement droppee.

| Fonction | Ligne | Ce qui est droppe | Impact |
|----------|-------|-------------------|--------|
| `backwardVertexToEdge()` | ~1018-1030 | `dE_proj[c]` (gradient caps via attention) | dW perd ~50%, dE perd 100% path attention |
| `backwardEdgeToVertex()` | ~1298-1308 | `dH_proj[t]` (gradient tools via attention) | dW perd ~50%, dH perd 100% path attention |
| `backwardEdgeToEdge()` | ~1167-1177 | `dTgt_proj` (gradient target via attention) | dW perd ~50%, dE_target perd 100% |

**L'ancien SHGAT traite correctement les deux moities** (vertex-to-edge-phase.ts:346-350).

**Impact** : ~50% du gradient d'attention est perdu sur dW, 100% perdu sur les embeddings cibles via le path d'attention. Bug le plus impactant de l'audit.

**Fix** : Ajouter le traitement de la 2e moitie de `dConcat` dans chaque backward function. Pour chaque :
```typescript
// AJOUTER apres le bloc existant pour la 1ere moitie :
for (let d = 0; d < headDim; d++) {
  const dTarget_proj_d = dConcat[headDim + d];
  // -> dW contribution
  // -> dTarget_embedding contribution (backprop through W)
}
```

---

### C3. L2 Normalization post-MP sans gradient backward

- [ ] **FIXE**
- **Source** : audit-ml Finding 6, audit-hypergraph F10, audit-diff #2
- **Fichier** : `sparse-mp.ts` ~lignes 402-408 (forward) vs ~lignes 760-762 (backward)
- **Notes** :

Le forward applique L2 normalization apres le residual mixing :
```typescript
H[i][j] /= norm;  // Forward
```
Mais le backward ignore completement le Jacobien de la normalisation :
```typescript
dH_enriched[i][j] = (1 - alpha) * dH_input[i][j];  // Backward - PAS de d(normalize)
```

La derivee de `x/||x||` est `(I - x*x^T/||x||^2) / ||x||` - une matrice de projection.

**L'ancien SHGAT ne fait PAS de L2 normalization post-MP.** Il utilise un simple additive residual.

**Impact** : Gradients MP mathematiquement incorrects. Les poids MP apprennent dans la mauvaise direction.

**Fix recommande** : Retirer la L2 normalization du forward (match le vieux SHGAT). Plus simple et plus safe que d'implementer le Jacobien complet.

---

### C4. Missing W_out Projection apres Multi-Head Concat

- [ ] **FIXE**
- **Source** : audit-ml Finding 7, audit-hypergraph F1
- **Fichier** : `sparse-mp.ts` ~lignes 471-518
- **Notes** :

Le concat multi-head remplit `E_new[c]` dimension par dimension :
```
[head0_dim0..head0_dim63, head1_dim0..head1_dim63, ...]  // espace projete
```
Puis le residual ajoute l'embedding BGE-M3 original :
```
H[i][j] = (1-alpha) * H_enriched[i][j] + alpha * H_init[i][j]  // espace original
```

**Ces deux espaces sont differents.** Il manque une projection `W_out @ concat(heads)` pour revenir dans l'espace d'embedding avant le residual. C'est standard dans Transformer et GAT.

**Impact** : Les embeddings post-MP sont dans un espace corrompu. Le scoring K-head opere sur des embeddings incoherents.

**Note** : Fonctionne "par accident" quand `numHeads * headDim == embDim` (16*64 = 1024), mais la semantique des dimensions est fausse. Fix gros (~100 lignes + nouveaux params) donc medium term.

---

### C5. Missing V2V Co-occurrence Phase

- [ ] **FIXE**
- **Source** : audit-diff #1
- **Fichier** : Absent du nouveau code. Existe dans `src/graphrag/algorithms/shgat/message-passing/vertex-to-vertex-phase.ts`
- **Notes** :

L'ancien SHGAT enrichit les embeddings tools avec les donnees de co-occurrence AVANT le upward pass. Le nouveau n'a rien de comparable.

**Impact** : Les tools ne recoivent aucun signal inter-tool. Le MP hierarchique doit tout compenser seul. Fix gros (~200 lignes) donc long term.

---

## FINDINGS IMPORTANTS (Priorite 2 - Performance)

### H1. Gradient Scale Mismatch : MP vs K-head

- [ ] **FIXE**
- **Source** : audit-ml Finding 4
- **Fichier** : `autograd-trainer.ts` ~ligne 1185
- **Notes** :

Le `dLogits` manuel pour le MP ne divise PAS par `examples.length`, mais le autograd path le fait via `loss.div(examples.length)`. Les gradients MP sont donc `batchSize` fois trop grands par rapport aux gradients K-head.

**Impact** : Avec batchSize=32, les poids MP recoivent des gradients 32x trop larges.

**Note** : Si on fix C1 (retirer loss.div), ce mismatch disparait. Les 2 paths seront en batch-sum scale.

---

### H2. dH Non-Zero envoye au MP Backward

- [ ] **SKIP** - Decision : on GARDE dH non-zero
- **Source** : audit-diff #5
- **Fichier** : `autograd-trainer.ts` ~ligne 1339 vs ancien `shgat.ts` ~ligne 1745
- **Notes** :

L'ancien SHGAT passe `dH_final = null` au backward MP. Le nouveau passe `dH_accum` non-zero. L'ancien ne remontait pas les gradients nodes (ex-tools) dans le MP, le nouveau si.

**Decision** : On garde le comportement nouveau (dH non-zero). Le vocabulaire tool/capability est unifie en "nodes" dans le nouveau code. Remonter les gradients de tous les nodes dans le MP est coherent avec l'architecture unifiee. L'absence dans le vieux code etait probablement une limitation, pas un choix optimal. Tests anterieurs suggerent que c'est pas forcement mauvais de les remonter.

---

### H3. No Dropout During Training

- [ ] **FIXE**
- **Source** : audit-diff #4
- **Fichier** : `sparse-mp.ts`, `autograd-trainer.ts`
- **Notes** :

L'ancien SHGAT applique du dropout sur H et E apres le MP forward (`multi-level-orchestrator.ts:811-816`). Le nouveau n'a aucun dropout.

**Impact** : Pas de regularisation, risque d'overfitting.

---

### H4. PER Importance Sampling Weights Ignorees

- [ ] **FIXE**
- **Source** : audit-ml Finding 5
- **Fichier** : `autograd-trainer.ts` ~ligne 520-535
- **Notes** :

L'ancien SHGAT multiplie le loss par `isWeight` (poids IS de PER). Le nouveau ne le fait pas. L'API `trainBatch()` n'accepte meme pas de parametre isWeights.

**Impact** : Avec PER sampling, les estimations de gradient sont biaisees (surrepresentation des exemples difficiles sans correction).

---

### H5. W_q/W_k Separation vs Shared

- [ ] **FIXE**
- **Source** : audit-ml Finding 2, audit-diff #6
- **Notes** :

| | Ancien | Nouveau |
|---|--------|---------|
| Matrices | W_q et W_k separees (mais `W_q === W_k` meme reference JS) | Un seul W_k |
| Gradient | `dW_q + dW_k` appliques 2x sur la meme memoire (amplification implicite) | Autograd somme Q+K contributions |

Le score devient `intent^T W_k^T W_k emb` (matrice PSD) au lieu de `intent^T W_q^T W_k emb` (general).

**Impact** : Capacite reduite, mais le double-write de l'ancien etait probablement un facteur d'amplification non-intentionnel. A evaluer avec A/B benchmark.

---

### H6. Projection Partagee Tools/Caps dans le MP

- [ ] **FIXE**
- **Source** : audit-hypergraph F11
- **Fichier** : `sparse-mp.ts` ~lignes 463-464, 666-667
- **Notes** :

L'ancien SHGAT utilise `W_child` pour projeter les tools et `W_parent` pour les caps dans les 2 directions. Le nouveau utilise un seul W pour source ET target.

**Impact** : Tools et caps ne peuvent pas habiter des sous-espaces differents par head. Reduit l'expressivite MP de 2x par phase.

---

## FINDINGS MINEURS (Priorite 3)

### M1. Temperature 0.07 vs 0.10

- [ ] **FIXE**
- **Source** : audit-diff #9
- **Notes** : Le nouveau utilise temperature=0.07 (plus sharp), l'ancien 0.10. Le bench override a 0.10 donc n'affecte que le default.

### M2. Pas de L2 Regularization sur les poids MP

- [ ] **FIXE**
- **Source** : audit-diff #8
- **Notes** : `applySparseMPGradients()` n'applique pas de L2 reg ni de per-element clip sur les poids MP. L'ancien applique les deux.

### M3. Dense vs Sparse Forward Inconsistency

- [ ] **FIXE**
- **Source** : audit-hypergraph F4
- **Notes** : Le dense `attentionAggregation()` decompose l'attention en contributions additives separees. Le sparse concatene puis applique LeakyReLU conjointement. Resultats differents.

### M4. Adam Optimizer Alloue mais Non-Utilise

- [ ] **FIXE**
- **Source** : audit-ml Finding 10
- **Notes** : `this.optimizer = tf.train.adam(lr)` est cree mais jamais utilise dans le path sparse. Gaspillage memoire.

### M5. Scoring Train vs Eval : Scaled Dot Product vs Cosine Similarity

- [ ] **FIXE**
- **Source** : audit-ml Finding 1, audit-diff #7
- **Notes** : Le `kHeadScoring()` (training) utilise scaled dot product. Le `khead-scorer.ts::scoreNodes()` (inference librairie) utilise cosine similarity. Si le deploiement utilise `scoreNodes()`, il y a un mismatch train/eval.

### M6. Dimension Fragile : numHeads * headDim == embDim

- [ ] **FIXE**
- **Source** : audit-hypergraph F1
- **Notes** : Le code assume `16 * 64 = 1024 = embDim`. Vrai pour la config par defaut mais fragile.

### M7. MP Gradient Norm Sous-Rapporte (sqrt(2) factor)

- [ ] **FIXE**
- **Source** : audit-ml Addendum A5
- **Fichier** : `autograd-trainer.ts` ~lignes 1346-1358
- **Notes** : Le calcul de la norme du gradient MP n'inclut que `dW_up` et `da_up`, pas `dW_down` et `da_down`. La norme rapportee est ~0.707x la vraie valeur. N'affecte PAS le training mais masque des instabilites potentielles dans les logs.

### M8. Commentaire Trompeur "MP params regularized separately"

- [ ] **FIXE**
- **Source** : audit-ml Addendum A6
- **Fichier** : `autograd-trainer.ts` ~ligne 1254
- **Notes** : Le commentaire dit que les MP params sont regularises separement mais il n'y a PAS de regularisation MP en mode sparse.

### M9. residualWeights tf.Variable Dead Code

- [ ] **FIXE**
- **Source** : audit-ml F8, audit-hypergraph F3
- **Fichier** : `autograd-trainer.ts` ~lignes 154-157
- **Notes** : `residualWeights` est initialise comme tf.Variable mais jamais utilise. Alpha est hardcode a 0.3.

---

## CLUSTER DE BUGS INTERCONNECTES

### Cluster MP Gradient Pipeline (5 findings = severite collective HIGH+)

**Source** : cross-analyse audit-hypergraph + audit-ml

Ces 5 bugs individuellement LOW-MEDIUM forment un probleme systematique HIGH :

1. **Pas de L2 reg sur MP params** (M2)
2. **Jacobien L2 manquant dans backward** (C3)
3. **Pas de gradient clipping MP** (audit-hypergraph)
4. **~50% du signal gradient d'attention perdu** (C2 - dConcat)
5. **Divergence sparse/dense L2 norm** (M3)

Les parametres MP recoivent des gradients **systematiquement incorrects en direction ET en magnitude**, sans regularisation ni clipping pour compenser. Le K-head scoring (autograd correct) compense partiellement mais les poids MP eux-memes ne convergent pas.

---

## PIPELINE COMPARISON TABLE (31 etapes)

| # | Etape | Match? | Severite |
|---|-------|--------|----------|
| 1 | Init graph | PARTIAL | LOW |
| 2 | V->E^0 (W_child/W_parent vs W_up shared) | **NON** | **CRITIQUE** |
| 3 | E^k->E^(k+1) (meme W sharing) | **NON** | **CRITIQUE** |
| 4 | Downward E^L->E^0 | **NON** | **CRITIQUE** |
| 5 | E^0->V | PARTIAL | **CRITIQUE** |
| 6 | Residual (additive vs weighted+L2norm) | **NON** | **CRITIQUE** |
| 7 | Flatten embeddings | OUI | NONE |
| 8 | Intent projection | PARTIAL | LOW |
| 9 | Q projection (W_q vs W_k shared) | **NON** | **CRITIQUE** |
| 10 | K projection | OUI | NONE |
| 11 | Score (scaled dot product) | OUI | NONE |
| 12 | Head fusion (mean) | OUI | NONE |
| 13 | Reliability multiplier | OUI | NONE |
| 14 | Loss (InfoNCE, epsilon/IS diff) | PARTIAL | MEDIUM |
| 15 | Temperature (0.07 vs 0.10) | **NON** | MEDIUM |
| 16 | Negative sampling | OUI | NONE |
| 17 | dLogit computation (IS weights) | PARTIAL | MEDIUM |
| 18 | dW_q (absent dans new) | **NON** | **CRITIQUE** |
| 19 | dW_k | OUI | LOW |
| 20 | dW_intent | OUI | LOW |
| 21 | dCapEmbedding chain rule | PARTIAL | MEDIUM |
| 22 | MP backward (residual scaling diff) | PARTIAL | MEDIUM |
| 23 | Gradient clipping K-head | OUI | NONE |
| 24 | L2 regularization | PARTIAL | LOW |
| 25 | Gradient application K-head | OUI | NONE |
| 26 | Gradient application MP | OUI | NONE |
| 27 | Optimizer (Adam dead code) | **NON** | LOW |
| 28 | PER IS weights | **NON** | MEDIUM |
| 29 | TD error for PER | **NON** | MEDIUM |
| 30 | V2V params (learnable vs fixed) | **NON** | MEDIUM |
| 31 | Embedding update (frozen) | OUI | NONE |

**Resume** : 9 etapes CRITIQUES, 7 MEDIUM, 4 LOW, 11 identiques.

---

## PLAN DE REMEDIATION (Ordre de Priorite)

| # | Fix | Finding | Severite | Effort | Fichier | Status |
|---|-----|---------|----------|--------|---------|--------|
| 1 | Retirer `loss.div(examples.length)` | C1 | CRITIQUE | 1 ligne | autograd-trainer.ts | [ ] |
| 2 | Retirer L2 normalization post-MP | C3 | CRITIQUE | ~10 lignes | sparse-mp.ts | [ ] |
| 3 | Ajouter 2e moitie dConcat dans 3 backward functions | C2 | CRITIQUE | ~60 lignes | sparse-mp.ts | [ ] |
| 4 | ~~Passer dH=zeros au MP backward~~ | H2 | ~~IMPORTANT~~ | SKIP | autograd-trainer.ts | SKIP - on garde dH non-zero |
| 5 | Fix gradient scale mismatch MP/khead | H1 | IMPORTANT | ~5 lignes | autograd-trainer.ts | [ ] |
| 6 | Ajouter W_out projection post-concat | C4 | CRITIQUE | ~100 lignes | sparse-mp.ts | [ ] |
| 7 | Ajouter dropout post-MP | H3 | IMPORTANT | ~20 lignes | sparse-mp.ts | [ ] |
| 8 | Ajouter PER IS weights | H4 | IMPORTANT | ~10 lignes | autograd-trainer.ts | [ ] |
| 9 | Ajouter L2 reg + clip aux poids MP | M2 | MOYEN | ~15 lignes | sparse-mp.ts | [ ] |
| 10 | Implementer V2V enrichment | C5 | CRITIQUE | ~200 lignes | nouveau fichier | [ ] |

### Quick Wins (Fix 1-5) : Faisables immediatement, gros impact attendu

Le Fix 1 seul devrait multiplier le learning rate par 32x.
Le Fix 2 + 3 devrait corriger les gradients MP.
Le Fix 4 + 5 devrait stabiliser l'entrainement.

### Medium Term (Fix 6-9) : Importants mais plus de travail

### Long Term (Fix 10) : Architecture complete avec V2V

---

## BENCH CRASH : WASM Aborted() a l'epoch 6

```
Epoch  6/10: loss=2.9835 acc=0.615 tau=0.070 grad=1.466 [medium] (23 batches)
Aborted()
error: Uncaught RuntimeError: Aborted(). Build with -sASSERTIONS for more info.
  at Object.batchMatMul [as kernelFunc]
```

Probablement une fuite memoire tenseur dans la boucle d'entrainement. Les tenseurs crees pendant `tf.variableGrads` ou la chain rule manuelle ne sont pas tous disposes.

---

## REFERENCES FICHIERS CLES

| Fichier | Role |
|---------|------|
| `lib/shgat-tf/src/training/autograd-trainer.ts` | Trainer principal (K-head, loss, SGD) |
| `lib/shgat-tf/src/training/sparse-mp.ts` | Message passing sparse (forward + backward + apply) |
| `lib/shgat-tf/src/attention/khead-scorer.ts` | Scorer inference (cosine similarity) |
| `lib/shgat-tf/src/core/types.ts` | Config defaults |
| `lib/shgat-tf/src/initialization/parameters.ts` | Init params |
| `src/graphrag/algorithms/shgat.ts` | ANCIEN SHGAT (reference ~80% Hit@1) |
| `src/graphrag/algorithms/shgat/training/multi-level-trainer-khead.ts` | ANCIEN K-head gradient apply |
| `src/graphrag/algorithms/shgat/training/batched-khead.ts` | ANCIEN K-head forward/backward |
| `src/graphrag/algorithms/shgat/message-passing/multi-level-orchestrator.ts` | ANCIEN MP orchestrator |
| `src/graphrag/algorithms/shgat/message-passing/vertex-to-edge-phase.ts` | ANCIEN V->E (reference backward correct) |
