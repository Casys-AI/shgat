# SHGAT-TF Training Optimization Plan — 2026-02-13

## Contexte

Le training SHGAT-TF avec le nouveau `train-from-bench.ts` crash systématiquement (OOM 19GB+).
Le message passing (MP) était désactivé dans le training loop (`mpContext = undefined`), ce qui signifiait que W_up, W_down, a_up, a_down ne recevaient **jamais** de gradients.

### Résultats des runs précédents

| Run | LR | KL | MP | Epochs | Hit@1 | RAM Peak | Status |
|-----|-----|-----|-----|--------|-------|----------|--------|
| Run 1 (tmux) | 0.05 | OFF | OFF | 4/10 (crash) | N/A (eval@5) | 19 GB | OOM epoch 4 |
| Run 2 (report) | 0.001 | OFF | OFF | 5/5 | 10.3% | 22.4 GB | Sous-performe |
| Baseline (ancien) | 0.001 | N/A | N/A | 10 | 16.4% | ~4 GB | 525 tools flat |

### Dataset

- 8984 nodes (1901 leaves + 7083 non-leaf)
- 1155 prod exemples (385 traces × 3 oversample)
- 35189 n8n soft targets (44147 total, 80/20 split)
- Embeddings 1024D (BGE-M3)

---

## Diagnostic — Sources de fuite mémoire

### 1. Tape autograd massif (5-8 GB)
Dans `trainBatch()`, chaque appel à `forwardScoring()` dans la boucle per-example crée des tensors intermédiaires que le tape de `tf.variableGrads` retient pour le backward pass. Avec batch=32 × ~100 tensors/exemple = ~3200 tensors dans le tape.

### 2. Matrices d'attention MP dense (5-6 GB)
`denseAttentionAggregation` crée une matrice [7083 caps, 1901 tools] = 54 MB par head. Avec 16 heads × 2 directions (up+down), le tape retient ~5-6 GB.

### 3. `arraySync()` dans le tape
Ligne ~1218 de autograd-trainer.ts : `scores.arraySync()` force la sync GPU→CPU pendant que le tape est ouvert, causant des copies doublées.

### 4. Pas de `tf.tidy()` entre les batches
Les tensors intermédiaires du batch précédent ne sont pas disposés avant le batch suivant.

---

## BUG CRITIQUE : hiérarchie multi-niveaux jamais active

### Découverte (gnn-researcher, 2026-02-13)

Le MP multi-niveaux (`V→E^0→E^1→E^2→...→V`) n'a **JAMAIS fonctionné** dans la pipeline actuelle. C'est un bug de wiring dans `builder.ts`, pas un choix architectural.

**Cause** : dans `builder.ts` lignes 392-408, tous les children sont traités uniformément comme `toolsUsed` :
```typescript
// ACTUEL (bugué) :
capInfos.push({
  id: node.id,
  toolsUsed: validChildren,  // ← tools ET sous-caps mélangés
  // children: JAMAIS peuplé → cap.children = undefined
});
```

Dans `buildGraphStructure()` (autograd-trainer.ts ~ligne 1370), le code cherche `cap.children` pour construire les niveaux — mais c'est toujours `undefined`. Résultat : `maxLevel = 0`, `capToCapMatrices` vide.

### Fix requis (~15 lignes dans builder.ts)

```typescript
// CORRIGÉ :
const childTools = validChildren.filter(id => leafIds.has(id));
const childCaps = validChildren.filter(id => !leafIds.has(id));
capInfos.push({
  id: node.id,
  toolsUsed: childTools.length > 0 ? childTools : validChildren,
  children: childCaps.length > 0 ? childCaps : undefined,
});
```

Nécessite un premier passe pour identifier les `leafIds` (noeuds sans children), puis un deuxième passe pour construire les capInfos.

### Impact
- **Avant fix** : `maxLevel = 0`, MP flat (V↔E^0 seulement), les 59 edges DB hiérarchiques sont écrasées
- **Après fix** : `maxLevel = 2-3`, MP complet (V↔E^0↔E^1↔E^2), la hiérarchie récursive est active
- Le code MP multi-niveaux (`messagePassingForward()` lignes 446-573) est **déjà correct** — il suffit de fixer l'input

---

## Décision architecturale : le Message Passing DOIT rester actif

### Pourquoi

Le MP est la raison d'être du SHGAT. Sans MP, c'est juste du cosine similarity sur les embeddings bruts BGE-M3. Le MP propage le contexte structurel à travers la hiérarchie pour distinguer des tools quasi-identiques dans un grand vocabulaire.

### Littérature de support

- [Graph Attention Networks for Recommendation](https://dl.acm.org/doi/10.1145/3292500.3330989) — l'attention MP discrimine l'importance des voisins
- [GNN Training Systems Benchmark](https://arxiv.org/html/2406.00552v3) — mini-batch ≈ full-graph en accuracy avec fraction de mémoire
- [Scalable MPNN](https://arxiv.org/html/2411.00835v1) — MP efficace sans attention globale
- [LMC: Fast GNN Training via Subgraph Sampling](https://arxiv.org/abs/2302.00924) — convergence prouvée avec historical gradients

---

## Stratégie retenue : Ancestral Path Sampling

### Le problème de la hiérarchie récursive

Les capabilities sont récursives : L0 (tool) → L1 (capability) → L2 (meta-capability) → ... → L∞.
Le MP complet fait V→E^0→E^1→E^2→...→E^L→...→E^0→V.

### Options évaluées et rejetées

| Option | Verdict | Raison |
|--------|---------|--------|
| GraphSAGE 1-hop horizontal | ❌ Rejeté | Coupe les chemins verticaux, fan-out explosif |
| Random Walk (GraphSAINT) | ❌ Rejeté | Sur graphe bipartite hiérarchique, équivalent à Ancestral Path déterministe. La stochasticité n'apporte rien. |
| METIS Partitioning | ❌ Rejeté | Incompatible avec contrastive learning : METIS regroupe les tools proches, mais InfoNCE a BESOIN de tools distants comme négatifs. Dépendance C native. Roadmap 50K+ seulement. |
| Co-occurrence weighting | ⚠️ Pas prioritaire | Optimise quels chemins choisir, mais le bottleneck est la taille du sous-graphe, pas la qualité des chemins. Phase 3+ éventuellement. |
| GRU-informed sampling | ❌ Rejeté | Couplage circulaire SHGAT↔GRU. Les co-occurrences sont déjà capturées par les workflows n8n (hyperedges E^0). |

### Option retenue : Ancestral Path Sampling ✅

Pour chaque tool du batch, remonter TOUTE la chaîne ancestrale (L0→L1→L2→...→root) + sampler K siblings par parent.

**Déjà implémenté** dans `sampleSubgraph()` (autograd-trainer.ts lignes 1547-1723) :
- Sampling horizontal V↔E^0 avec K_horizontal=16 (Fisher-Yates)
- Sampling vertical E^0→E^1→...→root avec K_vertical=4
- Construction mini `capToCapMatrices` par paire de niveaux
- Retourne `GraphStructure` avec `maxLevel` dynamique

**Bloqué par le bug du builder** : tant que `maxLevel = 0`, le sampling vertical ne fait rien.

### Estimation mémoire (avec hiérarchie activée, 3 niveaux)

```
V ↔ E^0 : [~1200 tools, ~200 caps]  → attention ~614 MB  (99.98%)
E^0 ↔ E^1 : [~200 caps, ~20 parents] → attention ~0.1 MB
E^1 ↔ E^2 : [~20 caps, ~5 parents]   → attention ~0.01 MB
──────────────────────────────────────────────────────────
Total dans le tape : ~614 MB
```

Les niveaux supérieurs sont **quasi-gratuits** en mémoire (pyramide s'amincit exponentiellement).

---

## État actuel du code (après modifications agents 2026-02-13)

### ✅ FAIT (dans autograd-trainer.ts, par gnn-researcher)

| # | Modification | Lignes |
|---|-------------|--------|
| 1 | `AdjacencyCache` multi-niveaux (childToParents, parentToChildren) | 1462-1473 |
| 2 | `buildAdjacencyCache()` construit le cache depuis capToCapMatrices | 1479-1528 |
| 3 | `sampleK()` Fisher-Yates partial shuffle seedable | 1535-1545 |
| 4 | `sampleSubgraph()` Ancestral Path Sampling multi-niveaux | 1547-1723 |
| 5 | `buildSubgraphContext()` construit mini mpContext depuis batch | 1842-1945 |
| 6 | `trainBatch()` : `mpContext = this.buildSubgraphContext(examples)` au lieu de `undefined` | 1967 |
| 7 | Gradient scaling W_up/W_down par `config.mpLearningRateScale` | 1294-1306 |
| 8 | `setSubgraphK(K)` et `setSubgraphSampling(K, rng)` API publiques | — |
| 9 | Dispose propre du subgraph après trainStep | — |

### ✅ FAIT (dans train-from-bench.ts, par ml-optimizer)

| # | Modification | Détail |
|---|-------------|--------|
| 1 | LR configurable via `--lr` | Default 0.01 (ancien: 0.05) |
| 2 | eval-every configurable via `--eval-every` | Default 5, recommandé 1 |
| 3 | Flags `--no-per` `--no-curriculum` | Pour baseline simplifiée |

### ✅ FAIT (par agents, implémenté dans le code)

| # | Action | Fichier | Statut |
|---|--------|---------|--------|
| **A** | **Fix bug builder** : 2-pass sépare childTools/childCaps | `builder.ts` lignes 399-433 | **FAIT** — maxLevel > 0 quand DB hierarchy edges existent |
| **B** | **arraySync() hors du tape** : `tf.argMax().dataSync()[0]` au lieu de `scores.arraySync()` | `autograd-trainer.ts` ligne 1221 | **FAIT** — économise ~1-2 GB |

### ✅ FAIT (quick wins corrigés 2026-02-13)

| # | Action | Fichier | Statut |
|---|--------|---------|--------|
| **C3** | **arraySync() hors du tape KL** : `dataSync()[0]` après `tf.variableGrads` dans `trainStepKL` | `autograd-trainer.ts` ~ligne 985 | **FAIT** — même pattern que trainStep ligne 1272 |
| **C4** | **tf.engine().startScope()/endScope()** autour de chaque `trainBatch` et `trainBatchKL` | `train-from-bench.ts` ~lignes 551, 573, 653, 656 | **FAIT** — catch les tensors orphelins du subgraph MP |
| **Conv** | **Convention matrice [source, target]** : capToCapMatrices en [numChildren, numParents] | `autograd-trainer.ts` buildGraphStructure, buildAdjacencyCache, sampleSubgraph | **FAIT** — fixe le broadcast error `shapes 7047,9 and 9,7047` |

### ❌ À FAIRE — Quick wins (memory, impact estimé -5-13GB)

| # | Sévérité | Action | Fichier | Impact | Phase |
|---|----------|--------|---------|--------|-------|
| **C1** | CRITIQUE | **Sparse toolToCapMatrix** : remplacer dense [1928×7047] par COO/CSR | `autograd-trainer.ts` buildGraphStructure | -54MB par graph, prérequis pour sparse attention | 2 |
| **C5** | HAUTE | **Chunking threshold** : abaisser ATTENTION_CHUNK_THRESHOLD ou chunker le MP subgraph | `autograd-trainer.ts` ~ligne 203 | -2-4GB pendant MP dans le tape | 2 |
| **H2** | HAUTE | **Adjacency cache sparse** : construire depuis sparse incidence, pas arraySync dense | `autograd-trainer.ts` buildAdjacencyCache | -67MB (k-fold) + dépendance C1 | 2 |

### ❌ À FAIRE — Speedup (impact estimé 2-3x)

| # | Sévérité | Action | Fichier | Impact | Phase |
|---|----------|--------|---------|--------|-------|
| **H1** | HAUTE | **Batch InfoNCE scoring** : remplacer la boucle per-example (lignes 1186-1222) par projection batched. `batchContrastiveLoss()` existe mais n'est pas utilisé. | `autograd-trainer.ts` trainStep | 2-3x speedup (8.5s→~3s/epoch) | 2 |
| **M2** | MOYENNE | **Grad norm dans tf.tidy()** : wrapper le calcul square/sum dans un scope | `autograd-trainer.ts` ~lignes 1275-1283 | -5-10% pression GC | 3 |
| **M3** | MOYENNE | **Pool mini graph** : réutiliser les structures subgraph au lieu de réallouer 480×/run | `autograd-trainer.ts` buildSubgraphContext | -216MB churn total | 3 |
| **M4** | BASSE | **Reservoir sampling pour KL negs** : au lieu de shuffleInPlace(1918 items) + slice(128) | `train-from-bench.ts` ~ligne 632 | -1-2% | 3 |

### ❌ À FAIRE — Design decisions (Phase 3+)

| # | Sévérité | Action | Fichier | Impact | Phase |
|---|----------|--------|---------|--------|-------|
| **C2** | CRITIQUE (design) | **KL + MP** : actuellement KL ne passe PAS par le MP → W_up/W_down ne sont entraînés que sur les 1155 exemples prod, pas les 35K n8n. Intentionnel (pour éviter OOM) mais sous-optimal. Options : (A) MP dans le tape KL, (B) MP léger 1 niveau, (C) documenter seulement. | `autograd-trainer.ts` trainBatchKL | W_up/W_down sous-entraînés | 3 |
| **L1** | BASSE (futur) | **Phase V2V inutilisée** : 707 lignes implémentées (forward, backward, 2 params learnable) mais `v2vResidual: 0` par défaut et aucun code ne charge les données de co-occurrence. Infrastructure prête, activation manquante. | `vertex-to-vertex-phase.ts` | Code mort ou feature future | 3+ |
| **D** | MOYENNE | **Gradient accumulation micro-batch=4 pour KL** | `autograd-trainer.ts` trainBatchKL() | Réduit peak memory KL | 3 |
| **E** | MOYENNE | **KL weight warmup** (0→0.1 sur 5 epochs) — déjà implémenté dans train-from-bench.ts via `scheduleKLWeight()` | `train-from-bench.ts` | Déjà fait | — |
| **F** | MOYENNE | **Sampled softmax pour KL** (128 négatifs au lieu de 1901) — déjà implémenté via `KL_NUM_NEGS` | `train-from-bench.ts` | Déjà fait | — |
| **H3** | MOYENNE | **precomputeEnrichedEmbeddings full-graph MP** : 136MB peak pendant eval. Options : (A) cache (déjà fait), (B) subgraph MP pour eval, (C) skip MP pour eval. | `autograd-trainer.ts` | +136MB peak transitoire | 3 |

### Vérifications confirmées (pas de bug)

| # | Item | Résultat |
|---|------|----------|
| **M1** | Convention capToCapMatrices [children, parents] | ✅ Correct partout (buildGraphStructure, sampleSubgraph, buildAdjacencyCache) |
| **L2** | tf.memory() diagnostic calls | ✅ Pas de fuite (tf.memory() lit des compteurs, n'alloue pas) |

---

## Plan d'exécution en 3 phases

### Phase 1 — Baseline avec MP actif + hiérarchie

**Objectif** : valider que le subgraph MP + multi-level fonctionne sans OOM.

**Actions** : items A, B, C3, C4, convention matrices — tous **FAIT**

**Commande** :
```bash
NODE_OPTIONS="--max-old-space-size=8192" npx tsx tools/train-from-bench.ts \
  --epochs 20 --batch-size 32 --lr 0.01 --temperature 0.07 \
  --no-per --no-curriculum --no-kl --eval-every 1
```

**Cibles** : Hit@1 > 16.4% (battre l'ancien baseline), RAM < 6 GB, pas de crash

**Résultats préliminaires (run 2026-02-13, avant C3/C4 fix)** :
- Epoch 1 : Hit@1=18.7%, Hit@3=45.8%, MRR=0.334, RAM=11.5GB — **bat déjà le baseline (16.4%)**
- Epoch 2 : Hit@1=8.4% (chute due au LR warmup 0.0034→0.0067→0.01)
- Training stoppé pour appliquer C3+C4. À relancer après ces fixes pour vérifier la réduction mémoire.
- **RAM cible post-fix : 6-8GB** (C3 = -2-5GB, C4 = -3-8GB)

### Phase 2 — Sparse matrices + batch InfoNCE + valider multi-niveaux

**Objectif** : passer les matrices en sparse (C1, H2), batcher le scoring InfoNCE (H1), valider maxLevel > 0.

**Actions** : C1 (sparse toolToCapMatrix), H1 (batch InfoNCE), H2 (adjacency sparse), C5 (chunking)

**Validation** : vérifier que `maxLevel > 0` dans les logs, que `capToCapMatrices` n'est plus vide, et que l'embedding delta augmente.

**Cibles** : Hit@1 > Phase 1, RAM < 6 GB, temps/epoch < 60s (vs ~276s actuel)

### Phase 3 — KL + design decisions + polish

**Objectif** : exploiter les 44K exemples n8n, décider du KL+MP (C2), activer V2V (L1).

**Actions** : C2 (KL + MP design), D (gradient accumulation), L1 (V2V activation), M2-M3-M4 (polish)

**Stratégie** :
- KL utilise les embeddings **pré-enrichies** par le MP (pas de MP dans le tape pour KL) — design decision C2 à trancher
- KL warmup (E) et sampled softmax (F) sont **déjà implémentés**
- Micro-batch=4 avec gradient accumulation (D) si nécessaire
- V2V : charger les données de co-occurrence, activer `v2vResidual > 0`

**Cibles** : Hit@1 +3-5% vs Phase 2, RAM < 8 GB

---

## Cibles globales

| Métrique | Avant fixes | Phase 1 (C3+C4) | Phase 2 (sparse+batch) | Phase 3 (KL+V2V) |
|----------|-------------|-----------------|----------------------|-------------------|
| Hit@1 | 18.7% (ep1) | 18-22% | 22-28% | 28-35% |
| RAM peak | 11.5 GB | **6-8 GB** | <6 GB | <8 GB |
| MP actif | Oui (subgraph) | Oui (subgraph) | Oui (multi-level sparse) | Oui + KL + V2V |
| W_up/W_down gradients | InfoNCE only | InfoNCE only | InfoNCE only | InfoNCE + KL (C2) |
| Temps/epoch | ~276s | ~250s | **~60s** (H1 batch) | ~120s |

---

## Impact attendu sur le GRU downstream

Le GRU utilise les embeddings enrichis par SHGAT comme `similarity_head` (frozen).
Meilleur SHGAT = meilleurs embeddings enrichis = meilleur GRU.

- GRU actuel (embeddings bruts) : K-fold Hit@1 = 62.5% ±2.2%
- GRU attendu (après Phase 2) : Hit@1 = 63-65% (+2-4%)
- GRU attendu (après Phase 3 + KL) : Hit@1 = 65-68% (+3-5% supplémentaires)

Le gain principal viendra des cas où le GRU hésite entre des tools sémantiquement proches — exactement ce que le SHGAT contrastif avec MP est censé résoudre.

---

## Points d'incertitude

1. **mpLearningRateScale** : recommandé 50, mais pourrait être 10-100. Nécessite ablation study (comparer normes gradient W_up vs W_k).
2. **Impact réel des 59 DB edges** : 99.5% des caps restent à L0. Le gain multi-niveaux sera mesurable seulement si ces 59 edges connectent des tools fréquemment interrogés.
3. **K=8 vs K=16** : K=16 dans train-from-bench (choix ml-optimizer), K=8 par défaut dans sampleSubgraph. K=16 = meilleure couverture mais plus de mémoire. Avec 59 DB edges, K=8 suffit probablement pour les niveaux supérieurs.
4. **toolsUsed=[] après fix builder** : quand une capability a SEULEMENT des enfants-capabilities, `toolsUsed: []`. Vérifié : `buildGraphStructure` gère ce cas correctement (`[].includes(toolId)` → false).
5. **Fan-out sibling explosion** : un tool populaire (ex: http_request) peut être dans 200+ workflows × 16 siblings = 3200. Avec K=16 uniforme et 512 caps, le sous-graphe peut atteindre 8000+ tools. **Solution à implémenter si nécessaire** : budget-based sampling (toolBudget=512, capBudget=256/level).

---

## Décisions archivées

### METIS Partitioning
Évalué et **rejeté**. Raison principale : incompatible avec contrastive learning (InfoNCE). METIS regroupe les tools proches dans un cluster, mais InfoNCE a BESOIN de négatifs distants. Des négatifs cross-cluster sont possibles (style Cluster-GCN + LMC historical embeddings) mais ajoutent de la complexité (cache d'embeddings stales, rafraîchissement périodique). Pour 8984 nodes, l'Ancestral Path Sampling est plus simple. À reconsidérer à 50K+ nodes si l'adjacency cache ne tient plus en RAM.

### Random Walk (GraphSAINT)
Évalué et **rejeté**. Sur un graphe bipartite hiérarchique (tools↔caps_L0↔caps_L1↔...), un random walk est forcé d'alterner entre niveaux. C'est équivalent à l'Ancestral Path Sampling déterministe. La stochasticité n'apporte rien.

### GRU-informed sampling
Évalué et **rejeté** pour la v1. Créerait un couplage circulaire SHGAT↔GRU (le GRU utilise les embeddings SHGAT en input). Les co-occurrences sont déjà capturées par les workflows n8n (hyperedges E^0). Possible en v2 via des "importance sampling weights" pré-calculés.

### GraphSAGE 1-hop pur
Implémenté initialement puis **remplacé** par l'Ancestral Path Sampling multi-niveaux qui préserve la hiérarchie complète.

---

## Références

- Panel d'experts ML (4 experts, 2026-02-13) — analyse mémoire et hyperparamètres
- gnn-researcher (agent) — diagnostic MP, subgraph sampling, découverte bug builder
- ml-optimizer (agent) — diagnostic hyperparamètres, stratégie KL
- Recherche : GraphSAGE, Cluster-GCN, GraphSAINT, Scalable MPNN, LMC
- Training report : `lib/gru/data/shgat-training-report-2026-02-12T17-13-52-273Z.json`
- Log run LR=0.05 : `/tmp/shgat-train-ab-nomp.log`
