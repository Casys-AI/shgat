# Audit Consolidé SHGAT-TF + GRU — 2026-02-13

> **Statut** : Training SHGAT-TF en cours (train-ob.ts, backward manuel + OpenBLAS FFI, ~7min/epoch, 15 epochs).
> **Dernière action** : Nettoyage massif du code (~1850 lignes mortes supprimées du training stack).

---

## 1. Vue d'ensemble architecture

```
                          ┌─────────────────────────────────────────────┐
                          │              Pipeline Complet               │
                          └─────────────────────────────────────────────┘

  Intent (texte)                                                    Séquence d'outils
       │                                                                  ▲
       ▼                                                                  │
 ┌───────────┐         67D (64+3)          ┌───────────┐    buildPath*    │
 │  SHGAT-TF │ ──────────────────────────► │  GRU      │ ───────────────►│
 │  (scoring) │  capInput: 64D (hierarchy) │  v0.3.0   │    greedy /     │
 │            │  compositeInput: 3D        │  (258K)   │    beam@3       │
 │  K-head    │    (scores)                │           │                 │
 │  + MP (?)  │                            │  5 inputs │                 │
 │  + InfoNCE │    VocabNode unifié        │  GRU(64)  │                 │
 └───────────┘ ◄──────────────────────────►└───────────┘                 │
       │          shared vocabulary (870)        │                       │
       │          644 tools + 226 caps           │                       │
       ▼                                         ▼                       │
 ┌───────────┐                           ┌───────────────┐               │
 │ LiveMCP   │                           │ Production    │               │
 │ Bench     │                           │ Traces (52)   │               │
 │ (282 ex.) │                           │ + n8n (35K)   │               │
 │ eval-only │                           │ = 38K+ ex.    │               │
 └───────────┘                           └───────────────┘
```

### Rôles

| Composant | Rôle | Entrées | Sorties |
|-----------|------|---------|---------|
| **SHGAT-TF** | Scoring sémantique hiérarchique | Intent embedding (1024D BGE-M3), hypergraph (servers→tools) | 67D pour GRU : capInput (64D hierarchy) + compositeInput (3D scores), recall@K standalone |
| **GRU v0.3.0** | Séquencement autorégressif | 5 inputs dont 67D du SHGAT | Prochaine tool ID, probabilité de terminaison |
| **VocabNode** | Vocabulaire unifié | — | L0 = leaf tools, L1+ = capabilities. Shared entre SHGAT et GRU |

### Pipeline de production recommandée

1. **1er outil** : GRU(ctx=[]) — le GRU avec contexte vide surpasse SHGAT-first (63.3% vs 17.9%)
2. **Outils suivants** : GRU(itération) — autorégressif avec contexte enrichi
3. **SHGAT** : scoring/vocabulary + features (67D) — fournit capInput + compositeInput au GRU

---

## 2. SHGAT-TF — État actuel

### 2.1 K-head Scoring (mature)

Le K-head scoring est le coeur du SHGAT entraînable :

- **Forward** : `computeMultiHeadKHeadScoresWithCache()` — multi-tête Q·K/√dim → sigmoid
- **Backward** : `backpropMultiHeadKHeadLogit()` — gradients analytiques dQ, dK, dW_q, dW_k
- **Loss** : InfoNCE contrastive avec température τ (annealing 0.1→0.06)
- **Optimizer** : Adam plain JS (pas TF.js) avec bias correction et gradient clipping

**Fichiers** :
- `src/training/multi-level-trainer-khead.ts` (~244 lignes) — forward + backward K-head
- `src/training/multi-level-trainer.ts` (~104 lignes) — types + gradient init/reset
- `src/training/infonce-loss.ts` — InfoNCE loss contrastive
- `src/training/adam-optimizer.ts` — Adam optimizer plain JS

**Nettoyage effectué (2026-02-13)** :
- ~1850 lignes mortes supprimées (3 générations : cosine scorer → K-head autograd → OB manuel)
- `batched-khead.ts` supprimé (402 lignes, 0 imports)
- `multi-level-trainer.ts` réduit de 834→104 lignes
- `multi-level-trainer-khead.ts` réduit de 680→244 lignes
- Tests mis à jour (500→219 lignes, 21/21 pass)

### 2.2 Message Passing (en cours d'évaluation avec 36K exemples)

Le MP propage l'information dans l'hypergraphe à travers 4 phases :

| Phase | Direction | Fichier | Rôle |
|-------|-----------|---------|------|
| VE | Vertex → Edge | `vertex-to-edge-phase.ts` | Agrège les embeddings des noeuds vers les hyperedges |
| EE | Edge → Edge | `edge-to-edge-phase.ts` | Propagation inter-edges (voisinage) |
| EV | Edge → Vertex | `edge-to-vertex-phase.ts` | Redistribue l'info enrichie vers les noeuds |
| V2V | Vertex → Vertex | `vertex-to-vertex-phase.ts` | Propagation directe parent↔enfant (implémenté, non activé) |

**Orchestrator** : `multi-level-orchestrator.ts` — coordonne les 4 phases avec support backward complet.

**QUESTION OUVERTE : le MP aide-t-il avec 36K exemples ?**

| Dataset | Exemples | Résultat MP |
|---------|----------|-------------|
| LiveMCPBench | 282 | **INUTILE** — Flat R@1=16.4%, Hier-PDR(0.5) R@1=11.2% |
| Training principal (n8n+prod) | 38K+ | **EN COURS** — résultats attendus epoch 5+ |

Le constat LiveMCPBench (282 ex.) : "K-head scoring apprend bien, mais W_up/W_down nécessitent 1000+ exemples". Avec 38K exemples, les conditions sont radicalement différentes.

### 2.3 Training OB — Backward Manuel + OpenBLAS FFI (approche ACTUELLE)

**Ce n'est PAS un compromis — c'est un choix délibéré** après avoir testé TF.js autograd :

| Critère | TF.js autograd (LEGACY) | Backward manuel + OpenBLAS (ACTUEL) |
|---------|------------------------|--------------------------------------|
| Runtime | Deno WASM (lent) | Deno + OpenBLAS FFI (natif) |
| Contrôle gradients | Boîte noire | Total — chaque gradient vérifié analytiquement |
| RAM | Incontrôlable (tape TF.js) | Prévisible (~12GB pour 38K ex.) |
| Débogage | Difficile (graphe TF) | Direct (valeurs numériques accessibles) |
| Vitesse/epoch | ~35s/epoch (282 ex. WASM) | ~7min/epoch (38K ex., OpenBLAS) |

**Pipeline** (`tools/train-ob.ts`) :
1. Export dataset via `export-dataset.ts` → msgpack.gz (pako + @msgpack/msgpack)
2. Charge embeddings 1024D, hypergraphe, exemples prod+n8n
3. Forward K-head + MP → scores
4. InfoNCE loss contrastive (τ annealing, négatifs par batch)
5. Backward analytique → gradients K-head + W_intent + MP
6. Adam optimizer step (LR warmup 3 epochs)
7. KL divergence pour exemples n8n (warmup 3 epochs, weight 0.2)
8. Eval tous les 5 epochs

**Config actuelle** : 15 epochs, LR 0.01 (warmup 3), batch 32, seed 42, 5.25M params, ~12GB RAM

### 2.4 autograd-trainer.ts — LEGACY (benchmark uniquement)

Conservé **uniquement** pour le benchmark LiveMCPBench (282 exemples, 525 tools, 69 MCP servers).

Résultats LiveMCPBench (5-fold CV, seed=42) :
- B-Flat (LR=0.001, 10ep) : R@1=16.4%, R@3=37.6%, NDCG@5=44.6% ← **BEST**
- Overfit : R@1=33.5% — prouve la capacité du modèle, 282 exemples = bottleneck

### 2.5 Divergences specs/code identifiées

| Divergence | Détail | Statut |
|------------|--------|--------|
| Q/K sharing | Spec mentionne partage, implem a W_q et W_k séparés | Intentionnel (meilleur gradient flow) |
| downwardResidual | `autograd-trainer.ts` hardcodait alpha=0.3 | Corrigé 2026-02-09 |
| KL/MP asymétrie | KL loss n8n ne passe pas par MP | Design intentionnel |
| V2V | Implémenté dans orchestrator, non activé dans train-ob | Attend données de cooccurrence |
| batchContrastiveLoss | Fonction existante, jamais appelée | Dead code (à nettoyer) |

---

## 3. GRU v0.3.0 — Pilier du système

### 3.1 Architecture (5 inputs, 258K params)

| # | Input | Dimension | Source | Projection |
|---|-------|-----------|--------|------------|
| 1 | contextInput | [maxSeq=20, 1024D] | Embeddings BGE-M3 | → 128D (input_proj) |
| 2 | transFeatsInput | [maxSeq=20, 5D] | Jaccard, bigram, structural | Direct |
| 3 | intentInput | 1024D | Embedding BGE-M3 | → 64D (intent_proj) |
| 4 | **capInput** | **64D** | **Capability fingerprint — hiérarchie L1+ du SHGAT** | → 16D (cap_proj) |
| 5 | **compositeInput** | **3D** | **scoreNodes() du SHGAT** | → 8D (composite_proj) |

**Couplage SHGAT→GRU = 67D** (inputs #4 + #5)

**Processing :**
```
contextInput[1024] → input_proj(128) ──┐
transFeatsInput[5] ─────────────────────┤→ concat[133] → GRU(64)

intentInput[1024] → intent_proj(64) ────┐
capInput[64] → cap_proj(16) ────────────┤→ fusion[152] → similarity_head → tool scores
compositeInput[3] → composite_proj(8) ──┤
gruOutput[64] ──────────────────────────┘

Termination: [gruOutput(64) + intentProj(64)] = 128D → Dense(32,relu) → Dense(1,sigmoid)
```

### 3.2 Résultats

**5-fold cross-validation** (seed=42, 870 vocab, 3100 exemples) :

| Métrique | Moyenne | Std |
|----------|---------|-----|
| Hit@1 | 60.6% | ±5.2 |
| Hit@3 | 80.9% | ±4.1 |
| MRR | 0.712 | ±0.04 |
| Termination | 73.3% | ±3.8 |
| Beam@3 | 26.9% | ±12 |

**Finding clé** : GRU-first (63.3%) >> SHGAT-first (17.9%) pour le 1er outil.

### 3.3 Pipeline n8n data augmentation

| Source | Exemples | Loss |
|--------|----------|------|
| Production (52 traces × 3 oversample) | 3465 | Focal cross-entropy + termination BCE |
| n8n workflows (filtré) | 35189 | KL divergence (soft targets T=0.005) |
| **Total** | **~38654** | Multi-task |

### 3.4 Les 67D du SHGAT dans le GRU

**capInput (64D)** — le capability fingerprint :
- Encode la structure hiérarchique L1+ du SHGAT
- Chaque dimension représente l'appartenance/affinité à une capability
- Dérivé de la hiérarchie du graphe (servers → capabilities → tools)
- Information **structurelle** sur la position de l'outil dans le graphe

**compositeInput (3D)** — les scores SHGAT :
- `[0]` bestCompositeScore — score [0,1] du meilleur match SHGAT
- `[1]` compositeCoverage — overlap sémantique intent ↔ composite embedding
- `[2]` compositeLevel — niveau hiérarchique normalisé (0=L0, 0.5=L1, 1.0=L2)

**67D = signal SHGAT complet** — pas juste "3 features", mais structure (64D) + scoring (3D).

---

## 4. Intégration SHGAT ↔ GRU : couplage 67D

### 4.1 Points de couplage

| Point | Dimension | Direction | Nature |
|-------|-----------|-----------|--------|
| **capInput** | **64D** | SHGAT → GRU | Structurel (hiérarchie L1+) |
| **compositeInput** | **3D** | SHGAT → GRU | Scoring (scores continus) |
| VocabNode unifié | — | Partagé | Vocabulaire (870 noeuds) |
| Embeddings BGE-M3 | 1024D | Partagé | Embeddings tools + intent |

### 4.2 Analyse du couplage

67D = **couplage fort** :
- Ce n'est pas une interface légère — c'est une représentation riche (structure + scoring)
- Le GRU dépend structurellement du SHGAT pour 67/152 = **44% de sa couche de fusion**
- Toute modification de la hiérarchie SHGAT ou du scoring impacte le GRU
- Unidirectionnel (GRU ne renvoie rien au SHGAT) mais la dépendance est profonde

### 4.3 Proposition : adaptateur `shgat-for-gru`

Module adaptateur qui formalise l'interface 67D :

```
lib/shgat-for-gru/
├── src/
│   ├── adapter.ts          # Interface formelle: SHGAT → 67D features
│   ├── cap-fingerprint.ts  # Calcul capInput (64D) depuis hiérarchie
│   ├── composite-scorer.ts # Calcul compositeInput (3D) depuis scoreNodes
│   ├── types.ts            # SHGATForGRUFeatures { cap: float[64], composite: float[3] }
│   └── cosine-scorer.ts    # Cold-start: scorer minimal pour outils inconnus du GRU
└── tests/
```

**Avantages** :
- **Contrat formel** : si l'interface change, un seul endroit à modifier
- **Testabilité** : on peut tester que SHGAT produit les bonnes features sans lancer le GRU
- **Cold-start** : scorer minimal pour les outils pas encore vus par le GRU
- **Découplage** : le GRU importe `shgat-for-gru`, pas `shgat-tf` directement
- **Production** : l'adaptateur peut cacher la complexité SHGAT derrière une API simple

---

## 5. Question fusion vs séparation

### 5.1 État actuel : 3 modules

| Module | Statut | Runtime |
|--------|--------|---------|
| `lib/shgat-tf/` | Actif | Deno + OpenBLAS FFI |
| `lib/gru/` | Actif | Node.js + TF.js |
| `lib/shgat-ob/` | Legacy | Deno (pas de training) |

### 5.2 Recommandation : architecture 3-tier avec adaptateur

```
shgat-tf (standalone)  →  shgat-for-gru (adaptateur 67D)  →  gru (séquenceur)
```

- `shgat-tf` reste autonome (scoring, training, benchmark)
- `shgat-for-gru` formalise le contrat 67D (capInput + compositeInput)
- `gru` importe l'adaptateur, pas shgat-tf directement
- `shgat-ob` → archiver dans `lib/_archived/`

**Arguments pour cette approche** :
- Runtimes différents (Deno + OpenBLAS vs Node.js + TF.js)
- Cycles de dev différents (SHGAT mature vs GRU actif)
- SHGAT utile standalone (LiveMCPBench, API discover, playground)
- Le contrat 67D formalise un couplage qui existe déjà de facto

---

## 6. DX et refactoring

### 6.1 Fait (2026-02-13)

- ~1850 lignes mortes supprimées du training stack SHGAT-TF
- 3 fichiers nettoyés, 1 fichier supprimé
- Tests mis à jour et passent (21/21)

### 6.2 Reste à faire

| Priorité | Item | Effort |
|----------|------|--------|
| **P0** | Attendre résultats training OB (epoch 5+) | — |
| P1 | Créer `shgat-for-gru` adaptateur (contrat 67D) | Moyen |
| P1 | Logging structuré dans `train-ob.ts` (batch accuracy, grad norms, ETA) | Faible |
| P2 | Archiver `shgat-ob` dans `lib/_archived/` | Faible |
| P2 | Disposer tensors entre folds k-fold GRU (OOM entre folds) | Moyen |
| P3 | Documentation formelle de l'interface 67D | Faible |
| P3 | Unifier config types entre les modules | Moyen |

### 6.3 shgat-ob → archiver

Consensus : `lib/shgat-ob/` a les 4 mêmes phases MP mais sans backward training. Déplacer vers `lib/_archived/shgat-ob/` avec README explicatif.

---

## 7. Prochaines étapes

### Immédiat

1. **Résultats train-ob.ts** — epoch 5 = premier eval
   - MP aide → investir backward MP, optimiser résiduals
   - MP n'aide pas → geler MP, focus K-head seul
2. **Créer `shgat-for-gru`** — formaliser le contrat 67D

### Court terme

3. **Intégrer les nouveaux poids SHGAT** entraînés sur 38K exemples → recalculer composite features
4. **Re-run k-fold GRU** avec les nouvelles features SHGAT
5. **Fix DAG ancestors** — 43.7% des contextes GRU contaminés par branches parallèles

### Décisions suspendues

| Décision | Dépend de | Échéance |
|----------|-----------|----------|
| MP : investir ou geler | Résultats epoch 5+ | Cette semaine |
| shgat-ob : supprimer | Validation backward MP | Après training |
| Fusion complète des libs | Stabilité des interfaces | Pas avant v1.0 |

---

## Annexe A — Les 5 inputs du GRU

| # | Input | Dimension | Source | Lien SHGAT |
|---|-------|-----------|--------|------------|
| 1 | contextInput | [20, 1024D] | Embeddings BGE-M3 outils en contexte | Embeddings partagés |
| 2 | transFeatsInput | [20, 5D] | Features structurelles (Jaccard, bigram) | Aucun |
| 3 | intentInput | 1024D | Embedding BGE-M3 intent | Embedding partagé |
| 4 | **capInput** | **64D** | **Fingerprint capabilities — hiérarchie L1+ SHGAT** | **Structurel (64D)** |
| 5 | **compositeInput** | **3D** | **scoreNodes() du SHGAT** | **Scoring (3D)** |
| | **Total SHGAT** | **67D** | | **44% couche fusion** |

## Annexe B — Glossaire

| Terme | Définition |
|-------|------------|
| **K-head** | Projection multi-tête Q·K/√dim pour scorer les outils |
| **MP** | Message Passing — propagation dans l'hypergraphe (4 phases) |
| **InfoNCE** | Loss contrastive : maximise score positif vs négatifs |
| **OB / OpenBLAS** | Bibliothèque d'algèbre linéaire, utilisée via FFI depuis Deno |
| **VocabNode** | Noeud unifié L0 (tool) / L1+ (capability) partagé SHGAT/GRU |
| **composite_features** | 3D : [bestScore, coverage, level] depuis SHGAT scoreNodes |
| **capInput** | 64D : fingerprint de capabilities depuis hiérarchie SHGAT L1+ |
| **67D coupling** | capInput (64D) + compositeInput (3D) = signal SHGAT dans GRU |
| **BGE-M3** | Modèle d'embedding 1024D (intent + outils) |
| **Focal CE** | Cross-entropy focale — pondère les exemples difficiles |
| **KL divergence** | Loss pour soft targets n8n |
