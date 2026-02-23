# Fondements Mathematiques de la Pipeline SHGAT-TF + GRU Compact Informed
## Et roadmap d'ameliorations

**Date** : 2026-02-10
**Auteurs** : Panel de 3 experts (Paper Analyst, TF.js Architect, Reporter)
**Scope** : Documentation des formules mathematiques, de l'architecture, et priorisation des ameliorations

---

## 1. Vue d'ensemble de l'architecture

La pipeline de retrieval et sequencement d'outils combine deux composants complementaires :

```
Intent (texte)
    |
    v
BGE-M3 encoder (1024D embedding)
    |
    +-----------------------------+
    |                             |
    v                             v
SHGAT-TF                    GRU Compact Informed
(scoring hierarchique)       (sequencement)
    |                             |
    | scoreNodes()                | buildPath()
    | → scores [N]                | → [tool_1, tool_2, ..., tool_k]
    | → compositeFeatures [3]     |
    |                             |
    +-----> compositeInput ------>+
    |                             |
    v                             v
Capability existante?        Nouveau path
    |                             |
    | Oui → execution directe    | Non → GRU construit la sequence
    +-----------------------------+
```

**Roles** :
- **SHGAT-TF** : scoring de pertinence intent→outils via attention multi-tete sur un graphe hierarchique. Fournit le vocabulaire enrichi et les composite features.
- **GRU Compact Informed** : prediction sequentielle du prochain outil et detection de terminaison. Consomme les composite features du SHGAT pour un spectre continu de delegation.

**Graphe hierarchique** (3 niveaux en production) :
```
Niveau 0 : Tools           (N_T ~ 644 outils)
Niveau 1 : MCP Servers     (N_1 ~ 212 capabilities L0)
Niveau 2 : Categories      (N_2 ~ 26 capabilities L1)
```

---

## 2. SHGAT-TF : Fondements mathematiques

### 2.1 Structure du graphe hierarchique

Le graphe est un n-SuperHyperGraph biparti (Smarandache, 2022) avec :

- `H_init` in R^{N_T x d} : embeddings des outils (d = 1024, BGE-M3)
- `E_init^(l)` in R^{N_l x d} : embeddings des capabilities au niveau l
- `A_TC` in {0,1}^{N_T x N_0} : matrice d'incidence Tool → Capability (niveau 0)
- `A_l` in {0,1}^{N_{l-1} x N_l} : matrice d'incidence Cap(l-1) → Cap(l)
- maxLevel = 2 en production

**Fichier** : `autograd-trainer.ts` (GraphStructure, lignes 52-64)

### 2.2 K-Head Attention Scoring

Le scoring calcule la pertinence de chaque noeud (outil ou capability) pour un intent donne, via K=16 tetes d'attention independantes.

**Parametres** :
- `W_k[h]` in R^{d x d_h} : matrice de projection par tete h (d=1024, d_h=64)
- `W_intent` in R^{d x d} : projection de l'intent (desactivee quand preserveDim=true)

**Formules** :

Pour chaque tete h in {0, ..., K-1} :
```
K_h = X @ W_k[h]                                    // [N, d_h]  keys
Q_h = q @ W_k[h]                                    // [d_h]     query (meme projection)

score_h[i] = (K_h[i] . Q_h) / (||K_h[i]|| * ||Q_h|| + eps)   // cosine similarity
```

Score final (mean pooling des tetes) :
```
score[i] = (1/K) * sum_{h=0}^{K-1} score_h[i]       // [N]
```

L'utilisation de la meme matrice `W_k[h]` pour la query et la key (Q et K partageant la projection) est un choix delibere qui reduit les parametres de moitie par rapport au GAT original (ou W_q != W_k).

**Fichiers** : `autograd-trainer.ts:469-503` (kHeadScoring), `khead-scorer.ts:55-106` (inference)

### 2.3 Message Passing hierarchique (upward/downward)

Le MP enrichit les embeddings avant le scoring via une propagation bidirectionnelle dans le graphe hierarchique.

#### Attention Aggregation (par tete)

Pour une paire source/target avec une matrice de connectivite C :

```
Projections :
  S_proj = S @ W_source[h]           // [N_s, d_h]
  T_proj = T @ W_target[h]           // [N_t, d_h]

Scores d'attention (split decomposition pour eviter O(N_s * N_t * 2*d_h)) :
  a[h] = [a_tgt | a_src]             // vecteur d'attention split en 2 x d_h
  score_tgt[t] = LeakyReLU(T_proj[t]) @ a_tgt        // [N_t]
  score_src[s] = LeakyReLU(S_proj[s]) @ a_src        // [N_s]
  scores[t,s] = score_tgt[t] + score_src[s]          // [N_t, N_s] broadcasting

Masquage et normalisation :
  scores[t,s] = -inf   si C^T[t,s] = 0
  alpha[t,s] = softmax_s(scores[t,s])                // row-stochastic

Aggregation :
  result[t] = ELU( sum_s( alpha[t,s] * S_proj[s] ) )    // [N_t, d_h]
```

**Multi-head** : Les K tetes sont concatenees → [N_t, K * d_h] = [N_t, 1024] (preserve la dimension d).

**Fichier** : `autograd-trainer.ts:185-267`

#### Upward Pass (V → E^0 → E^1 → ... → E^L)

```
Pour l = 0 a maxLevel :
  si l == 0 :
    E^0 = MultiHeadMP(H, E^0_init, A_TC, W_up[0], a_up[0], K)
  sinon :
    E^l = MultiHeadMP(E^{l-1}, E^l_init, A_l, W_up[l], a_up[l], K)
```

Chaque capability aggrege les features de ses outils/sous-capabilities via l'attention apprise.

#### Downward Pass (E^L → ... → E^0 → V)

```
dr = downwardResidual                    // defaut 0, controle le blend

Pour l = maxLevel-1 a 0 :
  E^l_prop = MultiHeadMP(E^{l+1}, E^l, A_{l+1}^T, W_down[l+1], a_down[l+1], K)
  E^l = (1 - dr) * E^l_prop + dr * E^l_avant_downward

Final (E^0 → V) :
  H_prop = MultiHeadMP(E^0, H, A_TC^T, W_down[1], a_down[1], K)
  H = (1 - dr) * H_prop + dr * H_avant_downward
```

Les outils recoivent l'information contextuelle de leurs capabilities parentes.

#### Post-MP Residual

```
pdr = preserveDimResidual               // defaut 0.3

H_final = (1 - pdr) * H_apres_MP + pdr * H_init
E^l_final = (1 - pdr_l) * E^l_apres_MP + pdr_l * E^l_init
```

Le residual preserve les embeddings BGE-M3 originaux pour eviter le collapse dimensionnel.

**Fichier** : `autograd-trainer.ts:275-454`

### 2.4 Training contrastif (focal CE + hard negatives)

#### InfoNCE Loss

```
L_InfoNCE = -log( exp(s_pos / tau) / sum_j exp(s_j / tau) )
```

ou `s_pos` est le score de l'outil correct, `s_j` les scores de tous les candidats (positif + negatifs), et `tau = 0.07` la temperature.

#### Batch Contrastive Loss (symmetric)

```
sim = (Q_norm @ K_norm^T) / tau           // [B, B]
L = 0.5 * ( CE(I, sim) + CE(I, sim^T) )  // symmetric InfoNCE
```

Chaque element du batch sert de negatif pour les autres. Avec B=32, cela fournit 31 negatifs par exemple.

#### Regularisation

```
L_total = L_contrastive + lambda * sum(||W||^2)
```

- lambda = 0.0001 (standard), 10x plus fort pour le projection head
- Gradient clipping : max norm = 1.0
- Adam optimizer : lr = 0.001

**Fichier** : `autograd-trainer.ts:558-619`

### 2.5 Nombre de parametres et complexite

| Composant | Dimensions | Parametres |
|-----------|-----------|------------|
| W_k (16 tetes) | 16 x [1024, 64] | 1,048,576 |
| W_intent | [1024, 1024] | 0 (skip si preserveDim) |
| W_up (3 niveaux x 16 tetes) | 3 x 16 x [1024, 64] | 3,145,728 |
| W_down (3 niveaux x 16 tetes) | 3 x 16 x [1024, 64] | 3,145,728 |
| a_up (3 niveaux x 16 tetes) | 3 x 16 x [128] | 6,144 |
| a_down (3 niveaux x 16 tetes) | 3 x 16 x [128] | 6,144 |
| residualWeights | [3] | 3 |
| **Total scoring (K-head seul)** | | **~1.05M** |
| **Total avec MP** | | **~7.35M** |

**Complexite compute** (pour N=800, M=70, d=1024, d_h=64) :
- K-head scoring : O(N * K * d * d_h) = O(800 * 16 * 1024 * 64) ~ 840M FLOPs
- MP par couche : O(N * M * d_h * K) ~ 57M FLOPs
- Total MP (3 niveaux up + 3 niveaux down) : ~340M FLOPs
- **Total : ~1.2 GFLOPs** → <2s en TF.js Node CPU

**Note importante** : En production, seul le scoring K-head est entraine (1.05M params). Le MP n'est pas entraine faute de donnees suffisantes (282 exemples pour 6.3M params MP = overfitting garanti). Le MP avec poids random degrade les performances (R@1 = 11.2% vs 16.4% flat).

---

## 3. GRU Compact Informed : Fondements mathematiques

### 3.1 Architecture (5 inputs, GRU(64), similarity head)

Le GRU est un modele sequentiel a 5 entrees et 2 sorties :

```
Entrees :
  contextInput      [B, S, 1024]    // S = maxSeqLen = 20
  transFeatsInput   [B, S, 5]       // 5 features de transition par timestep
  intentInput       [B, 1024]       // embedding intent
  capInput          [B, C]          // fingerprint capabilities (C ~ 212)
  compositeInput    [B, 3]          // features composites SHGAT

Branche contextuelle :
  input_proj(1024 → 128, linear)
  concat([proj_ctx, trans_feats]) → [B, S, 133]
  GRU(133 → 64, recurrentDropout=0.25) → [B, 64]

Branche statique :
  intent_proj(1024 → 64, ReLU) → [B, 64]
  cap_proj(C → 16, ReLU)       → [B, 16]
  comp_proj(3 → 8, ReLU)       → [B, 8]

Fusion :
  concat([GRU_out, intent, cap, composite]) → [B, 152]
  Dense(152 → 64, ReLU)
  Dropout(0.4)

Sorties :
  emb_proj(64 → 1024, linear) → similarity_head(1024 → numTools, softmax, FROZEN)
  termination_head(64 → 1, sigmoid)
```

**Fichier** : `gru-model.ts:100-265`

### 3.2 Prediction de sequence (next-tool + termination)

#### Cellule GRU (formulation standard Keras)

```
z_t = sigma(W_z @ x_t + U_z @ h_{t-1} + b_z)                  // gate de mise a jour
r_t = sigma(W_r @ x_t + U_r @ h_{t-1} + b_r)                  // gate de reset
h_tilde_t = tanh(W_h @ x_t + U_h @ (r_t * h_{t-1}) + b_h)     // candidat
h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde_t                   // etat final
```

Ou `x_t` in R^133 (concatenation de l'embedding projete [128] et des features de transition [5]) et `h_t` in R^64.

#### Similarity Head (frozen)

La tete de sortie est un produit scalaire dans l'espace d'embedding, non-trainable :

```
kernel = E^T / tau           // E in R^{numTools x 1024}, tau = temperature
P(tool_i | h) = softmax( emb_proj(h) @ kernel )_i
```

La temperature est annealee par cosine schedule :
```
T(epoch) = T_end + (T_start - T_end) * 0.5 * (1 + cos(pi * epoch / (R * epochs_total)))
```
Avec T_start = 0.20, T_end = 0.12, R (annealingStopRatio) = 0.7.

#### Structural Bias (post-inference)

Apres la prediction softmax, des biais structurels sont ajoutes en log-space :

```
log_p_adj[i] = log(p[i]) + alpha * Jaccard(last_tool, i) + beta * Bigram(last_tool, i)
p_adj = softmax(log_p_adj)
```

- alpha = 0.5 : poids de la similarite Jaccard (capabilities partagees)
- beta = 0.3 : poids de la frequence bigramme (co-occurrence dans les traces)

#### Features de Transition (5 par timestep)

```
[0] jaccardSim    : Jaccard(caps(tool_{t-1}), caps(tool_t))
[1] bigramFreq    : P(tool_t | tool_{t-1}) from training data
[2] sharedCaps    : |caps(tool_{t-1}) ∩ caps(tool_t)| / numCaps
[3] capNovelty    : |caps(tool_t) \ union(caps(context))| / |caps(tool_t)|
[4] positionInSeq : t / maxSeqLen
```

**Fichier** : `structural-bias.ts`

### 3.3 Beam search avec length normalization

#### Greedy Decoding

```
path = [first_tool]
pour step = 0 a maxPathLength-1 :
  probs, termProb = model.predict(intent, path)
  probs_adj = structuralBias(probs, path)
  probs_adj = stickyBias(probs_adj, path)     // penalise les repetitions >= 3
  next_tool = argmax(probs_adj)
  path.append(next_tool)
  si termProb > 0.5 : break
```

#### Beam Search

```
score_normalise = logProb / len^alpha           // alpha = 0.7

A chaque step, pour chaque candidat :
  Branche A (TERMINATE) : score += log(termProb)
  Branche B (CONTINUE) : score += log(1 - termProb) + log(topK_prob)

Garder les top-beamWidth candidats par score normalise.
Dedupliquer les paths identiques.
```

La normalisation par la longueur (`len^alpha`) est essentielle pour eviter que les paths courts dominent systematiquement (score cumule plus eleve car moins d'etapes de penalisation).

**Fichier** : `gru-model.ts:1076-1300`

### 3.4 Multi-task training (focal CE + KL divergence + termination BCE)

Le training est realise en deux passes de gradient separees.

#### Pass 1 : Next-tool (exclut termination_head du gradient)

**a) Focal CE avec label smoothing (exemples production)** :
```
y_smooth = (1 - eps) * onehot(target) + eps / numTools     // eps = 0.1
p_target = probs[target_index]
focal_weight = (1 - p_target)^gamma                        // gamma = 2.0
L_focal = focal_weight * (-sum(y_smooth * log(probs)))

L_nextTool = sum(L_focal * multiToolMask * prodMask) / sum(prodMask)
```

La focal loss (Lin et al., 2017) penalise davantage les exemples difficiles (p_target faible) que la CE standard. Avec gamma=2.0, un exemple facile (p=0.9) est pondere a 0.01 alors qu'un exemple difficile (p=0.1) est pondere a 0.81.

**b) KL Divergence (exemples n8n augmentation)** :
```
L_KL = sum( softTargets * log(softTargets / probs) ) / numN8n

L_pass1 = L_nextTool + w_n8n * L_KL       // w_n8n = 0.3
```

Les soft targets sont des distributions de probabilite construites par similarite cosine entre les embeddings n8n et les embeddings MCP, avec temperature T=0.005 et seuil cosine >= 0.70.

#### Pass 2 : Termination (tous les parametres, optimiseur separe)

```
L_term = -(wPos * y * log(p) + wNeg * (1-y) * log(1-p))
wPos = numNeg / numTotal                   // class balancing
wNeg = numPos / numTotal
L_pass2 = w_term * mean(L_term * prodMask)  // w_term = 10.0
```

La termination loss est masquee pour les exemples n8n (pas de signal de terminaison fiable).

**Fichier** : `gru-model.ts:583-830`

### 3.5 Data augmentation n8n (soft targets, temperature, filtrage)

#### Pipeline

1. **Scrape** : 7654 workflows n8n, 103213 edges, 811 types de noeuds
2. **Embeddings** : 2114 types uniques n8n, BGE-M3 1024D
3. **Filtrage** : exclusion des prefixes langchain, triggers, plumbing → 1978 types gardes
4. **Soft targets** : Pour chaque transition n8n (node_A → node_B), calculer :
   ```
   sim[i] = cosine(emb_n8n_B, emb_mcp_i)    // pour chaque outil MCP i
   softTarget[i] = exp(sim[i] / T) / sum_j exp(sim[j] / T)    // T = 0.005
   ```
   Seuil : conserver uniquement si max(sim) >= 0.70
5. **Training** : exemples n8n (KL) + exemples prod 3x oversample (focal CE)

#### Resultats (seed=42, n8n v1 = 2465 exemples)

| Config | Next-tool | Beam@3 exact | Beam@3 tools |
|--------|-----------|-------------|-------------|
| Baseline prod | 34.6% | 37.3% | 45.1% |
| +n8n w=0.3 | 40.4% | **52.9%** | **58.8%** |
| +n8n w=0.05 | **44.2%** | 41.2% | 47.1% |

Config recommandee : w=0.3 (beam@3 = metrique prod, +15.6 pts).

**Fichiers** : `lib/gru/src/n8n/` (scrape-n8n.ts, embed-n8n-nodes.ts, build-soft-targets.ts)

### 3.6 Nombre de parametres GRU

| Couche | Dimensions | Parametres |
|--------|-----------|------------|
| input_proj | Dense(1024, 128) | 131,200 |
| GRU | input=133, hidden=64 | 37,824 |
| intent_proj | Dense(1024, 64) | 65,600 |
| cap_proj | Dense(212, 16) | 3,408 |
| comp_proj | Dense(3, 8) | 32 |
| fusion_dense | Dense(152, 64) | 9,792 |
| emb_proj | Dense(64, 1024) | 66,560 |
| termination_head | Dense(64, 1) | 65 |
| **Total trainable** | | **~258K** |
| similarity_head (FROZEN) | Dense(1024, ~645) | ~660K |

---

## 4. Pipeline combinee : GRU wraps SHGAT

### 4.1 Role de chaque composant

| Composant | Role | Entree | Sortie |
|-----------|------|--------|--------|
| **BGE-M3** | Encodage de l'intent et des outils | Texte | Embedding 1024D |
| **SHGAT scoring** | Pertinence intent → noeuds hierarchiques | Intent + graphe | Scores [N], compositeFeatures [3] |
| **GRU sequencement** | Prediction du prochain outil + terminaison | Intent + contexte + SHGAT features | Sequence d'outils |
| **Structural bias** | Ajustement post-inference (Jaccard, bigram) | Predictions GRU + matrices statiques | Predictions ajustees |

Le SHGAT fournit trois choses au GRU :
1. **Les embeddings** des outils (colonnes de la similarity_head frozen)
2. **Les composite features** [bestScore, coverage, level] — spectre continu qui remplace le seuil binaire
3. **Le vocabulaire** des capabilities connues (futures : integration dans le vocabulaire GRU)

### 4.2 GRU-first vs SHGAT-first (pourquoi GRU gagne)

Le benchmark E2E (seed=42, 30 traces test) montre un ecart decisif pour le choix du 1er outil :

| Mode | 1er outil @1 | 1er outil @3 | Greedy exact | Beam@3 exact |
|------|-------------|-------------|-------------|-------------|
| SHGAT-first | 16.7% | 30.0% | 3.3% | 6.7% |
| **GRU-first** | **70.0%** | **86.7%** | **36.7%** | 23.3% |

**Explication mathematique** :

Le SHGAT optimise `argmax_i cos(W_k @ q, W_k @ e_i)` — la similarite semantique globale entre l'intent et chaque outil. Pour l'intent "Read deno.json and hash the content", le SHGAT pousse `hash_checksum` en #1 (plus proche semantiquement de l'intent complet) alors que le 1er outil logique est `read_file`.

Le GRU avec contexte vide (ctx=[]) apprend `P(tool_1 | intent, ctx=[])` a partir des sequences reelles. Son hidden state `h_0 = GRU(zeros)` est un vecteur fixe qui represente "debut de sequence". La fusion avec `intent_proj` permet au GRU de choisir l'outil qui DEMARRE les sequences similaires, pas celui qui est semantiquement le plus proche de l'intent global.

### 4.3 Capabilities comme vocabulaire unifie

La pipeline de production recommandee integre les capabilities comme raccourcis :

```
1. SHGAT scoreNodes(intent) → si bestCompositeScore > seuil_haut
     → executer la capability directement (court-circuit)
2. Sinon : GRU(ctx=[], compositeFeatures) → 1er outil
3. GRU(iterations) → suite du path
4. Enregistrer la nouvelle sequence comme capability
```

Le spectre continu (compositeInput) remplace le seuil binaire : la tete de terminaison du GRU apprend QUAND le composite suffit, eliminant la decision fragile par seuil.

---

## 5. Resultats benchmarks actuels

### 5.1 SHGAT-TF (LiveMCPBench, 5-fold CV, seed=42)

| Config | R@1 | R@3 | R@5 | NDCG@5 |
|--------|-----|-----|-----|--------|
| Cosine baseline | 14.4% | 29.6% | 35.3% | 32.5% |
| K-head flat (no MP) | 14.7% | 27.1% | 35.4% | 31.9% |
| **K-head flat trained (B-Flat)** | **16.4%** | **37.6%** | **45.1%** | **44.6%** |
| K-head hier trained (B-Hier, PDR .5) | 11.2% | 23.8% | 29.5% | 29.2% |
| Production (easy, 180 queries) | 99.4% | — | — | — |

### 5.2 GRU Compact Informed (seed=42, split par trace)

| Config | Next-tool @1 | Termination | Beam@3 exact | Beam@3 tools |
|--------|-------------|------------|-------------|-------------|
| Baseline prod | 34.6% | 70.1% | 37.3% | 45.1% |
| +n8n w=0.3 | 40.4% | 69.2% | **52.9%** | **58.8%** |
| +n8n w=0.05 | **44.2%** | 69.2% | 41.2% | 47.1% |

### 5.3 Pipeline E2E (seed=42)

| Metrique | Valeur |
|----------|--------|
| Hit@1 / Hit@3 / MRR | 64.9% / 81.9% / 0.743 |
| GRU-first greedy exact | 35.7% (28 traces) |
| GRU-first beam@3 exact | 39.3% |
| SHGAT-first @1 → E2E | 17.9% → 0-7% |
| GRU 1er outil (ctx=[]) @1 | 70.0% |

### 5.4 Limites statistiques

Avec N=52 exemples test, l'ecart-type d'une proportion binomiale est `sigma = sqrt(p(1-p)/N)`. Pour p=0.35 : sigma = 6.6%, IC 95% = [22%, 48%]. La difference baseline (37.3%) vs +n8n (52.9%) = +15.6 pts n'est **pas statistiquement significative** avec un seul split. K-fold CV est un prerequis pour valider ces resultats.

---

## 6. Roadmap d'ameliorations

### Vue d'ensemble par priorite

| Priorite | # | Amelioration | Effort | ROI attendu | Risque |
|----------|---|-------------|--------|------------|--------|
| **P0** | 1 | Fix DAG ancestors | 1-2j | +5-10 pts next-tool | Faible |
| **P0** | 2 | K-fold cross-validation | 1j | 0 pts (prerequis) | Nul |
| **P1** | 3 | Capabilities dans vocabulaire GRU | 2-3j | +5-15 pts E2E | Moyen |
| **P1** | 4 | Tete termination separee | 1-2j | +2-5 pts term | Moyen |
| **P2** | 5 | Intent paraphrasing | 2-3j | +3-8 pts next-tool | Moyen |
| **P2** | 6 | Full softmax SHGAT | 0.5j | +1-3 pts R@1 | Faible |
| **P2** | 7 | ToolBench pre-training | 3-5j | +5-15 pts R@1 SHGAT | Eleve |
| **P3** | 8 | Online learning | 3-5j | Graduel | Eleve |

---

### P0-1 : Fix DAG Ancestors (linearisation topologique)

**Probleme** : 43.7% des contextes GRU sont contamines par des outils de branches paralleles du DAG. Pour une execution parallele `[A, B] → [C, D]`, le contexte peut voir `[A, C, B, D]` au lieu du vrai historique causal.

**Justification mathematique** : Le GRU encode l'historique via `h_t = GRU(x_t, h_{t-1})`. Si `x_t` vient d'une branche parallele sans relation causale avec `x_{t-1}`, le signal pollue la gate de reset `r_t` et cree des correlations spurieuses. Sur 52 exemples test, ~23 ont des contextes contamines. Si la moitie causent une erreur corrigible : ~+10 pts next-tool.

**Implementation** : Lineariser le DAG par tri topologique. Pour chaque noeud, son contexte = ses ancetres transitifs (parents recursifs), pas tous les noeuds executes avant.

**Effort** : 1-2 jours | **ROI** : +5-10 pts | **Risque** : Faible (changement isole dans la preparation des donnees)

---

### P0-2 : K-fold Cross-Validation

**Probleme** : N=52 exemples test. IC 95% = +/-13 pts. Aucune comparaison n'est statistiquement significative.

**Justification mathematique** : K-fold avec K=5 reduit la variance de l'estimateur :
```
Var(p_kfold) ≈ Var(p_single) / K
```
L'IC 95% passe de +/-13 pts a +/-6 pts. Cela permet enfin de distinguer les vrais gains du bruit.

**Implementation** : Boucler train+eval sur 5 folds de ~43 traces. Le code de split seeded existe deja.

**Effort** : 1 jour | **ROI** : 0 pts direct, mais prerequis pour toute conclusion fiable | **Risque** : Nul

---

### P1-3 : Capabilities comme items du vocabulaire GRU

**Probleme** : Le GRU predit parmi 644 tools uniquement. Les 126 capabilities multi-tool existantes ne sont pas dans le vocabulaire. Quand l'intent matche une capability connue, le GRU reconstruit la sequence outil par outil au lieu de raccourcir.

**Justification mathematique** : Si une capability `c` a un embedding `e_c` proche de l'intent `q`, alors dans la similarity head : `cos(emb_proj(h), e_c) > cos(emb_proj(h), e_t)` pour tout tool individuel. Le GRU predit la capability en 1 step au lieu de N steps.

**Implementation** :
1. `setToolVocabulary()` accepte tools + caps (770 items au lieu de 644)
2. La similarity_head passe de [1024, 644] a [1024, 770] (+126 colonnes, toujours frozen)
3. `buildPath()` : si prediction = capability → expand `tools_used` dans le contexte
4. Generer des exemples dual-level (L0 tools + L1 caps)

**Parametres supplementaires** : 0 (la similarity_head est frozen ; seul emb_proj reste a [64, 1024]).

**Effort** : 2-3 jours | **ROI** : +5-15 pts E2E exact | **Risque** : Moyen (equilibre caps/tools dans le training)

---

### P1-4 : Tete termination separee

**Probleme** : Deux optimiseurs Adam s'appliquent aux memes parametres partages (GRU, fusion_dense). Les moments du 1er ordre sont incoherents :
```
optimizer1 : m1_t = beta1 * m1_{t-1} + (1-beta1) * g_nextTool
optimizer2 : m2_t = beta1 * m2_{t-1} + (1-beta1) * g_term
```

**Solution** : Brancher la tete termination depuis `[gruOutput, intentProj]` directement (pas `fusionDropout`). Un MLP dedie [128→32→1] isole completement les gradients. Un seul optimiseur Adam avec loss combinee `L = L_next + lambda * L_term`.

**Parametres supplementaires** : Dense(128→32) + Dense(32→1) = ~4.2K (+1.6%).

**Effort** : 1-2 jours | **ROI** : +2-5 pts termination | **Risque** : Moyen (re-entrainement complet)

---

### P2-5 : Intent Paraphrasing

**Probleme** : 457 exemples prod pour 258K params = ratio 565 params/exemple (zone d'overfitting).

**Justification mathematique** : Avec K=5 paraphrases par trace, le ratio passe a 258K/2285 = 113. L'ecart de generalisation diminue comme `O(sqrt(params / N))`, soit une reduction d'un facteur `sqrt(5) = 2.24`.

**Implementation** : Paraphrase LLM (gpt-5-mini) + re-embedding BGE-M3. Memes tool sequences, seul l'intent change.

**Effort** : 2-3 jours | **ROI** : +3-8 pts next-tool | **Risque** : Moyen (qualite des paraphrases)

---

### P2-6 : Full Softmax SHGAT

**Probleme** : InfoNCE avec ~8 negatifs. Le signal gradient est borne par `log(K+1) = log(9) = 2.2 nats`.

**Justification mathematique** : Avec full softmax (K=524) : borne = `log(525) = 6.3 nats`. Le gradient est 3x plus informatif. Cependant, le bottleneck reste le nombre de positifs (282), pas les negatifs.

**Implementation** : Modifier `trainStep()` pour passer les 525 tools comme candidats.

**Effort** : 0.5 jour | **ROI** : +1-3 pts R@1 | **Risque** : Faible

---

### P2-7 : ToolBench Pre-training

**Probleme** : 282 exemples LiveMCPBench = bottleneck fondamental du SHGAT.

**Justification mathematique** : ToolBench offre ~16K queries avec des hierarchies Category → Tool → API. Le pre-training contrastif sur 16K queries fournit ~128K comparaisons uniques par epoch (vs 2,256 avec 282 ex). Le fine-tuning adapte ensuite a notre catalogue MCP.

**Effort** : 3-5 jours | **ROI** : +5-15 pts R@1 SHGAT | **Risque** : Eleve (domain gap ToolBench vs MCP)

---

### P3-8 : Online Learning

**Probleme** : Modele statique, les nouvelles traces ne sont pas utilisees.

**Justification mathematique** : Online learning avec PER buffer. Le regret cumule est `O(sqrt(T * log(B)))` au lieu de `O(T)` pour un modele statique. Cependant, ~10 traces/jour est trop peu pour un impact mesurable a court terme. Risque de catastrophic forgetting.

**Effort** : 3-5 jours | **ROI** : +1-3 pts/mois (graduel) | **Risque** : Eleve

---

### Plan d'execution recommande

```
Sprint 1 (2-3 jours) — Base fiable
├── P0-1 : Fix DAG ancestors
└── P0-2 : K-fold cross-validation
    → Prerequis pour mesurer tout le reste

Sprint 2 (3-4 jours) — Gains structurels
├── P1-3 : Capabilities dans vocabulaire GRU
└── P1-4 : Tete termination separee
    → Gain cumule estime : +10 a +20 pts beam@3 E2E

Sprint 3 (3-5 jours) — Data augmentation
├── P2-5 : Intent paraphrasing
│   OU
└── P2-7 : ToolBench pre-training
    → Choisir selon resultats du sprint 2

Sprint 4 (continu) — Optimisations incrementales
├── P2-6 : Full softmax SHGAT
└── P3-8 : Online learning
```

**Gain cumule estime** apres sprints 1-2 : de ~53% beam@3 E2E a ~65-78%.

---

## Annexe : Formules de reference

### A.1 — K-Head Scoring SHGAT

```
score[i] = (1/K) * sum_{h=0}^{K-1} cos(W_k[h] @ q, W_k[h] @ x_i)
```

### A.2 — Attention Aggregation (MP, par tete)

```
scores[t,s] = LeakyReLU(T_proj[t]) @ a_tgt + LeakyReLU(S_proj[s]) @ a_src
alpha[t,s] = softmax_s( scores[t,s] * mask(C) )
result = ELU( alpha @ S_proj )
```

### A.3 — GRU Cell

```
z_t = sigma(W_z x_t + U_z h_{t-1})
r_t = sigma(W_r x_t + U_r h_{t-1})
h_t = (1-z_t) * h_{t-1} + z_t * tanh(W_h x_t + U_h (r_t * h_{t-1}))
```

### A.4 — Focal CE Loss

```
L_focal = (1 - p_target)^gamma * CE(y_smooth, probs)
```

### A.5 — KL Divergence (n8n soft targets)

```
L_KL = sum( q * log(q / p) )     // q = soft target, p = predicted
```

### A.6 — InfoNCE

```
L_InfoNCE = -log( exp(s_pos / tau) / sum_j exp(s_j / tau) )
```

### A.7 — Beam Score Normalization

```
score_normalise = sum(log_probs) / len^alpha     // alpha = 0.7
```

### A.8 — Temperature Annealing (cosine)

```
T(t) = T_end + (T_start - T_end) * 0.5 * (1 + cos(pi * t / (R * T_total)))
```

---

## Annexe : Fichiers de reference

| Fichier | Contenu |
|---------|---------|
| `lib/shgat-tf/src/training/autograd-trainer.ts` | Training SHGAT, message passing, loss functions |
| `lib/shgat-tf/src/attention/khead-scorer.ts` | K-head scoring inference |
| `lib/shgat-tf/src/core/builder.ts` | Builder pattern SHGAT |
| `lib/shgat-tf/src/core/types.ts` | SHGATConfig, hierarchie, parametres |
| `lib/gru/src/transition/gru-model.ts` | Architecture GRU, training, beam search |
| `lib/gru/src/transition/types.ts` | CompactGRUConfig, TransitionExample |
| `lib/gru/src/transition/structural-bias.ts` | Features de transition, Jaccard, bigram |
| `lib/gru/src/n8n/build-soft-targets.ts` | Pipeline n8n soft targets |
| `lib/gru/src/benchmark-e2e.ts` | Pipeline benchmark E2E |
| `lib/gru/docs/2026-02-10-benchmark-e2e-results.md` | Resultats benchmark |

---

*Document genere le 2026-02-10 par le panel d'experts SHGAT-TF + GRU Compact Informed.*
*Sources : code source (lib/shgat-tf, lib/gru), benchmarks internes (LiveMCPBench, E2E seed=42), panels precedents.*
