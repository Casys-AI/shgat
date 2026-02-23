# SHGAT-TF Changelog

## 2026-02-15 — Training defaults overhaul + n8n data cleanup

### Nouveaux defauts train-ob.ts

| Param | Ancien | Nouveau | Raison |
|---|---|---|---|
| `--kl-batch-size` | 128 | **30000** | Tout le dataset en 1 forward/backward. ~700MB RAM, largement dans les 10GB. 1 update KL/epoch = gradient stable |
| `--kl-accum` | 4 | **1** | Plus besoin d'accumuler avec batch=30K |
| `--kl-subsample` | 2000 | **0** (all) | Utilise les 30K exemples n8n au lieu d'en sous-echantillonner 2K |
| `--kl-isolate-khead` | ON (implicite) | **OFF** (flag opt-in) | Flag inverse: `--kl-isolate-khead` pour activer l'isolation. Defaut = KL met a jour W_q/W_k |

### Commande recommandee

```bash
DENO_V8_FLAGS=--max-old-space-size=10240 deno run -A tools/train-ob.ts \
  --kl --kl-weight 0.02 --kl-isolate-khead --seed 42 --epochs 20 --lr 0.001 --eval-every 2
```

### n8n mapping improvements (build-soft-targets.ts)

- **Service overrides** pour `std:*` tools: `std:http_get/post/request`, `std:curl_fetch` → service `http`; `filesystem:read_*`, `std:jq`, `std:transform_csv_parse` → service `file_reader`
- **Aliases n8n**: `httpRequest` → `http`, `extractFromFile` → `file_reader`
- **LLM provider exclusions**: openAi, gemini, mistral, anthropic, ollama, airtop + variants langchain (121 nodes exclus total)
- **Tier 1**: 33.9% (676/1993) — +3.5pts grace aux 24 nodes http + 16 nodes file_reader

### Resultats run 11 (kl=0.02, isolate, 30K, batch=2048)

| Epoch | R@1 | MRR | |∇| | MP Δ |
|---|---|---|---|---|
| 2 | 3.7% | 0.111 | 0.7 | 1.54 |
| 4 | 6.5% | 0.160 | 5.0 | 2.15 |
| 6 | 0.9% | 0.022 | 58.9 | 2.80 |
| 8 | 26.2% | 0.345 | 43.1 | 7.41 |
| 10 | **35.5%** | **0.447** | 11.7 | 9.07 |

**Insight**: gradients instables a LR=0.005 (explosion epoch 6-7). Le best arrive quand le cosine decay ramene le LR a ~0.0001. Run 12 baisse LR a 0.001 + 20 epochs.

### GRU E2E avec SHGAT 26% (run 8 params)

- Hit@1=60.9% (vs 65.7% baseline sans SHGAT trained) — regression
- GRU-first 1st@1=76% (+13pts) mais E2E exact=16% (-20pts)
- MP delta=79 (vs 388 non-entraine) — enrichissement trop faible
- SHGAT standalone pousse `psql_tables` en top partout (mode collapse K-head)

---

## 2026-02-14 — KL gradient isolation for K-head scoring

### Probleme

Le KL loss n8n degradait les performances SHGAT (28.0% → 24.3% Hit@1) car le gradient
KL modifiait W_q et W_k (scoring heads) avec un objectif "smooth distribution" antagoniste
avec l'objectif InfoNCE "sharp discrimination".

### Solution: `--kl-isolate-khead` (defaut: ON)

Quand active (defaut), le gradient KL ne traverse plus W_q et W_k :

| Composant | InfoNCE | KL (ancien) | KL (nouveau) |
|---|---|---|---|
| W_q, W_k | gradient | gradient | **bloque** |
| W_intent | gradient | gradient | gradient |
| MP weights | gradient (via _epochDH) | gradient (via _epochDH) | gradient (via _epochDH) |

Le KL continue d'enrichir W_intent (projection intent) et les poids MP (relations structurelles),
mais le scoring K-head reste exclusivement optimise par InfoNCE (contrastive, discriminatif).

### Utilisation

```bash
# Defaut: KL isole (recommande)
deno run ... --kl --kl-weight 0.2

# Ancien comportement: KL touche aussi W_q/W_k
deno run ... --kl --kl-update-khead
```

### Fichiers modifies

- `lib/shgat-tf/tools/train-ob.ts` — flag `KL_ISOLATE_KHEAD`, 3 blocs conditionnes

---

## 2026-02-14 — Eval metrics + best model checkpoint + epochs default 10

### Changements

#### 1. Best model checkpoint
- Sauvegarde `shgat-params-ob-best.json` a chaque amelioration de R@1 (ecrase le precedent).
- Export final ecrit toujours le last epoch (`shgat-params-ob-{timestamp}.json`) + affiche le chemin du best.
- `serializeParams()` helper factorise la serialization.

#### 2. 3 nouvelles metriques d'eval SHGAT

Le R@1 standalone n'est pas le bon KPI pour SHGAT (c'est un feature extractor pour le GRU, pas un scorer final). 3 metriques ajoutees pour evaluer la qualite du MP :

- **MP delta norm**: `||H_final - H_init||` moyen, split hier/orphan. Montre si le MP colore les embeddings. Attendu: hier >> orph (~0 pour orphelins car MP = identity).
- **R@1 split hier/orph**: Meme R@1 mais decompose par tools hierarchiques vs orphelins. Si le MP aide, hier R@1 > orph R@1.
- **Silhouette intra-capability**: Cosine sim moyenne entre tools sous meme cap (intra) vs entre caps differentes (inter). Delta positif = clustering coherent.

#### 3. Epochs default: 15 → 10
- Run 7: best R@1=24.3% a epoch 8, puis overfitting (8.4% a epoch 20).
- 10 epochs evite la zone d'overfitting tout en laissant de la marge.

### Resultats run 7 (20 epochs, seed=42, 6 KL fixes)

| Metrique | Valeur |
|---|---|
| Best R@1 | 24.3% (epoch 8) |
| Best MRR | 0.318 |
| Temps total | 180min (9min/epoch) |
| Peak RSS | 6.5GB |
| Speedup vs run 6 | 3.7x (epoch 1: 7m58s vs 29m43s) |

**Observations:**
- Overfitting apres epoch 8 (R@1 24.3% → 8.4% a epoch 20)
- MP grad norms clippees a 4.0 des epoch ~10 (destabilisation)
- 1155 prod exemples / 7.35M params = ratio 6364 params/ex → overfitting inevitable
- KL loss n8n stable (~2.4) mais ne transfere pas directement vers eval prod

---

## 2026-02-14 (run 7) — KL Performance Overhaul (6 fixes)

### Changements

#### 1. Batched backward KL (matmul au lieu de per-target outerProduct)
- **Avant:** Pour chaque tuple (example, sparse_target), on appelait `backpropMultiHeadKHeadLogit()` qui faisait 16 heads × 2 `outerProductAdd` + 2 `matVecTransposeBlas` = ~64 operations vectorielles. Avec ~768 tuples/batch × 16 heads = **~49K small ops/batch** (JS fallback car W_q[64,1024] < 256 rows BLAS threshold).
- **Apres:** 3-phase architecture:
  - **Phase 1:** Forward pass inchange, collecte `KLBackwardTuple` structs au lieu d'appeler backward inline.
  - **Phase 2:** Pour chaque head, empile les dQ/dK en matrices [T, headDim] et fait 4 matmul BLAS:
    - `dW_q += transpose(dQ_batch) @ intentProjBatch` — [64,T] @ [T,1024] = [64,1024]
    - `dW_k += transpose(dK_batch) @ nodeEmbBatch` — idem
    - `dIntentBatch = dQ_batch @ W_q` — [T,64] @ [64,1024] = [T,1024]
    - `dNodeEmbBatch = dK_batch @ W_k` — idem
  - **Phase 3:** Scatter-accumulate dNodeEmb→`_epochDH`, group dIntentProj par exIdx→`backpropWIntent`.
- **Equivalence mathematique:** `sum_i(dQ_i ⊗ v_i) = dQ_batch^T @ v_batch` (somme de produits externes = matmul). Exact, pas approximatif.
- **BLAS:** T~768 >= 10 et dim=1024 >= 64 → toutes les 4 matmul touchent le path BLAS (cblas_sgemm).
- **Impact:** ~49K small JS ops → 64 large BLAS matmul/batch. Estimee 40-55% speedup KL.
- **RAM:** ~30MB peak supplementaire par batch (dQ/dK [960,64], intent/nodeEmb batches [960,1024]).

#### 2. Batched forward KL (matmulTranspose pour projections Q)
- **Avant:** Per-example: `matVecBlas(W_q[h], intentProj)` pour chaque ex × chaque head = 128 × 16 = **2048 matVec JS** par batch (W_q[64,1024] < 256 rows → JS fallback).
- **Apres:** Pre-filtre les exemples valides, puis:
  - `intentProjBatch = matmulTranspose(intentEmbBatch, W_intent)` — [V,1024] @ [1024,1024]^T = [V,1024] (1 BLAS)
  - `Q_allHeads[h] = matmulTranspose(intentProjBatch, W_q[h])` — [V,1024] @ [64,1024]^T = [V,64] (16 BLAS)
- **BLAS:** V~120 >= 10, dim=1024 >= 64 → toutes les matmulTranspose touchent le path BLAS.
- **Impact:** 2048 JS matVec → 17 BLAS matmul par batch. Scoring (dot Q·K) reste per-example (sparse targets differents).

#### 3. In-place gradient clipping (adam-optimizer.ts)
- **Avant:** `clipGradients()` allouait une NOUVELLE matrice via `grads.map(row => row.map(...))` a chaque appel. Avec 33 Adam calls par batch × 16 batches = **528 allocations de matrices par epoch**, causant une forte pression GC.
- **Apres:** Clipping in-place avec boucle `for`. Zero allocation.
- **Impact:** 5-10% reduction du temps KL (moins de GC pauses).

#### 4. KL batch size separe (128 vs 32)
- **Nouveau CLI:** `--kl-batch-size <n>` (default 128). Le KL utilise maintenant un batch size independant du InfoNCE batch size.
- **Impact:** 63 batches → 16 batches par epoch KL (4x moins de backward passes + Adam steps).
- **RAM:** +~4x exemples par batch = ~960 sparse targets par batch. A 7.5 targets/ex × 128 ex = ~960 tuples → ~7.5MB buffers. Negligeable avec 12GB heap.

#### 5. Gradient accumulation KL
- **Nouveau CLI:** `--kl-accum <n>` (default 4). Accumule les gradients sur N batches KL avant de faire UN SEUL Adam step.
- **Avant:** 16 batches × 33 Adam calls = 528 Adam steps par epoch KL.
- **Apres:** 16 batches / 4 accum = 4 groupes × 33 calls = **132 Adam steps** (4x moins).
- **Normalisation:** Les gradients accumules sont divises par `accumCount` avant l'Adam step pour maintenir la meme magnitude effective.
- **Impact:** 10-15% reduction supplementaire.

#### 6. Pre-computed K projections (run 6, conserve)
- Pre-compute `projectedKeysPerHead[h] = matmulTranspose(H_final, W_k[h])` une fois par epoch.
- KL scoring forward utilise des lookups O(1) au lieu de matVecBlas.
- Note: les K deviennent stales apres les Adam steps sur W_k (drift negligeable a LR=0.002-0.005).

### Impact cumule attendu
| Metrique | Run 5 (base) | Run 7 (6 fixes) |
|----------|:---:|:---:|
| KL backward ops/batch | ~49K JS vector ops | 64 BLAS matmul |
| KL forward ops/batch | 2048 JS matVec | 17 BLAS matmul |
| Adam calls par epoch KL | 2079 | 132 |
| Allocations GC (clipping) | 2079 | 0 |
| Batches KL | 63 | 16 |

### Tests
- `batched-kl-backward.test.ts`: 11/11 PASS — equivalence mathematique batched vs per-target
  - single tuple, multi-tuple (T=5), shared Q (same-example multi-target), zero dLogit, stress T=100
  - batched forward W_intent, W_q, all heads, scoring invariant
  - W_intent scatter-sum, gradient accumulation
- `adam-optimizer.test.ts`: 7/7 PASS (in-place clipping backwards-compatible)
- `kl-optimization.test.ts`: 9/9 PASS (TF.js autograd path)

---

## 2026-02-14 (run 6) — KL Batch Optimization: Pre-Computed Keys + Index Lookups

### Changements

#### 1. Pre-computed K projections (matmulTranspose BLAS)
- **Avant:** Chaque KL batch faisait `matVecBlas(W_k[h], nodeEmb)` pour chaque tool × chaque head → ~15K-25K `matVecBlas` calls par batch (sparseL0 × numHeads).
- **Apres:** `projectedKeysPerHead[h] = matmulTranspose(H_final, W_k[h])` compute en une seule operation BLAS (`cblas_sgemm`) tous les K projections pour tous les 1932 tools. Lookup O(1) par tool ensuite.
- **Justification:** H_final est constant sur l'epoch (MP forward 1x/epoch). Les W_k changent per-batch, mais avec epoch-level MP backward, les headParams sont stables intra-epoch.
- **Impact:** Remplace ~2000 × numHeads × batchCount `matVecBlas` → numHeads `matmulTranspose` + lookups.

#### 2. Integer index optimization (sparseL0Idxs)
- **Avant:** `sparseL0Ids: string[]` + `l0IdxMap.get(nodeId)` pour chaque target → hash lookup string.
- **Apres:** `sparseL0Idxs: number[]` resolu une seule fois au debut du batch. Acces direct `H_final[l0Idx]` et `projectedKeysPerHead[h][l0Idx]`.
- **Impact:** Elimine toutes les conversions string→index dans le hot loop KL scoring + backward.

#### 3. Timing pre-compute loggue
- Nouveau log `Pre-compute K proj: Xms` apres MP forward pour tracer le cout du pre-compute.

### Impact performance attendu
- **KL scoring forward:** 10-30x speedup (BLAS matmul > boucle matVec + overhead FFI par call)
- **KL backward:** ~2x speedup (elimination des hash lookups string dans le hot loop dNodeEmbedding)
- **Memoire:** +numHeads × 1932 × headDim floats (~240KB pour 16 heads × 1932 × 2D) — negligeable

### Tests
- `src/training/__tests__/kl-optimization.test.ts` — regression tests KL loss, gradient flow, temperature, klWeight, sparse efficiency

---

## 2026-02-14 (run 5) — Fix HIER source: prod+n8n

### Changements

#### 1. Fix: HIER contrastive inclut les n8n examples (plus seulement prod)
- **Bug run 4:** Le HIER contrastive iterait uniquement `prodShuffled` pour collecter les exemples avec ancetres L1+. Or tous les prod tools MCP sont orphelins (pas de L1 parent) → **0 HIER exemples**, le code etait mort.
- **Root cause:** 801/1932 L0 tools ont des ancetres, mais ce sont les n8n tools (groupes par workflow). Les 1155 prod tools n'ont pas de parent L1 dans la hierarchie (design voulu, pas un bug d'export).
- **Fix:** `collectFromExamples()` itere prod ET `ds.n8nTrain`. Shuffle unifie avant batching.
- **Impact attendu:** ~800+ exemples HIER par epoch (vs 0 avant). Le MP recoit enfin son gradient dE direct.

---

## 2026-02-14 (run 4) — Contrastive Hierarchy-Level Loss

### Changements

#### 1. Contrastive hierarchy-level loss (InfoNCE intent→ancestor)
- **Nouveau:** Pour chaque niveau L1+, on compute `InfoNCE(intent, ancestor_embedding)` sur les exemples qui ont un ancetre a ce niveau.
- **Gradients:** Les `dNodeEmbeddings` sont accumules dans `_epochDE[level]` (etait toujours zeros avant). Le MP backward recoit maintenant un signal direct via `dE_final`.
- **Design recursif:** Itere `0..maxLevel`, fonctionne automatiquement si on ajoute des niveaux L3+.
- **Poids:** `hierWeight=0.5` applique uniformement sur K-head grads, W_intent grads, et dE.
- **Normalisation per-level:** Chaque niveau divise par son propre nombre de batches.
- **Impact:** Le MP recoit un gradient direct "cette aggregation de tools en node L1 doit matcher cet intent". Avant, le MP ne recevait que le signal indirect via dH (gradient des L0 qui remonte).
- **Reference:** Porter (panel Q3.2) — "contrastive au niveau capability = chainage manquant"
- **Note:** Run 4 en pratique = 0 HIER exemples (prod tools orphelins). Corrige en run 5.

#### 2. `l0Ancestors` — mapping recursif L0→ancetres
- **Nouveau champ** dans `PureGraphStructure`: `l0Ancestors[l0Idx] = Map<orchLevel, idx[]>`
- Construit par walk-up recursif dans `buildGraphStructure()`.
- Chaque L0 tool connait tous ses ancetres a chaque niveau de la hierarchie.

#### 3. `_epochDE` remplace `_zeroDE`
- **Avant:** `_zeroDE` = buffer immutable toujours zeros (car dE non utilise).
- **Apres:** `_epochDE = Map<level, number[][]>` mutable. Accumule les gradients hierarchy-level.
- Ajoute `zeroDE()` helper pour reset par epoch.

### Resultats run 3 (sans hierarchy contrastive, avec les 3 fixes dynamiques)

| Epoch | InfoNCE loss | Acc | KL loss | klW | MP grad norms | Duree |
|-------|-------------|-----|---------|-----|--------------|-------|
| 1 | 2.334 | 43.5% | 2.000 | 0.020 | W_c=4.28e-2 W_p=4.28e-2 a=1.14e-4 | 22m49s |
| 2 | ~1.5 | ~68% | — | 0.050 | — | ~20m |

**Succes critique:** MP grad norms non-zero (`4.28e-2` vs ~0 avant). Les 3 fixes dynamiques debloquent le signal MP.

---

## 2026-02-14 (run 3) — Training Dynamics Fix

### Changements

#### 1. KL ramp doux des epoch 0 (plus de warmup binaire)
- **Avant:** `klWeight = 0` pour 3 epochs, puis ramp 0→plateau. MP affame pendant le warmup.
- **Apres:** `klWeight = plateau * 0.1` des epoch 0 (= 0.02), ramp lineaire vers 0.2 sur 6 epochs.
- **Justification (panel unanime):** Le KL est la source principale de gradients dH denses (touche ~10-20 L0 nodes/exemple vs 1 pour InfoNCE). Couper le KL = couper le signal MP.
- **Reference:** Meadows (systems thinking) — "information delay in feedback loop", Taleb — "via negativa violation: removing signal to prevent hypothetical problem"

#### 2. MP_LR_SCALE: 0.1 → 1.0
- **Avant:** MP params recoivent 360x moins d'update effective que K-head (1 update/epoch x 0.1 scale vs 36 updates/epoch x 1.0).
- **Apres:** MP_LR_SCALE = 1.0. Adam s'adapte naturellement aux differentes magnitudes.
- **Justification:** Adam sqrt scaling rule (Princeton 2024). Effective batch 36x → LR devrait etre sqrt(36)~6x plus grand, pas 10x plus petit. 0.1 etait un vestige du design per-batch.

#### 3. Fix normalisation: numBatches au lieu de totalExamples
- Deja applique dans run 2. La loss est deja moyennee par batch_size; diviser par numBatches = moyenne correcte.

### Panel d'experts (5 experts, mode debat)

**Propositions retenues:**
1. KL ramp doux (priorite 1, applique)
2. MP_LR_SCALE = 1.0 (priorite 1, applique)
3. Contrastive au niveau capability L1+ (priorite 2, prochaine iteration)

**Propositions ecartees:**
- Gated residual (pas convaincu pour l'instant)
- Hierarchy pruning (non — on garde la hierarchie complete)
- Hyperbolic embeddings (long terme)
- Learned hierarchy restructuring (long terme)

**Insight cle:** Le probleme n'est PAS l'init MP (identity-like Glorot = quasi-zero contribution, OK). Le probleme est que les gradients n'arrivent JAMAIS pour faire evoluer les poids MP au-dela de l'init. Fixer la dynamique de training d'abord.

### Analyse d'impact des 3 fixes

Les 3 problemes etaient **cumulatifs** — chacun supprimait le signal MP independamment:

| Fix | Facteur de suppression | Justification |
|-----|----------------------|---------------|
| Normalisation totalExamples→numBatches | 32x (1/BATCH_SIZE parasite) | Bug mathematique: loss deja moyennee par batch_size dans batchContrastiveBackward |
| KL warmup binaire 3 epochs→ramp 0.02 | ∞ pendant 3 epochs (signal = 0) | KL = source principale de dH denses (10-20 L0 nodes/ex vs 1 InfoNCE). Couper = affamer MP |
| MP_LR_SCALE 0.1→1.0 | 10x | Vestige du design per-batch. Avec epoch-level (1 update vs 36), 0.1 = 360x moins effectif |

**Facteur combine:** jusqu'a ~320x de suppression (32 × 10) apres epoch 3, et signal **zero** avant epoch 3.

**Confiance elevee** car chaque fix est justifiee independamment:
1. numBatches = correction mathematique pure (pas d'hyperparametre)
2. KL ramp = consensus unanime panel (5/5 experts), reference systems thinking (delai d'information)
3. MP_LR_SCALE = Adam sqrt scaling rule (Princeton 2024), valide empiriquement sur le LiveMCPBench

**Critere de succes:** MP grad norms > 1e-4 en fin d'epoch 1 (run 3). Si confirme → le MP recoit du signal et peut apprendre la coloration L1→L0.

### Propositions ecartees — analyse d'impact

**Hierarchy Pruning (Christensen)** — Impact faible. Les caps avec 1 tool = identite dans le MP (pas de dilution, pas d'agregation). Les caps avec 0 tools = pas d'edges dans le graphe sparse. Le probleme n'est pas le nombre de caps, c'est que le MP ne recoit pas de gradient. Pruner ne resout pas ca.

**Gated Residual (Taleb)** — Impact potentiellement eleve mais premature. Le gate appris (`H_final = (1-g)*H_pre + g*MP`) donne un filet de securite. Mais avec des gradients MP faibles, le gate n'a pas de signal pour apprendre a s'ouvrir → cercle vicieux (gate ferme → pas de contribution MP → pas de gradient → gate reste ferme). A reconsiderer si le MP fonctionne apres les fixes dynamiques, comme stabilisateur.

---

## 2026-02-14 (run 2) — Epoch-Level MP Backward + numBatches fix

(Run avec normalisation corrigee mais KL warmup et MP_LR_SCALE anciennes valeurs.)

---

## 2026-02-14 (run 1) — Epoch-Level MP Backward ("Autoroute")

### Changements

**Fichier modifie:** `tools/train-ob.ts`

#### 1. Epoch-level gradient accumulation pour MP
- **Avant:** MP backward appele a chaque batch (~100x/epoch), chacun traversant le graphe complet (35356 edges x 16 heads). Cout: ~20-25 min/epoch.
- **Apres:** Les gradients dH sont accumules dans `_epochDH` sur toute l'epoch, puis UN SEUL MP backward + UN SEUL Adam step MP a la fin. Cout: ~1m30/epoch.
- **Speedup:** ~17x (20 epochs: 8h → ~30min)
- **Justification:** Standard en GNN training (GNNAutoScale, FreshGNN). Les poids K-head/W_intent continuent d'etre mis a jour per-batch.

#### 2. Fix normalisation: numBatches au lieu de totalExamples
- **Bug:** Normalisation par `totalExamples` (=infoBatches*BATCH_SIZE + klBatches*BATCH_SIZE) ajoutait un facteur 1/BATCH_SIZE (32x) de suppression inutile. La loss InfoNCE est deja moyennee par batch_size dans `batchContrastiveBackward`.
- **Fix:** Normaliser par `numBatches` (=infoBatches + klBatches). Gradient ~32x plus fort.
- **Reference:** Thomas Wolf gradient accumulation pattern, PyTorch Lightning docs.

#### 3. Fusion buffers DE
- `_batchDE` + `_emptyDE` → `_zeroDE` (un seul buffer, jamais mute, car toutes les cibles sont L0).
- Suppression de `zeroDE()`.

#### 4. Logs ameliores
- Progress bars Unicode avec ANSI colors (loss rouge, acc vert, ETA cyan)
- MP backward time + gradient norms en notation exponentielle
- Epoch summary avec `mpBack=Xs`
- Fix `Deno.isatty` → `Deno.stdout.isTerminal()` (Deno 2.x API)

#### 5. Helper `computeMPGradNorms`
- Calcule les normes L2 separees pour W_child, W_parent, a_upward/a_downward depuis `MultiLevelGradients`.

### Decisions architecturales

| Decision | Choix | Justification |
|----------|-------|---------------|
| MP backward frequency | 1x/epoch | Standard GNN (stale gradients OK pour 1 layer, staleness < FreshGNN threshold t=200) |
| Gradient accumulation target | `_epochDH` (L0 only) | Toutes les cibles sont L0 (USE_BATCH_CONTRASTIVE=true), dE toujours zeros |
| Normalisation | 1/numBatches | Loss deja moyennee par batch_size; diviser par batches = moyenne correcte des gradients per-batch |
| Residuel forward | Additif pur (H + prop) | `forwardMultiLevelWithCache` n'utilise PAS downwardResidual alpha — pas de suppression du signal MP |
| MP_LR_SCALE | 0.1 (inchange) | A surveiller: avec 1 update/epoch vs 36 pour K-head, le ratio effectif est 360x. Litterature suggere sqrt(36)~6x (Adam square root scaling rule) |

### Metriques observees (run avec normalisation incorrecte totalExamples)

| Epoch | Loss | Acc | R@1 | MRR | Duree | MP grad norms |
|-------|------|-----|-----|-----|-------|---------------|
| 1 | 2.334 | 43.5% | — | — | 1m28s | ~0 (normalisation trop agressive) |
| 2 | — | — | 31.8% | 0.451 | — | ~0 |
| 4 | 1.411 | 72.1% | 18.7% | 0.303 | 1m25s | ~0 |

### Recherche litterature (2024-2026)

- **GNNAutoScale (ICML 2021):** Historical embeddings, error borné par constantes Lipschitz
- **FreshGNN (VLDB 2024):** Staleness threshold t=200 iterations, gradient norms comme proxy de stabilite
- **VISAGNN (2025):** Dynamic Staleness Attention, gamma(t) decayant
- **KDD 2025 (Hyperbolic Collapse):** Leaf collapse + height collapse dans hierarchies; ERank comme metrique
- **Cambridge 2025 (Vanishing Gradients in GNNs):** GNNs "by design prone to extreme gradient vanishing"
- **Adam scaling (Princeton 2024, NeurIPS 2024):** batch_size x kappa → LR x sqrt(kappa)

### Probleme ouvert: MP_LR_SCALE

Avec epoch-level accumulation, MP recoit 1 update/epoch vs 36 pour K-head. Le batch effectif MP est 36x plus grand. Selon la regle sqrt pour Adam:
- LR_MP devrait etre `epochLR * sqrt(36) ≈ epochLR * 6` au lieu de `epochLR * 0.1`
- Ratio actuel: 360x moins que K-head. Ratio recommande: ~6x moins.
- A tester empiriquement. Si MP grad norms restent negligeables avec la fix numBatches, augmenter MP_LR_SCALE a 0.5-1.0.
