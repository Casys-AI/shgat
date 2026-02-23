# Expert Panel: MP Training Dynamics & Innovation
**Date:** 2026-02-14
**Mode:** Debate (5 experts)
**Contexte:** Gradients MP quasi-zeros, MP ne contribue pas au scoring malgre l'architecture epoch-level backward "autoroute"

## Panelistes
- Donella Meadows (Systems Thinking)
- Clayton Christensen (Disruption Theory)
- Nassim Nicholas Taleb (Antifragility & Risk)
- Jim Collins (Disciplined Execution)
- Michael Porter (Competitive Strategy)

---

## Q1: KL Warmup — supprimer, reduire, ou decoupler?

### Diagnostic (Meadows)
Le warmup KL cree un **delai structurel** dans la boucle de feedback. Le MP a deux boucles:
- K-head: signal chaque batch, 36 updates/epoch — reponse immediate
- MP: signal 1x/epoch, et pendant 3 epochs = zero

C'est un probleme classique de **delai d'information**. Quand le KL s'active (epoch 3), les K-head ont deja converge vers un optimum local sans information structurelle. Le MP doit alors "lutter" contre un paysage de scoring fige — **resistance politique** du systeme.

### Analyse de risque (Taleb)
Le risque de KL a epoch 0 est **convexe** (pas concave):
- Si les gradients KL+MP sont du bruit aleatoire → ils s'annulent (CLT sur les batches)
- Si ils contiennent un signal → direction utile pour MP
- Pire cas (corruption embedding) borne par le residuel additif: `H_final = H_pre + MP` preserve les embeddings originaux

Le warmup est une **violation via negativa** — on supprime du signal pour prevenir un probleme hypothetique.

### Consensus
**Unanime: supprimer le warmup binaire.** Remplacer par ramp doux:
- `klWeight = plateau * 0.1` (= 0.02) des epoch 0
- Ramp lineaire vers plateau (0.2) sur `warmupEpochs * 2` (6 epochs)
- Signal directionnel immediat sans magnitude excessive

---

## Q2: MP_LR_SCALE — 0.1 est-il trop conservateur?

### Les chiffres brutaux (Collins)

| Param group | Updates/epoch | Effective LR/step | Produit total |
|---|---|---|---|
| K-head | ~36 | 0.005 | 0.18 |
| MP (ancien) | 1 | 0.005 * 0.1 | 0.0005 |

360x moins d'update effective = **famine**, pas prudence.

### Adam sqrt scaling (Collins + Meadows)
Effective batch MP = epoch entiere (~1155 ex). K-head batch = 32.
```
kappa = 1155 / 32 = 36
sqrt(kappa) = 6
```
LR MP devrait etre `baseLR * 6.0`, pas `baseLR * 0.1`. Ecart de **60x**.

Mais Adam est adaptatif (moments s'ajustent). La correction est moins dramatique que pour SGD.

### Staleness (Taleb)
L'accumulation epoch-level cree une asymetrie informationnelle: K-head voit chaque batch et s'ajuste, MP ne voit que la moyenne. Les gradients MP sont partiellement **stale** par rapport a l'etat K-head.

Proposition avancee (future): LR MP adaptatif lie a la convergence K-head:
```typescript
const adaptiveMPScale = Math.max(0.5, Math.min(5.0, 1.0 / (kheadGradNorm + 0.1)));
```

### Consensus
**Unanime: MP_LR_SCALE = 1.0.** Adam s'adapte. Reduire a 0.5 si oscillation.

---

## Q3: Innovations non-standard pour le retrieval hierarchique

### 1. Gated Residual (Taleb) — ECARTE par l'utilisateur
```
gate[i] = sigmoid(W_gate @ [H_pre[i]; MP_contribution[i]])
H_final[i] = (1 - gate[i]) * H_pre[i] + gate[i] * MP_contribution[i]
```
Si MP mauvais → gate ferme, pas de degradation. +2049 params/level.

### 2. Contrastive au niveau Capability (Porter) — RETENU, prochaine iteration
**Job to be done:** le K-head scoring optimise intent→tool. Mais le MP optimise rien directement. Ajouter:
```
Loss_cap = InfoNCE(intent, target_capability_embedding)
```
L'embedding capability vient du V→E forward (aggregation). Ce loss fournit un gradient DIRECT au MP V→E phase — pas besoin de passer par le scoring K-head.

**Impact attendu:** Le MP recoit un signal direct "cette aggregation de tools en capability doit matcher cet intent". C'est le chainage manquant.

### 3. Hierarchy Pruning (Christensen) — ECARTE par l'utilisateur
6916 caps avec avg 0.28 tools/cap. Beaucoup de caps avec 0-1 tool = bruit. Pruner caps < 3 tools → ~600 caps utiles. **Non retenu** — on garde la hierarchie complete.

### 4. Hierarchical Positional Encoding (Meadows) — A EXPLORER
Encoder la position dans l'arbre comme feature supplementaire:
```
pos[tool_42] = [super_cap_idx / num_super, cap_idx / num_caps, tool_idx / num_tools]
```
GPS coordinate dans la hierarchie, utilisable comme biais d'attention.

### 5. Multi-Resolution Attention (Meadows)
Mecanismes d'attention differents par niveau:
- L0→L1: Fine-grained (quels outils dans cette cap sont pertinents?)
- L1→L2: Coarse (quels clusters de capabilities sont actives?)

Deja partiellement supporte par les params per-level separés.

---

## Q4: Le probleme "MP contribution = bruit aleatoire"

### Diagnostic (Taleb)
Avec W_child/W_parent non entraines (Glorot init), `MP_contribution = f(W_init, graph, H_pre)` est du **bruit structure** — pas aleatoire, mais non-informe.

3 cas:
1. MP_contribution petit (init identity-like) → perturbation negligeable, OK
2. MP_contribution grand → corruption. Catastrophique mais **improbable** avec Glorot
3. MP_contribution modere → biases systematiques par topologie (caps avec beaucoup d'outils = dilution, caps avec peu = identite). **C'est de l'information structurelle utile** meme sans entrainement.

### Consensus
**NE PAS faire de warmup MP inverse** (0→1). L'init identity-like est un bon point de depart. Le probleme est que les gradients n'arrivent pas pour faire evoluer les poids, pas que l'init est mauvaise.

---

## Q5: Vision long-terme

### Court terme (cette semaine)
1. ~~Fixer dynamique training~~ **FAIT** (KL ramp + MP_LR_SCALE = 1.0)
2. Evaluer si MP contribue (|H_final - H_pre| / |H_pre|)
3. Si oui → contrastive capability-level

### Moyen terme (2-4 semaines)

| Si MP fonctionne | Si MP echoue |
|---|---|
| Contrastive capability (Porter) | Hierarchical regularization (Christensen) |
| Positional encoding (Meadows) | Simplifier: K-head + regularisation |

### Long terme

| Approche | Quand | Proposeur |
|---|---|---|
| Learned hierarchy restructuring | 50K+ tools | Porter |
| Hyperbolic embedding space | Si collapse mesure empiriquement | Meadows |
| Adaptive staleness-aware LR | Si staleness est un probleme mesure | Taleb |

### Hierarchical Regularization (Christensen) — alternative si MP echoue
```
Loss = InfoNCE(intent, tool) + lambda * HierarchicalConsistency(tool_scores, sibling_scores)
```
Penalise quand un tool score haut mais ses siblings scorent bas. Injecte l'info hierarchique SANS message passing — plus simple, plus rapide.

---

## Q6: Gate adaptatif pour le MP — filtrage de gradients (post-panel)

### Probleme
Le gated residual (Q3.1) a un cercle vicieux: le gate ne peut pas apprendre a s'ouvrir si les gradients MP sont faibles. Peut-on entrainer le gate en parallele avec un signal independant?

### Option A: Gradient agreement filtering (statique) — ECARTE
```
cos_angle = dot(grad_MP, grad_Khead) / (|grad_MP| * |grad_Khead|)
if cos_angle > 0: appliquer grad_MP
else: ignorer grad_MP
```
Inspire de PCGrad (Yu et al., NeurIPS 2020) pour le multi-task learning. ~5 lignes, pas de params supplementaires. **Ecarte** — heuristique statique, pas de capacite d'apprentissage, le seuil `> 0` est arbitraire.

### Option B: Gate appris avec signal differentiel — RETENU (priorite 3)
Briser le cercle vicieux en donnant au gate un signal supervise direct:

1. Forward **sans** MP → `loss_baseline` (K-head scoring sur H_pre)
2. Forward **avec** MP → `loss_mp` (K-head scoring sur H_final)
3. `delta[i] = loss_baseline[i] - loss_mp[i]` par noeud (positif = MP a aide)
4. Le gate apprend a predire `sigmoid(delta) > 0.5` par noeud

```typescript
// Gate training signal (per-node, per-batch)
gate_target[i] = loss_without_mp[i] > loss_with_mp[i] ? 1.0 : 0.0;
gate_loss = BCE(gate_pred[i], gate_target[i]);
```

**Avantages:**
- Le gate recoit un signal clair et direct (difference de loss), independant des gradients MP
- Pas de cercle vicieux: le gate apprend meme si les gradients MP sont faibles
- Apprentissage par noeud: certaines capabilities beneficient du MP, d'autres non
- Le gate peut se specialiser (ex: caps avec 10+ tools = MP utile, caps avec 1 tool = ignorer)

**Cout:** 2x forward K-head par batch (le forward MP est deja cache 1x/epoch). Le K-head scoring sans MP est rapide (~0.5s/batch), donc surcout modere.

**Prerequis:** Confirmer d'abord que les fixes run 3 (KL ramp + MP_LR_SCALE + normalisation) generent des gradients MP non-zero. Si le MP fonctionne sans gate → pas besoin. Si le MP aide certains noeuds mais en degrade d'autres → le gate differentiel est la bonne solution.

**Reference:** Meta-learning / learned loss weighting (Shu et al., ICML 2019), gradient-based data reweighting.

---

## Metriques a surveiller

1. **MP grad norms** (en notation exponentielle) — doivent etre > 1e-4
2. **Embedding delta**: `||H_final - H_pre|| / ||H_pre||` — mesure l'effet reel du MP
3. **R@1 avec vs sans MP** — le delta est la valeur ajoutee du MP
4. **Ratio grad MP / grad K-head** — devrait etre ~0.01-0.1 (pas 0.0000)

---

## References

- GNNAutoScale (Fey et al., ICML 2021) — historical embeddings, staleness bounds
- FreshGNN (VLDB 2024) — gradient norms comme proxy stabilite, t_stale=200
- VISAGNN (2025) — Dynamic Staleness Attention
- KDD 2025 (Hyperbolic Collapse) — leaf collapse, ERank comme metrique
- Cambridge 2025 — vanishing gradients in GNNs
- Adam sqrt scaling (Princeton 2024, NeurIPS 2024) — batch×kappa → LR×sqrt(kappa)
- Thomas Wolf — gradient accumulation reference implementation
