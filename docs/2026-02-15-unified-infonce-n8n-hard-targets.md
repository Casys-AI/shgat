# Unified InfoNCE: n8n as Hard Targets

**Date:** 2026-02-15
**Status:** Proposition

## Contexte

Le training SHGAT-TF utilise actuellement deux loss separees :
- **InfoNCE** (cross-entropy focale) sur 1155 exemples prod — hard targets (1 tool correct)
- **KL divergence** sur 4000 exemples n8n (subsample de 30K) — soft targets (distribution cosine sim)

Ratio params/data : 7.35M / 1155 = **6364 params/exemple** → overfit systematique.

## Resultats experimentaux (2026-02-14/15, seed=42)

| Run | KL | Isolation K-head | kl-weight | R@1 | MRR | R@5 | Best epoch |
|---|---|---|---|---|---|---|---|
| KL classique | oui | non | 0.2 | **24.3%** | 0.318 | ~50% | 8/20 |
| No-KL | non | — | — | 21.5% | **0.349** | **55.1%** | 12/15 |
| KL isole | oui | oui | 0.05 | 20.6% | 0.294 | 26.2% | 4/10 |
| KL isole | oui | oui | 0.2 | 12.1% | 0.216 | — | 2/10 (killed) |

**Conclusion** : KL classique = meilleur R@1, mais overfit dans tous les cas (train acc 100% des epoch 3).

## Proposition : Unifier n8n et prod sous InfoNCE

### Observation cle

Les workflows n8n sont fondamentalement la **meme tache** que les workflows prod : predire le prochain outil dans une sequence. La distinction KL/InfoNCE est artificielle.

De plus :
- Les 1155 exemples prod sont **tous orphelins** (0 parent L1 dans la hierarchie)
- Les 3998 exemples HIER viennent deja des n8n mappes
- Le MP ne recoit de gradient direct que via les n8n

### Changements proposes

1. **ArgMax hard targets** : Remplacer les distributions soft (T=0.005, cosine sim) par le tool avec la similarite maximale. Avec avg top-1 sim = 0.796 et T=0.005, les distributions sont deja quasi-peaked — autant prendre le argmax.

2. **Un seul InfoNCE** : Supprimer la branche KL entierement. Tous les exemples (prod + n8n) passent par la meme loss InfoNCE.

3. **Dataset unifie** : 1155 prod (3x oversample = 3465) + 30141 n8n = ~33K exemples.

4. **Ratio params/data** : 7.35M / 33K = **223 params/ex** (27x mieux que 6364).

### Impact attendu

| Metrique | Avant (KL) | Apres (unifie) |
|---|---|---|
| Exemples effectifs | 1155 (InfoNCE) + 4000 (KL indirect) | 33K (InfoNCE) |
| Params/exemple | 6364 | 223 |
| Overfit | Systematique (ep 3-4) | Devrait disparaitre |
| Complexite code | 2 losses + masques + warmup | 1 loss unique |
| HIER contrastive | Inchange (n8n mappes ont des ancetres L1) | Inchange |

### Risques

1. **Bruit de mapping** : ~20% des argmax n8n→vocab pourraient etre incorrects. Le InfoNCE avec temperature basse tolere mal les mauvais labels.
   - Mitigation : filtrer les mappings avec sim < 0.80, ou label smoothing

2. **Domain shift** : Les workflows n8n (automation no-code) != workflows prod (MCP tool calling). Les patterns sequentiels pourraient differer.
   - Mitigation : oversample prod 3x (comme actuellement)

3. **Perte du signal distribue** : KL donnait un gradient dense (10-20 tools par exemple). InfoNCE = 1 tool. Le MP recevait plus de signal via KL.
   - Mitigation : le volume 27x superieur compense largement

### Fichiers a modifier

- `lib/gru/src/n8n/build-soft-targets.ts` → ajouter mode `--hard` (argmax)
- `lib/shgat-tf/tools/train-ob.ts` → supprimer branche KL, unifier dataset
- `lib/gru/data/` → nouveau fichier hard targets

## Questions ouvertes (panel)

1. ArgMax strict ou top-K avec label smoothing ?
2. Seuil de qualite mapping (sim >= 0.70 ? 0.80 ?)
3. Faut-il un poids differentiel prod/n8n ou traiter tout pareil ?
4. Impact sur le HIER contrastive (plus d'exemples avec ancetres = mieux ?)
5. Faut-il garder un KL residuel comme regularisateur ?
