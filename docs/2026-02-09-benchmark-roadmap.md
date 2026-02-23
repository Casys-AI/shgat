# SHGAT-TF Benchmark Roadmap

**Date:** 2026-02-09
**Context:** Suite de la décision no-training (ADR 2026-02-08). L'Option A (projection orthogonale)
est validée sur nos données de prod (99.2-99.7% Hit@1). Il faut maintenant prouver que le
message passing hiérarchique apporte un delta réel vs cosine brut sur des benchmarks publics
avec des ontologies profondes.

**Objectif:** Démontrer que SHGAT (graph structure + MP + projection) > cosine similarity brute
quand la hiérarchie est profonde et que les siblings sont sémantiquement proches.

---

## Benchmarks à réaliser

### Phase 1 : LiveMCPBench (priorité haute)

- **Source:** https://huggingface.co/datasets/ICIP/LiveMCPBench
- **Paper:** https://arxiv.org/abs/2508.01780
- **Hiérarchie:** 3 niveaux — Category (8) → MCP Server (70) → Tool (527)
- **Taille:** 527 tools, 95 tasks
- **Pourquoi:** MCP-natif, structure identique à notre cas d'usage. Paper montre que
  retrieval = 50% des échecs. Weighted server+tool similarity insuffisant → exactement
  ce que le MP hiérarchique devrait résoudre.
- **Embedding:** ~2 min (527 tools + 95 queries via BGE-M3 Deno)
- **Métriques:** Recall@k, NDCG@k, MAP@k, comparaison Cosine vs SHGAT-Flat vs SHGAT-Hier
- **Status:** DONE (2026-02-09)
- **Résultats (5-fold CV, seed=42):**

  | Config | R@1 | R@3 | R@5 | NDCG@5 |
  |--------|-----|-----|-----|--------|
  | Cosine baseline | 14.4% | 29.6% | 35.3% | 32.5% |
  | Option A — Flat (random proj) | 14.7% | 27.1% | 35.4% | 31.9% |
  | Option A — Hier (PDR .99/.5/.3) | 14.2% | — | — | — |
  | Option A — Hier (DR=0, no residual) | 5.2% | — | — | — |
  | **Option B — Flat (LR=0.001, 10ep)** | **16.4%** | **37.6%** | **45.1%** | **44.6%** |
  | Option B — Hier (PDR .99) | 16.4% | 38.4% | 45.1% | 44.7% |
  | Option B — Hier (PDR .5/.3/.1) | 11.2% | 23.8% | 29.5% | 29.2% |

- **Conclusion LiveMCPBench:**
  - Le training K-head (W_k, W_intent) fonctionne : +28% R@5 vs cosine
  - Le message passing hiérarchique N'AIDE PAS avec 282 exemples d'entraînement
  - Hier avec residuals élevés (0.99) = identique à flat (MP = no-op)
  - Hier avec residuals bas (0.5) = pire que cosine (MP destructif)
  - Le bottleneck est le volume de données, pas l'architecture
  - **Best config prod : B-Flat, LR=0.001, 10 epochs**

### Phase 2 : ToolBench full (stress test)

- **Source:** https://github.com/OpenBMB/ToolBench
- **Paper:** ICLR 2024, https://arxiv.org/abs/2307.16789
- **HuggingFace:** https://huggingface.co/datasets/tuandunghcmut/toolbench-v1
- **Hiérarchie:** 4 niveaux — Category (49) → Collection (500+) → Tool (3,451) → API (16,464)
- **Taille:** 16,464 APIs
- **Pourquoi:** Gold standard. Queries intra-collection testent la disambiguation entre
  siblings sémantiquement proches — le cas clé où le MP fait la différence.
  ToolRet aplatit cette hiérarchie (2 niveaux seulement), ToolBench la conserve.
- **Embedding:** ~1h (16K APIs via BGE-M3 Deno)
- **Métriques:** NDCG@1/5, split par généralisation (Inst/Tool/Cat)
- **Status:** TODO

### Phase 3 : SNOMED CT (preuve théorique, optionnel)

- **Source:** https://arxiv.org/html/2511.16698 (paper hiérarchique)
- **Hiérarchie:** 7+ niveaux en moyenne (DAG, pas arbre), 19 top-level concepts
- **Taille:** ~200,000 concepts
- **Pourquoi:** Seul benchmark publié prouvant que hiérarchique bat cosine (+24% MRR relatif).
  Ontologie extrêmement profonde. Pas du tool retrieval mais preuve de concept sur
  la valeur du graph structure pour le retrieval.
- **Embedding:** ~14h CPU (200K concepts) — nécessite GPU ou embedding partiel
- **Résultat publié:** OnT (hyperbolic) MRR 0.68 vs flat SBERT MRR 0.55 (d=5)
- **Status:** OPTIONNEL — à faire si Phases 1-2 montrent un delta positif

### Bonus : MCP-Bench (distracteurs)

- **Source:** https://github.com/Accenture/mcp-bench, https://arxiv.org/abs/2508.20453
- **Hiérarchie:** 3 niveaux — Domain → Server (28) → Tool (250)
- **Spécificité:** 10 serveurs distracteurs par task (100+ tools parasites)
- **Pourquoi:** Stress-test de la résistance au bruit — le pruning hiérarchique via
  graph attention devrait exceller ici.
- **Status:** OPTIONNEL

---

## Infrastructure existante

- `lib/shgat-tf/benchmark/` — pipeline benchmark (Node/tsx pour scoring, Deno pour embedding)
- `scripts/download-toolret.py` — download HuggingFace via Python `datasets`
- `scripts/embed-toolret.ts` — embedding BGE-M3 via `EmbeddingModel` Deno
- `src/run.ts` — runner benchmark (Cosine vs SHGAT-Flat vs SHGAT-Hier)
- `src/metrics.ts` — métriques IR (Recall@k, NDCG@k, MAP@k, Precision@k)

## Ce qu'on a prouvé / réfuté

1. **Cosine seul ≈ SHGAT-Flat (no-training)** — PROUVÉ (14.4% vs 14.7% R@1)
2. **SHGAT-Hier > Cosine sur 3 niveaux** — RÉFUTÉ sur LiveMCPBench (282 exemples insuffisants pour MP)
3. **Le delta croît avec la profondeur** — NON TESTÉ (besoin de ToolBench 4 niveaux)
4. **Disambiguation siblings** — NON TESTÉ (besoin d'analyse par catégorie)
5. **K-head trained > Cosine** — PROUVÉ (+28% R@5, +14% R@1 avec B-Flat LR=0.001)

## Hypothèses révisées

- Le MP hiérarchique nécessite probablement **1000+ exemples** pour apprendre des attention patterns utiles (W_up, W_down)
- Le scoring K-head (W_k, W_intent) apprend efficacement même avec 282 exemples
- **ToolBench (16K APIs, 4 niveaux)** est le prochain test décisif pour le MP — plus de données ET plus de profondeur

## Références

- ADR no-training : `lib/shgat-tf/docs/2026-02-08-no-training-decision.md`
- Johnson-Lindenstrauss (1984), Dasgupta-Gupta (2003)
- Prabhu et al. "RanDumb" (NeurIPS 2024)
- Qin et al. "ToolLLM" (ICLR 2024)
- Li et al. "LiveMCPBench" (2025)
- Xu et al. "MCP-Bench" (NeurIPS 2025 Workshop)
