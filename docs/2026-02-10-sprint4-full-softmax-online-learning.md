# Sprint 4 : Full Softmax SHGAT (P2-6) + Audit Online Learning (P3-8)

**Date** : 2026-02-10
**Auteur** : Reporter (panel d'experts)
**Prerequis** : Document `2026-02-10-mathematical-foundations-and-roadmap.md`, sections P2-6 et P3-8

---

## Partie 1 : P2-6 — Full Softmax SHGAT

### 1.1 Diagnostic du code actuel

Le training SHGAT-TF utilise une loss InfoNCE **per-example** avec un nombre restreint de negatifs (~8). Le code se trouve dans `lib/shgat-tf/src/training/autograd-trainer.ts`.

**Flux actuel dans `trainStep()` (ligne 654)** :

```
pour chaque example:
  candidateId = ex.candidateId        (1 positif)
  negativeCapIds = ex.negativeCapIds  (~8 negatifs semi-hard)
  → total candidats = ~9

  scores = forwardScoring(intent, [positif, neg1, ..., neg8])
  loss += infoNCELoss(positiveScore, negativeScores, temperature)
```

**Probleme** : La borne superieure du signal gradient est `log(K+1) = log(9) = 2.2 nats`. Avec full softmax sur 525 outils, la borne passe a `log(525) = 6.3 nats` — un facteur 3x d'information supplementaire par gradient update.

### 1.2 Localisation precise du code a modifier

**Fichier** : `lib/shgat-tf/src/training/autograd-trainer.ts`

**1. Generation des negatifs (lignes 705-714)** — Actuellement restreint a `ex.negativeCapIds` :

```typescript
// ACTUEL - lignes 706-714
const allNodeIds = new Set<string>();
for (const ex of examples) {
  allNodeIds.add(ex.candidateId);
  for (const negId of ex.negativeCapIds || []) {
    if (nodeEmbeddings.has(negId)) {
      allNodeIds.add(negId);
    }
  }
}
```

**Modification proposee** : Remplacer `ex.negativeCapIds` par **tous les nodeIds** disponibles dans `nodeEmbeddings` :

```typescript
// PROPOSE
const allNodeIds = new Set<string>();
// Ajouter TOUS les outils disponibles comme candidats
for (const nodeId of nodeEmbeddings.keys()) {
  allNodeIds.add(nodeId);
}
// S'assurer que les candidats des exemples sont inclus
for (const ex of examples) {
  allNodeIds.add(ex.candidateId);
}
```

**2. Boucle per-example (lignes 784-818)** — Remplacement des negatifs per-example par full softmax :

```typescript
// ACTUEL - lignes 784-791
for (const ex of examples) {
  const nodeIds: string[] = [ex.candidateId];
  for (const negId of ex.negativeCapIds || []) {
    if (nodeEmbeddings.has(negId)) {
      nodeIds.push(negId);
    }
  }
  // ...
}
```

**Modification proposee** : Utiliser TOUS les nodes comme negatifs (sauf le positif) :

```typescript
// PROPOSE
// Pre-compute: tous les nodeIds sauf chaque positif
const allNodeIdsList = Array.from(allNodeIds);

for (const ex of examples) {
  // Tous les candidats: positif en premier, puis tous les autres
  const nodeIds: string[] = [ex.candidateId];
  for (const nid of allNodeIdsList) {
    if (nid !== ex.candidateId) {
      nodeIds.push(nid);
    }
  }
  // ... reste identique (forwardScoring, infoNCELoss)
}
```

**3. Fonction `batchContrastiveLoss()` (lignes 578-619)** — Existe deja dans le fichier mais n'est PAS utilisee par `trainStep()`. Cette fonction calcule une matrice de similarite [B x B] en in-batch. Elle pourrait etre une alternative plus efficace si le batch contient suffisamment d'exemples divers. Cependant, elle ne couvre pas le cas full-softmax sur 525 outils car elle est limitee au batch.

### 1.3 Impact sur les performances

| Dimension | Actuel (~9 candidats) | Full softmax (~525) |
|---|---|---|
| Borne InfoNCE | 2.2 nats | 6.3 nats |
| Forward pass / exemple | 9 scores | 525 scores |
| Memoire par exemple | ~9 x 1024 floats = 36 KB | ~525 x 1024 = 2.1 MB |
| Temps par epoch (estime) | ~1s | ~5-10s |

**Mitigation de la memoire** : Les 525 embeddings sont deja charges une seule fois dans `allEmbsTensor` (lignes 774-778). Le gather par exemple est O(525) mais le tenseur est reutilise.

### 1.4 Optimisation potentielle : batch-level full softmax

Au lieu de gather 525 embeddings par exemple, on peut pre-calculer les scores de TOUS les outils en batch :

```typescript
// OPTIMISE: score tous les outils en une matrice
const intentBatch = stack(examples.map(e => e.intentEmbedding));  // [B, 1024]
const allEmbs = allEmbsTensor;  // [525, 1024]
const allScores = batchForwardScoring(intentBatch, allEmbs, params, config);
// allScores: [B, 525]

// Puis pour chaque exemple, extraire le label du positif
for (let i = 0; i < examples.length; i++) {
  const positiveIdx = allNodeIdToIdx.get(examples[i].candidateId);
  // softmax CE avec label = positiveIdx
}
```

Cela est plus efficace que 525 per-example car le `matMul` est batche. Cependant, cela necessite de refactorer `forwardScoring()` pour accepter un batch d'intents.

### 1.5 Risques et precautions

1. **Memoire GPU/CPU** : 525 x batchSize x 1024 floats. Avec batchSize=32 : ~64 MB. Acceptable.
2. **Convergence** : Plus de negatifs = loss initiale plus elevee. Ajuster le learning rate si necessaire.
3. **Negatifs trop faciles** : Avec 525 outils, la majorite sont des negatifs faciles (score ~0). Le gradient sera domine par les hard negatives proches du positif, ce qui est souhaitable.
4. **Backward compatibility** : Les exemples du `per-training.ts` (production V1) continueront d'envoyer `negativeCapIds`. L'option full softmax doit etre un **flag optionnel** dans `TrainerConfig` :

```typescript
interface TrainerConfig {
  // ...existant...
  /** Use all available node embeddings as negatives (full softmax) */
  fullSoftmax?: boolean;
}
```

### 1.6 Estimation d'effort

- Modification de `trainStep()` : 2-3 heures
- Flag `fullSoftmax` dans config : 30 min
- Tests : 1-2 heures
- **Total** : 0.5 jour (confirme l'estimation du roadmap)
- **ROI estime** : +1-3 pts R@1 (borne par le nombre de positifs, pas de negatifs)

---

## Partie 2 : P3-8 — Audit de l'Online Learning existant

### 2.1 Inventaire complet du code existant

L'online learning est **deja implemente en production**. Voici l'inventaire exhaustif des composants.

#### Composant A : OnlineLearningController

**Fichier** : `src/graphrag/learning/online-learning.ts`

**Fonctionnement** :
- Ecoute l'evenement `execution.trace.saved` via `eventBus`
- Recupere la trace avec `traceStore.getTraceById()`
- Appelle `trainSHGATOnExecution(shgat, { intentEmbedding, targetCapId, outcome })`
- Emet `learning.online.trained` apres chaque training
- Stats internes : `trainingCount`, `totalLoss` (pas de persistence)

**Architecture** :
```
eventBus("execution.trace.saved")
    |
    v
OnlineLearningController.start()
    |
    v  (filtre: skip si pas de capabilityId ou intentEmbedding)
trainSHGATOnExecution(shgat, ...)   ← SHGAT V1 K-head, PAS l'autograd-trainer
    |
    v
eventBus.emit("learning.online.trained")
```

**Limites identifiees** :
1. Utilise `trainSHGATOnExecution()` (V1 K-head) et non `trainStep()` (autograd-trainer TF.js)
2. Pas de PER — chaque trace est traitee individuellement sans priorite
3. Pas de protection contre le catastrophic forgetting (pas d'EWC, pas de replay buffer)
4. Pas de rate limiting — si 10 traces arrivent en 1s, 10 trainings sequentiels
5. Stats en memoire uniquement — perdues au restart

#### Composant B : PER Training (Batch)

**Fichier** : `src/graphrag/learning/per-training.ts`

**Fonctionnement** :
- Sampling de traces par priorite via `traceStore.sampleByPriority()`
- Aplatissement hierarchique des paths via `flattenExecutedPath()`
- Generation multi-exemples par trace via `traceToTrainingExamples()` (1 exemple par noeud du path)
- Semi-hard negative mining : middle tier des candidats tries par similarite cosinus a l'intent
- Curriculum learning : stocke `allNegativesSorted` (24 negatifs hard→easy) pour selection dynamique par le worker
- Exclusion des outils du cluster de l'ancre (community Louvain) pour eviter les faux negatifs
- Training dans un subprocess Deno via `spawnSHGATTraining()`
- Update des priorites PER post-training via `batchUpdatePrioritiesFromTDErrors()`

**Architecture** :
```
trainSHGATOnPathTracesSubprocess()
    |
    +--→ traceStore.sampleByPriority(maxTraces=100, minPriority=0.1, alpha=0.6)
    |
    +--→ flattenExecutedPath() × N traces  (recursif, inclut enfants)
    |
    +--→ traceToTrainingExamples()  (1 ex/noeud, semi-hard negatives)
    |
    +--→ spawnSHGATTraining()  (subprocess Deno, non-bloquant)
    |       |
    |       +--→ train-worker.ts (batchSize=32, epochs=25/1, InfoNCE, PER, curriculum)
    |       |
    |       +--→ retourne: params, tdErrors, finalLoss, finalAccuracy
    |
    +--→ shgat.importParams(result.params)
    |
    +--→ batchUpdatePrioritiesFromTDErrors(traces, tdErrors)
```

#### Composant C : TrainSHGATUseCase

**Fichier** : `src/application/use-cases/execute/train-shgat.use-case.ts`

**Fonctionnement** :
- Use case Clean Architecture appele apres chaque execution
- Verifie `shgatTrainer.shouldTrain()` (seuil d'accumulation + lock)
- Delegue a `ISHGATTrainer.train()` avec config live : epochs=1, temperature=0.07, usePER=false, useCurriculum=false, learningRate=0.03
- Met a jour Thompson Sampling avec les outcomes par outil

**Architecture** :
```
TrainSHGATUseCase.execute()
    |
    +--→ updateThompsonSampling(taskResults)  ← Beta distribution par outil
    |
    +--→ shgatTrainer.shouldTrain() ?
    |       |
    |       +--→ Non → return { trained: false }
    |       +--→ Oui → shgatTrainer.train(input, liveConfig)
    |
    +--→ onTrainingComplete?.()  ← callback pour sauvegarder params
```

#### Composant D : PostExecutionService

**Fichier** : `src/application/services/post-execution.service.ts`

**Fonctionnement** :
- Orchestre toutes les taches post-execution (fire and forget)
- Appelle `runPERBatchTraining()` en background avec `trainingLock`
- Config live identique : epochs=1, temperature=0.07, usePER=false, learningRate=0.03

**Architecture** :
```
PostExecutionService.process()
    |
    +--→ 1. updateDRDSP()           ← hyperedges DR-DSP
    +--→ 2. registerSHGATNodes()     ← ajout noeuds au graphe
    +--→ 3. updateThompsonSampling() ← Beta distributions
    +--→ 4. learnFromTaskResults()   ← fan-in/fan-out edges
    +--→ 5. enrichToolOutputSchemas()← ADR-061, output schemas
    +--→ 6. runPERBatchTraining()    ← PER subprocess (background, lock)
            |
            +--→ trainingLock.acquire("PER") ?
            +--→ db.query("SELECT ... FROM workflow_pattern ...")  ← 500 capabilities max
            +--→ trainSHGATOnPathTracesSubprocess(shgat, traceStore, ...)
```

#### Composant E : PER Priority

**Fichier** : `src/capabilities/per-priority.ts`

**Fonctionnement** :
- Calcul du TD Error : `actual - predicted` (SHGAT prediction vs outcome reel)
- Priority = `|TD Error|` clampee a [0.01, 1.0]
- Cold start : priority = 0.5 quand SHGAT n'a pas de noeuds
- `storeTraceWithPriority()` : calcule TD Error et stocke avec la trace
- `batchUpdatePriorities()` : recalcule apres training (forward pass SHGAT)
- `batchUpdatePrioritiesFromTDErrors()` : update depuis TD errors pre-calcules (plus efficace)

#### Composant F : Training Lock

**Fichier** : `src/graphrag/learning/training-lock.ts`

**Fonctionnement** :
- Mutex simple in-process : `acquire(owner)` / `release(owner)`
- Previent les trainings concurrents (BATCH vs PER)
- Limitation : pas de synchronisation inter-workers (chaque worker a son propre lock)

#### Composant G : SHGATTrainerAdapter

**Fichier** : `src/infrastructure/di/adapters/execute/shgat-trainer-adapter.ts`

**Fonctionnement** :
- Adaptateur Clean Architecture entre `ISHGATTrainer` et le `SHGATLiveTrainer` concret
- Initialisation lazy (`setLiveTrainer()`, `setThompsonSampling()`)
- Delegue `shouldTrain()` et `train()` au trainer concret

### 2.2 Flux complet de bout en bout

```
Execution terminee
    |
    v
PostExecutionService.process()
    |
    +--→ registerSHGATNodes()      ← noeuds dans le graphe
    +--→ updateThompsonSampling()  ← Beta dist par outil
    +--→ runPERBatchTraining()     ← subprocess avec PER sampling
    |
    v (en parallele)
TrainSHGATUseCase.execute()
    |
    +--→ shgatTrainer.train()      ← live training (1 epoch, LR=0.03)
    |
    v (en parallele)
eventBus.emit("execution.trace.saved")
    |
    v
OnlineLearningController
    +--→ trainSHGATOnExecution()   ← V1 K-head direct
```

**Probleme majeur identifie** : Il y a **3 chemins de training concurrents** qui s'executent potentiellement en parallele apres chaque execution :
1. `PostExecutionService.runPERBatchTraining()` — protege par `trainingLock`
2. `TrainSHGATUseCase.execute()` — verifie `shouldTrain()` mais pas `trainingLock`
3. `OnlineLearningController` — aucune protection

### 2.3 Gaps identifies

| # | Gap | Severite | Description |
|---|-----|----------|-------------|
| G1 | **Concurrence non coordonnee** | HAUTE | 3 chemins de training concurrents sans coordination unifiee. Le `trainingLock` ne protege que le chemin PER. |
| G2 | **V1 vs Autograd** | MOYENNE | `OnlineLearningController` utilise V1 K-head (`trainSHGATOnExecution`), le reste utilise le subprocess avec le worker V1. Aucun chemin n'utilise `autograd-trainer.ts` (TF.js). |
| G3 | **Pas de replay buffer** | MOYENNE | Chaque trace est traitee une fois par l'online controller. Le PER batch re-sample par priorite mais sans buffer circulaire explicite (depende de la DB). |
| G4 | **Catastrophic forgetting** | HAUTE | Aucune protection (EWC, distillation, experience replay). Avec ~10 traces/jour, un seul outlier peut deplacer les poids significativement (LR=0.03 est agressif pour du online). |
| G5 | **Stats non persistees** | BASSE | `OnlineLearningController.trainingCount` et `totalLoss` sont en memoire. Perdus au redemarrage. |
| G6 | **Pas de rate limiting** | BASSE | Si N executions arrivent en rafale, N trainings sequentiels. Le lock PER protege le batch mais pas l'online controller. |
| G7 | **Thompson Sampling duplique** | BASSE | `updateThompsonSampling()` est appele a la fois dans `PostExecutionService` et `TrainSHGATUseCase`. Potentiel double-count. |
| G8 | **Training lock mono-process** | BASSE | En multi-worker (cluster), chaque worker a son propre lock. Pas de coordination inter-workers. |

### 2.4 Recommandations

#### R1 : Unifier les chemins de training (adresse G1, G2)

Supprimer `OnlineLearningController` (V1) et `TrainSHGATUseCase` direct training. Garder un seul chemin : `PostExecutionService.runPERBatchTraining()` avec le subprocess.

**Avantage** : Un seul point d'entree, un seul lock, un seul format de training.
**Cout** : Verifier que tous les call sites passent par `PostExecutionService`.

#### R2 : Ajouter EWC ou distillation (adresse G4)

Pour le catastrophic forgetting, deux options :
- **EWC (Elastic Weight Consolidation)** : penalise les changements sur les poids importants. Necessite de stocker la matrice de Fisher (~2x la taille des params).
- **Distillation** : avant chaque online update, sauvegarder les predictions sur un set de reference. La loss inclut un terme KL(predictions_old || predictions_new).

**Recommandation** : Commencer par un **replay buffer** simple (garder les N dernieres traces en DB, re-sampler 50% du batch depuis le buffer). Le PER fait deja cela naturellement via `sampleByPriority()`.

#### R3 : Rate limiting sur le training (adresse G6)

Ajouter un debounce : au lieu de trainer a chaque execution, accumuler les traces et trainer toutes les N executions (deja implemente partiellement via `shouldRunBatchTraining(interval=10)`). Verifier que ce seuil est respecte dans tous les chemins.

#### R4 : Persister les stats de training (adresse G5)

Sauvegarder `trainingCount`, `avgLoss`, `lastTrainedAt` dans la base de donnees (table `shgat_training_stats` ou similaire). Permet le monitoring et la detection de regression.

#### R5 : Deduplication Thompson Sampling (adresse G7)

Appeler `updateThompsonSampling()` dans un seul endroit. Recommandation : uniquement dans `PostExecutionService`.

### 2.5 Etat actuel vs. roadmap P3-8

Le roadmap P3-8 decrivait le besoin d'online learning comme si c'etait a creer. En realite, le systeme est **deja en place** avec :

| Feature | Statut | Qualite |
|---------|--------|---------|
| Event-driven training | Implemente (`OnlineLearningController`) | Fonctionnel mais V1 K-head |
| PER sampling | Implemente (`per-training.ts`) | Production, alpha=0.6 |
| TD Error priority | Implemente (`per-priority.ts`) | Production |
| Subprocess training | Implemente (`spawn-training.ts`) | Non-bloquant |
| Training lock | Implemente (`training-lock.ts`) | Mono-process |
| Thompson Sampling | Implemente (`adaptive-threshold.ts`) | Production |
| Path flattening | Implemente (`per-training.ts`) | Recursif hierarchique |
| Semi-hard negatives | Implemente (`per-training.ts`) | Middle tier + Louvain cluster exclusion |
| Curriculum learning | Implemente (worker) | 24 negatifs sorted, tier dynamique |
| Live config (1ep, LR=0.03) | Implemente | Differenciee du batch |
| **EWC / Forgetting protection** | **NON** | Gap G4 |
| **Chemin unifie** | **NON** | Gap G1 — 3 chemins concurrents |
| **Stats persistees** | **NON** | Gap G5 |
| **Inter-worker coordination** | **NON** | Gap G8 |

**Verdict** : L'online learning est fonctionnel et en production. Les gaps identifies sont des problemes d'ingenierie (coordination, persistence, protection) et non des problemes d'architecture fondamentale. La priorite est G1 (unifier les chemins) avant d'ajouter de la complexite.

---

## Resume executif

### P2-6 : Full Softmax SHGAT

- **Quoi** : Remplacer ~8 negatifs par les 525 outils disponibles dans `trainStep()`
- **Ou** : `lib/shgat-tf/src/training/autograd-trainer.ts`, lignes 705-818
- **Comment** : Changer la collecte des `allNodeIds` pour inclure tous les `nodeEmbeddings.keys()`, et dans la boucle per-example utiliser tous les nodes comme negatifs
- **Impact** : Borne InfoNCE passe de 2.2 a 6.3 nats. +1-3 pts R@1 estime.
- **Effort** : 0.5 jour
- **Risque** : Faible. Temps de training ~5-10x plus long par epoch, memoire ~525x par exemple (2 MB vs 36 KB, total ~64 MB pour batch=32)

### P3-8 : Audit Online Learning

- **Statut** : DEJA EN PRODUCTION avec 7 composants
- **Gaps critiques** : G1 (3 chemins non coordonnes), G4 (catastrophic forgetting)
- **Gaps mineurs** : G5 (stats non persistees), G6 (rate limiting), G7 (Thompson duplique)
- **Action prioritaire** : Unifier les chemins de training (R1) avant d'ajouter des protections
- **Ce qui n'est PAS a faire** : Recreer l'online learning from scratch — le systeme existe et fonctionne

---

## Annexe : Fichiers references

| Fichier | Role |
|---------|------|
| `lib/shgat-tf/src/training/autograd-trainer.ts` | trainStep, infoNCELoss, batchContrastiveLoss (P2-6) |
| `src/graphrag/learning/online-learning.ts` | OnlineLearningController (V1 event-driven) |
| `src/graphrag/learning/per-training.ts` | PER batch training, path flattening, negative mining |
| `src/graphrag/learning/path-level-features.ts` | Path success rate, frequency, decision rate |
| `src/graphrag/learning/training-lock.ts` | Mutex mono-process |
| `src/graphrag/learning/mod.ts` | Exports du module learning |
| `src/graphrag/algorithms/shgat/spawn-training.ts` | Subprocess Deno pour training non-bloquant |
| `src/capabilities/per-priority.ts` | TD Error, PER priority, cold start |
| `src/application/use-cases/execute/train-shgat.use-case.ts` | Use case Clean Architecture |
| `src/application/services/post-execution.service.ts` | Orchestrateur post-execution |
| `src/infrastructure/di/adapters/execute/shgat-trainer-adapter.ts` | Adaptateur ISHGATTrainer |
| `src/domain/interfaces/shgat-trainer.ts` | Interface ISHGATTrainer |
