# Rapport de projet

Cours: 8INF804 - Vision Artificielle
Projet: Choix 1 - Détection et suivi d'objets dans des vidéos
Équipe: Anna-Eve Mercier, Laure Warlop, Flavien Baron, Evan Schwaller, Clément Auvray
Date: 27/04/2026

## Résumé exécutif
Ce projet vise à construire un pipeline complet de vision par ordinateur pour la détection et le suivi d'objets dans des vidéos. La solution implémentée couvre la préparation des données, l'entraînement et l'inférence de détecteurs, l'évaluation quantitative robuste, le suivi multi-objets, ainsi qu'un scénario de reproduction de bout en bout. Une comparaison entre un détecteur classique (YOLOv11, 0.672 mAP) et une approche basée Vision Transformer (DETR, 0.318 mAP) est intégrée avec un protocole d'évaluation amélioré incluant validation de cohérence, tests anti-régression et métriques opérationnelles.

## 1. Objectif de l'application
L'objectif principal est de développer une application reproductible capable de détecter des personnes dans des séquences vidéo, de suivre ces personnes dans le temps à l'aide d'identifiants persistants, et de produire des sorties à la fois quantitatives et qualitatives pour analyser les performances du pipeline.

## 2. Intention et motivation
La détection et le suivi d'objets constituent des briques essentielles pour de nombreuses applications réelles, qu'il s'agisse d'analyse de trafic, de sécurité, de robotique mobile ou de supervision vidéo. Le choix de ce projet permet donc de mobiliser des compétences complémentaires en modélisation et apprentissage profond, en ingénierie de pipeline de données, en évaluation expérimentale et en analyse critique, tout en accordant une attention particulière à la reproductibilité et à la qualité logicielle.

La motivation de l'équipe est de livrer une solution complète plutôt qu'un simple prototype de détection isolée.

## 3. Dataset choisi
Le pipeline s'appuie sur MOT17 pour les séquences vidéo et le suivi, ainsi que sur des représentations dérivées pour l'entraînement et l'évaluation de la détection. Les données brutes sont conservées dans data/raw/MOT17, tandis que les annotations de vérité terrain ont été converties en format COCO dans data/processed/val_gt.json et data/processed/train_gt.json. Pour l'entraînement du détecteur de référence, une structure YOLO complète a également été générée dans data/processed/MOT17_YOLO_DATASET, avec un fichier de configuration dédié dans data/processed/mot17_pedestrian_yolo.yaml.

Ce choix de dataset s'explique d'abord par son adéquation au suivi multi-objets, puisqu'il fournit des séquences urbaines réalistes où plusieurs personnes apparaissent simultanément, avec des situations de recouvrement, de mouvement et d'occlusion. Il a aussi l'avantage de s'intégrer naturellement dans un protocole d'évaluation standardisé grâce au format COCO, tout en restant compatible avec une phase d'expérimentation rapide via la préparation en format YOLO.

## 4. Méthodologie complète
### 4.1 Choix du dataset
Le choix de MOT17 répond à trois objectifs méthodologiques complémentaires. D'abord, il permet de disposer de séquences vidéo réelles dont la densité d'objets varie fortement d'une scène à l'autre, ce qui oblige les modèles à faire face à des cas plus proches de la réalité qu'un corpus synthétique ou trop homogène. Ensuite, MOT17 est un benchmark reconnu en suivi multi-objets, ce qui rend les résultats comparables à des travaux connus et facilite la justification des choix techniques. Enfin, il assure une cohérence forte entre les volets détection et tracking, puisque les mêmes séquences alimentent à la fois l'entraînement, l'inférence et l'évaluation.

Dans ce projet, la classe principale étudiée est la personne. Ce cadrage réduit l'ambiguïté du problème, facilite la comparaison entre approches, et correspond directement aux cas d'usage visés, à savoir les scènes urbaines et piétonnes.

### 4.2 Préparation des données
La préparation repose sur une chaîne de transformation contrôlée afin d'éviter les écarts de format entre les composants du projet. Les annotations MOT sont d'abord converties vers COCO pour l'évaluation quantitative, puis vers YOLO pour l'entraînement du détecteur classique. En parallèle, un split train/validation est défini au niveau des séquences et non image par image, ce qui limite le risque de fuite d'information temporelle entre les ensembles. Une vidéo source stable est également générée pour garantir que les inférences puissent être répétées dans des conditions identiques.

### 4.3 Conception de l'architecture
Le pipeline est découpé en modules spécialisés et interconnectés par un contrat de données explicite. Le sous-système object_detection prend en charge l'entraînement, l'inférence et le benchmark du détecteur classique. Le module vision_transformer_p3 fournit le détecteur alternatif de type Vision Transformer. Le pipeline-suivi-P4 se charge ensuite de l'association temporelle et de la génération des trajectoires, tandis que p5 assure la conversion, la validation de schéma et l'évaluation comparative. L'ensemble est orchestré par project/cli.py, qui sert de point d'entrée unifié.

Le principe retenu est celui du couplage faible: chaque module peut être exécuté indépendamment tant qu'il respecte les formats d'entrée et de sortie attendus. Cette organisation facilite le débogage, la réutilisation et l'évolution ultérieure du projet.

### 4.4 Entraînement
L'entraînement est piloté en deux niveaux. Le premier correspond au baseline prioritaire, construit autour de YOLOv11 afin d'obtenir rapidement un modèle exploitable par le reste du pipeline. Le second niveau repose sur une branche complémentaire Faster R-CNN, conservée pour comparaison et pour d'éventuelles extensions futures.

Le protocole d'entraînement du baseline comprend un paramétrage explicite des principaux hyperparamètres, notamment le nombre d'epochs, la taille d'image, la taille de batch et le device d'exécution. Avant le lancement d'un entraînement plus coûteux, les entrées sont vérifiées au moyen d'une validation du dataset YAML, afin d'éviter les erreurs de configuration. Enfin, les sorties d'entraînement sont sauvegardées pour pouvoir être auditées et reproduites ultérieurement.

Cette stratégie permet d'itérer rapidement sur un modèle performant tout en conservant une ouverture vers d'autres familles de détecteurs.

### 4.5 Évaluation quantitative
L'évaluation est conduite selon trois axes complémentaires. Le premier concerne la qualité de détection, mesurée à l'aide des métriques COCO standard (mAP@50-95, mAP@50, mAP@75) et de dérivées opérationnelles (Précision, Rappel, F1 à seuil configurable). Le deuxième porte sur la qualité du suivi, observée au moyen des métriques MOT (MOTA, MOTP, ID switches, IDF1) appliquées aux trajectoires. Le troisième évalue la performance système à travers le FPS, la latence moyenne et la consommation mémoire GPU.

Depuis la version améliorée du pipeline (P0-P4), l'évaluation intègre également:
- **Validation protocole (P0)**: Vérification explicite des modes (single-sequence/multi-sequence), cohérence source-vérité terrain, et checklist détaillée
- **Métriques opérationnelles (P1)**: Calcul de Précision/Rappel/F1 par seuil IoU configurableà la place d'une simple détermination binaire
- **Suivi avancé (P2)**: Ajout de IDF1 et HOTA (fallback) en complément de MOTA/MOTP
- **Tests qualité (P3)**: Sanity checks avec baselines empty (0.0) et oracle (1.0) pour détecter les régressions

Les résultats de P2 (YOLOv11) et P3 (DETR) sont convertis vers un format COCO commun avant mesure afin de garantir une comparaison équitable. L'évaluation est organisée par séquence avec agrégation statistique (mean, std, min, max) pour mesurer la stabilité inter-séquences.

Un verdict final **PASS/FAIL** est attribué selon la réussite conjointe des vérifications protocole et des sanity checks, validant ainsi que les résultats sont fiables et sans régression.

### 4.6 Analyse critique des résultats
L'analyse suit une grille systématique en cinq dimensions:

1. **Compromis précision-vitesse**: Comparaison quantifiée entre YOLOv11 (classique, rapide) et DETR (Vision Transformer, généraliste mais plus lent).

2. **Sensibilité aux hyperparamètres**: Documentation explicite des seuils configurables (seuil opérationnel pour Precision/Recall/F1, seuil IoU pour matching, paramètres ByteTrack).

3. **Robustesse inter-séquences**: Mesure de la stabilité des performances au-delà d'une seule séquence, via agrégation statistique (mean, std, min, max).

4. **Analyse d'erreurs qualitative**: Identification des catégories d'échecs (occlusions longues, objets proches, variations d'échelle, flou de mouvement) à partir des vidéos annotées.

5. **Validation anti-régression**: Suites de tests sanity (baselines empty=0.0 et oracle=1.0) et checklist protocole pour garantir la fiabilité de chaque exécution.

Les observations qualitatives, obtenues à partir des vidéos annotées et des fichiers de trajectoires, sont croisées avec les métriques quantitatives (COCO, MOT) pour construire un diagnostic multidimensionnel et éviter une conclusion fondée sur une seule source d'évidence.

### 4.7 Pistes d'amélioration
Les pistes d'amélioration découlent directement des limites observées. Une première direction consiste à systématiser l'optimisation des hyperparamètres au moyen d'une recherche structurée. Une seconde vise à améliorer la robustesse du suivi dans les situations d'occlusion longue. Une troisième porte sur l'extension du protocole de test à un plus grand nombre de séquences afin de mieux mesurer la généralisation. Enfin, l'ajout de validations automatiques sur la qualité minimale des sorties renforcerait encore la fiabilité du pipeline.

En complément, la reproductibilité est soutenue par un enchaînement de commandes standardisées pour la préparation, l'entraînement, l'inférence, et l'évaluation, ce qui permet de reconstruire le pipeline complet de manière déterministe.

## 5. Choix d'architecture et raisons
Les choix techniques principaux reposent sur YOLOv11 pour le détecteur classique, DETR pour la comparaison avec une famille Vision Transformer, et un pipeline dédié basé sur ByteTrack pour le suivi. L'orchestration est assurée par une CLI unifiée permettant d'exécuter les étapes de préparation des données, d'entraînement, d'inférence, d'évaluation et de smoke tests.

Ces choix ont été retenus pour trois raisons principales. Ils permettent d'abord de livrer rapidement une baseline exploitable par le suivi. Ils maintiennent ensuite un format de sortie commun entre les détecteurs, ce qui simplifie l'évaluation croisée. Ils facilitent enfin la reproductibilité et l'intégration continue des composants.

## 6. Analyse des résultats
L'analyse révèle un pipeline production reproductible et exécutable de bout en bout, avec un protocole d'évaluation robuste et anti-régression. Les implémentations P0 à P4 assurent une validation explicite, des métriques opérationnelles détaillées, et des tests de qualité systématiques.

### 6.1 Évaluation comparative P2 (YOLOv11) vs P3 (DETR) - Mode single-sequence

La comparaison a été effectuée sur la séquence MOT17-02-FRCNN en mode single-sequence avec seuil opérationnel 0.25. Les résultats COCO montrent une domination claire de P2 sur P3:

| Métrique | YOLOv11 (P2) | DETR (P3) |
|----------|------------|-----------|
| mAP@50-95 | 0.672 | 0.318 |
| mAP@50 | 0.837 | 0.445 |
| mAP@75 | 0.714 | 0.313 |
| Precision (seuil 0.25) | 0.895 | 0.489 |
| Recall (seuil 0.25) | 0.880 | 0.686 |
| F1-score (seuil 0.25) | 0.887 | 0.571 |

Ces métriques confirment que YOLOv11 offre un meilleur équilibre précision-rappel dans le contexte piéton du dataset MOT17. L'écart de mAP (0.354 points) reflète une meilleure calibration du détecteur sur les petits objets, typiques en suivi piéton urbain.

Source des résultats: `results/evaluation_pipeline_best_seq02/comparison_report.json`

### 6.2 Performances système

**Détection classique (P2 / YOLOv11)**
- Inférence sur 600 frames: ~50 FPS
- Débit compatible avec applications temps réel
- 600 frames vues et mappées avec succès (100% mapping rate)

**Vision Transformer (P3 / DETR)**
- Inférence: 0.98 FPS (benchmark: 0.97 FPS)
- Latence moyenne: 1024.49 ms/frame (benchmark: 1033.95 ms)
- Compromis vitesse/qualité défavorable sur CPU malgré meilleure généralité
- Intérêt potentiel sur GPU pour certaines applications spécialisées

### 6.3 Protocole d'évaluation robuste (P0-P4)

Le pipeline implémente un protocole complet validant la cohérence et la qualité:

**Validation protocole (P0)**
- Mode d'évaluation: single-sequence (garantit GT mono-séquence)
- Vérification source/GT coherence: **PASS** ✓
- Mapping frames: 600/600 (0 non mappés)
- Catégories vérifiées: person/pedestrian ✓

**Métriques opérationnelles (P1)**
- Seuil opérationnel configurable (défaut: 0.25)
- Précision, Rappel, F1 calculés par matching IoU
- Agrégation par séquence avec statistiques (mean, std, min, max)

**Sanity checks (P3)**
- Empty predictions baseline: mAP = 0.0 ✓ (attendu)
- Oracle predictions baseline: mAP = 1.0 ✓ (validation modèle)
- Status: **PASS** ✓

**Artefacts de sortie**
- `comparison_report.json`: Synthèse métriques P2/P3 avec verdict
- `protocol_checklist.json`: Checklist détaillée (source, mapping, catégories)
- `sanity_checks.json`: Baselines vides et oracle
- `comparison_report_per_sequence.json`: Résultats par séquence avec agrégats

### 6.4 Suivi (P4)

Les métriques MOT22 sur ByteTrack montrent que P2 reste la meilleure base pour le tracking:

| Métrique | P2-YOLO | P3-DETR |
|----------|---------|---------|
| MOTA | 0.176 | -0.979 |
| MOTP | 0.183 | 0.245 |
| ID Switches | 57 | 614 |

Le MOTA négatif de P3 indique que les erreurs de détection (fausses alarmes et non-détections) dominent, surchargeant l'algorithme d'association temporelle. Cela confirme que la qualité de détection prime sur les autres facteurs pour cette tâche.

### 6.5 Verdict final

Le pipeline atteint le verdict **PASS** ✓ pour le protocole complet, validant:
- Cohérence source-vérité terrain
- Sanity checks (baselines empty et oracle)
- Absence de régression

Cette validation assure que les résultats sont fiables et comparables pour des analyses ultérieures.
## 7. Travaux futurs
Les travaux futurs s'articulent autour de plusieurs pistes prioritaires, organisées par ordre d'impact estimé:

1. **Optimisation des hyperparamètres**: Conduire une recherche systématique (grille ou Bayesien) sur les paramètres clefs (confidence threshold, IoU threshold, ByteTrack parameters) afin d'identifier un point de fonctionnement plus robuste. Le protocole P1 (seuil opérationnel) facilite cette exploration.

2. **Extension du protocole de test**: Valider les performances sur l'ensemble complet de MOT17 (plutôt que MOT17-02 seule) pour mieux mesurer la généralisation et la stabilité inter-séquences.

3. **Amélioration du suivi en occlusion longue**: Intégrer de meilleures heuristiques pour maintenir les identités lors d'occlusions prolongées, et/ou ajouter un ré-identification (re-ID) de faible rang pour les relinkages.

4. **Optimisation inférentielle**: Déployer sur GPU, implémenter la quantification ou la distillation, afin de rapprocher P3 du temps réel et de réduire l'écart avec P2.

5. **Enrichissement de la validation**: Étendre les sanity checks à d'autres baselines (détecteurs pré-entraînés, résultats historiques) et à des indicateurs de confiance supplémentaires (variance par classe, distribution spatiale des erreurs).

6. **Documentation expérimentale**: Générer automatiquement des rapports détaillés par séquence et par classe, avec visualisations des courbes P/R, matrices de confusion et cartes d'erreur spatiales.

En parallèle, la reproductibilité reste un atout clef du projet: toutes les commandes standardisées sont documentées dans commandes.md et peuvent être exécutées de manière déterministe pour reconstruire le pipeline complet.

## 8. Synthèse technique
Le pipeline repose sur une architecture modulaire orchestrée par une CLI unifié (project/cli.py). Chaque module respecte un contrat de données explicite:

- **object_detection**: Entraînement et inférence YOLOv11, benchmark système
- **vision_transformer_p3**: Inférence DETR, benchmark système comparatif
- **evaluation_pipeline**: Conversion COCO, validation protocole (P0-P4), évaluation comparative, rapport structuré
- **pipeline-suivi-P4**: Tracking ByteTrack, métriques MOT avec IDF1/HOTA

Les formats de sortie sont standardisés (JSON pour prédictions/métriques, CSV pour rapports détaillés), facilitant l'intégration avec des outils d'analyse externes.

La robustesse est assurée par:
- Validation des schémas de données à chaque étape
- Gestion explicite des modes d'évaluation (single-sequence vs multi-sequence)
- Tests systématiques (sanity checks avec baselines)
- Rapports structurés incluant verdict et evidence trail

## 9. Reproductibilité et commande de référence
Le projet fournit des commandes de référence standardisées dans commandes.md permettant de reproduire les résultats rapportés. La commande complète d'évaluation comparative (P5) sur la séquence MOT17-02-FRCNN avec les nouveaux résultats est:

```powershell
python -m project.cli evaluation-pipeline \
  --skip-prepare-data \
  --eval-mode single-sequence \
  --operating-threshold 0.25 \
  --gt-json data/processed/gt_mot17_02_frcnn.json \
  --p2-preds results/object_detection/inference_best/predictions.json \
  --p3-preds results/p3/predictions.json \
  --output-dir results/evaluation_pipeline_best_seq02
```

Cette commande génère automatiquement:
- `comparison_report.json`: Synthèse métriques P2/P3
- `protocol_checklist.json`: Vérifications protocole P0
- `sanity_checks.json`: Tests baselines P3
- `comparison_report_per_sequence.json`: Résultats par séquence avec agrégats

L'ensemble des étapes (préparation données, entraînement, inférence, évaluation) peut être exécuté de manière déterministe en suivant les commandes documentées, garantissant la reproductibilité complète du pipeline.

## 10. Conclusion
Le projet livre un pipeline de vision par ordinateur reproductible et complet, couvrant détection, suivi et évaluation robuste. La baseline YOLOv11 (P2) atteint 0.672 mAP@50-95 et 50 FPS, démontrant un équilibre exploitable pour des applications proches du temps réel. La comparaison avec DETR (P3) révèle un écart de 0.354 points mAP au profit de YOLOv11, avec un coût calcul prohibitif (0.98 FPS).

Au-delà des scores absolus, le projet introduit un protocole d'évaluation P0-P4 robuste incluant:
- Validation explicite des modes d'évaluation et cohérence source-vérité terrain (P0)
- Métriques opérationnelles (Precision/Recall/F1) et agrégation par séquence (P1)
- Métriques de suivi enrichies (IDF1, HOTA fallback) (P2)
- Tests de qualité systématiques (sanity checks) (P3)
- Reproductibilité assurée et documentée (P4)

Cet investissement dans la robustesse et l'instrumentation pose une base solide pour les itérations futures. Les travaux futurs doivent porter sur l'optimisation des hyperparamètres, l'extension multi-séquences et l'amélioration du suivi en occlusion, tout en conservant la simplicité de reproduction qui constitue un point fort actuel du système.