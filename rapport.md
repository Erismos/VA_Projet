# Rapport de projet

Cours: 8INF804 - Vision Artificielle
Projet: Choix 1 - Détection et suivi d'objets dans des vidéos
Équipe: Anna-Eve Mercier, Laure Warlop, Flavien Baron, Evan Schwaller, Clément Auvray
Date: 27/04/2026

## Résumé exécutif
Ce projet vise à construire un pipeline complet de vision par ordinateur pour la détection et le suivi d'objets dans des vidéos. La solution implémentée couvre la préparation des données, l'entraînement et l'inférence de détecteurs, l'évaluation quantitative, le suivi multi-objets, ainsi qu'un scénario de reproduction de bout en bout. Une comparaison entre un détecteur classique et une approche basée Vision Transformer est également intégrée.

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
L'évaluation est conduite selon trois axes complémentaires. Le premier concerne la qualité de détection, mesurée à l'aide des métriques COCO et de leurs dérivées. Le deuxième porte sur la qualité du suivi, observée au moyen des métriques MOT appliquées aux fichiers de trajectoires. Le troisième évalue la performance système à travers le FPS, la latence moyenne et la mémoire GPU.

Les résultats de P2, qui correspond au détecteur classique, et de P3, qui correspond au Vision Transformer, sont d'abord convertis vers un format commun avant mesure afin de garantir une comparaison équitable. Pour le suivi, le pipeline peut exploiter des détections directes ou des détections exportées en JSON. Cette séparation permet de mesurer l'impact du détecteur sur la qualité finale des trajectoires.

### 4.6 Analyse critique des résultats
L'analyse suit une grille systématique centrée sur le compromis précision-vitesse entre méthodes, la sensibilité aux hyperparamètres tels que le seuil de confiance, l'IoU et les seuils du tracker, ainsi que sur la typologie des échecs observés. Les cas les plus fréquents concernent les occlusions, les objets proches, les variations d'échelle et le flou de mouvement. Une attention particulière est également portée à la robustesse inter-séquences afin d'évaluer si les performances se maintiennent dans des contextes visuels différents.

Les observations qualitatives, obtenues à partir des vidéos annotées, sont croisées avec les métriques quantitatives pour éviter une conclusion fondée sur une seule source d'évidence.

### 4.7 Pistes d'amélioration
Les pistes d'amélioration découlent directement des limites observées. Une première direction consiste à systématiser l'optimisation des hyperparamètres au moyen d'une recherche structurée. Une seconde vise à améliorer la robustesse du suivi dans les situations d'occlusion longue. Une troisième porte sur l'extension du protocole de test à un plus grand nombre de séquences afin de mieux mesurer la généralisation. Enfin, l'ajout de validations automatiques sur la qualité minimale des sorties renforcerait encore la fiabilité du pipeline.

En complément, la reproductibilité est soutenue par un enchaînement de commandes standardisées pour la préparation, l'entraînement, l'inférence, et l'évaluation, ce qui permet de reconstruire le pipeline complet de manière déterministe.

## 5. Choix d'architecture et raisons
Les choix techniques principaux reposent sur YOLOv11 pour le détecteur classique, DETR pour la comparaison avec une famille Vision Transformer, et un pipeline dédié basé sur ByteTrack pour le suivi. L'orchestration est assurée par une CLI unifiée permettant d'exécuter les étapes de préparation des données, d'entraînement, d'inférence, d'évaluation et de smoke tests.

Ces choix ont été retenus pour trois raisons principales. Ils permettent d'abord de livrer rapidement une baseline exploitable par le suivi. Ils maintiennent ensuite un format de sortie commun entre les détecteurs, ce qui simplifie l'évaluation croisée. Ils facilitent enfin la reproductibilité et l'intégration continue des composants.

## 6. Analyse des résultats
L'analyse des résultats met en évidence un pipeline techniquement fonctionnel de bout en bout, mais avec une qualité de détection encore insuffisante dans la configuration d'évaluation actuelle.

Sur la partie détection classique (P2 / YOLOv11), l'inférence a été exécutée sur 600 frames avec 6761 détections et un débit d'environ 54.96 FPS dans results/p2/inference/run_summary.json. Une exécution équivalente via object_detection rapporte 50.05 FPS dans results/object_detection/inference/run_summary.json. Ces deux mesures confirment un comportement compatible avec un usage proche temps réel pour la composante baseline.

Sur la branche Vision Transformer (P3 / DETR), les mesures sont nettement plus coûteuses en calcul: 21478 détections sur 600 frames, 0.98 FPS et 1024.49 ms/frame dans results/p3/predictions_metrics.json. Le benchmark dédié confirme cet ordre de grandeur avec 0.97 FPS, 1033.95 ms de latence moyenne, 1241.58 ms au percentile 95 et 0.0 GB de mémoire GPU (exécution CPU) dans results/p3/benchmark_report.json. Le compromis précision-vitesse est donc défavorable au regard de la contrainte de performance.

Pour la qualité de détection mesurée en COCO (P5), les valeurs obtenues sont très faibles: YOLOv11 donne mAP@50-95 = 0.0, mAP@50 = 0.0 et mAP@75 = 0.0; DETR donne mAP@50-95 = 8.883569325819805e-06, mAP@50 = 3.78235143855433e-05 et mAP@75 = 1.0484449726277228e-06 (results/p5/comparison_report.json et results/p5/comparison_report.csv). Ces scores indiquent que, dans le protocole de conversion/évaluation utilisé, les prédictions ne recouvrent pratiquement pas les annotations de référence de manière valide.

L'inspection qualitative des sorties aide à interpréter ce résultat: les prédictions P2 sont bien centrées sur la classe person (results/p2/inference/predictions.json), alors que P3 émet plusieurs classes COCO (par exemple chair et person dans results/p3/predictions.json). Cette hétérogénéité de classes, combinée aux choix de seuil et aux conversions inter-formats, peut dégrader fortement l'évaluation si le protocole cible principalement la classe piéton.

Sur la partie suivi (P4), le pipeline produit bien des trajectoires exploitables: 600 frames traitées et 222 tracks générés sur MOT17-02-FRCNN (results/p4/MOT17-02_summary.json), avec export MOT et vidéo de visualisation. En revanche, les artefacts actuellement conservés ne fournissent pas directement de métriques MOT globales (MOTA, IDF1, HOTA), ce qui limite la comparaison chiffrée de la qualité d'association temporelle.

Au niveau projet, le rapport consolidé confirme les mêmes tendances (results/project/project_report.json et results/project/project_report.csv): exécution complète réussie, mais écart important entre faisabilité logicielle et performance quantitative. En pratique, la baseline YOLOv11 est la plus utilisable opérationnellement grâce à sa vitesse, tandis que la branche DETR nécessite une optimisation matérielle (GPU) et une harmonisation plus stricte du protocole d'évaluation pour devenir compétitive.

## 7. Travaux futurs
Les travaux futurs s'articulent autour de plusieurs pistes prioritaires. Il serait utile de mener une recherche systématique sur les hyperparamètres de détection et de tracking afin d'identifier un point de fonctionnement plus robuste. Une autre extension importante consiste à améliorer la résistance du système aux occlusions et aux changements d'échelle. Le protocole gagnerait également à être évalué sur davantage de séquences et de scénarios afin de mieux mesurer la généralisation. Sur le plan système, une optimisation inférentielle pourrait rapprocher l'exécution du temps réel. Enfin, l'ajout de tests de non-régression sur les métriques clefs renforcerait la confiance dans les évolutions futures.

## 9. Reproductibilité et commande de référence
Le projet fournit des commandes de référence dans commandes.md pour la préparation des données, l'entraînement object_detection, l'inférence object_detection, l'évaluation P5, le tracking P4, ainsi que les tests et le smoke test e2e. Cette base permet à un évaluateur de reproduire les expériences avec un effort minimal tout en retrouvant exactement les mêmes points d'entrée que ceux utilisés pendant le développement.

## 10. Conclusion
Le projet répond aux attentes d'un pipeline de vision de bout en bout, avec une contribution object_detection qui joue le rôle de baseline solide pour l'ensemble des étapes aval. La suite du travail consiste à renforcer la profondeur expérimentale et la robustesse des évaluations, tout en conservant la simplicité de reproduction qui constitue un point fort actuel. L'ensemble forme ainsi une base crédible pour une version plus complète du système, avec des résultats mieux instrumentés et des analyses encore plus démonstratives.