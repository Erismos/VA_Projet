# DONE - Suivi d'implémentation

Date de début: 2026-04-21
Objectif: implémenter progressivement les actions listées dans TODO.md jusqu'à un pipeline inter-parties réellement fonctionnel.

## Itération 1 - Intégration P5 sur sorties réelles

### 1) Adaptateur de format prédictions -> COCO
Statut: fait

Actions réalisées:
- Ajout de `p5/eval/adapters.py`.
- Conversion supportée:
  - format natif P2/P3 (`frame_id`, `class_name`/`category_id`, `bbox`, `score`)
  - format déjà COCO (`image_id`, `category_id`, `bbox`, `score`)
- Mapping `frame_id -> image_id` construit à partir du `file_name` du GT COCO.
- Gestion du décalage fréquent 0-based -> 1-based (`frame_id + 1`).
- Mapping de classes par défaut: `person` et `pedestrian` -> `category_id = 1`.

### 2) Pipeline P5 principal branché sur vraies prédictions
Statut: fait

Actions réalisées:
- Refonte de `main_p5.py`:
  - mode par défaut = évaluation sur prédictions réelles P2/P3;
  - mode mock conservé uniquement via `--use-mock`.
- Nouveaux arguments CLI:
  - `--p2-preds`
  - `--p3-preds`
  - `--gt-json`
  - `--skip-prepare-data`
  - `--output-dir`
- Conversion automatique des prédictions vers COCO dans `results/p5/converted/`.
- Génération d'un rapport de comparaison:
  - JSON: `results/p5/comparison_report.json`
  - CSV: `results/p5/comparison_report.csv`

### 3) Bridge données P5 -> entraînement P2
Statut: partiellement fait

Actions réalisées:
- Intégration de `create_yolo_dataset(...)` dans `main_p5.py`.
- Génération automatique de `data/processed/mot17_pedestrian_yolo.yaml`.
- Le pipeline produit maintenant une structure YOLO train/val prête pour Ultralytics.

Reste à faire:
- Vérifier sur données réelles volumineuses la robustesse des copies image/label et des cas limites fichiers manquants.

### 4) Dépendances
Statut: partiellement fait

Actions réalisées:
- Ajout de `transformers>=4.38.0` dans `requirements.txt` pour P3.

Reste à faire:
- Valider versions exactes compatibles GPU/CPU selon environnement cible.

### 5) Qualité / tests
Statut: fait (premier niveau)

Actions réalisées:
- Ajout d'un test unitaire: `tests/test_p5_adapters.py`.
- Le test vérifie la conversion P2/P3 -> COCO et le mapping `frame_id -> image_id`.

## Prochaines étapes planifiées

1. Refactorer `pipeline-suivi-P4/track-bytetrack.py` en CLI configurable (suppression hardcoding).
2. Brancher P4 sur détections exportées P2/P3 (et non modèle local fixé).
3. Unifier les conventions de noms de sorties entre P2/P3/P5.
4. Ajouter validations de schéma JSON avant l'évaluation.

## Itération 2 - Refactor P4 tracking

### 1) CLI configurable sans hardcoding
Statut: fait

Actions réalisées:
- Réécriture complète de `pipeline-suivi-P4/track-bytetrack.py`.
- Nouveaux paramètres CLI:
  - `--mot-seq-dir`
  - `--output-dir`
  - `--output-name`
  - `--detector-backend`
  - `--weights`
  - `--detections-json`
  - `--conf`, `--iou`, `--device`
  - `--tracker-iou`, `--tracker-max-age`

### 2) Backends de détection/tracking
Statut: fait

Backends supportés:
- `ultralytics-track`: tracking ByteTrack natif Ultralytics.
- `p2-yolo`: détection via P2 YOLO + suivi IoU local.
- `p2-fasterrcnn`: détection via P2 Faster R-CNN + suivi IoU local.
- `detections-json`: consommation directe d'un export JSON P2/P3 + suivi IoU local.

### 3) Sorties standardisées P4
Statut: fait

Sorties générées:
- Vidéo annotée: `results/p4/<output-name>.mp4`
- Résultats MOT: `results/p4/<output-name>.txt`
- Résumé JSON: `results/p4/<output-name>_summary.json`
- Fichier de config exemple: `configs/p4_tracking_example.yaml`

### 4) Points restants
Statut: à faire

- Ajouter un test d'intégration minimal pour le mode `detections-json`.

## Itération 3 - Reproductibilité globale et documentation

### 1) Point d'entrée unifié du projet
Statut: fait

Actions réalisées:
- Ajout du package `project/`.
- Ajout de `project/cli.py` pour lancer P1, P2, P3, P3 benchmark, P4, P4 eval, P4 visualize et P5 via une commande unique:
  - `python -m project.cli <component> ...`

### 2) Harmonisation P3 (docs + sorties)
Statut: fait

Actions réalisées:
- Mise à jour des exemples dans `vision_transformer_p3/detector.py`:
  - utilisation de `python -m vision_transformer_p3.detector`
- Mise à jour des chemins de sortie par défaut:
  - détections: `results/p3/predictions.json`
  - benchmark: `results/p3/benchmark_report.json`

### 3) Validation de schéma avant évaluation P5
Statut: fait

Actions réalisées:
- Ajout de `validate_predictions_schema(...)` dans `p5/eval/adapters.py`.
- Validation appelée dans `main_p5.py` avant conversion/evaluation.
- Extension des tests dans `tests/test_p5_adapters.py`.

### 4) Documentation workflow
Statut: fait

Actions réalisées:
- Ajout dans `README.md`:
  - section point d'entrée unifié;
  - commandes d'exemple P1..P5;
  - section workflow bout-en-bout.

## Itération 4 - Harmonisation des chemins P4

### 1) Evaluation P4 sans hardcoding
Statut: fait

Actions réalisées:
- Réécriture de `pipeline-suivi-P4/evaluate.py` avec arguments CLI:
  - `--gt-file`
  - `--pred-file`
  - `--output-csv`

### 2) Visualisation P4 sans hardcoding
Statut: fait

Actions réalisées:
- Réécriture de `pipeline-suivi-P4/visualize.py` avec arguments CLI:
  - `--seq-img-dir`
  - `--pred-file`
  - `--output`
  - `--trail-len`
  - `--fps`

Résultat:
- Les scripts P4 utilisent désormais par défaut `data/raw/MOT17/...` et `results/p4/...`.

## Itération 5 - Validation du bridge P5 -> P2 + test d'intégration P4

### 1) Vérification images/labels et splits YOLO
Statut: fait

Actions réalisées:
- Renforcement de `p5/data/split_data.py`:
  - protection contre dossiers manquants pendant la copie;
  - copie image conditionnelle si le fichier source existe;
  - ajout de `validate_yolo_dataset(...)`.
- Validation automatique dans `main_p5.py` après création du dataset YOLO.
- Le pipeline échoue explicitement si le split train/val est vide ou si des paires image/label sont incohérentes.

### 2) Vérification de la commande d'entraînement P2 sur sortie P5
Statut: fait

Actions réalisées:
- Ajout de `validate_yolo_training_inputs(...)` dans `p2/train.py`.
- Ajout d'un mode dry-run dans `p2/cli.py`:
  - `python -m project.cli p2 train --model yolo --dataset-yaml <...> --dry-run`
- Intégration dans `main_p5.py`:
  - génération automatique d'une commande recommandée dans `results/p5/p2_train_command.txt`;
  - option `--verify-p2-train-cmd` pour valider la faisabilité de l'entraînement sans lancer de training lourd.

### 3) Test d'intégration minimal P4 en mode detections-json
Statut: fait

Actions réalisées:
- Ajout de `tests/test_p4_tracking_json.py`.
- Test couvrant:
  - création d'une mini séquence MOT synthétique;
  - exécution du backend `detections-json`;
  - vérification de la génération des sorties MOT et résumé JSON.

### 4) Tests bridge P5 -> P2
Statut: fait

Actions réalisées:
- Ajout de `tests/test_p5_data_bridge.py`.
- Vérifie:
  - cohérence image/label/split via `validate_yolo_dataset(...)`;
  - validation du `dataset_yaml` via `validate_yolo_training_inputs(...)`.

### 5) Résultat de validation locale
Statut: fait

Commande exécutée:
- `python -m pytest -q tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`

Résultat:
- 5 tests passés.

## Itération 9 - Consolidation métriques et couverture de validation

### 1) Tableau consolidé projet
Statut: fait

Actions réalisées:
- Ajout d'un artefact consolidé au niveau projet dans `project/e2e.py`.
- Génération automatique de:
  - `project_report.json`
  - `project_report.csv`
- Les métriques consolidées couvrent:
  - P5 détection (`p2_map_50_95`, `p3_map_50_95`)
  - P4 tracking (`frames`, `tracks`)

### 2) Tests de validation normalisée
Statut: fait

Actions réalisées:
- Ajout de `tests/test_p5_validation.py`.
- Vérification du contrat de retour normalisé:
  - `ok`
  - `reason`
  - `details`
- Cas couverts:
  - prédictions valides/invalides;
  - layout YOLO valide;
  - entrée d'entraînement P2 valide.

### 3) Couverture e2e complète et sans P4
Statut: fait

Actions réalisées:
- Ajout de `tests/test_project_e2e.py`.
- Couverture des deux chemins:
  - `skip_p4=True`
  - `skip_p4=False`
- Vérification de la génération des artefacts projet consolidés.

### 4) Vérification après consolidation
Statut: fait

Commande exécutée:
- `python -m pytest -q tests/test_p5_validation.py tests/test_project_e2e.py tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`

Résultat:
- 11 tests passés.

## Itération 11 - Restauration de Kaggle comme source par défaut

### 1) Téléchargement MOT17 prioritaire via Kaggle
Statut: fait

Actions réalisées:
- Restauration du chemin Kaggle dans `p5/data/download_mot17.py`.
- Le téléchargement tente maintenant Kaggle en premier, puis bascule sur le miroir HTTP si nécessaire.
- Extraction automatique de la première archive ZIP téléchargée.

### 2) Dépendance réajoutée
Statut: fait

Actions réalisées:
- Réintégration de `kaggle` dans `requirements.txt`.
- Le projet redevient compatible avec un setup Kaggle standard via `kaggle.json`.

### 3) Vérification après restauration
Statut: fait

Commande exécutée:
- `python -m pytest -q tests/test_p5_validation.py tests/test_project_e2e.py tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`

Résultat:
- 11 tests passés.

## Itération 10 - Suppression de la dépendance Kaggle

### 1) Téléchargement MOT17 sans Kaggle
Statut: fait

Actions réalisées:
- Réécriture de `p5/data/download_mot17.py` pour utiliser uniquement le miroir HTTP public.
- Suppression du chemin d'exécution Kaggle.
- Ajout d'un `timeout` et d'un `raise_for_status()` pour rendre l'échec explicite.

### 2) Dépendances allégées
Statut: fait

Actions réalisées:
- Suppression de `kaggle` dans `requirements.txt`.
- Le projet ne requiert plus de configuration `kaggle.json` pour préparer MOT17.

### 3) Vérification après nettoyage
Statut: fait

Commande exécutée:
- `python -m pytest -q tests/test_p5_validation.py tests/test_project_e2e.py tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`

Résultat:
- 11 tests passés.

## Itération 8 - Suppression des modes mock et dummy fallback

### 1) Pipeline P5 strictement réel
Statut: fait

Actions réalisées:
- Suppression du support `use_mock` dans `p5/pipeline.py`.
- Suppression du support `allow_mock` et `allow_dummy_fallback`.
- `evaluate_p5(...)` ne traite plus que des prédictions réelles P2/P3.

### 2) Suppression du fallback dummy implicite
Statut: fait

Actions réalisées:
- Suppression de l'appel implicite à `create_dummy_mot17(...)` dans le flux P5.
- En cas d'absence de MOT17 ou d'échec de téléchargement, le pipeline s'arrête avec exception.

### 3) Nettoyage CLI et documentation
Statut: fait

Actions réalisées:
- Retrait des flags mock/dummy du CLI P5 dans `main_p5.py`.
- Mise à jour du guide principal dans `README.md` pour ne plus mentionner le mode mock comme chemin d'exécution.
- Retrait du paramètre obsolète dans `project/e2e.py`.

### 4) Suppression des modules de secours orphelins
Statut: fait

Actions réalisées:
- Suppression de `p5/data/create_dummy_data.py`.
- Suppression de `p5/eval/generate_mock_results.py`.
- Le dépôt ne conserve plus de chemin d'exécution mock/dummy pour P5.

### 5) Vérification après suppression
Statut: fait

Commande exécutée:
- `python -m pytest -q tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`

Résultat:
- 5 tests passés.

## Itération 6 - Simplification structurelle du projet

### 1) Découpage du pipeline P5 (préparation vs évaluation)
Statut: fait

Actions réalisées:
- Ajout de `p5/pipeline.py` avec:
  - `P5Config` pour centraliser les paramètres/chemins par défaut;
  - `prepare_p5_data(...)` pour la partie data bridge;
  - `evaluate_p5(...)` pour la partie évaluation;
  - `run_p5_pipeline(...)` comme orchestration légère.
- Allègement de `main_p5.py` en point d'entrée CLI minimal délégant à `p5.pipeline`.

### 2) Unification des validateurs
Statut: fait

Actions réalisées:
- Ajout de `p5/validation.py` pour regrouper:
  - validation schéma prédictions;
  - validation layout dataset YOLO;
  - validation des entrées d'entraînement P2.
- Harmonisation des retours de validation dans un format commun:
  - `ok`, `reason`, `details`.

### 3) Commande pipeline simplifiée
Statut: fait

Actions réalisées:
- Ajout de `project/e2e.py` pour exécuter un smoke pipeline unique:
  - évaluation P5 réelle (sans préparation);
  - tracking P4 optionnel en mode `detections-json`.
- Ajout du composant CLI `e2e-smoke` dans `project/cli.py`.

### 4) Vérifications après simplification
Statut: fait

Commandes exécutées:
- `python -m pytest -q tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`
- `python -m project.cli e2e-smoke --help`

Résultats:
- Régression tests: 5 tests passés.
- Commande unifiée `e2e-smoke` disponible et documentée par l'aide CLI.

## Itération 7 - Nettoyage des modes de secours non essentiels

### 1) Mock sorti du flux principal de prod
Statut: fait

Actions réalisées:
- Durcissement de `p5/pipeline.py`:
  - `use_mock` ne fonctionne plus sans opt-in explicite.
  - ajout de `allow_mock` dans `P5Config`.
  - erreur explicite si `--use-mock` est utilisé sans `--allow-mock`.

### 2) Fallback dummy rendu explicite
Statut: fait

Actions réalisées:
- Suppression du fallback implicite vers `create_dummy_mot17(...)` en cas d'échec de téléchargement.
- Ajout de `allow_dummy_fallback` dans `P5Config`.
- Le fallback dummy n'est exécuté que si opt-in explicite via CLI.
- Sinon, le pipeline s'arrête avec message d'erreur clair.

### 3) Alignement CLI
Statut: fait

Actions réalisées:
- Mise à jour de `main_p5.py`:
  - ajout de `--allow-mock` (requis avec `--use-mock`);
  - ajout de `--allow-dummy-fallback`.
- Les messages d'aide indiquent clairement que ces modes sont réservés aux runs demo/test.

### 4) Vérification après nettoyage
Statut: fait

Commande exécutée:
- `python -m pytest -q tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py`

Résultat:
- 5 tests passés.
