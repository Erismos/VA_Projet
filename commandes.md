# Commandes du projet

Séquence complète pour exécuter le projet de bout en bout sur Windows PowerShell.

## 0) Pré-requis

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Data

### 1.1 Préparer les données MOT17 et les fichiers de travail P5
Cette commande prépare le dataset, génère le GT COCO et la structure YOLO pour P2.

```powershell
python -m project.cli p5 --mot-root data/raw/MOT17 --gt-json data/processed/val_gt.json --output-dir results/p5 --verify-p2-train-cmd --prepare-only
```

Sorties attendues:
- `data/processed/val_gt.json`
- `data/processed/train_gt.json`
- `data/processed/mot17_pedestrian_yolo.yaml`
- `results/p5/p2_train_command.txt`

## 2) Train

### 2.1 Entraîner P2 YOLO
Utiliser le YAML généré à l'étape data.

```powershell
python -m project.cli p2 train --model yolo --weights yolo11n.pt --dataset-yaml data/processed/mot17_pedestrian_yolo.yaml --epochs 30 --imgsz 640 --batch 16 --device cpu --project models/p2 --name yolo_baseline
```

### 2.2 Entraîner P2 Faster R-CNN (optionnel)

```powershell
python -m project.cli p2 train --model fasterrcnn --dataset-root data/processed/MOT17_YOLO_DATASET --epochs 10 --device cpu --output-dir models/p2/fasterrcnn_baseline
```

## 3) Inference / Export des prédictions

### 3.1 P2 inference

```powershell
python -m project.cli p2 infer --model yolo --weights yolo11n.pt --source data/sample_video.mp4 --output-dir results/p2/inference --conf 0.25 --iou 0.45 --device cpu --save-video
```

### 3.2 P3 inference

```powershell
python -m project.cli p3 --source data/sample_video.mp4 --model detr --output results/p3/predictions.json
```

### 3.3 P3 benchmark

```powershell
python -m project.cli p3-benchmark --source data/sample_video.mp4 --model detr --output results/p3/benchmark_report.json
```

## 4) Evaluate

### 4.1 Evaluation P5 sur prédictions réelles
Cette commande compare P2 et P3 avec le GT COCO.

```powershell
python -m project.cli p5 --skip-prepare-data --gt-json data/processed/val_gt.json --p2-preds results/p2/inference/predictions.json --p3-preds results/p3/predictions.json --output-dir results/p5
```

Sorties attendues:
- `results/p5/comparison_report.json`
- `results/p5/comparison_report.csv`
- `results/p5/converted/p2_coco_predictions.json`
- `results/p5/converted/p3_coco_predictions.json`

### 4.2 Tracking P4 avec détections P2/P3

```powershell
python -m project.cli p4 --detector-backend detections-json --mot-seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --detections-json results/p2/inference/predictions.json --output-dir results/p4 --output-name MOT17-02
```

### 4.3 Evaluation tracking P4

```powershell
python -m project.cli p4-eval --gt-file data/raw/MOT17/train/MOT17-02-FRCNN/gt/gt.txt --pred-file results/p4/MOT17-02.txt --output-csv results/p4/metrics_bytetrack.csv
```

### 4.4 Visualisation tracking P4

```powershell
python -m project.cli p4-visualize --seq-img-dir data/raw/MOT17/train/MOT17-02-FRCNN/img1 --pred-file results/p4/MOT17-02.txt --output results/p4/demo_visualization.mp4 --trail-len 40 --fps 25
```

## 5) Test

### 5.1 Tests unitaires et d'intégration

```powershell
python -m pytest -q tests/test_p5_validation.py tests/test_project_e2e.py tests/test_p5_adapters.py tests/test_p5_data_bridge.py tests/test_p4_tracking_json.py
```

### 5.2 Smoke test projet complet

```powershell
python -m project.cli e2e-smoke --output-dir results/project --mot-seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --gt-json data/processed/val_gt.json --p2-preds results/p2/inference/predictions.json --p3-preds results/p3/predictions.json --p5-output-dir results/project/p5 --p4-output-dir results/project/p4 --output-name MOT17-02-smoke
```

Sorties attendues:
- `results/project/project_report.json`
- `results/project/project_report.csv`
- `results/project/p5/comparison_report.json`
- `results/project/p4/MOT17-02-smoke.txt`

## 6) Ordre recommandé

1. Installer l'environnement.
2. Lancer la préparation data P5.
3. Entraîner P2.
4. Produire les prédictions P2 et P3.
5. Lancer l'évaluation P5.
6. Lancer le tracking P4.
7. Exécuter les tests.
8. Lancer le smoke test projet complet.
