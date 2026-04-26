# VA Projet - Detection et suivi de personnes (MOT17)

## 1) Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Preparation des donnees

```powershell
python -m project.cli evaluation-pipeline --mot-root data/raw/MOT17 --gt-json data/processed/val_gt.json --output-dir results/evaluation_pipeline --verify-p2-train-cmd --prepare-only
```

Cette commande genere notamment:
- data/processed/val_gt.json
- data/processed/train_gt.json
- data/processed/mot17_pedestrian_yolo.yaml

## 3) Entrainement P2 (YOLO)

```powershell
python -m project.cli object-detection train --model yolo --weights yolo11n.pt --dataset-yaml data/processed/mot17_pedestrian_yolo.yaml --epochs 30 --imgsz 640 --batch 16 --device 0 --project models/object_detection --name yolo_baseline
```

Poids entraines attendus:
- models/object_detection/yolo_baseline/weights/best.pt

## 4) Inference P2 et P3

### 4.1 Generer une video source stable

```powershell
python -m project.cli mot17-video --seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --output data/sample_video.mp4 --fps 25
```

### 4.2 Inference P2 (poids entraines)

```powershell
python -m project.cli object-detection infer --model yolo --weights models/object_detection/yolo_baseline/weights/best.pt --source data/sample_video.mp4 --output-dir results/object_detection/inference_best --conf 0.25 --iou 0.45 --device 0 --save-video
```

### 4.3 Inference P3 (DETR)

```powershell
python -m project.cli p3 --source data/sample_video.mp4 --model detr --output results/p3/predictions.json
```

## 5) Evaluation P5 (mAP)

Le protocole supporte maintenant 2 modes explicites:
- `--eval-mode single-sequence`: GT doit contenir exactement une sequence et doit correspondre a la source d'inference.
- `--eval-mode multi-sequence`: les predictions doivent contenir une metadata sequence (`sequence`, `sequence_name`, `seq`, `video`, `source`).

Le pipeline calcule aussi Precision/Recall/F1 au seuil operationnel `--operating-threshold` (defaut `0.25`).

### 5.1 Generer GT COCO mono-sequence (MOT17-02-FRCNN)

```powershell
python -c "from evaluation_pipeline.data.mot_to_coco import mot_to_coco; mot_to_coco('data/raw/MOT17','data/processed/gt_mot17_02_frcnn.json',['MOT17-02-FRCNN'])"
```

### 5.2 Evaluer P2 vs P3

Exemple `single-sequence` (recommande pour comparer deux inferences sur une video unique):

```powershell
python -m project.cli evaluation-pipeline --skip-prepare-data --eval-mode single-sequence --operating-threshold 0.25 --gt-json data/processed/gt_mot17_02_frcnn.json --p2-preds results/object_detection/inference_best/predictions.json --p3-preds results/p3/predictions.json --output-dir results/evaluation_pipeline_best_seq02
```

Exemple `multi-sequence` (predictions avec metadata sequence obligatoire):

```powershell
python -m project.cli evaluation-pipeline --skip-prepare-data --eval-mode multi-sequence --operating-threshold 0.25 --gt-json data/processed/val_gt.json --p2-preds results/object_detection/inference_best/predictions.json --p3-preds results/p3/predictions.json --output-dir results/evaluation_pipeline_multi
```

Fichiers utiles:
- results/evaluation_pipeline_best_seq02/comparison_report.json
- results/evaluation_pipeline_best_seq02/comparison_report.csv
- results/evaluation_pipeline_best_seq02/comparison_report_per_sequence.json
- results/evaluation_pipeline_best_seq02/protocol_checklist.json
- results/evaluation_pipeline_best_seq02/sanity_checks.json

## 6) Tracking P4

### 6.1 Tracking

```powershell
python -m project.cli p4 --detector-backend detections-json --mot-seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --detections-json results/object_detection/inference_best/predictions.json --output-dir results/p4 --output-name MOT17-02
```

### 6.2 Evaluation tracking

```powershell
python -m project.cli p4-eval --gt-file data/raw/MOT17/train/MOT17-02-FRCNN/gt/gt.txt --pred-file results/p4/MOT17-02.txt --output-csv results/p4/metrics_bytetrack.csv --output-json results/p4/metrics_bytetrack.json
```

Rapport tracking contient maintenant:
- MOTA
- MOTP
- ID switches
- IDF1
- HOTA (fallback explicite si bibliotheque non disponible)

## Notes

- Pour CPU seulement, remplacer --device 0 par --device cpu.
