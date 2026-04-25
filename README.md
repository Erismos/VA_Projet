# VA_Projet

## P2 - Detecteur classique (implementation)

Cette implementation couvre la mission P2:
- entrainement baseline (YOLOv11 prioritaire),
- inference image/video,
- export predictions JSON et CSV,
- benchmark FPS et memoire GPU.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Structure ajoutee

- object_detection/train.py: entrainement YOLO + squelette Faster R-CNN
- object_detection/inference.py: inference image/video et export commun
- object_detection/benchmark.py: mesure FPS et memoire GPU
- object_detection/cli.py: interface unique
- configs/p2_yolo_example.yaml: exemple de configuration

## Commandes

### 1) Entrainement YOLO baseline

```bash
python main.py train --model yolo --weights yolo11n.pt --dataset-yaml data/coco_pedestrian_subset/data.yaml --epochs 30 --imgsz 640 --batch 16 --device 0 --project models/object_detection --name yolo_baseline
```

### 2) Entrainement Faster R-CNN (starter)

```bash
python main.py train --model fasterrcnn --dataset-root data/coco_pedestrian_subset --epochs 10 --device cpu --output-dir models/object_detection/fasterrcnn_baseline
```

### 3) Inference + export JSON/CSV

```bash
python main.py infer --model yolo --weights yolo11n.pt --source data/sample_video.mp4 --output-dir results/object_detection/inference --conf 0.25 --iou 0.45 --device 0 --save-video
```

### 4) Benchmark FPS/GPU

```bash
python main.py benchmark --model yolo --weights yolo11n.pt --source data/sample_video.mp4 --conf 0.25 --iou 0.45 --device 0 --warmup-frames 10 --max-frames 200 --output results/object_detection/benchmark_report.json
```

## Format de sortie commun

Fichier JSON: results/object_detection/inference/predictions.json

Schema minimal par detection:
- frame_id
- class_name
- category_id
- bbox au format [x, y, w, h]
- score

Fichier CSV: results/object_detection/inference/predictions.csv

Colonnes:
- frame_id,class_name,x,y,w,h,score

## Sorties generees

- models/object_detection/...: checkpoints et resume d entrainement
- results/object_detection/inference/: predictions.json, predictions.csv, run_summary.json, video/image annotee
- results/object_detection/benchmark_report.json: rapport FPS et memoire GPU

## Notes

- Le mode Faster R-CNN est un starter pour laisser le choix technique, mais le baseline prioritaire est YOLOv11 pour livrer vite P4.
- Pour CUDA, remplacer --device cpu par --device 0.

## Checklist validation P2

- [x] Detecteur classique integre et executable.
- [x] conf, IoU, NMS exposes et documentes dans les commandes.
- [x] Export JSON/CSV conforme au format commun.
- [x] Mesure FPS et memoire GPU disponible via benchmark.
- [x] Baseline YOLO livrable rapidement pour P4.
- [x] Instructions de reproduction presentes.

## Point d'entree unifie (P1 a P5)

Un lanceur unique est disponible via `python -m project.cli`.

Exemples:

```bash
# P2
python -m project.cli object-detection infer --model yolo --weights yolo11n.pt --source data/sample_video.mp4 --output-dir results/object_detection/inference --save-video

# P3 detection
python -m project.cli p3 --source data/sample_video.mp4 --model detr --output results/p3/predictions.json

# P3 benchmark
python -m project.cli p3-benchmark --source data/sample_video.mp4 --model detr --output results/p3/benchmark_report.json

# P4 tracking
python -m project.cli p4 --detector-backend detections-json --mot-seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --detections-json results/object_detection/inference/predictions.json --output-dir results/p4 --output-name MOT17-02

# P5 evaluation (reel, sans mock)
python -m project.cli p5 --skip-prepare-data --gt-json data/processed/val_gt.json --p2-preds results/object_detection/inference/predictions.json --p3-preds results/p3/predictions.json --output-dir results/p5

# Video source reproductible pour inference (depuis MOT17)
python -m project.cli mot17-video --seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --output data/sample_video.mp4 --fps 25
```

## Workflow Bout-en-Bout recommande

1. Preparer les donnees MOT17 et le GT COCO (P5):

```bash
python -m project.cli p5 --mot-root data/raw/MOT17
```

2. Generer une video source stable pour P2/P3 (a partir de MOT17):

```bash
python -m project.cli mot17-video --seq-dir data/raw/MOT17/train/MOT17-02-FRCNN --output data/sample_video.mp4 --fps 25
```

3. Produire des predictions P2 et P3 (JSON).
4. Lancer l'evaluation reelle P5 avec les chemins de predictions.
5. Lancer P4 en mode `detections-json` pour suivre a partir des detections exportees.

Sorties principales:
- `results/object_detection/inference/predictions.json`
- `results/p3/predictions.json`
- `results/p5/comparison_report.json`
- `results/p4/<output-name>.txt` (format MOT)