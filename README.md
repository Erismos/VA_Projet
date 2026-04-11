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

- p2/train.py: entrainement YOLO + squelette Faster R-CNN
- p2/inference.py: inference image/video et export commun
- p2/benchmark.py: mesure FPS et memoire GPU
- p2/cli.py: interface unique
- configs/p2_yolo_example.yaml: exemple de configuration

## Commandes

### 1) Entrainement YOLO baseline

```bash
python main.py train --model yolo --weights yolo11n.pt --dataset-yaml data/coco_pedestrian_subset/data.yaml --epochs 30 --imgsz 640 --batch 16 --device 0 --project models/p2 --name yolo_baseline
```

### 2) Entrainement Faster R-CNN (starter)

```bash
python main.py train --model fasterrcnn --dataset-root data/coco_pedestrian_subset --epochs 10 --device cpu --output-dir models/p2/fasterrcnn_baseline
```

### 3) Inference + export JSON/CSV

```bash
python main.py infer --model yolo --weights yolo11n.pt --source data/sample_video.mp4 --output-dir results/p2/inference --conf 0.25 --iou 0.45 --device 0 --save-video
```

### 4) Benchmark FPS/GPU

```bash
python main.py benchmark --model yolo --weights yolo11n.pt --source data/sample_video.mp4 --conf 0.25 --iou 0.45 --device 0 --warmup-frames 10 --max-frames 200 --output results/p2/benchmark_report.json
```

## Format de sortie commun

Fichier JSON: results/p2/inference/predictions.json

Schema minimal par detection:
- frame_id
- class_name
- category_id
- bbox au format [x, y, w, h]
- score

Fichier CSV: results/p2/inference/predictions.csv

Colonnes:
- frame_id,class_name,x,y,w,h,score

## Sorties generees

- models/p2/...: checkpoints et resume d entrainement
- results/p2/inference/: predictions.json, predictions.csv, run_summary.json, video/image annotee
- results/p2/benchmark_report.json: rapport FPS et memoire GPU

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