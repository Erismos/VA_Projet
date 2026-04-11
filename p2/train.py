from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from p2.export_utils import save_json


def train_yolo(
    weights: str,
    dataset_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
) -> dict[str, Any]:
    model = YOLO(weights)
    result = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
    )
    summary = {
        "model": "yolo",
        "weights_start": weights,
        "dataset_yaml": dataset_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "save_dir": str(result.save_dir),
    }
    save_json([summary], Path(project) / name / "train_summary.json")
    return summary


def train_fasterrcnn_placeholder(
    dataset_root: str,
    epochs: int,
    device: str,
    output_dir: str,
) -> dict[str, Any]:
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(torch.device(device))

    summary = {
        "model": "fasterrcnn",
        "dataset_root": dataset_root,
        "epochs": epochs,
        "device": device,
        "note": "Starter scaffold created. Add your dataset adapter and DataLoader to complete training.",
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json([summary], out / "train_summary.json")
    torch.save(model.state_dict(), out / "fasterrcnn_initial_weights.pth")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 training runner")
    sub = parser.add_subparsers(dest="model", required=True)

    yolo = sub.add_parser("yolo")
    yolo.add_argument("--weights", default="yolo11n.pt")
    yolo.add_argument("--dataset-yaml", required=True)
    yolo.add_argument("--epochs", type=int, default=30)
    yolo.add_argument("--imgsz", type=int, default=640)
    yolo.add_argument("--batch", type=int, default=16)
    yolo.add_argument("--device", default="cpu")
    yolo.add_argument("--project", default="models/p2")
    yolo.add_argument("--name", default="yolo_baseline")

    frcnn = sub.add_parser("fasterrcnn")
    frcnn.add_argument("--dataset-root", required=True)
    frcnn.add_argument("--epochs", type=int, default=10)
    frcnn.add_argument("--device", default="cpu")
    frcnn.add_argument("--output-dir", default="models/p2/fasterrcnn_baseline")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.model == "yolo":
        summary = train_yolo(
            weights=args.weights,
            dataset_yaml=args.dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
        )
    else:
        summary = train_fasterrcnn_placeholder(
            dataset_root=args.dataset_root,
            epochs=args.epochs,
            device=args.device,
            output_dir=args.output_dir,
        )
    print(summary)


if __name__ == "__main__":
    main()
