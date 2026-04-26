from __future__ import annotations

import argparse
import yaml
from pathlib import Path
from typing import Any

from object_detection.export_utils import save_json


def validate_yolo_training_inputs(dataset_yaml: str) -> dict[str, Any]:
    yaml_path = Path(dataset_yaml)
    if not yaml_path.exists():
        return {"ok": False, "reason": f"Dataset YAML not found: {dataset_yaml}"}

    with yaml_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    required_keys = {"path", "train", "val", "names"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        return {"ok": False, "reason": f"Dataset YAML missing keys: {missing}"}

    root = Path(payload["path"]).expanduser()
    train_dir = root / str(payload["train"])
    val_dir = root / str(payload["val"])

    if not root.exists():
        return {"ok": False, "reason": f"Dataset root not found: {root}"}
    if not train_dir.exists():
        return {"ok": False, "reason": f"Train path not found: {train_dir}"}
    if not val_dir.exists():
        return {"ok": False, "reason": f"Val path not found: {val_dir}"}

    train_images = list(train_dir.glob("*.jpg"))
    val_images = list(val_dir.glob("*.jpg"))
    if not train_images:
        return {"ok": False, "reason": f"No train images found in: {train_dir}"}
    if not val_images:
        return {"ok": False, "reason": f"No val images found in: {val_dir}"}

    return {
        "ok": True,
        "dataset_yaml": str(yaml_path),
        "dataset_root": str(root),
        "train_images": len(train_images),
        "val_images": len(val_images),
    }


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
    from ultralytics import YOLO

    project_dir = Path(project).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights)
    result = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=name,
    )
    save_dir = Path(result.save_dir).resolve()
    trained_weights = save_dir / "weights" / "best.pt"
    summary = {
        "model": "yolo",
        "weights_start": weights,
        "dataset_yaml": dataset_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "project": str(project_dir),
        "save_dir": str(save_dir),
        "trained_weights": str(trained_weights),
    }
    save_json([summary], project_dir / name / "train_summary.json")
    return summary


def train_fasterrcnn_placeholder(
    dataset_root: str,
    epochs: int,
    device: str,
    output_dir: str,
) -> dict[str, Any]:
    import torch
    from torchvision.models.detection import (
        FasterRCNN_ResNet50_FPN_Weights,
        fasterrcnn_resnet50_fpn,
    )

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
    yolo.add_argument("--project", default="models/object_detection")
    yolo.add_argument("--name", default="yolo_baseline")

    frcnn = sub.add_parser("fasterrcnn")
    frcnn.add_argument("--dataset-root", required=True)
    frcnn.add_argument("--epochs", type=int, default=10)
    frcnn.add_argument("--device", default="cpu")
    frcnn.add_argument("--output-dir", default="models/object_detection/fasterrcnn_baseline")

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
