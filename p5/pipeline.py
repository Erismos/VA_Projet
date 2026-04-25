from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from p5.data.download_mot17 import download_mot17
from p5.data.mot_to_coco import mot_to_coco
from p5.data.mot_to_yolo import convert_mot_to_yolo
from p5.data.split_data import create_yolo_dataset, split_sequences
from p5.eval.adapters import convert_predictions_to_coco
from p5.eval.evaluate import compare_models, evaluate_coco
from p5.validation import (
    validate_p2_training_inputs,
    validate_prediction_file,
    validate_yolo_dataset_layout,
)


@dataclass
class P5Config:
    mot_root: str = "data/raw/MOT17"
    gt_path: str = "data/processed/val_gt.json"
    p2_preds: str = "results/object_detection/inference/predictions.json"
    p3_preds: str = "results/p3/predictions.json"
    output_dir: str = "results/p5"
    skip_prepare_data: bool = False
    prepare_only: bool = False
    verify_p2_train_cmd: bool = False


@dataclass
class P5PrepareResult:
    train_seqs: list[str]
    val_seqs: list[str]
    yolo_yaml: str
    p2_train_command_file: str


def _write_yolo_yaml(yolo_dataset_dir: str, output_yaml: str) -> str:
    payload = {
        "path": str(Path(yolo_dataset_dir).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": ["person"],
    }
    output = Path(output_yaml)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("path: " + payload["path"] + "\n")
        handle.write("train: " + payload["train"] + "\n")
        handle.write("val: " + payload["val"] + "\n")
        handle.write("names:\n")
        handle.write("  0: person\n")
    return str(output)


def _write_comparison_report(report_path: str, rows: list[dict[str, float]]) -> None:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    csv_path = report_file.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "mAP_50_95", "mAP_50", "mAP_75"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_p2_train_command(dataset_yaml: str, device: str = "cpu") -> str:
    return (
        "python -m project.cli object-detection train "
        "--model yolo "
        "--weights yolo11n.pt "
        f"--dataset-yaml {dataset_yaml} "
        "--epochs 30 --imgsz 640 --batch 16 "
        f"--device {device} "
        "--project models/object_detection --name yolo_baseline"
    )


def prepare_p5_data(cfg: P5Config) -> P5PrepareResult:
    if not os.path.exists(cfg.mot_root):
        download_mot17("data/raw")

    if not os.path.exists(cfg.mot_root):
        raise FileNotFoundError(f"{cfg.mot_root} not found even after download attempt.")

    train_root = os.path.join(cfg.mot_root, "train")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"MOT17 train directory not found: {train_root}")

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    train_seqs, val_seqs = split_sequences(train_root)
    print(f"Train sequences: {train_seqs}")
    print(f"Val sequences: {val_seqs}")

    yolo_labels_dir = "data/processed/MOT17_YOLO"
    yolo_dataset_dir = "data/processed/MOT17_YOLO_DATASET"

    convert_mot_to_yolo(cfg.mot_root, yolo_labels_dir)
    create_yolo_dataset(yolo_labels_dir, yolo_dataset_dir, train_seqs, val_seqs)

    yolo_yaml = _write_yolo_yaml(yolo_dataset_dir, "data/processed/mot17_pedestrian_yolo.yaml")
    print(f"YOLO dataset manifest written to: {yolo_yaml}")

    yolo_check = validate_yolo_dataset_layout(yolo_dataset_dir, train_seqs, val_seqs)
    print(f"YOLO dataset validation: {yolo_check}")
    if not yolo_check["ok"]:
        raise ValueError(f"Invalid YOLO dataset generated: {yolo_check}")

    train_command = _build_p2_train_command(yolo_yaml)
    cmd_file = Path(cfg.output_dir) / "p2_train_command.txt"
    cmd_file.parent.mkdir(parents=True, exist_ok=True)
    cmd_file.write_text(train_command + "\n", encoding="utf-8")
    print(f"P2 train command saved to: {cmd_file}")

    if cfg.verify_p2_train_cmd:
        dry_run = validate_p2_training_inputs(yolo_yaml)
        print(f"P2 train dry-run validation: {dry_run}")
        if not dry_run["ok"]:
            raise ValueError(f"P2 train command verification failed: {dry_run}")

    mot_to_coco(cfg.mot_root, "data/processed/val_gt.json", val_seqs)
    mot_to_coco(cfg.mot_root, "data/processed/train_gt.json", train_seqs)

    return P5PrepareResult(
        train_seqs=train_seqs,
        val_seqs=val_seqs,
        yolo_yaml=yolo_yaml,
        p2_train_command_file=str(cmd_file),
    )


def evaluate_p5(cfg: P5Config) -> dict[str, Any]:
    if not os.path.exists(cfg.gt_path):
        raise FileNotFoundError(
            f"Ground truth not found: {cfg.gt_path}. Run data preparation or provide --gt-json."
        )

    output = Path(cfg.output_dir)
    converted_dir = output / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(cfg.p2_preds):
        raise FileNotFoundError(f"P2 predictions not found: {cfg.p2_preds}")
    if not os.path.exists(cfg.p3_preds):
        raise FileNotFoundError(f"P3 predictions not found: {cfg.p3_preds}")

    p2_schema = validate_prediction_file(cfg.p2_preds)
    p3_schema = validate_prediction_file(cfg.p3_preds)
    print(f"P2 schema validation: {p2_schema}")
    print(f"P3 schema validation: {p3_schema}")
    if not p2_schema["ok"]:
        raise ValueError(f"Invalid P2 prediction schema: {p2_schema}")
    if not p3_schema["ok"]:
        raise ValueError(f"Invalid P3 prediction schema: {p3_schema}")

    p2_eval_file = converted_dir / "p2_coco_predictions.json"
    p3_eval_file = converted_dir / "p3_coco_predictions.json"

    p2_info = convert_predictions_to_coco(cfg.p2_preds, cfg.gt_path, p2_eval_file)
    p3_info = convert_predictions_to_coco(cfg.p3_preds, cfg.gt_path, p3_eval_file)
    print(f"P2 conversion: {p2_info}")
    print(f"P3 conversion: {p3_info}")

    stats_p2 = evaluate_coco(cfg.gt_path, str(p2_eval_file))
    stats_p3 = evaluate_coco(cfg.gt_path, str(p3_eval_file))
    compare_models({"YOLOv11": stats_p2, "DETR": stats_p3})

    report_rows = [
        {
            "model": "YOLOv11",
            "mAP_50_95": float(stats_p2[0]),
            "mAP_50": float(stats_p2[1]),
            "mAP_75": float(stats_p2[2]),
        },
        {
            "model": "DETR",
            "mAP_50_95": float(stats_p3[0]),
            "mAP_50": float(stats_p3[1]),
            "mAP_75": float(stats_p3[2]),
        },
    ]
    report_path = output / "comparison_report.json"
    _write_comparison_report(str(report_path), report_rows)
    print(f"Comparison report written to: {report_path}")

    return {
        "report_json": str(report_path),
        "report_csv": str(report_path.with_suffix('.csv')),
        "p2_map_50_95": float(stats_p2[0]),
        "p3_map_50_95": float(stats_p3[0]),
    }


def run_p5_pipeline(
    mot_root: str = "data/raw/MOT17",
    gt_path: str = "data/processed/val_gt.json",
    p2_preds: str = "results/object_detection/inference/predictions.json",
    p3_preds: str = "results/p3/predictions.json",
    output_dir: str = "results/p5",
    skip_prepare_data: bool = False,
    prepare_only: bool = False,
    verify_p2_train_cmd: bool = False,
) -> dict[str, Any]:
    cfg = P5Config(
        mot_root=mot_root,
        gt_path=gt_path,
        p2_preds=p2_preds,
        p3_preds=p3_preds,
        output_dir=output_dir,
        skip_prepare_data=skip_prepare_data,
        prepare_only=prepare_only,
        verify_p2_train_cmd=verify_p2_train_cmd,
    )

    print("Starting Person 5 Pipeline...")
    prepared = None
    if not cfg.skip_prepare_data:
        prepared = prepare_p5_data(cfg)

    if cfg.prepare_only:
        return {
            "prepared": prepared.__dict__ if prepared else None,
            "evaluation": None,
        }

    print("\nRunning Evaluation...")
    evaluation = evaluate_p5(cfg)

    return {
        "prepared": prepared.__dict__ if prepared else None,
        "evaluation": evaluation,
    }
