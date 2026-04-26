from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from evaluation_pipeline.data.download_mot17 import download_mot17
from evaluation_pipeline.data.mot_to_coco import mot_to_coco
from evaluation_pipeline.data.mot_to_yolo import convert_mot_to_yolo
from evaluation_pipeline.data.split_data import create_yolo_dataset, split_sequences
from evaluation_pipeline.eval.adapters import convert_predictions_to_coco
from evaluation_pipeline.eval.evaluate import (
    compare_models,
    compute_operating_metrics,
    evaluate_coco,
    evaluate_coco_subset,
    split_coco_image_ids_by_sequence,
)
from evaluation_pipeline.validation import (
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
    output_dir: str = "results/evaluation_pipeline"
    eval_mode: str = "single-sequence"
    operating_threshold: float = 0.25
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


def _write_comparison_report(report_path: str, payload: dict[str, Any]) -> None:
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = report_file.with_suffix(".csv")
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model",
                "mAP_50_95",
                "mAP_50",
                "mAP_75",
                "precision",
                "recall",
                "f1",
                "operating_threshold",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _extract_sequence_from_text(value: str, gt_sequences: set[str]) -> str | None:
    token = value.lower().replace("\\", "/")
    for sequence in sorted(gt_sequences):
        if sequence.lower() in token:
            return sequence
    return None


def _extract_source_sequence_from_sidecar(predictions_path: str | Path, gt_sequences: set[str]) -> dict[str, Any]:
    preds_path = Path(predictions_path)
    sidecars = [
        preds_path.parent / "run_summary.json",
        preds_path.with_name(preds_path.stem + "_metrics.json"),
    ]

    for candidate in sidecars:
        if not candidate.exists():
            continue
        try:
            payload = _load_json(candidate)
        except Exception:
            continue

        source_value = None
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            source_value = payload[0].get("source") or payload[0].get("video")
        elif isinstance(payload, dict):
            source_value = payload.get("source") or payload.get("video")

        if source_value is None:
            continue

        source_text = str(source_value)
        source_sequence = _extract_sequence_from_text(source_text, gt_sequences)
        return {
            "sidecar": str(candidate),
            "source": source_text,
            "sequence": source_sequence,
        }

    return {
        "sidecar": None,
        "source": None,
        "sequence": None,
    }


def _build_source_coherence_check(
    cfg: P5Config,
    gt_sequences: set[str],
    p2_source: dict[str, Any],
    p3_source: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    details = {
        "eval_mode": cfg.eval_mode,
        "gt_sequence_count": len(gt_sequences),
        "gt_sequences": sorted(gt_sequences),
        "p2_source": p2_source,
        "p3_source": p3_source,
    }

    if cfg.eval_mode == "single-sequence":
        if len(gt_sequences) != 1:
            errors.append(
                "single-sequence mode requires GT JSON with exactly one sequence. "
                "Action: regenerate GT with one MOT17 sequence or switch --eval-mode multi-sequence."
            )
        target_sequence = sorted(gt_sequences)[0] if len(gt_sequences) == 1 else None

        for model_name, source in (("P2", p2_source), ("P3", p3_source)):
            sequence = source.get("sequence")
            if sequence is not None and target_sequence and sequence != target_sequence:
                errors.append(
                    f"{model_name}: source sequence '{sequence}' does not match GT sequence '{target_sequence}'."
                )

    if cfg.eval_mode == "multi-sequence":
        for model_name, source in (("P2", p2_source), ("P3", p3_source)):
            sequence = source.get("sequence")
            if sequence is not None and sequence not in gt_sequences:
                errors.append(
                    f"{model_name}: inferred source sequence '{sequence}' is not present in GT sequences."
                )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "details": details,
    }


def _aggregate_metrics(rows: list[dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    if not rows:
        return result
    for key in keys:
        values = [float(row[key]) for row in rows]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(variance)
        result[key] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min(values)),
            "max": float(max(values)),
        }
    return result


def _run_sanity_checks(gt_path: str, output_dir: Path) -> dict[str, Any]:
    gt_payload = _load_json(gt_path)
    if not isinstance(gt_payload, dict):
        raise ValueError("GT payload must be a COCO dict for sanity checks.")

    sanity_dir = output_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)

    empty_preds_path = sanity_dir / "predictions_empty.json"
    _write_json(empty_preds_path, [])
    try:
        empty_stats = evaluate_coco(gt_path, str(empty_preds_path))
    except (IndexError, Exception):
        # Empty predictions cannot be loaded by pycocotools. Expected behavior: all zeros.
        empty_stats = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    empty_operating = compute_operating_metrics(gt_path, empty_preds_path, score_threshold=0.25)

    oracle_preds: list[dict[str, Any]] = []
    for ann in gt_payload.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        oracle_preds.append(
            {
                "image_id": int(ann.get("image_id", -1)),
                "category_id": int(ann.get("category_id", 1)),
                "bbox": [float(v) for v in bbox],
                "score": 1.0,
            }
        )
    oracle_preds_path = sanity_dir / "predictions_oracle.json"
    _write_json(oracle_preds_path, oracle_preds)
    oracle_stats = evaluate_coco(gt_path, str(oracle_preds_path))
    oracle_operating = compute_operating_metrics(gt_path, oracle_preds_path, score_threshold=0.25)

    empty_ok = float(empty_stats[0]) <= 0.05
    oracle_min_map = 0.90
    oracle_ok = float(oracle_stats[0]) >= oracle_min_map
    if not oracle_ok:
        raise ValueError(
            f"Oracle sanity check failed: mAP_50_95={float(oracle_stats[0]):.4f} is below expected minimum {oracle_min_map:.2f}. "
            "This indicates a protocol mismatch between GT and prediction conversion."
        )

    return {
        "empty_predictions": {
            "mAP_50_95": float(empty_stats[0]),
            "mAP_50": float(empty_stats[1]),
            "mAP_75": float(empty_stats[2]),
            "precision": float(empty_operating["precision"]),
            "recall": float(empty_operating["recall"]),
            "f1": float(empty_operating["f1"]),
            "status": "PASS" if empty_ok else "FAIL",
        },
        "oracle_predictions": {
            "mAP_50_95": float(oracle_stats[0]),
            "mAP_50": float(oracle_stats[1]),
            "mAP_75": float(oracle_stats[2]),
            "precision": float(oracle_operating["precision"]),
            "recall": float(oracle_operating["recall"]),
            "f1": float(oracle_operating["f1"]),
            "status": "PASS" if oracle_ok else "FAIL",
            "minimum_expected_map_50_95": oracle_min_map,
        },
        "artifacts": {
            "empty_predictions_json": str(empty_preds_path),
            "oracle_predictions_json": str(oracle_preds_path),
        },
        "status": "PASS" if (empty_ok and oracle_ok) else "FAIL",
    }


def _load_category_ids_from_coco(path: str | Path) -> set[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return set()
    categories = payload.get("categories", [])
    ids: set[int] = set()
    for category in categories:
        if not isinstance(category, dict):
            continue
        category_id = category.get("id")
        if category_id is None:
            continue
        ids.add(int(category_id))
    return ids


def _load_category_ids_from_predictions(path: str | Path) -> set[int]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return set()
    ids: set[int] = set()
    for pred in payload:
        if not isinstance(pred, dict):
            continue
        category_id = pred.get("category_id")
        if category_id is None:
            continue
        ids.add(int(category_id))
    return ids


def _assert_non_empty_category_overlap(model_name: str, gt_ids: set[int], pred_ids: set[int]) -> None:
    overlap = gt_ids & pred_ids
    print(f"{model_name} category ids in GT: {sorted(gt_ids)}")
    print(f"{model_name} category ids in predictions: {sorted(pred_ids)}")
    print(f"{model_name} category overlap: {sorted(overlap)}")
    if not overlap:
        raise ValueError(
            f"{model_name} category mismatch: no overlap between GT and converted predictions. "
            "Check class filtering/category remapping in conversion."
        )


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

    sequence_image_ids = split_coco_image_ids_by_sequence(cfg.gt_path)
    gt_sequences = set(sequence_image_ids.keys())

    p2_source = _extract_source_sequence_from_sidecar(cfg.p2_preds, gt_sequences)
    p3_source = _extract_source_sequence_from_sidecar(cfg.p3_preds, gt_sequences)
    source_check = _build_source_coherence_check(cfg, gt_sequences, p2_source, p3_source)
    if not source_check["ok"]:
        raise ValueError(
            "Protocol source/GT coherence check failed:\n- " + "\n- ".join(source_check["errors"])
        )

    p2_info = convert_predictions_to_coco(
        cfg.p2_preds,
        cfg.gt_path,
        p2_eval_file,
        eval_mode=cfg.eval_mode,
        class_name_to_category_id={"person": 1, "pedestrian": 1},
        default_category_id=1,
        category_id_remap={0: 1},
    )
    p3_info = convert_predictions_to_coco(
        cfg.p3_preds,
        cfg.gt_path,
        p3_eval_file,
        eval_mode=cfg.eval_mode,
        class_name_to_category_id={"person": 1, "pedestrian": 1},
        default_category_id=1,
        allowed_class_names={"person", "pedestrian"},
    )
    print(f"P2 conversion: {p2_info}")
    print(f"P3 conversion: {p3_info}")

    for model_name, info in (("P2", p2_info), ("P3", p3_info)):
        total_input = int(info.get("total_input", 0))
        total_output = int(info.get("total_output", 0))
        filtered_out = int(info.get("filtered_out", 0))
        dropped = int(info.get("dropped", 0))
        kept_ratio = (total_output / total_input) if total_input > 0 else 0.0
        print(
            f"{model_name} conversion summary: input={total_input}, kept={total_output}, "
            f"filtered={filtered_out}, dropped={dropped}, kept_ratio={kept_ratio:.2%}"
        )

    gt_category_ids = _load_category_ids_from_coco(cfg.gt_path)
    p2_category_ids = _load_category_ids_from_predictions(p2_eval_file)
    p3_category_ids = _load_category_ids_from_predictions(p3_eval_file)
    _assert_non_empty_category_overlap("P2", gt_category_ids, p2_category_ids)
    _assert_non_empty_category_overlap("P3", gt_category_ids, p3_category_ids)

    stats_p2 = evaluate_coco(cfg.gt_path, str(p2_eval_file))
    stats_p3 = evaluate_coco(cfg.gt_path, str(p3_eval_file))
    operating_p2 = compute_operating_metrics(
        cfg.gt_path,
        p2_eval_file,
        score_threshold=cfg.operating_threshold,
    )
    operating_p3 = compute_operating_metrics(
        cfg.gt_path,
        p3_eval_file,
        score_threshold=cfg.operating_threshold,
    )
    compare_models({"YOLOv11": stats_p2, "DETR": stats_p3})

    report_rows: list[dict[str, float]] = [
        {
            "model": "YOLOv11",
            "mAP_50_95": float(stats_p2[0]),
            "mAP_50": float(stats_p2[1]),
            "mAP_75": float(stats_p2[2]),
            "precision": float(operating_p2["precision"]),
            "recall": float(operating_p2["recall"]),
            "f1": float(operating_p2["f1"]),
            "operating_threshold": float(cfg.operating_threshold),
        },
        {
            "model": "DETR",
            "mAP_50_95": float(stats_p3[0]),
            "mAP_50": float(stats_p3[1]),
            "mAP_75": float(stats_p3[2]),
            "precision": float(operating_p3["precision"]),
            "recall": float(operating_p3["recall"]),
            "f1": float(operating_p3["f1"]),
            "operating_threshold": float(cfg.operating_threshold),
        },
    ]

    per_sequence_rows: dict[str, list[dict[str, float]]] = {
        "YOLOv11": [],
        "DETR": [],
    }
    for sequence, image_ids in sorted(sequence_image_ids.items()):
        if not image_ids:
            continue
        p2_stats_seq = evaluate_coco_subset(cfg.gt_path, str(p2_eval_file), image_ids)
        p3_stats_seq = evaluate_coco_subset(cfg.gt_path, str(p3_eval_file), image_ids)
        p2_operating_seq = compute_operating_metrics(
            cfg.gt_path,
            p2_eval_file,
            score_threshold=cfg.operating_threshold,
            image_ids=set(image_ids),
        )
        p3_operating_seq = compute_operating_metrics(
            cfg.gt_path,
            p3_eval_file,
            score_threshold=cfg.operating_threshold,
            image_ids=set(image_ids),
        )

        per_sequence_rows["YOLOv11"].append(
            {
                "sequence": sequence,
                "mAP_50_95": float(p2_stats_seq[0]),
                "mAP_50": float(p2_stats_seq[1]),
                "mAP_75": float(p2_stats_seq[2]),
                "precision": float(p2_operating_seq["precision"]),
                "recall": float(p2_operating_seq["recall"]),
                "f1": float(p2_operating_seq["f1"]),
            }
        )
        per_sequence_rows["DETR"].append(
            {
                "sequence": sequence,
                "mAP_50_95": float(p3_stats_seq[0]),
                "mAP_50": float(p3_stats_seq[1]),
                "mAP_75": float(p3_stats_seq[2]),
                "precision": float(p3_operating_seq["precision"]),
                "recall": float(p3_operating_seq["recall"]),
                "f1": float(p3_operating_seq["f1"]),
            }
        )

    per_sequence_payload = {
        "eval_mode": cfg.eval_mode,
        "operating_threshold": float(cfg.operating_threshold),
        "models": {
            model: {
                "per_sequence": rows,
                "aggregate": _aggregate_metrics(
                    rows,
                    ["mAP_50_95", "mAP_50", "mAP_75", "precision", "recall", "f1"],
                ),
            }
            for model, rows in per_sequence_rows.items()
        },
    }

    sanity_checks = _run_sanity_checks(cfg.gt_path, output)
    sanity_path = output / "sanity_checks.json"
    _write_json(sanity_path, sanity_checks)

    verdict_status = "PASS" if (source_check["ok"] and sanity_checks.get("status") == "PASS") else "FAIL"
    protocol_checklist_payload = {
        "eval_mode": cfg.eval_mode,
        "source_gt_coherence": source_check,
        "mapping": {
            "p2": {
                "mapping_type": p2_info.get("mapping_type"),
                "frames_seen": p2_info.get("frames_seen"),
                "frames_mapped": p2_info.get("frames_mapped"),
                "frames_unmapped": p2_info.get("frames_unmapped"),
            },
            "p3": {
                "mapping_type": p3_info.get("mapping_type"),
                "frames_seen": p3_info.get("frames_seen"),
                "frames_mapped": p3_info.get("frames_mapped"),
                "frames_unmapped": p3_info.get("frames_unmapped"),
            },
        },
        "categories": {
            "gt": sorted(gt_category_ids),
            "p2_predictions": sorted(p2_category_ids),
            "p3_predictions": sorted(p3_category_ids),
        },
        "verdict": {
            "protocol": {
                "status": verdict_status,
                "checks": {
                    "source_gt_coherence": "PASS" if source_check["ok"] else "FAIL",
                    "sanity_checks": sanity_checks.get("status", "FAIL"),
                },
            }
        },
    }
    protocol_checklist_path = output / "protocol_checklist.json"
    _write_json(protocol_checklist_path, protocol_checklist_payload)

    comparison_payload = {
        "eval_mode": cfg.eval_mode,
        "operating_threshold": float(cfg.operating_threshold),
        "models": report_rows,
        "verdict": protocol_checklist_payload["verdict"],
    }
    report_path = output / "comparison_report.json"
    _write_comparison_report(str(report_path), comparison_payload)
    print(f"Comparison report written to: {report_path}")

    per_sequence_path = output / "comparison_report_per_sequence.json"
    _write_json(per_sequence_path, per_sequence_payload)
    print(f"Per-sequence report written to: {per_sequence_path}")
    print(f"Protocol checklist written to: {protocol_checklist_path}")
    print(f"Sanity checks written to: {sanity_path}")

    return {
        "report_json": str(report_path),
        "report_csv": str(report_path.with_suffix('.csv')),
        "report_per_sequence_json": str(per_sequence_path),
        "protocol_checklist_json": str(protocol_checklist_path),
        "sanity_checks_json": str(sanity_path),
        "eval_mode": cfg.eval_mode,
        "operating_threshold": float(cfg.operating_threshold),
        "verdict_protocol": verdict_status,
        "p2_map_50_95": float(stats_p2[0]),
        "p3_map_50_95": float(stats_p3[0]),
    }


def run_p5_pipeline(
    mot_root: str = "data/raw/MOT17",
    gt_path: str = "data/processed/val_gt.json",
    p2_preds: str = "results/object_detection/inference/predictions.json",
    p3_preds: str = "results/p3/predictions.json",
    output_dir: str = "results/evaluation_pipeline",
    eval_mode: str = "single-sequence",
    operating_threshold: float = 0.25,
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
        eval_mode=eval_mode,
        operating_threshold=operating_threshold,
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
