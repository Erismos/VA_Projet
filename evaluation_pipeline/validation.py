from __future__ import annotations

from pathlib import Path
from typing import Any

from object_detection.train import validate_yolo_training_inputs
from evaluation_pipeline.data.split_data import validate_yolo_dataset
from evaluation_pipeline.eval.adapters import validate_predictions_schema


def _normalize(result: dict[str, Any], *, details_key: str = "details") -> dict[str, Any]:
    ok = bool(result.get("ok", False))
    reason = str(result.get("reason", "ok" if ok else "validation failed"))
    return {
        "ok": ok,
        "reason": reason,
        details_key: result,
    }


def validate_prediction_file(predictions_path: str | Path) -> dict[str, Any]:
    return _normalize(validate_predictions_schema(predictions_path))


def validate_yolo_dataset_layout(
    output_dir: str | Path,
    train_seqs: list[str],
    val_seqs: list[str],
) -> dict[str, Any]:
    return _normalize(validate_yolo_dataset(str(output_dir), train_seqs, val_seqs))


def validate_p2_training_inputs(dataset_yaml: str | Path) -> dict[str, Any]:
    return _normalize(validate_yolo_training_inputs(str(dataset_yaml)))
