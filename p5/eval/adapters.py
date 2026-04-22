from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(payload: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _build_frame_index(gt_path: str | Path) -> dict[int, int]:
    gt = _load_json(gt_path)
    images = gt.get("images", []) if isinstance(gt, dict) else []

    mapping: dict[int, int] = {}
    for image in images:
        file_name = str(image.get("file_name", ""))
        image_id = image.get("id")
        if image_id is None:
            continue

        stem = Path(file_name).stem
        if not stem.isdigit():
            continue

        frame_id = int(stem)
        mapping[frame_id] = int(image_id)

    return mapping


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_coco_record(
    pred: dict[str, Any],
    frame_to_image: dict[int, int],
    class_name_to_category_id: dict[str, int],
    default_category_id: int,
    prefer_zero_based_shift: bool,
) -> dict[str, Any] | None:
    if all(key in pred for key in ("image_id", "category_id", "bbox", "score")):
        bbox = pred.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        return {
            "image_id": int(pred["image_id"]),
            "category_id": int(pred["category_id"]),
            "bbox": [_as_float(v) for v in bbox],
            "score": _as_float(pred.get("score"), 0.0),
        }

    frame_id_raw = pred.get("frame_id", pred.get("frame", pred.get("image_id")))
    if frame_id_raw is None:
        return None

    try:
        frame_id = int(frame_id_raw)
    except (TypeError, ValueError):
        return None

    image_id = None
    if prefer_zero_based_shift:
        image_id = frame_to_image.get(frame_id + 1)
        if image_id is None:
            image_id = frame_to_image.get(frame_id)
    else:
        image_id = frame_to_image.get(frame_id)
        if image_id is None:
            # Common mismatch: model outputs are 0-based while MOT/COCO frame ids are 1-based.
            image_id = frame_to_image.get(frame_id + 1)
    if image_id is None:
        return None

    bbox = pred.get("bbox")
    if bbox is None:
        bbox = [pred.get("x", 0.0), pred.get("y", 0.0), pred.get("w", 0.0), pred.get("h", 0.0)]
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    category_id = pred.get("category_id")
    if category_id is None:
        class_name = str(pred.get("class_name", "")).lower().strip()
        category_id = class_name_to_category_id.get(class_name, default_category_id)

    return {
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": [_as_float(v) for v in bbox],
        "score": _as_float(pred.get("score"), 0.0),
    }


def convert_predictions_to_coco(
    predictions_path: str | Path,
    gt_path: str | Path,
    output_path: str | Path,
    *,
    class_name_to_category_id: dict[str, int] | None = None,
    default_category_id: int = 1,
) -> dict[str, Any]:
    class_map = {"person": 1, "pedestrian": 1}
    if class_name_to_category_id:
        class_map.update({str(k).lower(): int(v) for k, v in class_name_to_category_id.items()})

    preds = _load_json(predictions_path)
    if not isinstance(preds, list):
        raise ValueError("Prediction file must contain a list of detections.")

    frame_to_image = _build_frame_index(gt_path)
    frame_values: list[int] = []
    for pred in preds:
        if isinstance(pred, dict) and "frame_id" in pred:
            try:
                frame_values.append(int(pred["frame_id"]))
            except (TypeError, ValueError):
                continue

    prefer_zero_based_shift = 0 in frame_values and 0 not in frame_to_image

    converted: list[dict[str, Any]] = []
    dropped = 0
    for pred in preds:
        if not isinstance(pred, dict):
            dropped += 1
            continue
        coco_pred = _to_coco_record(
            pred,
            frame_to_image,
            class_map,
            default_category_id,
            prefer_zero_based_shift,
        )
        if coco_pred is None:
            dropped += 1
            continue
        converted.append(coco_pred)

    _save_json(converted, output_path)
    return {
        "input": str(predictions_path),
        "output": str(output_path),
        "total_input": len(preds),
        "total_output": len(converted),
        "dropped": dropped,
    }


def validate_predictions_schema(predictions_path: str | Path) -> dict[str, Any]:
    """
    Lightweight validation for native detector outputs before conversion.
    Accepts native schema (frame_id + bbox + score) or COCO schema.
    """
    preds = _load_json(predictions_path)
    if not isinstance(preds, list):
        return {
            "ok": False,
            "reason": "Payload must be a list",
            "valid": 0,
            "invalid": 0,
        }

    valid = 0
    invalid = 0
    for pred in preds:
        if not isinstance(pred, dict):
            invalid += 1
            continue
        bbox = pred.get("bbox")
        bbox_ok = isinstance(bbox, list) and len(bbox) == 4
        has_native = "frame_id" in pred and "score" in pred
        has_coco = all(key in pred for key in ("image_id", "category_id", "score"))
        if bbox_ok and (has_native or has_coco):
            valid += 1
        else:
            invalid += 1

    return {
        "ok": valid > 0,
        "reason": "ok" if valid > 0 else "No valid prediction records found",
        "valid": valid,
        "invalid": invalid,
    }
