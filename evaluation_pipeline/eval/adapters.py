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


def _extract_sequence_from_file_name(file_name: str) -> str | None:
    normalized = file_name.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        return None
    # Expected MOT style: <sequence>/img1/<frame>.jpg
    return parts[0]


def _build_frame_index(gt_path: str | Path) -> tuple[dict[int, int], dict[tuple[str, int], int], set[str]]:
    gt = _load_json(gt_path)
    images = gt.get("images", []) if isinstance(gt, dict) else []

    mapping: dict[int, int] = {}
    sequence_mapping: dict[tuple[str, int], int] = {}
    sequences: set[str] = set()
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
        sequence = _extract_sequence_from_file_name(file_name)
        if sequence:
            sequences.add(sequence)
            sequence_mapping[(sequence, frame_id)] = int(image_id)

    return mapping, sequence_mapping, sequences


def _infer_sequence_token(pred: dict[str, Any], gt_sequences: set[str]) -> str | None:
    candidate_keys = ("sequence", "sequence_name", "seq", "seq_name", "video", "video_name", "source")
    gt_sequences_lc = {sequence.lower(): sequence for sequence in gt_sequences}

    for key in candidate_keys:
        value = pred.get(key)
        if value is None:
            continue
        token = str(value).strip()
        if not token:
            continue

        # Direct match first.
        direct = gt_sequences_lc.get(token.lower())
        if direct:
            return direct

        # Accept path-like values containing the sequence name.
        token_lc = token.lower().replace("\\", "/")
        for sequence_lc, original_sequence in gt_sequences_lc.items():
            if sequence_lc in token_lc:
                return original_sequence

    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_coco_record(
    pred: dict[str, Any],
    frame_to_image: dict[int, int],
    sequence_frame_to_image: dict[tuple[str, int], int],
    gt_sequences: set[str],
    require_sequence_match: bool,
    class_name_to_category_id: dict[str, int],
    default_category_id: int,
    category_id_remap: dict[int, int],
    prefer_zero_based_shift: bool,
) -> dict[str, Any] | None:
    if all(key in pred for key in ("image_id", "category_id", "bbox", "score")):
        bbox = pred.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        raw_category_id = int(pred["category_id"])
        mapped_category_id = category_id_remap.get(raw_category_id, raw_category_id)
        return {
            "image_id": int(pred["image_id"]),
            "category_id": int(mapped_category_id),
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

    sequence_token = _infer_sequence_token(pred, gt_sequences)

    image_id = None
    if sequence_token is not None:
        if prefer_zero_based_shift:
            image_id = sequence_frame_to_image.get((sequence_token, frame_id + 1))
            if image_id is None:
                image_id = sequence_frame_to_image.get((sequence_token, frame_id))
        else:
            image_id = sequence_frame_to_image.get((sequence_token, frame_id))
            if image_id is None:
                image_id = sequence_frame_to_image.get((sequence_token, frame_id + 1))
    elif require_sequence_match:
        # Ambiguous frame-only prediction against multi-sequence GT.
        return None
    else:
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
    category_id = category_id_remap.get(int(category_id), int(category_id))

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
    eval_mode: str = "single-sequence",
    class_name_to_category_id: dict[str, int] | None = None,
    default_category_id: int = 1,
    allowed_class_names: set[str] | None = None,
    category_id_remap: dict[int, int] | None = None,
) -> dict[str, Any]:
    class_map = {"person": 1, "pedestrian": 1}
    if class_name_to_category_id:
        class_map.update({str(k).lower(): int(v) for k, v in class_name_to_category_id.items()})
    allowed_names = {str(name).lower().strip() for name in allowed_class_names} if allowed_class_names else None
    remap = {int(k): int(v) for k, v in (category_id_remap or {}).items()}

    preds = _load_json(predictions_path)
    if not isinstance(preds, list):
        raise ValueError("Prediction file must contain a list of detections.")
    if eval_mode not in {"single-sequence", "multi-sequence"}:
        raise ValueError("eval_mode must be one of: single-sequence, multi-sequence")

    frame_to_image, sequence_frame_to_image, gt_sequences = _build_frame_index(gt_path)
    frame_values: list[int] = []
    needs_frame_mapping = False
    has_sequence_context = False
    mapped_by_sequence = 0
    mapped_by_frame_only = 0
    unmapped_frame_refs = 0
    distinct_frame_refs: set[int] = set()
    mapped_frame_refs: set[int] = set()
    for pred in preds:
        if not isinstance(pred, dict):
            continue
        if not all(key in pred for key in ("image_id", "category_id", "bbox", "score")):
            needs_frame_mapping = True
        if _infer_sequence_token(pred, gt_sequences) is not None:
            has_sequence_context = True
        if "frame_id" in pred:
            try:
                frame_values.append(int(pred["frame_id"]))
                distinct_frame_refs.add(int(pred["frame_id"]))
            except (TypeError, ValueError):
                continue

    if eval_mode == "single-sequence" and len(gt_sequences) != 1:
        preview = sorted(gt_sequences)[:5]
        raise ValueError(
            "Evaluation mode single-sequence requires a GT containing exactly one sequence. "
            f"Found {len(gt_sequences)} sequences. Use a per-sequence GT JSON or switch to "
            "--eval-mode multi-sequence. "
            f"GT sequences preview: {preview}."
        )

    if eval_mode == "multi-sequence" and needs_frame_mapping and not has_sequence_context:
        preview = sorted(gt_sequences)[:5]
        raise ValueError(
            "Evaluation mode multi-sequence requires sequence metadata in predictions but none was found "
            "sequence metadata (expected one of: sequence, sequence_name, seq, video, source). "
            "Action: export predictions with per-detection sequence tokens or switch to "
            "--eval-mode single-sequence with a GT JSON for the same source sequence. "
            f"GT sequences preview: {preview}."
        )

    if eval_mode == "single-sequence" and needs_frame_mapping and len(gt_sequences) > 1 and not has_sequence_context:
        preview = sorted(gt_sequences)[:5]
        raise ValueError(
            "Ambiguous frame mapping: predictions only provide frame ids while GT has multiple sequences. "
            "Action: provide a single-sequence GT JSON matching the inference source, or run with "
            "--eval-mode multi-sequence and include sequence metadata in predictions. "
            f"GT sequences preview: {preview}."
        )

    prefer_zero_based_shift = 0 in frame_values and 0 not in frame_to_image
    require_sequence_match = eval_mode == "multi-sequence" and needs_frame_mapping

    converted: list[dict[str, Any]] = []
    dropped = 0
    filtered_out = 0
    for pred in preds:
        if not isinstance(pred, dict):
            dropped += 1
            continue
        if allowed_names is not None:
            class_name = str(pred.get("class_name", "")).lower().strip()
            if class_name and class_name not in allowed_names:
                filtered_out += 1
                continue

        frame_id_raw = pred.get("frame_id", pred.get("frame", pred.get("image_id")))
        frame_id_int: int | None = None
        try:
            if frame_id_raw is not None:
                frame_id_int = int(frame_id_raw)
        except (TypeError, ValueError):
            frame_id_int = None
        sequence_token = _infer_sequence_token(pred, gt_sequences)

        coco_pred = _to_coco_record(
            pred,
            frame_to_image,
            sequence_frame_to_image,
            gt_sequences,
            require_sequence_match,
            class_map,
            default_category_id,
            remap,
            prefer_zero_based_shift,
        )
        if coco_pred is None:
            dropped += 1
            if frame_id_int is not None:
                unmapped_frame_refs += 1
            continue

        if frame_id_int is not None:
            mapped_frame_refs.add(frame_id_int)
            if sequence_token is not None:
                mapped_by_sequence += 1
            else:
                mapped_by_frame_only += 1
        converted.append(coco_pred)

    _save_json(converted, output_path)
    mapping_type = "coco-direct"
    if needs_frame_mapping:
        if mapped_by_sequence > 0 and mapped_by_frame_only == 0:
            mapping_type = "sequence+frame"
        elif mapped_by_sequence == 0 and mapped_by_frame_only > 0:
            mapping_type = "frame-only"
        elif mapped_by_sequence > 0 and mapped_by_frame_only > 0:
            mapping_type = "mixed"

    return {
        "input": str(predictions_path),
        "output": str(output_path),
        "eval_mode": eval_mode,
        "total_input": len(preds),
        "total_output": len(converted),
        "kept": len(converted),
        "filtered_out": filtered_out,
        "dropped": dropped,
        "mapping_type": mapping_type,
        "needs_frame_mapping": needs_frame_mapping,
        "has_sequence_metadata": has_sequence_context,
        "mapped_by_sequence": mapped_by_sequence,
        "mapped_by_frame_only": mapped_by_frame_only,
        "frames_mapped": len(mapped_frame_refs),
        "frames_seen": len(distinct_frame_refs),
        "frames_unmapped": max(0, len(distinct_frame_refs) - len(mapped_frame_refs)),
        "frames_unmapped_records": unmapped_frame_refs,
        "gt_sequences": sorted(gt_sequences),
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
