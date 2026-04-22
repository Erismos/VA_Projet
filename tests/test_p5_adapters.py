from __future__ import annotations

import json
from pathlib import Path

from p5.eval.adapters import convert_predictions_to_coco, validate_predictions_schema


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_convert_predictions_to_coco_maps_frame_id_to_image_id(tmp_path: Path) -> None:
    gt_path = tmp_path / "val_gt.json"
    preds_path = tmp_path / "p2_predictions.json"
    out_path = tmp_path / "converted.json"

    gt = {
        "images": [
            {"id": 101, "file_name": "MOT17-02-DPM/img1/000001.jpg", "width": 1920, "height": 1080},
            {"id": 102, "file_name": "MOT17-02-DPM/img1/000002.jpg", "width": 1920, "height": 1080},
        ],
        "annotations": [],
        "categories": [{"id": 1, "name": "pedestrian"}],
    }
    preds = [
        {"frame_id": 0, "class_name": "person", "bbox": [10, 20, 30, 40], "score": 0.9},
        {"frame_id": 1, "category_id": 1, "bbox": [11, 21, 31, 41], "score": 0.8},
    ]

    _write_json(gt_path, gt)
    _write_json(preds_path, preds)

    info = convert_predictions_to_coco(preds_path, gt_path, out_path)

    with out_path.open("r", encoding="utf-8") as handle:
        converted = json.load(handle)

    assert info["total_input"] == 2
    assert info["total_output"] == 2
    assert converted[0]["image_id"] == 101
    assert converted[0]["category_id"] == 1
    assert converted[1]["image_id"] == 102
    assert converted[1]["bbox"] == [11.0, 21.0, 31.0, 41.0]


def test_validate_predictions_schema_accepts_native_records(tmp_path: Path) -> None:
    preds_path = tmp_path / "predictions.json"
    _write_json(
        preds_path,
        [
            {"frame_id": 12, "class_name": "person", "bbox": [1, 2, 3, 4], "score": 0.7},
            {"frame_id": 13, "bbox": [3, 4, 5, 6], "score": 0.8},
        ],
    )

    result = validate_predictions_schema(preds_path)
    assert result["ok"] is True
    assert result["valid"] == 2
