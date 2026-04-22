from __future__ import annotations

import json
from pathlib import Path

from p5.validation import (
    validate_p2_training_inputs,
    validate_prediction_file,
    validate_yolo_dataset_layout,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validate_prediction_file_returns_normalized_details(tmp_path: Path) -> None:
    preds_path = tmp_path / "predictions.json"
    _write_json(preds_path, [{"frame_id": 1, "bbox": [1, 2, 3, 4], "score": 0.9}])

    result = validate_prediction_file(preds_path)

    assert result["ok"] is True
    assert result["reason"] == "ok"
    assert result["details"]["valid"] == 1


def test_validate_prediction_file_flags_invalid_payload(tmp_path: Path) -> None:
    preds_path = tmp_path / "bad_predictions.json"
    _write_json(preds_path, [{"frame_id": 1, "score": 0.9}])

    result = validate_prediction_file(preds_path)

    assert result["ok"] is False
    assert result["details"]["invalid"] == 1


def test_validate_yolo_dataset_layout_returns_normalized_details(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "train" / "seq1_sample.jpg").write_bytes(b"x")
    (root / "labels" / "train" / "seq1_sample.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    (root / "images" / "val" / "seq2_sample.jpg").write_bytes(b"x")
    (root / "labels" / "val" / "seq2_sample.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")

    result = validate_yolo_dataset_layout(root, ["seq1"], ["seq2"])

    assert result["ok"] is True
    assert result["details"]["splits"]["train"]["labels"] == 1


def test_validate_p2_training_inputs_returns_normalized_details(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "images" / "val").mkdir(parents=True)
    (dataset_root / "images" / "train" / "sample.jpg").write_bytes(b"x")
    (dataset_root / "images" / "val" / "sample.jpg").write_bytes(b"x")

    yaml_path = tmp_path / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
            ]
        ),
        encoding="utf-8",
    )

    result = validate_p2_training_inputs(yaml_path)

    assert result["ok"] is True
    assert result["details"]["train_images"] == 1
    assert result["details"]["val_images"] == 1
