from __future__ import annotations

from pathlib import Path

from p2.train import validate_yolo_training_inputs
from p5.data.split_data import validate_yolo_dataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_validate_yolo_dataset_detects_aligned_pairs(tmp_path: Path) -> None:
    root = tmp_path / "MOT17_YOLO_DATASET"
    images_train = root / "images" / "train"
    labels_train = root / "labels" / "train"
    images_val = root / "images" / "val"
    labels_val = root / "labels" / "val"

    _touch(images_train / "MOT17-02_000001.jpg")
    _touch(labels_train / "MOT17-02_000001.txt")
    _touch(images_val / "MOT17-04_000001.jpg")
    _touch(labels_val / "MOT17-04_000001.txt")

    result = validate_yolo_dataset(
        str(root),
        train_seqs=["MOT17-02"],
        val_seqs=["MOT17-04"],
    )

    assert result["ok"] is True
    assert result["splits"]["train"]["labels"] == 1
    assert result["splits"]["val"]["labels"] == 1


def test_validate_yolo_training_inputs_accepts_generated_yaml(tmp_path: Path) -> None:
    dataset_root = tmp_path / "MOT17_YOLO_DATASET"
    (dataset_root / "images" / "train").mkdir(parents=True)
    (dataset_root / "images" / "val").mkdir(parents=True)

    (dataset_root / "images" / "train" / "sample.jpg").write_bytes(b"x")
    (dataset_root / "images" / "val" / "sample.jpg").write_bytes(b"x")

    yaml_path = tmp_path / "mot17_pedestrian_yolo.yaml"
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

    result = validate_yolo_training_inputs(str(yaml_path))
    assert result["ok"] is True
    assert result["train_images"] == 1
    assert result["val_images"] == 1
