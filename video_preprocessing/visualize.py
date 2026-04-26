from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


def _stable_color(identifier: str | int) -> tuple[int, int, int]:
    digest = hashlib.md5(str(identifier).encode("utf-8")).hexdigest()
    red = max(int(digest[0:2], 16), 80)
    green = max(int(digest[2:4], 16), 80)
    blue = max(int(digest[4:6], 16), 80)
    return blue, green, red


def _load_annotations(path: str | Path) -> list[dict[str, Any]]:
    annotation_path = Path(path)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    if annotation_path.suffix.lower() == ".json":
        with annotation_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and "annotations" in payload:
            payload = payload["annotations"]
        if not isinstance(payload, list):
            raise ValueError("JSON annotations must be a list of detections/tracks")
        return [dict(item) for item in payload]

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _to_int_or_none(value: Any) -> int | None:
    if value in {None, "", "nan"}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value in {None, "", "nan"}:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_detection(det: dict[str, Any]) -> dict[str, Any]:
    bbox = det.get("bbox")
    if bbox is None:
        bbox = [det.get("x", 0), det.get("y", 0), det.get("w", 0), det.get("h", 0)]
    if isinstance(bbox, str):
        bbox = json.loads(bbox)

    frame_id = det.get("frame_id", det.get("frame", 0))
    track_id = det.get("track_id", det.get("id"))
    class_name = det.get("class_name", det.get("label", "object"))
    score = det.get("score", det.get("conf", 0.0))

    return {
        "frame_id": _to_int_or_none(frame_id) or 0,
        "track_id": _to_int_or_none(track_id),
        "class_name": str(class_name),
        "score": _to_float(score),
        "bbox": [float(value) for value in bbox],
    }


def _group_annotations(annotations: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for det in annotations:
        grouped[det["frame_id"]].append(det)
    return grouped


def _draw_box(frame: Any, det: dict[str, Any], *, with_track: bool = False) -> None:
    x, y, w, h = det["bbox"]
    p1 = (int(x), int(y))
    p2 = (int(x + w), int(y + h))
    color_key = det["track_id"] if with_track and det["track_id"] is not None else det["class_name"]
    color = _stable_color(color_key)

    cv2.rectangle(frame, p1, p2, color, 2)
    label_parts = []
    if det["track_id"] is not None:
        label_parts.append(f"ID {det['track_id']}")
    label_parts.append(det["class_name"])
    if det["score"]:
        label_parts.append(f"{det['score']:.2f}")
    label = " | ".join(label_parts)

    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_origin = (p1[0], max(0, p1[1] - 8))
    cv2.rectangle(
        frame,
        (p1[0], max(0, p1[1] - text_height - 10)),
        (p1[0] + text_width + 6, p1[1]),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (text_origin[0] + 3, text_origin[1] - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _resolve_source_frames(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(
            [
                path
                for path in source.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            ]
        )
    return [source]


def _frame_id_from_path(path: Path, fallback: int) -> int:
    match = re.search(r"frame_(\d+)", path.stem)
    if match is not None:
        return int(match.group(1))
    return fallback


def render_visualization(
    source: str | Path,
    annotations_path: str | Path,
    output_path: str | Path,
    *,
    mode: str = "boxes",
    trail_length: int = 30,
    fps: float | None = None,
) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")

    annotations = [_normalize_detection(item) for item in _load_annotations(annotations_path)]
    grouped = _group_annotations(annotations)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_is_dir = source_path.is_dir()
    image_paths = _resolve_source_frames(source_path)
    if not image_paths:
        raise RuntimeError(f"No frames found in source: {source_path}")

    capture = None
    if source_is_dir:
        first_frame = cv2.imread(str(image_paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Cannot read first frame image: {image_paths[0]}")
    else:
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {source_path}")
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            capture.release()
            raise RuntimeError(f"Cannot read first frame from: {source_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    height, width = first_frame.shape[:2]
    video_fps = fps
    if video_fps is None and capture is not None:
        video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    if video_fps is None:
        video_fps = 25.0

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))
    trails: dict[int, list[tuple[int, int]]] = defaultdict(list)

    frame_index = 0
    written_frames = 0
    if source_is_dir:
        frame_iter = (
            (_frame_id_from_path(path, idx), path, cv2.imread(str(path)))
            for idx, path in enumerate(image_paths)
        )
    else:
        def _video_frame_iter() -> Any:
            assert capture is not None
            idx = 0
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                yield idx, None, frame
                idx += 1

        frame_iter = _video_frame_iter()

    for frame_index, image_path, frame in frame_iter:
        if frame is None:
            if image_path is not None:
                raise RuntimeError(f"Cannot read frame image: {image_path}")
            break

        frame_annotations = grouped.get(frame_index, [])
        for det in frame_annotations:
            _draw_box(frame, det, with_track=mode == "tracks")
            if mode == "tracks" and det["track_id"] is not None:
                x, y, w, h = det["bbox"]
                center = (int(x + w / 2), int(y + h / 2))
                trail = trails[int(det["track_id"])]
                trail.append(center)
                if len(trail) > trail_length:
                    trail.pop(0)
                color = _stable_color(int(det["track_id"]))
                for idx in range(1, len(trail)):
                    cv2.line(frame, trail[idx - 1], trail[idx], color, 2, cv2.LINE_AA)
                cv2.circle(frame, center, 3, color, -1)

        cv2.putText(
            frame,
            f"Frame {frame_index}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        written_frames += 1

    if capture is not None:
        capture.release()
    writer.release()

    summary = {
        "source": str(source_path),
        "annotations": str(annotations_path),
        "output": str(output),
        "mode": mode,
        "frames_written": written_frames,
        "tracks_with_history": len(trails),
    }
    with output.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1 visualization helper")
    parser.add_argument("--source", required=True, help="Video file or exported frames directory")
    parser.add_argument("--annotations", required=True, help="CSV or JSON annotations file")
    parser.add_argument("--output", required=True, help="Output annotated video path")
    parser.add_argument("--mode", choices=["boxes", "tracks"], default="boxes")
    parser.add_argument("--trail-length", type=int, default=30)
    parser.add_argument("--fps", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = render_visualization(
        source=args.source,
        annotations_path=args.annotations,
        output_path=args.output,
        mode=args.mode,
        trail_length=args.trail_length,
        fps=args.fps,
    )
    print(result)


if __name__ == "__main__":
    main()
