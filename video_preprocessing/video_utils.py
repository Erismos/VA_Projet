from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


@dataclass
class FrameExportSummary:
    source: str
    output_dir: str
    frame_count: int
    exported_frames: int
    every_n: int
    image_pattern: str
    fps: float | None
    width: int | None
    height: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "output_dir": self.output_dir,
            "frame_count": self.frame_count,
            "exported_frames": self.exported_frames,
            "every_n": self.every_n,
            "image_pattern": self.image_pattern,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
        }


def _is_image_path(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def export_video_frames(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    every_n: int = 1,
    start_frame: int = 0,
    max_frames: int | None = None,
    image_ext: str = ".jpg",
    zero_pad: int = 6,
) -> dict[str, Any]:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    if start_frame < 0:
        raise ValueError("start_frame must be >= 0")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        frames = sorted(
            [
                path
                for path in source.iterdir()
                if path.is_file() and _is_image_path(path)
            ]
        )
        exported = 0
        selected_frames: list[dict[str, Any]] = []
        for index, image_path in enumerate(frames):
            if index < start_frame:
                continue
            if max_frames is not None and exported >= max_frames:
                break
            if (index - start_frame) % every_n != 0:
                continue

            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Cannot read frame image: {image_path}")

            frame_name = f"frame_{index:0{zero_pad}d}{image_ext}"
            frame_output = output / frame_name
            cv2.imwrite(str(frame_output), frame)
            selected_frames.append(
                {"frame_id": index, "source_path": str(image_path), "output_path": str(frame_output)}
            )
            exported += 1

        manifest = {
            "source": str(source),
            "output_dir": str(output),
            "frame_count": len(frames),
            "exported_frames": exported,
            "every_n": every_n,
            "image_pattern": f"frame_{{frame_id:0{zero_pad}d}}{image_ext}",
            "fps": None,
            "width": None,
            "height": None,
            "frames": selected_frames,
        }
        with (output / "frames_manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return manifest

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None

    exported = 0
    frame_index = 0
    selected_frames: list[dict[str, Any]] = []

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        if frame_index < start_frame:
            frame_index += 1
            continue

        if max_frames is not None and exported >= max_frames:
            break

        if (frame_index - start_frame) % every_n == 0:
            frame_name = f"frame_{frame_index:0{zero_pad}d}{image_ext}"
            frame_output = output / frame_name
            cv2.imwrite(str(frame_output), frame)
            selected_frames.append(
                {"frame_id": frame_index, "source_path": str(source), "output_path": str(frame_output)}
            )
            exported += 1

        frame_index += 1

    capture.release()

    summary = FrameExportSummary(
        source=str(source),
        output_dir=str(output),
        frame_count=frame_count,
        exported_frames=exported,
        every_n=every_n,
        image_pattern=f"frame_{{frame_id:0{zero_pad}d}}{image_ext}",
        fps=fps,
        width=width,
        height=height,
    )
    manifest = summary.as_dict()
    manifest["frames"] = selected_frames
    with (output / "frames_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def build_frame_path(output_dir: str | Path, frame_id: int, *, zero_pad: int = 6, image_ext: str = ".jpg") -> Path:
    return Path(output_dir) / f"frame_{frame_id:0{zero_pad}d}{image_ext}"
