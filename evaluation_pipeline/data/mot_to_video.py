from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def _resolve_img_dir(seq_dir: Path) -> Path:
    if (seq_dir / "img1").is_dir():
        return seq_dir / "img1"
    return seq_dir


def _list_frames(img_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def create_video_from_mot(
    seq_dir: str,
    output: str,
    fps: float = 25.0,
    max_frames: int | None = None,
) -> dict[str, str | int | float]:
    seq_path = Path(seq_dir)
    if not seq_path.exists():
        raise FileNotFoundError(f"MOT sequence path not found: {seq_dir}")

    img_dir = _resolve_img_dir(seq_path)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    frames = _list_frames(img_dir)
    if not frames:
        raise FileNotFoundError(f"No image frames found in: {img_dir}")

    if max_frames is not None and max_frames > 0:
        frames = frames[:max_frames]

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {frames[0]}")

    height, width = first.shape[:2]

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video for writing: {out_path}")

    written = 0
    try:
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {frame_path}")

            frame_h, frame_w = frame.shape[:2]
            if (frame_w, frame_h) != (width, height):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written == 0:
        raise RuntimeError("No frames were written to the output video")

    return {
        "seq_dir": str(seq_path),
        "img_dir": str(img_dir),
        "output": str(out_path),
        "fps": float(fps),
        "frames_written": int(written),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic sample video from a MOT17 sequence (img1 -> mp4)"
    )
    parser.add_argument(
        "--seq-dir",
        default="data/raw/MOT17/train/MOT17-02-FRCNN",
        help="Path to MOT sequence folder (containing img1) or directly to img1",
    )
    parser.add_argument(
        "--output",
        default="data/sample_video.mp4",
        help="Output video path",
    )
    parser.add_argument("--fps", type=float, default=25.0, help="Output frames per second")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for quick runs",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = create_video_from_mot(
        seq_dir=args.seq_dir,
        output=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    print(result)


if __name__ == "__main__":
    main()
