from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import cv2
import pandas as pd


def get_color(track_id: int) -> tuple[int, int, int]:
    digest = hashlib.md5(str(track_id).encode("utf-8")).hexdigest()
    r, g, b = int(digest[0:2], 16), int(digest[2:4], 16), int(digest[4:6], 16)
    return max(b, 80), max(g, 80), max(r, 80)


def render_tracking_video(
    seq_img_dir: str,
    pred_file: str,
    output_video: str,
    trail_len: int,
    fps: int,
) -> str:
    pred_cols = ["frame", "id", "x", "y", "w", "h", "conf", "x3d", "y3d", "z3d"]
    pred = pd.read_csv(pred_file, header=None, names=pred_cols)

    image_paths = sorted(glob.glob(str(Path(seq_img_dir) / "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No frame found in {seq_img_dir}")

    first = cv2.imread(image_paths[0])
    if first is None:
        raise RuntimeError(f"Cannot read first frame: {image_paths[0]}")

    h_vid, w_vid = first.shape[:2]
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w_vid, h_vid))

    trails: dict[int, list[tuple[int, int]]] = {}

    for frame_idx, img_path in enumerate(image_paths):
        frame_id = frame_idx + 1
        img = cv2.imread(img_path)
        if img is None:
            continue

        dets = pred[pred["frame"] == frame_id]

        for _, row in dets.iterrows():
            track_id = int(row["id"])
            x1, y1 = int(row["x"]), int(row["y"])
            x2, y2 = x1 + int(row["w"]), y1 + int(row["h"])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            conf = float(row["conf"])

            color = get_color(track_id)

            trails.setdefault(track_id, []).append((cx, cy))
            if len(trails[track_id]) > trail_len:
                trails[track_id].pop(0)

            pts = trails[track_id]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                thickness = max(1, int(alpha * 3))
                fade_color = tuple(int(c * alpha) for c in color)
                cv2.line(img, pts[i - 1], pts[i], fade_color, thickness, cv2.LINE_AA)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"#{track_id}  {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(
                img,
                label,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        overlay_text = f"Frame {frame_id:04d}"
        cv2.rectangle(img, (0, 0), (220, 32), (0, 0, 0), -1)
        cv2.putText(img, overlay_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    print(f"Visualization video written to: {output}")
    return str(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P4 tracking visualization")
    parser.add_argument("--seq-img-dir", default="data/raw/MOT17/train/MOT17-02-FRCNN/img1")
    parser.add_argument("--pred-file", default="results/p4/MOT17-02.txt")
    parser.add_argument("--output", default="results/p4/demo_visualization.mp4")
    parser.add_argument("--trail-len", type=int, default=40)
    parser.add_argument("--fps", type=int, default=25)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    render_tracking_video(
        seq_img_dir=args.seq_img_dir,
        pred_file=args.pred_file,
        output_video=args.output,
        trail_len=args.trail_len,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
