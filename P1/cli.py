from __future__ import annotations

import argparse

from P1.video_utils import export_video_frames
from P1.visualize import render_visualization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1 video pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract-frames", help="Export frames from a video")
    extract.add_argument("--source", required=True, help="Input video path")
    extract.add_argument("--output-dir", required=True, help="Directory where frames are written")
    extract.add_argument("--every-n", type=int, default=1, help="Keep one frame every N frames")
    extract.add_argument("--start-frame", type=int, default=0, help="First frame to export")
    extract.add_argument("--max-frames", type=int, default=None, help="Maximum number of exported frames")
    extract.add_argument("--image-ext", default=".jpg", help="Image extension, for example .jpg or .png")
    extract.add_argument("--zero-pad", type=int, default=6, help="Frame index padding")

    visualize = sub.add_parser("visualize", help="Draw boxes or tracks on video frames")
    visualize.add_argument("--source", required=True, help="Video file or frames directory")
    visualize.add_argument("--annotations", required=True, help="CSV or JSON annotations file")
    visualize.add_argument("--output", required=True, help="Annotated video output path")
    visualize.add_argument("--mode", choices=["boxes", "tracks"], default="boxes")
    visualize.add_argument("--trail-length", type=int, default=30)
    visualize.add_argument("--fps", type=float, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "extract-frames":
        result = export_video_frames(
            source_path=args.source,
            output_dir=args.output_dir,
            every_n=args.every_n,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            image_ext=args.image_ext,
            zero_pad=args.zero_pad,
        )
        print(result)
        return

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