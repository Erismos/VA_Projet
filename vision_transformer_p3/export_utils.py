"""
P3 export utilities.

Re-uses P2's save_json / save_csv so both detectors produce identical files.
If P2 is installed as a package, import from there directly; otherwise the
functions are duplicated here so P3 can run standalone.
"""
from __future__ import annotations

try:
    from object_detection.export_utils import save_csv, save_json  # noqa: F401 (re-export)
except ModuleNotFoundError:
    # ── Standalone fallback (same implementation as P2) ──────────────────────
    import csv
    import json
    from pathlib import Path
    from typing import Any

    def ensure_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def save_json(detections: list[dict[str, Any]], output_path: Path) -> None:
        ensure_parent(output_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2)

    def save_csv(detections: list[dict[str, Any]], output_path: Path) -> None:
        ensure_parent(output_path)
        fieldnames = ["frame_id", "class_name", "x", "y", "w", "h", "score"]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for det in detections:
                bbox = det.get("bbox", [0.0, 0.0, 0.0, 0.0])
                writer.writerow(
                    {
                        "frame_id": det.get("frame_id", det.get("image_id", "")),
                        "class_name": det.get("class_name", ""),
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2],
                        "h": bbox[3],
                        "score": det.get("score", 0.0),
                    }
                )