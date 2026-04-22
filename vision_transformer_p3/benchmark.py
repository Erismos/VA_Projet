"""
P3 – Benchmark de performance (FPS, latence, mémoire GPU).

Mesure les mêmes indicateurs que P2 pour permettre une comparaison directe
dans le rapport :
  - FPS moyen (frames par seconde)
  - Latence moyenne et médiane par frame (ms)
  - Mémoire GPU maximale allouée (Go)
  - Résultats sauvegardés en JSON avec le même schéma que P2

Usage :
    python -m vision_transformer_p3.benchmark --source data/video.mp4 --model detr --device cuda
    python -m vision_transformer_p3.benchmark --source data/video.mp4 --model dino_detr --device cuda --warmup 10 --frames 200
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import cv2
import torch

from vision_transformer_p3.detector import build_detector
from vision_transformer_p3.export_utils import save_json


def measure_gpu_memory_gb() -> float:
    """Retourne la mémoire GPU max allouée depuis le dernier reset (en Go)."""
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1e9, 4)
    return 0.0


def reset_gpu_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def load_frames(source: str, max_frames: int) -> list:
    """Charge jusqu'à max_frames frames en mémoire pour benchmark reproductible."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir : {source}")
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("Aucune frame lue depuis la source.")
    return frames


def run_benchmark(
    source: str,
    model: str,
    threshold: float,
    device: str,
    warmup: int,
    frames: int,
    output: str,
) -> dict:
    detector = build_detector(model=model, threshold=threshold, device=device)

    print(f"[P3 bench] Chargement de {frames + warmup} frames …")
    all_frames = load_frames(source, max_frames=frames + warmup)

    # ── Warmup (frames non comptabilisées) ───────────────────────────────────
    print(f"[P3 bench] Warmup ({warmup} frames) …")
    for frame in all_frames[:warmup]:
        detector.detect_frame(frame, frame_id=-1)

    # ── Benchmark ────────────────────────────────────────────────────────────
    reset_gpu_memory_stats()
    bench_frames = all_frames[warmup : warmup + frames]
    if not bench_frames:
        raise ValueError(
            f"Pas assez de frames : {len(all_frames)} disponibles, "
            f"{warmup} warmup + {frames} bench requis."
        )

    latencies_ms: list[float] = []
    print(f"[P3 bench] Mesure sur {len(bench_frames)} frames …")

    t_total_start = time.perf_counter()
    for i, frame in enumerate(bench_frames):
        t0 = time.perf_counter()
        detector.detect_frame(frame, frame_id=i)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(bench_frames)} frames traitées …")
    t_total = time.perf_counter() - t_total_start

    gpu_mem = measure_gpu_memory_gb()

    # ── Calcul des métriques ─────────────────────────────────────────────────
    metrics = {
        "model": model,
        "device": device,
        "threshold": threshold,
        "source": source,
        "warmup_frames": warmup,
        "benchmark_frames": len(bench_frames),
        # Performance
        "avg_fps": round(len(bench_frames) / t_total, 2),
        "avg_latency_ms": round(statistics.mean(latencies_ms), 2),
        "median_latency_ms": round(statistics.median(latencies_ms), 2),
        "p95_latency_ms": round(sorted(latencies_ms)[int(len(latencies_ms) * 0.95)], 2),
        "min_latency_ms": round(min(latencies_ms), 2),
        "max_latency_ms": round(max(latencies_ms), 2),
        # Mémoire
        "gpu_peak_memory_gb": gpu_mem,
    }

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    out_path = Path(output)
    save_json([metrics], out_path)
    print(f"\n[P3 bench] Résultats sauvegardés → {out_path}")
    print("[P3 bench] Résumé :")
    for k, v in metrics.items():
        print(f"      {k:30s}: {v}")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3 – Benchmark FPS / latence / mémoire GPU"
    )
    parser.add_argument("--source", required=True, help="Chemin vidéo source.")
    parser.add_argument(
        "--model",
        choices=["detr", "dino_detr"],
        default="detr",
    )
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=5, help="Frames de chauffe (non comptées).")
    parser.add_argument("--frames", type=int, default=100, help="Frames mesurées.")
    parser.add_argument(
        "--output",
        default="results/p3/benchmark_report.json",
        help="Fichier JSON de sortie.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_benchmark(
        source=args.source,
        model=args.model,
        threshold=args.threshold,
        device=args.device,
        warmup=args.warmup,
        frames=args.frames,
        output=args.output,
    )


if __name__ == "__main__":
    main()