"""
P3 – Vision Transformer object detector (DETR / DINO-DETR).

Produces the exact same JSON/CSV schema as P2 so P5 can compare both
detectors with a unified evaluation script.

Output record schema (one dict per detection):
    {
        "frame_id"   : int | str,   # index de frame ou nom d'image
        "class_name" : str,         # label humain (ex. "person")
        "bbox"       : [x, y, w, h],# coordonnées en pixels (top-left + taille)
        "score"      : float,       # confiance [0, 1]
        "model"      : str,         # "detr" ou "dino_detr"
    }

Usage rapide:
    detector = DetrDetector(threshold=0.7, device="cuda")
    records   = detector.detect_frame(frame_bgr, frame_id=42)

Usage ligne de commande:
    python -m vision_transformer_p3.detector --source video.mp4 --output results/p3/predictions.json
    python -m vision_transformer_p3.detector --source video.mp4 --model dino_detr --output results/p3/predictions_dino.json
"""
from __future__ import annotations

import argparse
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image

from vision_transformer_p3.export_utils import save_csv, save_json

# ── Constantes ────────────────────────────────────────────────────────────────
DETR_MODEL_ID = "facebook/detr-resnet-50"
DINO_MODEL_ID = "IDEA-Research/grounding-dino-base"  # via transformers >= 4.38


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base
# ══════════════════════════════════════════════════════════════════════════════

class BaseTransformerDetector(ABC):
    """Interface commune à DETR et DINO-DETR."""

    def __init__(self, threshold: float, device: str, cache_dir: str | None = None) -> None:
        self.threshold = threshold
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None: ...

    @abstractmethod
    def _run_inference(
        self, image_rgb: Image.Image
    ) -> tuple[list[list[float]], list[float], list[str]]:
        """Retourne (boxes_xyxy_pixels, scores, label_names)."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    # ── API publique ──────────────────────────────────────────────────────────

    def detect_frame(
        self,
        frame_bgr: "cv2.Mat",
        frame_id: int | str = 0,
    ) -> list[dict[str, Any]]:
        """
        Détecte les objets dans une frame OpenCV (BGR).

        Retourne une liste de dicts compatibles avec le format P2.
        """
        image_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes_xyxy, scores, label_names = self._run_inference(image_rgb)

        records: list[dict[str, Any]] = []
        for box, score, class_name in zip(boxes_xyxy, scores, label_names):
            x1, y1, x2, y2 = box
            records.append(
                {
                    "frame_id": frame_id,
                    "class_name": class_name,
                    # Format [x, y, w, h] en pixels — identique à P2
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "score": round(float(score), 4),
                    "model": self.model_name,
                }
            )
        return records


# ══════════════════════════════════════════════════════════════════════════════
# DETR (facebook/detr-resnet-50)
# ══════════════════════════════════════════════════════════════════════════════

class DetrDetector(BaseTransformerDetector):
    """
    Détecteur basé sur DETR (DEtection TRansformer, Carion et al. 2020).

    Points clés vs YOLO / Faster R-CNN (utiles pour le rapport) :
    - Pas de NMS : le modèle apprend des prédictions non-redondantes
      grâce aux object queries et au Hungarian matching.
    - Architecture end-to-end : backbone CNN → positional encoding
      → Transformer encoder-decoder → têtes de classification/régression.
    - 100 object queries par défaut → 100 prédictions candidates filtrées
      par le seuil de confiance.
    """

    @property
    def model_name(self) -> str:
        return "detr"

    def _load_model(self) -> None:
        from transformers import DetrForObjectDetection, DetrImageProcessor

        print(f"[P3] Chargement DETR depuis '{DETR_MODEL_ID}' …")
        load_kwargs: dict[str, str] = {}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cache_dir)
        self._processor = DetrImageProcessor.from_pretrained(DETR_MODEL_ID, **load_kwargs)
        self._model = DetrForObjectDetection.from_pretrained(DETR_MODEL_ID, **load_kwargs)
        self._model.to(self.device)
        self._model.eval()
        print("[P3] DETR chargé.")

    def _run_inference(
        self, image_rgb: Image.Image
    ) -> tuple[list[list[float]], list[float], list[str]]:
        inputs = self._processor(images=image_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Remapping vers taille originale (H, W)
        orig_h, orig_w = image_rgb.height, image_rgb.width
        target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)

        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.threshold,
            target_sizes=target_sizes,
        )[0]

        boxes = results["boxes"].cpu().tolist()       # [[x1,y1,x2,y2], ...]
        scores = results["scores"].cpu().tolist()
        label_ids = results["labels"].cpu().tolist()
        label_names = [
            self._model.config.id2label[lid] for lid in label_ids
        ]
        return boxes, scores, label_names


# ══════════════════════════════════════════════════════════════════════════════
# DINO-DETR  (IDEA-Research/grounding-dino-base)
# ══════════════════════════════════════════════════════════════════════════════

class DinoDETRDetector(BaseTransformerDetector):
    """
    Détecteur basé sur Grounding DINO (Liu et al. 2023).

    Grounding DINO améliore DINO-DETR en ajoutant du grounding textuel :
    on passe une requête en langage naturel (ex. "person . car . bicycle .")
    et le modèle ne détecte que ces classes.
    Résultats état-de-l'art sur COCO zero-shot.

    Nécessite : transformers >= 4.38
        pip install transformers>=4.38 torch torchvision
    """

    # Classes COCO 80 regroupées en une requête texte pour Grounding DINO
    COCO_TEXT_QUERY = (
        "person . bicycle . car . motorcycle . airplane . bus . train . truck . "
        "boat . traffic light . fire hydrant . stop sign . parking meter . bench . "
        "bird . cat . dog . horse . sheep . cow . elephant . bear . zebra . giraffe . "
        "backpack . umbrella . handbag . tie . suitcase . frisbee . skis . snowboard . "
        "sports ball . kite . baseball bat . baseball glove . skateboard . surfboard . "
        "tennis racket . bottle . wine glass . cup . fork . knife . spoon . bowl . "
        "banana . apple . sandwich . orange . broccoli . carrot . hot dog . pizza . "
        "donut . cake . chair . couch . potted plant . bed . dining table . toilet . "
        "tv . laptop . mouse . remote . keyboard . cell phone . microwave . oven . "
        "toaster . sink . refrigerator . book . clock . vase . scissors . teddy bear . "
        "hair drier . toothbrush ."
    )

    @property
    def model_name(self) -> str:
        return "dino_detr"

    def _load_model(self) -> None:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        print(f"[P3] Chargement DINO-DETR depuis '{DINO_MODEL_ID}' …")
        load_kwargs: dict[str, str] = {}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cache_dir)
        self._processor = AutoProcessor.from_pretrained(DINO_MODEL_ID, **load_kwargs)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID, **load_kwargs)
        self._model.to(self.device)
        self._model.eval()
        print("[P3] DINO-DETR chargé.")

    def _run_inference(
        self, image_rgb: Image.Image
    ) -> tuple[list[list[float]], list[float], list[str]]:
        inputs = self._processor(
            images=image_rgb,
            text=self.COCO_TEXT_QUERY,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        orig_h, orig_w = image_rgb.height, image_rgb.width
        results = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=self.threshold,
            text_threshold=self.threshold,
            target_sizes=[(orig_h, orig_w)],
        )[0]

        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        label_names = results["labels"]  # déjà des strings
        return boxes, scores, label_names


# ══════════════════════════════════════════════════════════════════════════════
# Fonctions utilitaires
# ══════════════════════════════════════════════════════════════════════════════

def build_detector(
    model: str,
    threshold: float,
    device: str,
    cache_dir: str | None = None,
) -> BaseTransformerDetector:
    """Factory — retourne le bon détecteur selon le nom."""
    if model == "detr":
        return DetrDetector(threshold=threshold, device=device, cache_dir=cache_dir)
    if model == "dino_detr":
        return DinoDETRDetector(threshold=threshold, device=device, cache_dir=cache_dir)
    raise ValueError(f"Modèle inconnu : '{model}'. Choisir 'detr' ou 'dino_detr'.")


def process_video(
    source: str,
    detector: BaseTransformerDetector,
    max_frames: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Parcourt une vidéo frame par frame et collecte les détections.

    Retourne (liste_détections, métriques_perf).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Impossible d'ouvrir la source : {source}")

    all_detections: list[dict[str, Any]] = []
    frame_times: list[float] = []
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_id >= max_frames:
                break

            t0 = time.perf_counter()
            dets = detector.detect_frame(frame, frame_id=frame_id)
            frame_times.append(time.perf_counter() - t0)

            all_detections.extend(dets)
            frame_id += 1

            if frame_id % 50 == 0:
                fps_now = 1.0 / (sum(frame_times[-50:]) / len(frame_times[-50:]))
                print(f"  frame {frame_id:5d} | FPS moyen (50 dernières) : {fps_now:.1f}")
    finally:
        cap.release()

    total_time = sum(frame_times)
    metrics = {
        "total_frames": frame_id,
        "total_detections": len(all_detections),
        "total_time_s": round(total_time, 3),
        "avg_fps": round(frame_id / total_time, 2) if total_time > 0 else 0.0,
        "avg_ms_per_frame": round(total_time / frame_id * 1000, 2) if frame_id > 0 else 0.0,
    }
    return all_detections, metrics


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P3 – Détection par Vision Transformer (DETR / DINO-DETR)"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Chemin vers la vidéo (ex. data/video.mp4) ou une image.",
    )
    parser.add_argument(
        "--model",
        choices=["detr", "dino_detr"],
        default="detr",
        help="Modèle à utiliser (défaut : detr).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Seuil de confiance [0-1] (défaut : 0.7).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device PyTorch : 'cpu', 'cuda', 'cuda:0', etc.",
    )
    parser.add_argument(
        "--cache-dir",
        default="models/vision_transformer_p3/hf_cache",
        help="Dossier de cache local des poids Transformers.",
    )
    parser.add_argument(
        "--output",
        default="results/p3/predictions.json",
        help="Chemin du fichier JSON de sortie.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Chemin optionnel du CSV de sortie (même données).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Nombre max de frames à traiter (debug).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    detector = build_detector(
        model=args.model,
        threshold=args.threshold,
        device=args.device,
        cache_dir=args.cache_dir,
    )

    print(f"[P3] Traitement de '{args.source}' avec {args.model.upper()} …")
    detections, metrics = process_video(
        source=args.source,
        detector=detector,
        max_frames=args.max_frames,
    )

    # ── Export JSON (requis par P5) ──────────────────────────────────────────
    output_path = Path(args.output)
    save_json(detections, output_path)
    print(f"[P3] {len(detections)} détections → {output_path}")

    # ── Export CSV (optionnel) ───────────────────────────────────────────────
    if args.output_csv:
        csv_path = Path(args.output_csv)
        save_csv(detections, csv_path)
        print(f"[P3] CSV → {csv_path}")

    # ── Métriques de performance ─────────────────────────────────────────────
    metrics_path = output_path.with_name(
        output_path.stem + "_metrics.json"
    )
    # Ajout des infos modèle dans les métriques
    metrics.update(
        {
            "model": args.model,
            "threshold": args.threshold,
            "device": args.device,
            "source": args.source,
        }
    )
    save_json([metrics], metrics_path)
    print(f"[P3] Métriques → {metrics_path}")
    print("[P3] Résumé :")
    for k, v in metrics.items():
        print(f"      {k:25s}: {v}")


if __name__ == "__main__":
    main()