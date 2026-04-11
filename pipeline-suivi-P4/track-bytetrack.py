import cv2
import glob
import os
import pandas as pd
import hashlib
from pathlib import Path
from ultralytics import YOLO
import sys
sys.path.append("..")
from p2.detectors import DetectorConfig, create_detector

def get_color(track_id):
    """Génère une couleur BGR unique et reproductible pour chaque ID."""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)

# config
MOT17_DIR    = "./MOT17/train/MOT17-02-FRCNN"
OUTPUT_DIR   = "./outputs"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/tracked_MOT17-02.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# chargement modèle 
# quand P2 te donne leur modèle, remplace "yolov8n.pt" par le chemin vers leur .pt
model = YOLO("yolov8n.pt")

# lecture images séquence MOT17
image_dir = Path(MOT17_DIR) / "img1"
image_paths = sorted(glob.glob(str(image_dir / "*.jpg")))
print(f"📁 {len(image_paths)} images trouvées dans {image_dir}")

# initialisation vidéo sortie
first_frame = cv2.imread(image_paths[0])
h, w = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 25, (w, h))

# tracking frame par frame
track_history = {}   # {track_id: [(cx, cy), ...]}
mot_results   = []

print("Lancement du tracking...")

for frame_idx, img_path in enumerate(image_paths):
    frame = cv2.imread(img_path)

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.3,
        iou=0.5,
        verbose=False
    )

    if results[0].boxes.id is None:
        video_writer.write(frame)
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy()
    ids   = results[0].boxes.id.cpu().numpy().astype(int)
    confs = results[0].boxes.conf.cpu().numpy()

    for box, track_id, conf in zip(boxes, ids, confs):
        x1, y1, x2, y2 = map(int, box)
        w_box = x2 - x1
        h_box = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        mot_results.append([
            frame_idx + 1, track_id,
            x1, y1, w_box, h_box,
            round(float(conf), 4),
            -1, -1, -1
        ])

        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        color = get_color(track_id)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        pts = track_history[track_id]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i-1], pts[i], color, thickness)

    video_writer.write(frame)

    if (frame_idx + 1) % 50 == 0:
        print(f"  Frame {frame_idx + 1}/{len(image_paths)} traitée")

video_writer.release()
print(f"\n Vidéo annotée sauvegardée : {OUTPUT_VIDEO}")

# sauvegarde résultats au format MOT
mot_df = pd.DataFrame(mot_results,
    columns=["frame","id","x","y","w","h","conf","x3d","y3d","z3d"])
results_file = f"{OUTPUT_DIR}/MOT17-02.txt"
mot_df.to_csv(results_file, header=False, index=False)
print(f"📄 Résultats MOT sauvegardés : {results_file}")