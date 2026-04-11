import cv2
import glob
import os
import pandas as pd
import hashlib
from pathlib import Path
from ultralytics import YOLO

# config
MOT17_DIR   = "./MOT17/train/MOT17-02-FRCNN"  # séquence de test
OUTPUT_DIR  = "./outputs"
OUTPUT_VIDEO = f"{OUTPUT_DIR}/tracked_MOT17-02.mp4"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# chargement modèle
model = YOLO("yolov8n.pt")   # à remplacer par le modèle de P2

# lecture images séquence mot17
image_dir = Path(MOT17_DIR) / "img1"
image_paths = sorted(glob.glob(str(image_dir / "*.jpg")))
print(f"📁 {len(image_paths)} images trouvées dans {image_dir}")

# initialisation video sortie
first_frame = cv2.imread(image_paths[0])
h, w = first_frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 25, (w, h))

# tracking frame par frame
# Stocke les positions passées pour dessiner les trajectoires
track_history = {}   # {track_id: [(cx, cy), ...]}
mot_results   = []   # résultats au format MOT pour l'évaluation

print("🚀 Lancement du tracking...")

for frame_idx, img_path in enumerate(image_paths):
    frame = cv2.imread(img_path)

    # ByteTrack via Ultralytics — persist=True pour garder les IDs entre frames
    results = model.track(
        frame,
        persist=True,          # IDs persistants entre frames
        tracker="bytetrack.yaml",
        classes=[0],           # classe 0 = personne dans COCO
        conf=0.3,              # seuil de confiance minimum
        iou=0.5,               # seuil IoU pour la suppression
        verbose=False
    )

    if results[0].boxes.id is None:
        # Aucun objet détecté/tracké dans cette frame
        video_writer.write(frame)
        continue

    boxes   = results[0].boxes.xyxy.cpu().numpy()    # [x1, y1, x2, y2]
    ids     = results[0].boxes.id.cpu().numpy().astype(int)
    confs   = results[0].boxes.conf.cpu().numpy()

    for box, track_id, conf in zip(boxes, ids, confs):
        x1, y1, x2, y2 = map(int, box)
        w_box = x2 - x1
        h_box = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Sauvegarde au format MOT
        # Format : frame, id, x, y, w, h, conf, -1, -1, -1
        mot_results.append([
            frame_idx + 1, track_id,
            x1, y1, w_box, h_box,
            round(float(conf), 4),
            -1, -1, -1
        ])

        # Historique de trajectoire
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        # On garde seulement les 30 dernières positions
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        # Couleur unique par ID
        color = get_color(track_id)

        # Dessin de la boîte et de l'ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID {track_id}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Dessin de la trajectoire
        pts = track_history[track_id]
        for i in range(1, len(pts)):
            alpha = i / len(pts)   # fade : plus opaque = plus récent
            thickness = max(1, int(alpha * 3))
            cv2.line(frame, pts[i-1], pts[i], color, thickness)

    video_writer.write(frame)

    if (frame_idx + 1) % 50 == 0:
        print(f"  Frame {frame_idx + 1}/{len(image_paths)} traitée")

video_writer.release()
print(f"\nVidéo annotée sauvegardée : {OUTPUT_VIDEO}")

# sauvegarde résultats

mot_df = pd.DataFrame(mot_results,
    columns=["frame","id","x","y","w","h","conf","x3d","y3d","z3d"])
results_file = f"{OUTPUT_DIR}/MOT17-02.txt"
mot_df.to_csv(results_file, header=False, index=False)
print(f"📄 Résultats MOT sauvegardés : {results_file}")


def get_color(track_id):
    """Génère une couleur BGR unique et reproductible pour chaque ID."""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)