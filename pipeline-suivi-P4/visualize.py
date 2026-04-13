import cv2
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import hashlib

def get_color(track_id):
    """Couleur BGR unique et reproductible par ID."""
    h = hashlib.md5(str(track_id).encode()).hexdigest()
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = max(r, 80); g = max(g, 80); b = max(b, 80)
    return (b, g, r)

def draw_dashed_box(img, x1, y1, x2, y2, color, thickness=2, dash_len=10):
    """Dessine une boîte en pointillés."""
    pts = [(x1,y1,x2,y1), (x2,y1,x2,y2), (x2,y2,x1,y2), (x1,y2,x1,y1)]
    for ax, ay, bx, by in pts:
        dist = int(((bx-ax)**2 + (by-ay)**2)**0.5)
        if dist == 0:
            continue
        dashes = dist // (dash_len * 2)
        for i in range(dashes + 1):
            t0 = (i * 2 * dash_len) / dist
            t1 = min((i * 2 * dash_len + dash_len) / dist, 1.0)
            p0 = (int(ax + t0*(bx-ax)), int(ay + t0*(by-ay)))
            p1 = (int(ax + t1*(bx-ax)), int(ay + t1*(by-ay)))
            cv2.line(img, p0, p1, color, thickness)

# config 
SEQ_NAME  = "MOT17-02-FRCNN"
MOT17_DIR = f"./MOT17/train/{SEQ_NAME}/img1"
PRED_FILE = "./outputs/MOT17-02.txt"
OUTPUT    = "./outputs/demo_visualization.mp4"
TRAIL_LEN = 40

# chargement prédictions
pred_cols = ["frame","id","x","y","w","h","conf","x3d","y3d","z3d"]
pred = pd.read_csv(PRED_FILE, header=None, names=pred_cols)

trail = {}

# initialisation vidéo
image_paths = sorted(glob.glob(f"{MOT17_DIR}/*.jpg"))
first = cv2.imread(image_paths[0])
h_vid, w_vid = first.shape[:2]
writer = cv2.VideoWriter(OUTPUT, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w_vid, h_vid))

print(f"Génération de la vidéo de démo ({len(image_paths)} frames)...")

for frame_idx, img_path in enumerate(image_paths):
    frame_id = frame_idx + 1
    img = cv2.imread(img_path)

    dets = pred[pred["frame"] == frame_id]
    active_ids = set()

    for _, row in dets.iterrows():
        tid = int(row["id"])
        x1, y1 = int(row["x"]), int(row["y"])
        x2, y2 = x1 + int(row["w"]), y1 + int(row["h"])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        conf = row["conf"]

        color = get_color(tid)
        active_ids.add(tid)

        if tid not in trail:
            trail[tid] = []
        trail[tid].append((cx, cy))
        if len(trail[tid]) > TRAIL_LEN:
            trail[tid].pop(0)

        pts = trail[tid]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            t = max(1, int(alpha * 3))
            fade_color = tuple(int(c * alpha) for c in color)
            cv2.line(img, pts[i-1], pts[i], fade_color, t, cv2.LINE_AA)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"#{tid}  {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)

        cv2.circle(img, (cx, cy), 3, color, -1)

    overlay_text = f"Frame {frame_id:04d}   Personnes : {len(active_ids)}"
    cv2.rectangle(img, (0, 0), (300, 32), (0, 0, 0), -1)
    cv2.putText(img, overlay_text, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    writer.write(img)

    if (frame_idx + 1) % 100 == 0:
        print(f"  {frame_idx + 1}/{len(image_paths)} frames")

writer.release()
print(f"\nVidéo de démo générée : {OUTPUT}")