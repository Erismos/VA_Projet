import pandas as pd
import motmetrics as mm
import numpy as np
from pathlib import Path

# config
SEQ_NAME    = "MOT17-02-FRCNN"
MOT17_DIR   = f"./MOT17/train/{SEQ_NAME}"
GT_FILE     = f"{MOT17_DIR}/gt/gt.txt"
PRED_FILE   = f"./outputs/MOT17-02.txt"

# lecture ground truth mot17
# Format GT : frame, id, x, y, w, h, conf, class, visibility
gt_cols = ["frame","id","x","y","w","h","conf","class","visibility"]
gt = pd.read_csv(GT_FILE, header=None, names=gt_cols)

# On garde seulement les piétons (class=1) bien visibles (visibility > 0.25)
gt = gt[(gt["class"] == 1) & (gt["visibility"] > 0.25)].copy()
print(f"📊 Ground truth : {len(gt)} annotations sur {gt['frame'].nunique()} frames")

# lecture prédictions
pred_cols = ["frame","id","x","y","w","h","conf","x3d","y3d","z3d"]
pred = pd.read_csv(PRED_FILE, header=None, names=pred_cols)
print(f"🔍 Prédictions : {len(pred)} détections trackées sur {pred['frame'].nunique()} frames")

# calcul métrique
acc = mm.MOTAccumulator(auto_id=True)

frames = sorted(gt["frame"].unique())

for frame_id in frames:
    gt_frame   = gt[gt["frame"] == frame_id]
    pred_frame = pred[pred["frame"] == frame_id]

    # IDs dans cette frame
    gt_ids   = gt_frame["id"].tolist()
    pred_ids = pred_frame["id"].tolist()

    # Boîtes au format [x, y, w, h]
    gt_boxes   = gt_frame[["x","y","w","h"]].values.tolist()
    pred_boxes = pred_frame[["x","y","w","h"]].values.tolist()

    # Matrice de distances IoU entre toutes les paires GT / Pred
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        distances = np.empty((len(gt_ids), len(pred_ids)))
        distances[:] = np.nan
    else:
        distances = mm.distances.iou_matrix(
            gt_boxes, pred_boxes, max_iou=0.5
        )

    acc.update(gt_ids, pred_ids, distances)

# affichage résultat
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=[
    "num_frames",
    "mota",       # Multiple Object Tracking Accuracy  (plus haut = mieux)
    "motp",       # Multiple Object Tracking Precision (plus bas = mieux)
    "num_switches",   # ID switches (moins = mieux)
    "num_misses",     # faux négatifs (personnes ratées)
    "num_false_positives",  # faux positifs
    "mostly_tracked",       # % objets trackés >80% du temps
    "mostly_lost",          # % objets perdus >80% du temps
], name="ByteTrack")

strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print("\n" + "="*60)
print("RÉSULTATS DU TRACKING")
print("="*60)
print(strsummary)
print("="*60)

# Interprétation rapide
mota = summary["mota"].values[0]
switches = summary["num_switches"].values[0]

print(f"\n MOTA = {mota:.1%}")
if mota > 0.6:
    print("   Bon résultat (>60% est correct pour MOT17)")
elif mota > 0.4:
    print("   Résultat moyen (essaie d'augmenter la conf ou l'IoU)")
else:
    print("   Résultat faible (vérifie les paramètres du modèle P2)")

print(f"\nID switches = {switches}")
print("   (chaque switch = une personne qui a changé d'ID, l'idéal est 0)")

# Sauvegarde des résultats dans un CSV pour P5
summary.to_csv("./outputs/metrics_bytetrack.csv")
print("\nMétriques sauvegardées dans ./outputs/metrics_bytetrack.csv")