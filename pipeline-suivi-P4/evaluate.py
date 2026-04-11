import pandas as pd
import motmetrics as mm
import numpy as np
from pathlib import Path

# config
SEQ_NAME  = "MOT17-02-FRCNN"
MOT17_DIR = f"./MOT17/train/{SEQ_NAME}"
GT_FILE   = f"{MOT17_DIR}/gt/gt.txt"
PRED_FILE = f"./outputs/MOT17-02.txt"

# lecture ground truth MOT17
gt_cols = ["frame","id","x","y","w","h","conf","class","visibility"]
gt = pd.read_csv(GT_FILE, header=None, names=gt_cols)
gt = gt[(gt["class"] == 1) & (gt["visibility"] > 0.25)].copy()
print(f"📊 Ground truth : {len(gt)} annotations sur {gt['frame'].nunique()} frames")

# lecture prédictions
pred_cols = ["frame","id","x","y","w","h","conf","x3d","y3d","z3d"]
pred = pd.read_csv(PRED_FILE, header=None, names=pred_cols)
print(f"🔍 Prédictions : {len(pred)} détections trackées sur {pred['frame'].nunique()} frames")

# calcul métriques
acc = mm.MOTAccumulator(auto_id=True)

for frame_id in sorted(gt["frame"].unique()):
    gt_frame   = gt[gt["frame"] == frame_id]
    pred_frame = pred[pred["frame"] == frame_id]

    gt_ids   = gt_frame["id"].tolist()
    pred_ids = pred_frame["id"].tolist()
    gt_boxes   = gt_frame[["x","y","w","h"]].values.tolist()
    pred_boxes = pred_frame[["x","y","w","h"]].values.tolist()

    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        distances = np.empty((len(gt_ids), len(pred_ids)))
        distances[:] = np.nan
    else:
        distances = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=0.5)

    acc.update(gt_ids, pred_ids, distances)

# affichage résultats
mh = mm.metrics.create()
summary = mh.compute(acc, metrics=[
    "num_frames",
    "mota",
    "motp",
    "num_switches",
    "num_misses",
    "num_false_positives",
    "mostly_tracked",
    "mostly_lost",
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

mota    = summary["mota"].values[0]
switches = summary["num_switches"].values[0]

print(f"\n MOTA = {mota:.1%}")
if mota > 0.6:
    print("   Bon résultat (>60% est correct pour MOT17)")
elif mota > 0.4:
    print("   Résultat moyen")
else:
    print("   Résultat faible (vérifie les paramètres du modèle P2)")

print(f"\nID switches = {switches}")
print("   (chaque switch = une personne qui a changé d'ID, l'idéal est 0)")

# sauvegarde CSV pour P5
summary.to_csv("./outputs/metrics_bytetrack.csv")
print("\n💾 Métriques sauvegardées dans ./outputs/metrics_bytetrack.csv")