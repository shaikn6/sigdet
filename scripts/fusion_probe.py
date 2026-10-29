"""Validation-only probe of cheap inference-time levers: test-time augmentation (TTA) and weighted boxes fusion (WBF) of n and s.

Scores the VAL split only (the test split is never read), so it can be used to choose a lever without touching test.
Writes results/fusion_probe.json.

    python scripts/fusion_probe.py [--device auto|cpu|mps|cuda:0]
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from device import HERE, pick_device, weights_for  # noqa: E402

SPLIT = pathlib.Path.home() / "datasets/signature_split"  # built by make_split.py
split = "val"
IOUV = np.linspace(0.5, 0.95, 10)

def gt_boxes(img):
    h, w = img_hw[img]
    lab = SPLIT / "labels" / split / (pathlib.Path(img).stem + ".txt")
    a = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
    c, x, y, bw, bh = a.T if len(a) else [np.zeros(0)] * 5
    return np.stack([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h], 1) if len(a) else np.zeros((0, 4))

def tp_matrix(pred, gt):
    """Same greedy matching as ultralytics DetectionValidator.match_predictions (single class)."""
    tp = np.zeros((len(pred), len(IOUV)), bool)
    if not len(pred) or not len(gt):
        return tp
    iou = box_iou(torch.tensor(gt, dtype=torch.float32), torch.tensor(pred, dtype=torch.float32)).numpy()
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1].astype(int), i] = True
    return tp

def wbf(lists, weights, thr=0.55):
    """Weighted boxes fusion (Solovyev et al.) for one class. lists: [(boxes Nx4, scores N)] per model."""
    items = [(s * w, b, w) for (bs, ss), w in zip(lists, weights) for b, s in zip(bs, ss)]
    items.sort(key=lambda t: -t[0])
    clusters = []
    for s, b, w in items:
        best, bi = thr, -1
        for k, c in enumerate(clusters):
            fb = c["box"]
            ix = max(0, min(b[2], fb[2]) - max(b[0], fb[0])) * max(0, min(b[3], fb[3]) - max(b[1], fb[1]))
            u = (b[2] - b[0]) * (b[3] - b[1]) + (fb[2] - fb[0]) * (fb[3] - fb[1]) - ix
            if u > 0 and ix / u > best:
                best, bi = ix / u, k
        if bi < 0:
            clusters.append(dict(boxes=[b], scores=[s], w=[w], box=b))
        else:
            c = clusters[bi]; c["boxes"].append(b); c["scores"].append(s); c["w"].append(w)
            sc = np.array(c["scores"]); c["box"] = (np.array(c["boxes"]) * sc[:, None]).sum(0) / sc.sum()
    tw = sum(weights)
    out_b = np.array([c["box"] for c in clusters]).reshape(-1, 4)
    out_s = np.array([sum(c["scores"]) / tw for c in clusters])  # = mean score scaled by model coverage
    return out_b, out_s

def stats_for(preds_per_img):
    out = []
    for img, (b, s) in zip(imgs, preds_per_img):
        gt = gt_boxes(img)
        out.append(dict(tp=tp_matrix(b, gt), conf=s, pred_cls=np.zeros(len(s)), target_cls=np.zeros(len(gt))))
    return out

def map50_95(stats):
    cat = {k: np.concatenate([s[k] for s in stats]) for k in ("tp", "conf", "pred_cls", "target_cls")}
    return float(ap_per_class(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[5].mean())

imgs = sorted(str(p) for p in (SPLIT / "images" / split).glob("*"))
img_hw = {}
raw = {}
ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
device = pick_device(ap.parse_args().device)
official = {}
for size in "ns":
    m = YOLO(str(weights_for(size)))
    for aug in (False, True):
        # The repo's own evaluator (as in bench.py), with and without ultralytics' built-in TTA.
        r = m.val(data=str(SPLIT / "signature_split.yaml"), split=split, device=device, verbose=False, plots=False, augment=aug)
        official[f"{size}{'+tta' if aug else ''}"] = round(float(r.box.map), 4)
        res = m.predict(imgs, conf=0.001, iou=0.7, device=device, verbose=False, augment=aug, imgsz=640, max_det=300)
        raw[(size, aug)] = [(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()) for r in res]
        for img, r in zip(imgs, res):
            img_hw[img] = r.orig_shape
results = {}
for k, v in raw.items():
    results[f"{k[0]}{'+tta' if k[1] else ''}"] = round(map50_95(stats_for(v)), 4)
combos = {"wbf n+s": [("n", False), ("s", False)], "wbf n+n_tta": [("n", False), ("n", True)],
          "wbf n+s+n_tta+s_tta": [("n", False), ("s", False), ("n", True), ("s", True)]}
for name, members in combos.items():
    for wn in (1, 2, 3):
        weights = [wn if k[0] == "n" else 1 for k in members]
        for thr in (0.55, 0.7):
            fused = [wbf([raw[k][i] for k in members], weights, thr) for i in range(len(imgs))]
            results[f"{name} w_n={wn} thr={thr}"] = round(map50_95(stats_for(fused)), 4)
out = {"split": split, "metric": "mAP50-95", "model_val": official, "predict_pipeline": results}
(HERE / "results" / "fusion_probe.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
