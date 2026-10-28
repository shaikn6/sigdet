"""Val and test accuracy plus per-image latency (chosen device and CPU) for each model. Test is never used to pick a checkpoint.

    python bench.py [--device auto|cpu|mps|cuda:0]
"""
import argparse
import json
import pathlib
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, ap_per_class
from ultralytics.utils.torch_utils import get_cpu_info
from device import HERE, pick_device, weights_for

SPLIT = pathlib.Path.home() / "datasets/signature_split"  # built by make_split.py

# Keep each image's match stats from model.val() so test mAP50-95 can be bootstrapped over images.
per_image = []
_update_stats = DetMetrics.update_stats


def _record(self, stat):
    per_image.append(stat)
    _update_stats(self, stat)


DetMetrics.update_stats = _record


def map50_95(stats):
    cat = {k: np.concatenate([s[k] for s in stats]) for k in ("tp", "conf", "pred_cls", "target_cls")}
    return ap_per_class(cat["tp"], cat["conf"], cat["pred_cls"], cat["target_cls"])[5].mean()


def bootstrap_ci(stats, n=2000):
    """95% percentile interval of mAP50-95 with images resampled with replacement (seed 0)."""
    rng = np.random.default_rng(0)
    vals = [map50_95([stats[i] for i in rng.integers(0, len(stats), len(stats))]) for _ in range(n)]
    return [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]


ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
device = pick_device(ap.parse_args().device)

imgs = sorted(str(p) for p in (SPLIT / "images/test").glob("*"))
out = dict(cpu=get_cpu_info(), device=device)
for size in ["n", "s"]:
    weights = weights_for(size)
    model = YOLO(str(weights))
    row = dict(weights=str(weights.relative_to(HERE)), params_m=round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2))
    for split in ["val", "test"]:
        per_image.clear()
        m = model.val(data=str(SPLIT / "signature_split.yaml"), split=split, device=device, verbose=False, plots=False)
        row[split] = dict(map50=round(float(m.box.map50), 4), map50_95=round(float(m.box.map), 4),
                          precision=round(float(m.box.mp), 4), recall=round(float(m.box.mr), 4),
                          images=len(per_image), map50_95_ci95=bootstrap_ci(per_image))
    for dev in dict.fromkeys([device, "cpu"]):
        model.predict(imgs[0], device=dev, verbose=False)
        ts = []
        for p in imgs:
            t = time.perf_counter(); model.predict(p, device=dev, verbose=False, imgsz=640); ts.append((time.perf_counter() - t) * 1e3)
        name = dev.split(":")[0]
        row[f"latency_ms_{name}_p50"] = round(float(np.percentile(ts, 50)), 1)
        row[f"latency_ms_{name}_p95"] = round(float(np.percentile(ts, 95)), 1)
    out[size] = row
    print(size, row, flush=True)
json.dump(out, open("bench.json", "w"), indent=2)
