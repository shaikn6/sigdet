"""Held-out accuracy plus per-image latency (MPS and CPU) for each trained model."""
import json
import pathlib
import time
import numpy as np
from ultralytics import YOLO

RUNS = pathlib.Path(__file__).resolve().parent / "runs"

imgs = sorted(str(p) for p in (pathlib.Path.home() / "datasets/signature/images/val").glob("*"))
out = {}
for size in ["n", "s"]:
    model = YOLO(str(RUNS / size / "weights" / "best.pt"))
    m = model.val(data="signature.yaml", device="mps", verbose=False, plots=False)
    row = dict(params_m=round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2),
               map50=round(float(m.box.map50), 4), map50_95=round(float(m.box.map), 4),
               precision=round(float(m.box.mp), 4), recall=round(float(m.box.mr), 4))
    for dev in ["mps", "cpu"]:
        model.predict(imgs[0], device=dev, verbose=False)
        ts = []
        for p in imgs:
            t = time.perf_counter(); model.predict(p, device=dev, verbose=False, imgsz=640); ts.append((time.perf_counter() - t) * 1e3)
        row[f"latency_ms_{dev}_p50"] = round(float(np.percentile(ts, 50)), 1)
        row[f"latency_ms_{dev}_p95"] = round(float(np.percentile(ts, 95)), 1)
    out[size] = row
    print(size, row, flush=True)
json.dump(out, open("bench.json", "w"), indent=2)
