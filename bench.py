"""Val and test accuracy plus per-image latency (MPS and CPU) for each trained model. Test is never used to pick a checkpoint."""
import json
import pathlib
import time
import numpy as np
from ultralytics import YOLO

RUNS = pathlib.Path(__file__).resolve().parent / "runs"
SPLIT = pathlib.Path.home() / "datasets/signature_split"  # built by make_split.py

imgs = sorted(str(p) for p in (SPLIT / "images/test").glob("*"))
out = {}
for size in ["n", "s"]:
    model = YOLO(str(RUNS / size / "weights" / "best.pt"))
    row = dict(params_m=round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2))
    for split in ["val", "test"]:
        m = model.val(data=str(SPLIT / "signature_split.yaml"), split=split, device="mps", verbose=False, plots=False)
        row[split] = dict(map50=round(float(m.box.map50), 4), map50_95=round(float(m.box.map), 4),
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
