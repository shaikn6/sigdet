"""Fine-tune YOLO11 on the leakage-aware signature split (see make_split.py); val picks the checkpoint, test is scored once.

    python train.py n s [--device auto|cpu|mps|cuda:0]
"""
import argparse
import json
import pathlib
from ultralytics import YOLO
from device import pick_device

RUNS = pathlib.Path(__file__).resolve().parent / "runs"
DATA = str(pathlib.Path.home() / "datasets/signature_split/signature_split.yaml")  # built by make_split.py

ap = argparse.ArgumentParser()
ap.add_argument("sizes", nargs="+", choices=["n", "s"])
ap.add_argument("--device", default="auto")
a = ap.parse_args()
device = pick_device(a.device)

for size in a.sizes:
    model = YOLO(f"yolo11{size}.pt")
    # 100 epochs at batch 8 won a 4-recipe x 3-seed grid on validation mAP50-95 for both sizes (results/grid.json,
    # scripts/modal_grid.py). The earlier 60 epochs at batch 16 is only ~420 steps and ended unstable for `s`.
    model.train(data=DATA, epochs=100, imgsz=640, batch=8, device=device, seed=0, deterministic=True,
                project=str(RUNS), name=size, exist_ok=True, plots=True, verbose=False)
    best = YOLO(str(RUNS / size / "weights" / "best.pt"))
    metrics = {}
    for split in ("val", "test"):
        m = best.val(data=DATA, split=split, device=device, plots=True)
        metrics[split] = dict(map50=m.box.map50, map50_95=m.box.map, precision=m.box.mp, recall=m.box.mr, speed_ms=m.speed)
    metrics["model"] = f"yolo11{size}"
    metrics["device"] = device
    json.dump(metrics, open(f"metrics_{size}.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))
