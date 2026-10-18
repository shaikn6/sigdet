"""Fine-tune YOLO11 on the leakage-aware signature split (see make_split.py); val picks the checkpoint, test is scored once."""
import json
import pathlib
import sys
from ultralytics import YOLO

RUNS = pathlib.Path(__file__).resolve().parent / "runs"
DATA = str(pathlib.Path.home() / "datasets/signature_split/signature_split.yaml")  # built by make_split.py

size = sys.argv[1]
model = YOLO(f"yolo11{size}.pt")
model.train(data=DATA, epochs=60, imgsz=640, batch=16, device="mps", seed=0,
            project=str(RUNS), name=size, exist_ok=True, plots=True, verbose=False)
best = YOLO(str(RUNS / size / "weights" / "best.pt"))
metrics = {}
for split in ("val", "test"):
    m = best.val(data=DATA, split=split, device="mps", plots=True)
    metrics[split] = dict(map50=m.box.map50, map50_95=m.box.map, precision=m.box.mp, recall=m.box.mr, speed_ms=m.speed)
metrics["model"] = f"yolo11{size}"
json.dump(metrics, open(f"metrics_{size}.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
