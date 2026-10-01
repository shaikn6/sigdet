"""Fine-tune YOLO11 on the public Ultralytics signature-detection dataset and log held-out metrics."""
import json
import pathlib
import sys
from ultralytics import YOLO

RUNS = pathlib.Path(__file__).resolve().parent / "runs"

size = sys.argv[1]
model = YOLO(f"yolo11{size}.pt")
model.train(data="signature.yaml", epochs=60, imgsz=640, batch=16, device="mps", seed=0,
            project=str(RUNS), name=size, exist_ok=True, plots=True, verbose=False)
m = YOLO(str(RUNS / size / "weights" / "best.pt")).val(data="signature.yaml", split="val", device="mps", plots=True)
metrics = dict(model=f"yolo11{size}", map50=m.box.map50, map50_95=m.box.map, precision=m.box.mp, recall=m.box.mr, speed_ms=m.speed)
json.dump(metrics, open(f"metrics_{size}.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))
