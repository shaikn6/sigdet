"""Detect signatures in document images: python detect.py scan.jpg [--out annotated.jpg] [--conf 0.25]"""
import argparse
import json
import pathlib

from ultralytics import YOLO

WEIGHTS = str(pathlib.Path(__file__).resolve().parent / "weights" / "signature_yolo11n.pt")


def detect(path, conf=0.25, weights=WEIGHTS):
    r = YOLO(weights).predict(path, conf=conf, verbose=False)[0]
    boxes = [dict(xyxy=[round(v, 1) for v in b.xyxy[0].tolist()], conf=round(float(b.conf), 3)) for b in r.boxes]
    return boxes, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("--out"); ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    boxes, r = detect(a.image, a.conf)
    print(json.dumps(boxes, indent=2))
    if a.out:
        r.save(filename=a.out)
