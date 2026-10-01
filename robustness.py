"""Measure how detector mAP degrades under realistic scan/photo corruptions of the held-out val images."""
import json, shutil, sys, tempfile, pathlib
import cv2, numpy as np
from ultralytics import YOLO

RUNS = pathlib.Path(__file__).resolve().parent / "runs"

SRC = pathlib.Path.home() / "datasets/signature"
rng = np.random.default_rng(0)


def blur(im, k):   return cv2.GaussianBlur(im, (k, k), 0)
def jpeg(im, q):   return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1)
def lowres(im, f): h, w = im.shape[:2]; return cv2.resize(cv2.resize(im, (w // f, h // f)), (w, h))
def noise(im, s):  return np.clip(im + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)
def dark(im, g):   return np.clip(im * g, 0, 255).astype(np.uint8)


# Rotation is omitted on purpose: axis-aligned boxes would need re-derivation and would confound the result.
CORRUPTIONS = {
    "clean": lambda im: im,
    "blur k=5": lambda im: blur(im, 5), "blur k=15": lambda im: blur(im, 15),
    "jpeg q=30": lambda im: jpeg(im, 30), "jpeg q=10": lambda im: jpeg(im, 10),
    "low-res /2": lambda im: lowres(im, 2), "low-res /4": lambda im: lowres(im, 4),
    "noise s=15": lambda im: noise(im, 15), "noise s=40": lambda im: noise(im, 40),
    "dim x0.5": lambda im: dark(im, 0.5), "dim x0.25": lambda im: dark(im, 0.25),
}


def build(tmp, fn):
    (tmp / "images/val").mkdir(parents=True); (tmp / "labels/val").mkdir(parents=True)
    for p in sorted((SRC / "images/val").iterdir()):
        cv2.imwrite(str(tmp / "images/val" / p.name), fn(cv2.imread(str(p))))
        shutil.copy(SRC / "labels/val" / (p.stem + ".txt"), tmp / "labels/val")
    (tmp / "d.yaml").write_text(f"path: {tmp}\ntrain: images/val\nval: images/val\nnames:\n  0: signature\n")
    return tmp / "d.yaml"


out = {}
for size in sys.argv[1:]:
    model = YOLO(str(RUNS / size / "weights" / "best.pt"))
    out[size] = {}
    for name, fn in CORRUPTIONS.items():
        with tempfile.TemporaryDirectory() as t:
            m = model.val(data=str(build(pathlib.Path(t), fn)), device="mps", verbose=False, plots=False)
        out[size][name] = dict(map50=round(float(m.box.map50), 4), map50_95=round(float(m.box.map), 4))
        print(size, name, out[size][name], flush=True)
json.dump(out, open("robustness.json", "w"), indent=2)
