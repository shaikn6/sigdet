import glob, pathlib
import pytest
from detect import detect

VAL = sorted(glob.glob(str(pathlib.Path.home() / "datasets/signature/images/val/*")))
LABELS = pathlib.Path.home() / "datasets/signature/labels/val"


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@pytest.mark.skipif(not VAL, reason="signature dataset not downloaded")
def test_detects_every_labelled_signature_on_a_val_image():
    path = VAL[0]
    boxes, r = detect(path)
    h, w = r.orig_shape
    gt = []
    for line in (LABELS / (pathlib.Path(path).stem + ".txt")).read_text().split("\n"):
        if line.strip():
            _, cx, cy, bw, bh = map(float, line.split())
            gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    assert gt and len(boxes) >= len(gt)
    assert all(max(iou(g, b["xyxy"]) for b in boxes) > 0.5 for g in gt)


def test_confidence_threshold_is_respected():
    if not VAL:
        pytest.skip("dataset missing")
    boxes, _ = detect(VAL[0], conf=0.9)
    assert all(b["conf"] >= 0.9 for b in boxes)
