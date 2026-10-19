import glob, pathlib
import pytest
from detect import WEIGHTS, detect

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


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_smoke_detects_signature_on_committed_sample():
    # Needs only committed files (sample.jpg + the n weights), so it also runs on CI.
    boxes, r = detect(str(ROOT / "sample.jpg"), weights=WEIGHTS)
    h, w = r.orig_shape
    assert boxes and all(0 < b["conf"] <= 1 for b in boxes)
    # Boxes must be well-formed and inside the image, not just present.
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_confidence_threshold_is_respected_on_committed_sample():
    # Same contract as the dataset-backed test above, but runnable on CI.
    sample = str(ROOT / "sample.jpg")
    default, _ = detect(sample)
    strict, _ = detect(sample, conf=0.9)
    assert default and all(b["conf"] >= 0.25 for b in default)
    assert all(b["conf"] >= 0.9 for b in strict)
    assert len(strict) <= len(default)
    assert detect(sample, conf=1.0)[0] == []


def test_missing_image_raises_clear_error():
    with pytest.raises(FileNotFoundError, match="input image not found"):
        detect(str(ROOT / "does_not_exist.jpg"))
