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


def test_split_manifest_has_no_shared_or_adjacent_frames():
    # Runs on CI: splits.json alone. Also proves the check fails on a leaked manifest.
    import copy, json
    from make_split import check_manifest
    manifest = json.loads((ROOT / "splits.json").read_text())
    check_manifest(manifest)
    leaked = copy.deepcopy(manifest)
    leaked["train"].append(manifest["test"][0])
    with pytest.raises(AssertionError):
        check_manifest(leaked)


@pytest.mark.skipif(not (pathlib.Path.home() / "datasets/signature_split/images/test").is_dir(), reason="split not built")
def test_split_dirs_share_no_image_bytes():
    import json
    from make_split import check_contents
    check_contents(json.loads((ROOT / "splits.json").read_text()))


def test_committed_weights_were_trained_on_the_leakage_aware_split():
    # The weights shipped before 2026-10-28 were trained on the stock split, which contains 22 of the 26 test images.
    from ultralytics import YOLO
    for w in sorted((ROOT / "weights").glob("*.pt")):
        args = YOLO(str(w)).ckpt["train_args"]
        assert args["data"].endswith("signature_split.yaml"), w.name
        # The documented recipe (see train.py): 100 epochs, batch 8, 640 px, seed 0.
        assert (args["epochs"], args["batch"], args["imgsz"], args["seed"]) == (100, 8, 640, 0), w.name


# --- Corruption transforms (robustness.py): synthetic, no model or dataset needed. ---

def _textured_image():
    import numpy as np
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (32, 48, 3), dtype="uint8")


def test_every_corruption_keeps_shape_and_dtype():
    # Labels are reused unchanged after corruption, so each transform must preserve image geometry and dtype.
    import robustness
    im = _textured_image()
    for name, fn in robustness.CORRUPTIONS.items():
        out = fn(im)
        assert out.shape == im.shape and out.dtype == im.dtype, name


def test_corruption_keys_match_the_committed_report():
    # The plot and README table index corruptions by these exact keys.
    import json
    import robustness
    report = json.loads((ROOT / "robustness.json").read_text())
    assert list(robustness.CORRUPTIONS) == list(report["n"]) == list(report["s"])


def test_clean_is_the_identity():
    import numpy as np
    import robustness
    im = _textured_image()
    assert np.array_equal(robustness.CORRUPTIONS["clean"](im), im)


def test_dim_darkens_by_the_stated_factor():
    import numpy as np
    import robustness
    im = np.full((8, 8, 3), 200, dtype="uint8")
    assert np.array_equal(robustness.dark(im, 0.5), np.full_like(im, 100))
    assert robustness.dark(im, 0.25).max() < robustness.dark(im, 0.5).max()


def test_noise_and_jpeg_stay_in_range_and_change_pixels():
    import numpy as np
    import robustness
    im = _textured_image()
    noisy = robustness.noise(im, 40)
    assert noisy.min() >= 0 and noisy.max() <= 255 and not np.array_equal(noisy, im)
    assert not np.array_equal(robustness.jpeg(im, 10), im)  # heavy compression must alter the texture


# --- Metric aggregation (bench.py): synthetic per-image match stats, no model needed. ---

def _stat(hit):
    import numpy as np
    return dict(tp=np.full((1, 10), hit, bool), conf=np.array([0.9 if hit else 0.3]),
                pred_cls=np.array([0.0]), target_cls=np.array([0.0]))


def test_map50_95_aggregates_across_images():
    import bench
    assert bench.map50_95([_stat(True)] * 5) > 0.9       # all detections correct
    assert bench.map50_95([_stat(False)] * 5) == 0.0     # none correct
    mixed = bench.map50_95([_stat(True)] * 7 + [_stat(False)] * 3)
    assert 0.0 < mixed < 0.9


def test_bootstrap_ci_is_ordered_bounded_and_deterministic():
    import bench
    mixed = [_stat(True)] * 7 + [_stat(False)] * 3
    lo, hi = bench.bootstrap_ci(mixed, n=300)
    assert 0.0 <= lo <= hi <= 1.0
    assert bench.bootstrap_ci(mixed, n=300) == [lo, hi]  # seeded, so reruns match


# --- Leakage-aware split construction (make_split.py): synthetic frame names, no dataset bytes. ---

def test_make_split_reproduces_the_committed_manifest(tmp_path, monkeypatch):
    import json
    import make_split
    manifest = json.loads((ROOT / "splits.json").read_text())
    roles = ("train", "val", "test", "dropped")
    (tmp_path / "images/train").mkdir(parents=True)
    for role in roles:
        for name in manifest[role]:
            (tmp_path / "images/train" / name).touch()  # make_split() only globs and parses names
    monkeypatch.setattr(make_split, "SRC", tmp_path)
    _, split = make_split.make_split()
    by_role = {r: sorted(n for n, v in split.items() if v == r) for r in roles}
    assert by_role == {r: manifest[r] for r in roles}


def test_check_manifest_rejects_a_val_chunk_adjacent_to_test():
    import copy, json
    from make_split import check_manifest
    manifest = json.loads((ROOT / "splits.json").read_text())
    bad = copy.deepcopy(manifest)
    bad["val"].append(manifest["test"][0])  # a val frame now sits on top of a test frame
    with pytest.raises(AssertionError):
        check_manifest(bad)


def test_check_contents_flags_identical_bytes_in_two_splits(tmp_path):
    from make_split import check_contents
    manifest = {"train": ["Frame_2.jpg"], "val": ["Frame_100.jpg"], "test": ["Frame_200.jpg"]}
    for role, names in manifest.items():
        (tmp_path / "images" / role).mkdir(parents=True)
        for name in names:
            (tmp_path / "images" / role / name).write_bytes(b"same-bytes")  # train and test collide
    with pytest.raises(AssertionError, match="same bytes"):
        check_contents(manifest, dst=tmp_path)
