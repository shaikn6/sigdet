# sigdet

Document **signature detection** with YOLO11, fine-tuned on the public Ultralytics signature dataset
with a **leakage-aware train/val/test split** (110 train / 26 val / 26 test; `make_split.py`), plus a
**robustness study**: how detection quality degrades under realistic scan and photo corruptions.
Relevant to KYC, cheque, and contract-processing pipelines.

![sample](sample.jpg)

## Results

The 178 images are every-2nd-frame grabs from two short videos, so adjacent frames are near-duplicates.
`make_split.py` cuts each clip into contiguous chunks and holds out whole chunks for val and test (dropping
the train frames next to them), so no near-copy of a test image sits in train. **Val selects the checkpoint;
the test split is scored once and never used for selection** (`splits.json`, `metrics_n.json`, `metrics_s.json`).

| Model | Params | Split | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|
| **YOLO11n** | 2.6M | val | 0.995 | 0.973 | 0.998 | 1.000 |
| **YOLO11n** | 2.6M | **test** | 0.995 | **0.904** | 0.998 | 1.000 |
| YOLO11s | 9.4M | val | 0.995 | 0.971 | 0.998 | 1.000 |
| YOLO11s | 9.4M | **test** | 0.995 | **0.905** | 0.997 | 1.000 |

mAP50 is saturated (0.995) on both splits; the honest generalization signal is mAP50-95, which drops ~7 points
from val to the held-out test (0.97 → 0.90) — the estimate the earlier val-only setup could not give. The two
model sizes are within noise of each other on both splits, so `n` is the shipped model
(`weights/signature_yolo11n.pt`, 60 epochs, 640px, seed 0) at ~1.7x lower CPU latency (15.0 ms vs 19.6 ms MPS,
34.4 ms vs 57.1 ms CPU; `bench.py`).

Only the `n` weights are committed; `s` is retrained via `python train.py s`. The split manifest and both
metrics files are committed, so the table reproduces from `python make_split.py && python train.py n s`.

## Robustness

Each corruption is applied to the 26 held-out **test** images (labels unchanged) and re-scored.

![robustness](robustness.png)

- Detection is robust: mAP50 stays 0.995 under every corruption (one exception: `n` at 0.25x brightness,
  0.991). The cost shows up in box tightness (mAP50-95), not in finding the signature.
- **Severe under-exposure is the worst case:** at 0.25x brightness `n` mAP50-95 falls from 0.898 (clean) to
  **0.585** (`s` to 0.710). Phone photos of documents in poor light are the main risk.
- Blur, 2–4x downsampling, and JPEG down to q=30 cost ~1 point of mAP50-95; q=10 is harder for `s` (0.724)
  than `n` (0.864).
- Heavy sensor noise (sigma=40) hurts box tightness most: `n` mAP50-95 drops to 0.76, and the larger model is
  *less* robust here (`s` 0.523), so extra capacity did not buy robustness on this dataset.

Full numbers: `robustness.json` (test split), `bench.json`; training curves: `runs/`.

## Caveats

The dataset is small (178 images from two videos): 26 test images give a usable but noisy estimate, and mAP50
is saturated, so read mAP50-95 and the robustness table rather than mAP50. The `clean` mAP50-95 in
`robustness.json` (`n` 0.898) is a few thousandths off the test row in `metrics_n.json` (0.904) because that
script re-scores re-encoded copies in a separate run; both were measured. Corruptions are synthetic
approximations of scan artefacts, rotation is excluded (it would need box re-derivation), and MPS training is
not bit-for-bit deterministic, so reruns can move mAP50-95 by a few thousandths.

## Use

```bash
pip install -r requirements.txt
python detect.py scan.jpg --out annotated.jpg        # prints JSON boxes
python make_split.py                                   # build the leakage-aware train/val/test split
python train.py n                                      # retrain (needs the dataset; ultralytics downloads it)
python bench.py && python robustness.py n s            # regenerate bench.json / robustness.json (test split)
python plot_robustness.py                              # redraw robustness.png from robustness.json
python -m pytest tests
```

CI runs lint, a compile check, and the tests. Two tests need the dataset under `~/datasets/signature` and are
skipped on CI; the smoke test (`detect()` on the committed `sample.jpg` with the committed `n` weights) and the
missing-file test run everywhere.

## License

AGPL-3.0. The dataset and the `ultralytics` library are AGPL-3.0, and this repo is licensed to match.
