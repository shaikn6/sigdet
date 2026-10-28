"""Recipe x seed x size grid for sigdet on Modal T4s (one container per run, all in parallel).

Selection rule fixed in advance: per model size, the recipe with the highest MEAN VALIDATION mAP50-95 over seeds
wins; the test split is only scored, never used to choose.

    SIGDET_REPO=<clone> .venv/bin/modal run --detach modal_grid.py
    .venv/bin/modal volume get sigdet-grid / <local dir>
"""
import json
import os
import pathlib
import re
import time

import modal

REPO = pathlib.Path(os.environ.get("SIGDET_REPO", "/unused"))
DATA = pathlib.Path.home() / "datasets" / "signature_split"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("ultralytics==8.4.191", "torch==2.14.1", "numpy", "matplotlib")
    .add_local_file(REPO / "yolo11n.pt", "/root/pretrained/yolo11n.pt")
    .add_local_file(REPO / "yolo11s.pt", "/root/pretrained/yolo11s.pt")
    .add_local_dir(DATA, "/root/datasets/signature_split")
)
vol = modal.Volume.from_name("sigdet-grid", create_if_missing=True)
app = modal.App("sigdet-grid")

RECIPES = {
    "A_60e_b16": dict(epochs=60, batch=16, imgsz=640),
    "B_150e_b16": dict(epochs=150, batch=16, imgsz=640),
    "C_100e_b8": dict(epochs=100, batch=8, imgsz=640),
    "D_100e_b8_800": dict(epochs=100, batch=8, imgsz=800),
}
SEEDS = (0, 1, 2)
SIZES = ("n", "s")
USD_PER_T4_HOUR = 0.59  # Modal list price at the time of writing


@app.function(image=image, gpu="T4", timeout=25 * 60, volumes={"/out": vol})  # worst case 24 x 25 min = ~$5.9
def run(size: str, recipe: str, seed: int) -> dict:
    from ultralytics import YOLO

    yaml = pathlib.Path("/root/datasets/signature_split/signature_split.yaml")
    # The yaml stores the absolute dataset path of the machine that built it; point it at the mounted copy.
    yaml.write_text(re.sub(r"(?m)^path:.*$", "path: /root/datasets/signature_split", yaml.read_text()))
    cfg = RECIPES[recipe]
    name = f"{size}_{recipe}_s{seed}"
    t0 = time.time()
    model = YOLO(f"/root/pretrained/yolo11{size}.pt")
    model.train(data=str(yaml), device="cuda:0", seed=seed, deterministic=True, project="/tmp/runs", name=name,
                exist_ok=True, plots=False, verbose=False, **cfg)
    best_path = pathlib.Path("/tmp/runs") / name / "weights" / "best.pt"
    best = YOLO(str(best_path))
    result = dict(size=size, recipe=recipe, seed=seed, **cfg)
    for split in ("val", "test"):
        m = best.val(data=str(yaml), split=split, device="cuda:0", plots=False, verbose=False)
        result[split] = dict(map50=m.box.map50, map50_95=m.box.map, precision=m.box.mp, recall=m.box.mr)
    result["seconds"] = time.time() - t0
    out = pathlib.Path("/out") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "best.pt").write_bytes(best_path.read_bytes())
    (out / "results.csv").write_text((pathlib.Path("/tmp/runs") / name / "results.csv").read_text())
    vol.commit()
    return result


@app.local_entrypoint()
def main():
    jobs = [(s, r, seed) for s in SIZES for r in RECIPES for seed in SEEDS]
    print(f"{len(jobs)} runs")
    outcomes = list(run.starmap(jobs, return_exceptions=True))
    results = [o for o in outcomes if isinstance(o, dict)]
    print(f"{len(results)} finished, {len(outcomes) - len(results)} failed: "
          f"{[str(o)[:80] for o in outcomes if not isinstance(o, dict)]}")
    secs = sum(r["seconds"] for r in results)
    print(f"total GPU seconds {secs:.0f}  (~${secs / 3600 * USD_PER_T4_HOUR:.2f} at ${USD_PER_T4_HOUR}/h)")
    pathlib.Path("grid_results.json").write_text(json.dumps(results, indent=2))
