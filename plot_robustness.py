"""Render robustness.json (test-split mAP50-95 per corruption) to robustness.png."""
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
data = json.load(open(HERE / "robustness.json"))
labels = list(data["n"].keys())
x = np.arange(len(labels))
w = 0.4

fig, ax = plt.subplots(figsize=(13, 4.5))
ax.bar(x - w / 2, [data["n"][k]["map50_95"] for k in labels], w, label="YOLO11n (2.6M)")
ax.bar(x + w / 2, [data["s"][k]["map50_95"] for k in labels], w, label="YOLO11s (9.4M)")
ax.set_ylabel("mAP50-95")
ax.set_ylim(0.5, 1.0)
ax.set_title("Signature detection under scan corruptions (26 held-out test images)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.legend()
fig.tight_layout()
fig.savefig(HERE / "robustness.png", dpi=110)
