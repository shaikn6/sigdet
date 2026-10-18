"""Build a leakage-aware train/val/test split of the Ultralytics signature dataset.

The 178 images are every-2nd-frame grabs from two short videos (frames 46-196 and 240-442), so adjacent
frames are near-duplicates. A random per-image split would put near-copies of test images in train. Instead
each clip is cut into contiguous chunks of CHUNK frames; one chunk per clip goes to test, one to val, the rest
to train, and the BUFFER train frames next to every held-out chunk are dropped.

    python make_split.py            # writes ~/datasets/signature_split and splits.json
"""
import json
import pathlib
import random
import re
import shutil

SEED, CHUNK, BUFFER = 0, 13, 2
SRC = pathlib.Path.home() / "datasets/signature"
DST = pathlib.Path.home() / "datasets/signature_split"
HERE = pathlib.Path(__file__).resolve().parent


def frame_no(p):
    return int(re.search(r"Frame_(\d+)", p.name).group(1))


def make_split():
    pool = sorted((p for s in ("train", "val") for p in (SRC / "images" / s).glob("*.jpg")), key=frame_no)
    assert len(pool) == 178, len(pool)
    # A jump of more than one 2-frame step marks a clip boundary.
    clips, cur = [], [pool[0]]
    for p in pool[1:]:
        if frame_no(p) - frame_no(cur[-1]) > 2:
            clips.append(cur)
            cur = []
        cur.append(p)
    clips.append(cur)

    rng = random.Random(SEED)
    split = {}
    for clip in clips:
        chunks = [clip[i:i + CHUNK] for i in range(0, len(clip), CHUNK)]
        order = list(range(len(chunks)))
        rng.shuffle(order)
        test_i = order[0]
        val_i = next(i for i in order[1:] if abs(i - test_i) > 1)  # never adjacent to test: no val/test leakage
        for ci, chunk in enumerate(chunks):
            if ci == test_i:
                role = "test"
            elif ci == val_i:
                role = "val"
            else:
                role = "train"
            for p in chunk:
                split[p.name] = role
        for ci in (test_i, val_i):  # buffer: drop train frames adjacent to a held-out chunk
            for nb, idx in ((ci - 1, -BUFFER), (ci + 1, 0)):
                if 0 <= nb < len(chunks) and split[chunks[nb][0].name] == "train":
                    for p in (chunks[nb][idx:] if idx else chunks[nb][:BUFFER]):
                        split[p.name] = "dropped"
    return pool, split


def main():
    pool, split = make_split()
    by_role = {r: sorted(n for n, v in split.items() if v == r) for r in ("train", "val", "test", "dropped")}
    # Overlap check: no image in more than one role, and nothing lost or invented.
    all_named = [n for r in by_role.values() for n in r]
    assert len(all_named) == len(set(all_named)) == len(pool)
    assert not set(by_role["test"]) & set(by_role["train"]) and not set(by_role["val"]) & set(by_role["train"])

    if DST.exists():
        shutil.rmtree(DST)
    for role in ("train", "val", "test"):
        (DST / "images" / role).mkdir(parents=True)
        (DST / "labels" / role).mkdir(parents=True)
        for name in by_role[role]:
            src = next(p for p in pool if p.name == name)
            shutil.copy(src, DST / "images" / role / name)
            lab = src.parent.parent.parent / "labels" / src.parent.name / (src.stem + ".txt")
            shutil.copy(lab, DST / "labels" / role / lab.name)
    (DST / "signature_split.yaml").write_text(
        f"path: {DST}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: signature\n")
    manifest = dict(seed=SEED, chunk=CHUNK, buffer=BUFFER, counts={r: len(v) for r, v in by_role.items()}, **by_role)
    (HERE / "splits.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(manifest["counts"], "overlap check passed")


if __name__ == "__main__":
    main()
