"""Shared helpers: pick a torch device and find trained weights."""
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def pick_device(device="auto"):
    """'auto' -> CUDA, then Apple MPS, then CPU. Any other value is passed to ultralytics unchanged."""
    if device != "auto":
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def weights_for(size):
    """Fresh training output in runs/ if present, else the committed copy in weights/ (works from a clean clone)."""
    run = HERE / "runs" / size / "weights" / "best.pt"
    return run if run.exists() else HERE / "weights" / f"signature_yolo11{size}.pt"
