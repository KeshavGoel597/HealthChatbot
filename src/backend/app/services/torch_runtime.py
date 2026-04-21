"""Torch runtime selection helpers (CUDA, MPS, CPU)."""

from __future__ import annotations

import torch


def detect_torch_runtime() -> tuple[str, torch.device, torch.dtype, bool]:
    """Return (backend_name, device, dtype, use_device_map_auto).

    Backends:
    - CUDA: use device_map='auto' for larger models.
    - MPS: use a single-device placement on Apple Silicon.
    - CPU: float32 fallback.
    """
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda"), torch.float16, True

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps", torch.device("mps"), torch.float16, False

    return "cpu", torch.device("cpu"), torch.float32, False
