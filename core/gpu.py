"""Whether the GPU can run inference — MPS on Apple Silicon.

PyTorch's MPS backend uses Metal Performance Shaders on M-series chips.
Every M1 and later supports it. Some operators fall back to CPU silently,
and ultralytics' predict() can hit unsupported ops on older PyTorch builds.
The check here is simple: if MPS is built and available, use it.
"""

_DEVICE: str | None = None


def mps_usable() -> tuple[bool, str]:
    """`(usable, reason)` — whether Apple Silicon's MPS backend works."""
    try:
        import torch
    except Exception as e:
        return False, f"PyTorch did not load ({type(e).__name__})"

    try:
        if not hasattr(torch.backends, "mps"):
            return False, "MPS not available"
        if not torch.backends.mps.is_built():
            return False, "PyTorch was built without MPS support"
        if not torch.backends.mps.is_available():
            return False, "MPS not available (not Apple Silicon)"
        return True, "Apple Silicon (Metal)"
    except Exception as e:
        return False, f"could not check MPS ({type(e).__name__})"


def torch_device() -> str:
    """`"mps"` or `"cpu"` — the device every model and inference call must use.

    Passing this explicitly at every call site removes the dependency on
    ultralytics' auto-select, which would silently pick a device this build
    may not support.

    Cached: the check imports torch and queries the hardware, and this is
    called per frame. The answer cannot change inside one process.
    """
    global _DEVICE
    if _DEVICE is None:
        usable, _ = mps_usable()
        _DEVICE = "mps" if usable else "cpu"
    return _DEVICE
