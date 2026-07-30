import numpy as np
from torchcodec.decoders import VideoDecoder


def get_fps(vr: VideoDecoder) -> float:
    """Extract frames-per-second from a VideoDecoder's metadata."""
    try:
        md = vr.metadata
        fps = md.frame_rate
        if isinstance(fps, (int, float)):
            return float(fps)
        return float(fps.numerator) / float(fps.denominator)
    except Exception:
        pass
    return 30.0


def sample_indices(start: int, end: int, n: int) -> np.ndarray:
    """Uniformly sample *n* frame indices from [start, end).

    Handles edge cases where the span has fewer than *n* frames
    (pads by repeating the last available frame).
    """
    available = end - start
    if available <= 0:
        return np.array([max(0, start)] * n)
    if available <= n:
        idx = np.arange(start, end)
        pad = np.full(n - available, end - 1)
        return np.concatenate([idx, pad])
    return np.linspace(start, end - 1, n, dtype=int)
