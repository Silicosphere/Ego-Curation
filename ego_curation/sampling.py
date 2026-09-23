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


def sample_indices(start: int, n: int, step: float, num_frames: int) -> np.ndarray:
    """Return *n* frame indices starting at *start*, *step* source frames apart.

    *step* is ``source_fps / sample_fps`` and may be fractional; indices are
    rounded to the nearest frame. Indices past the end of the video repeat the
    last frame.
    """
    idx = start + (np.arange(n) * step + 0.5).astype(int)
    return np.minimum(idx, num_frames - 1)
