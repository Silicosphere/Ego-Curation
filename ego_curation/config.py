from dataclasses import dataclass
from typing import Optional, Literal

VJEPA_MODELS = {
    "base":     "apiantonio/vjepa2.1-vit-base-384",
    "large":    "apiantonio/vjepa2.1-vit-large-384",
    "giant":    "apiantonio/vjepa2.1-vit-giant-384",
    "gigantic": "apiantonio/vjepa2.1-vit-gigantic-384",
}

DEFAULT_MODEL_SIZE = "large"


@dataclass
class SurpriseConfig:
    context_frames: int = 32
    target_frames: int = 16
    context_duration: float = 2.0
    target_duration: float = 1.0
    stride_duration: Optional[float] = None
    agg: Literal["mean", "max"] = "mean"
    return_curve: bool = False
