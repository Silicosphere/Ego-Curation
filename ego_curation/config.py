from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class SurpriseConfig:
    context_frames: int = 32
    target_frames: int = 16
    context_duration: float = 2.0
    target_duration: float = 1.0
    stride_duration: Optional[float] = None
    agg: Literal["mean", "max"] = "mean"
    return_curve: bool = False
