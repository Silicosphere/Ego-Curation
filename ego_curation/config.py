from dataclasses import dataclass
from typing import Optional, Literal

# Only the checkpoints whose predictor writes into their own representation space
# are usable here. The base/large ports set `pred_teacher_embed_dim = 1664`, i.e.
# their predictor outputs ViT-G teacher features, which are not comparable with
# their own 768/1024-dim encoder output. giant/gigantic set it to null, so the
# predictor reproduces the encoder's own hierarchical features.
VJEPA_MODELS = {
    "giant":    "apiantonio/vjepa2.1-vit-giant-384",
    "gigantic": "apiantonio/vjepa2.1-vit-gigantic-384",
}

DEFAULT_MODEL_SIZE = "gigantic"


@dataclass
class SurpriseConfig:
    context_frames: int = 32
    target_frames: int = 16
    context_duration: float = 2.0
    target_duration: float = 1.0
    stride_duration: Optional[float] = None
    agg: Literal["mean", "max"] = "mean"
    return_curve: bool = False
