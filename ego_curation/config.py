from dataclasses import dataclass
from typing import Optional, Literal

# Official V-JEPA 2.1 release (github.com/facebookresearch/vjepa2), loaded through
# torch.hub. Pinned to a commit because torch.hub executes the repository's code.
VJEPA_HUB_REPO = "facebookresearch/vjepa2:204698b45b3712590f06245fbfba32d3be539812"
VJEPA_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/vjepa2/{}.pt"

# Only the checkpoints whose predictor writes into their own representation space
# are usable here. The base/large releases are distilled from ViT-G
# (`teacher_embed_dim = 1664`), i.e. their predictor outputs ViT-G teacher
# features, which are not comparable with their own encoder output. giant/gigantic
# have no teacher, so the predictor reproduces the encoder's own hierarchical
# features (5632 = 4 x 1408 and 6656 = 4 x 1664 respectively).
VJEPA_MODELS = {
    "giant": {
        "hub_entry": "vjepa2_1_vit_giant_384",
        "checkpoint": "vjepa2_1_vitg_384",
    },
    "gigantic": {
        "hub_entry": "vjepa2_1_vit_gigantic_384",
        "checkpoint": "vjepa2_1_vitG_384",
    },
}

DEFAULT_MODEL_SIZE = "gigantic"


# V-JEPA 2.1 was pretrained on clips sampled at 4 fps
# (configs/train_2_1/*/pretrain-256px-16f.yaml in the official repo).
DEFAULT_SAMPLE_FPS = 4.0


@dataclass
class SurpriseConfig:
    context_frames: int = 32
    target_frames: int = 16
    # Context and target share one sampling rate, so the predictor's temporal
    # grid (which continues the context's spacing) matches the target clip.
    sample_fps: float = DEFAULT_SAMPLE_FPS
    stride_duration: Optional[float] = None
    agg: Literal["mean", "max"] = "mean"
    return_curve: bool = False

    @property
    def context_duration(self) -> float:
        return self.context_frames / self.sample_fps

    @property
    def target_duration(self) -> float:
        return self.target_frames / self.sample_fps

    @property
    def window_duration(self) -> float:
        return self.context_duration + self.target_duration
