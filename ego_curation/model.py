import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from ego_curation.config import VJEPA_CHECKPOINT_URL, VJEPA_HUB_REPO, VJEPA_MODELS

TUBELET = 2
PATCH_SIZE = 16
CROP_SIZE = 384
# Clip length the released models were built for. Positions are RoPE-only, so this
# only sizes the predictor's mask-token grid and can safely be raised.
PRETRAINED_FRAMES = 64

_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)

# The official attention blocks wrap SDPA in the deprecated
# `torch.backends.cuda.sdp_kernel()` context, which warns on every call site.
warnings.filterwarnings(
    "ignore", message=r".*sdp_kernel\(\)` is deprecated", category=FutureWarning
)


def preprocess(frames: torch.Tensor) -> torch.Tensor:
    """(T, C, H, W) uint8 frames -> (1, C, T, CROP_SIZE, CROP_SIZE) model input.

    Mirrors the official eval transform
    (evals/video_classification_frozen/utils.py): short-side resize to
    crop * 256 / 224, center crop, ImageNet normalisation.
    """
    x = frames.float().div(255)
    h, w = x.shape[-2:]
    scale = int(CROP_SIZE * 256 / 224) / min(h, w)
    x = F.interpolate(
        x,
        size=(round(h * scale), round(w * scale)),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (x.shape[-2] - CROP_SIZE) // 2
    left = (x.shape[-1] - CROP_SIZE) // 2
    x = x[..., top : top + CROP_SIZE, left : left + CROP_SIZE]
    x = (x - _MEAN) / _STD
    return x.permute(1, 0, 2, 3).unsqueeze(0)


def _predictor_output_dim(model) -> int:
    """Width of the predictor's target predictions."""
    return model.predictor.predictor_proj.out_features


def _encoder_output_dim(model) -> int:
    """Width of the encoder's hierarchical output: distillation levels concatenated."""
    return len(model.encoder.out_layers_distillation) * model.encoder.embed_dim


@torch.no_grad()
def encode(model, pixel_values_videos):
    """Encode frames through the V-JEPA backbone (skip predictor).

    Returns the concatenated distillation levels rather than the last layer:
    that is both the predictor's expected input and the space its predictions
    live in, so encodings and predictions are directly comparable.
    """
    with torch.autocast(pixel_values_videos.device.type, dtype=torch.float16):
        return model.encoder(pixel_values_videos)


@torch.no_grad()
def predict_target(model, context_embeddings, n_target_tokens):
    """Predict target embeddings from context embeddings via V-JEPA predictor.

    Target tokens are given the positions that directly follow the context, so
    the predictor reads them as the temporal continuation of the context clip.
    """
    B, n_ctx_tokens, _ = context_embeddings.shape
    if n_ctx_tokens + n_target_tokens > model.predictor.num_patches:
        raise ValueError(
            f"context + target span {n_ctx_tokens + n_target_tokens} tokens but the "
            f"predictor was built for {model.predictor.num_patches}; pass a larger "
            "num_frames to load_jepa2"
        )
    dev = context_embeddings.device
    ctx_mask = torch.arange(n_ctx_tokens, device=dev).unsqueeze(0).repeat(B, 1)
    tgt_mask = torch.arange(
        n_ctx_tokens, n_ctx_tokens + n_target_tokens, device=dev
    ).unsqueeze(0).repeat(B, 1)
    with torch.autocast(dev.type, dtype=torch.float16):
        pred, _ = model.predictor(context_embeddings, ctx_mask, tgt_mask)
    return pred


def build_jepa2(hub_entry: str, num_frames: int = PRETRAINED_FRAMES):
    """Instantiate the official V-JEPA 2.1 encoder + predictor (random weights).

    `pretrained=False` because the upstream hub entrypoints point at a localhost
    test URL; the released weights are loaded by `load_jepa2` instead.
    """
    encoder, predictor = torch.hub.load(
        VJEPA_HUB_REPO,
        hub_entry,
        pretrained=False,
        num_frames=max(num_frames, PRETRAINED_FRAMES),
        trust_repo=True,
    )
    encoder.return_hierarchical = True
    return torch.nn.ModuleDict({"encoder": encoder, "predictor": predictor})


def check_comparable(model, name: str) -> None:
    """Raise unless predictor output and encoder output share one space."""
    pred_dim = _predictor_output_dim(model)
    enc_dim = _encoder_output_dim(model)
    if pred_dim != enc_dim:
        raise ValueError(
            f"{name} is not usable for surprise scoring: its predictor emits "
            f"{pred_dim}-dim features while its encoder emits {enc_dim}-dim features, "
            "so the two are not comparable (the predictor targets a distillation "
            "teacher's space). Use a checkpoint trained without a teacher."
        )


def _checkpoint_path(name: str) -> Path:
    """Download an official checkpoint once into the torch hub cache."""
    ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.pt"
    if not path.exists():
        torch.hub.download_url_to_file(VJEPA_CHECKPOINT_URL.format(name), str(path))
    return path


def _clean_backbone_key(state_dict):
    return {
        k.replace("module.", "").replace("backbone.", ""): v
        for k, v in state_dict.items()
    }


def load_jepa2(model_size: str, device: torch.device, num_frames: int = PRETRAINED_FRAMES):
    """Load an official V-JEPA 2.1 model and its preprocessing function.

    `num_frames` must cover context + target frames of one window. Weights are
    held in float16 and run under float16 autocast: the official RoPE attention
    promotes q/k to float32, so autocast is what brings q, k and v back to a
    common dtype for the SDPA kernel.

    The encoder is the EMA `target_encoder`, i.e. the network the predictor was
    trained to match.
    """
    spec = VJEPA_MODELS[model_size]
    model = build_jepa2(spec["hub_entry"], num_frames)
    check_comparable(model, spec["hub_entry"])
    model.half()

    # mmap: the files also hold optimizer state (15.7 / 28.2 GB), so only the
    # tensors actually copied into the model are read into memory.
    state = torch.load(
        _checkpoint_path(spec["checkpoint"]),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.encoder.load_state_dict(_clean_backbone_key(state["target_encoder"]))
    model.predictor.load_state_dict(_clean_backbone_key(state["predictor"]))
    del state

    return model.to(device).eval(), preprocess
