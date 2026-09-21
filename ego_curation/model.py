import torch
from transformers import AutoVideoProcessor, AutoModel

TUBELET = 2


def _predictor_output_dim(config) -> int:
    """Width of `predictor.last_hidden_state`, mirroring VJEPA21Predictor.__init__."""
    n_hier = len(config.predictor_hierarchical_layers)
    if config.pred_teacher_embed_dim is not None:
        out_dim = config.pred_teacher_embed_dim // n_hier
    else:
        out_dim = config.hidden_size
    return n_hier * out_dim


def _encoder_output_dim(config) -> int:
    """Width of `hierarchical_hidden_state`: the distillation levels concatenated."""
    return len(config.encoder_distillation_layers) * config.hidden_size


@torch.no_grad()
def encode(model, pixel_values_videos):
    """Encode frames through the V-JEPA backbone (skip predictor).

    Returns the concatenated distillation levels rather than `last_hidden_state`:
    that is both the predictor's expected input and the space its predictions
    live in, so encodings and predictions are directly comparable.
    """
    out = model(
        pixel_values_videos=pixel_values_videos,
        skip_predictor=True,
        return_hierarchical=True,
    )
    return out.hierarchical_hidden_state


@torch.no_grad()
def predict_target(model, context_embeddings, n_target_tokens):
    """Predict target embeddings from context embeddings via V-JEPA predictor.

    Target tokens are given the positions that directly follow the context, so
    the predictor reads them as the temporal continuation of the context clip.
    """
    B, n_ctx_tokens, _ = context_embeddings.shape
    dev = context_embeddings.device
    ctx_mask = torch.arange(n_ctx_tokens, device=dev).unsqueeze(0).repeat(B, 1)
    tgt_mask = torch.arange(
        n_ctx_tokens, n_ctx_tokens + n_target_tokens, device=dev
    ).unsqueeze(0).repeat(B, 1)
    out = model.predictor(
        encoder_hidden_states=context_embeddings,
        context_mask=[ctx_mask],
        target_mask=[tgt_mask],
    )
    return out.last_hidden_state


def load_jepa2(model_name: str, device: torch.device):
    """Load a V-JEPA model + processor from HuggingFace Hub.

    Weights are held in float16 with the SDPA attention kernel. float16 is far
    more accurate than bfloat16 for this family, but its attention logits can
    overflow under the eager kernel, so SDPA is requested explicitly rather than
    left to the default.
    """
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float16,
        attn_implementation="sdpa",
    ).to(device)
    processor = AutoVideoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    pred_dim = _predictor_output_dim(model.config)
    enc_dim = _encoder_output_dim(model.config)
    if pred_dim != enc_dim:
        raise ValueError(
            f"{model_name} is not usable for surprise scoring: its predictor emits "
            f"{pred_dim}-dim features while its encoder emits {enc_dim}-dim features, "
            f"so the two are not comparable (pred_teacher_embed_dim="
            f"{model.config.pred_teacher_embed_dim}). Use a checkpoint whose "
            "pred_teacher_embed_dim is null."
        )
    return model, processor
