import torch
from transformers import AutoVideoProcessor, AutoModel

TUBELET = 2


@torch.no_grad()
def encode(model, pixel_values_videos):
    """Encode frames through the V-JEPA backbone (skip predictor)."""
    dev = next(model.parameters()).device
    with torch.autocast(
        device_type="cuda" if dev.type == "cuda" else "cpu",
        dtype=torch.bfloat16,
        enabled=(dev.type == "cuda"),
    ):
        out = model(pixel_values_videos=pixel_values_videos, skip_predictor=True)
    return out.last_hidden_state


@torch.no_grad()
def predict_target(model, context_embeddings, n_target_tokens):
    """Predict target embeddings from context embeddings via V-JEPA predictor."""
    B, n_ctx_tokens, _ = context_embeddings.shape
    dev = context_embeddings.device
    ctx_mask = torch.arange(n_ctx_tokens, device=dev).unsqueeze(0).repeat(B, 1)
    tgt_mask = torch.arange(
        n_ctx_tokens, n_ctx_tokens + n_target_tokens, device=dev
    ).unsqueeze(0).repeat(B, 1)
    with torch.autocast(
        device_type="cuda" if dev.type == "cuda" else "cpu",
        dtype=torch.bfloat16,
        enabled=(dev.type == "cuda"),
    ):
        out = model.predictor(
            encoder_hidden_states=context_embeddings,
            context_mask=[ctx_mask],
            target_mask=[tgt_mask],
        )
    return out.last_hidden_state


def load_jepa2(model_name: str, device: torch.device):
    """Load a V-JEPA model + processor from HuggingFace Hub."""
    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoVideoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor
