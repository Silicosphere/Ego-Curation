import gc
import warnings
import torch
from torchcodec.decoders import VideoDecoder

from ego_curation.config import SurpriseConfig
from ego_curation.sampling import get_fps, sample_indices
from ego_curation.model import encode, predict_target, TUBELET


def aggregate(scores: list[float], agg: str) -> float:
    """Collapse per-window scores into a single video-level score."""
    return max(scores) if agg == "max" else sum(scores) / len(scores)


@torch.no_grad()
def calc_surprise_streaming(model, processor, video_file, config: SurpriseConfig):
    assert config.context_frames % TUBELET == 0 and config.target_frames % TUBELET == 0

    vr = VideoDecoder(video_file)
    num_frames = len(vr)
    fps = get_fps(vr)
    model_device = next(model.parameters()).device

    # Context and target are one continuous sampling grid: the target's first
    # frame is exactly one step after the context's last frame.
    step = fps / config.sample_fps
    if step < 1:
        warnings.warn(
            f"{video_file}: sample rate {config.sample_fps} fps exceeds the "
            f"video's {fps:.2f} fps; frames will be repeated"
        )
    n_total = config.context_frames + config.target_frames
    window = max(1, int(round(n_total * step)))

    if config.stride_duration is None:
        stride = window
    else:
        stride = max(1, int(round(config.stride_duration * fps)))

    starts = list(range(0, max(1, num_frames - window + 1), stride))

    scores = []
    for s in starts:
        idx = sample_indices(s, n_total, step, num_frames)
        frames = vr.get_frames_at(indices=idx.tolist()).data
        ctx_frames = frames[: config.context_frames]
        tgt_frames = frames[config.context_frames :]

        ctx_proc = processor(ctx_frames).to(model_device)
        tgt_proc = processor(tgt_frames).to(model_device)
        del ctx_frames, tgt_frames, frames

        ctx_emb = encode(model, ctx_proc)
        tgt_emb_true = encode(model, tgt_proc)
        tgt_emb_pred = predict_target(model, ctx_emb, tgt_emb_true.shape[1])

        per_window = (tgt_emb_pred.float() - tgt_emb_true.float()).abs().mean()
        scores.append(per_window.item())

        del ctx_proc, tgt_proc, ctx_emb, tgt_emb_true, tgt_emb_pred, per_window
        if model_device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del vr

    if config.return_curve:
        return scores, [s / fps for s in starts]
    return aggregate(scores, config.agg)
