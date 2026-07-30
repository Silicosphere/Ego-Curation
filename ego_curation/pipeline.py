import gc
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from ego_curation.config import SurpriseConfig
from ego_curation.sampling import get_fps, sample_indices
from ego_curation.model import encode, predict_target, TUBELET


@torch.no_grad()
def calc_surprise_streaming(model, processor, video_file, config: SurpriseConfig):
    assert config.context_frames % TUBELET == 0 and config.target_frames % TUBELET == 0

    vr = VideoDecoder(video_file)
    num_frames = len(vr)
    fps = get_fps(vr)
    model_device = next(model.parameters()).device

    ctx_frame_count = int(round(config.context_duration * fps))
    tgt_frame_count = int(round(config.target_duration * fps))
    window = ctx_frame_count + tgt_frame_count

    if config.stride_duration is None:
        stride = window
    else:
        stride = max(1, int(round(config.stride_duration * fps)))

    starts = list(range(0, max(1, num_frames - window + 1), stride))
    if not starts:
        starts = [0]

    scores = []
    for idx, s in enumerate(starts):
        ctx_end = min(s + ctx_frame_count, num_frames)
        tgt_end = min(ctx_end + tgt_frame_count, num_frames)

        ctx_idx = sample_indices(s, ctx_end, config.context_frames)
        tgt_idx = sample_indices(ctx_end, tgt_end, config.target_frames)

        ctx_frames = vr.get_frames_at(indices=ctx_idx).data
        tgt_frames = vr.get_frames_at(indices=tgt_idx).data

        ctx_proc = (
            processor(ctx_frames, return_tensors="pt")["pixel_values_videos"]
            .to(model_device)
        )
        tgt_proc = (
            processor(tgt_frames, return_tensors="pt")["pixel_values_videos"]
            .to(model_device)
        )
        del ctx_frames, tgt_frames

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
    return max(scores) if config.agg == "max" else float(np.mean(scores))
