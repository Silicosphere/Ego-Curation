import numpy as np
import torch
from torchcodec.decoders import VideoDecoder
import gc

TUBELET = 2


def _get_fps(vr):
    try:
        md = vr.metadata
        fps = md.frame_rate
        if isinstance(fps, (int, float)):
            return float(fps)
        return float(fps.numerator) / float(fps.denominator)
    except Exception:
        pass
    return 30.0


def _sample_indices(start, end, n):
    available = end - start
    if available <= 0:
        return np.array([max(0, start)] * n)
    if available <= n:
        idx = np.arange(start, end)
        pad = np.full(n - available, end - 1)
        return np.concatenate([idx, pad])
    return np.linspace(start, end - 1, n, dtype=int)


@torch.no_grad()
def _encode(model, pixel_values_videos):
    dev = next(model.parameters()).device
    with torch.autocast(
        device_type="cuda" if dev.type == "cuda" else "cpu",
        dtype=torch.bfloat16,
        enabled=(dev.type == "cuda"),
    ):
        out = model(pixel_values_videos=pixel_values_videos, skip_predictor=True)
    return out.last_hidden_state


@torch.no_grad()
def _predict_target(model, context_embeddings, n_target_tokens):
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


@torch.no_grad()
def calc_surprise_streaming(
    model,
    processor,
    video_file,
    context_frames=32,
    target_frames=16,
    context_duration=2.0,
    target_duration=1.0,
    stride_duration=None,
    agg="mean",
    batch_size=4,
    return_curve=False,
):
    assert context_frames % TUBELET == 0 and target_frames % TUBELET == 0

    vr = VideoDecoder(video_file)
    num_frames = len(vr)
    fps = _get_fps(vr)
    model_device = next(model.parameters()).device

    ctx_frame_count = int(round(context_duration * fps))
    tgt_frame_count = int(round(target_duration * fps))
    window = ctx_frame_count + tgt_frame_count

    if stride_duration is None:
        stride = window
    else:
        stride = max(1, int(round(stride_duration * fps)))

    starts = list(range(0, max(1, num_frames - window + 1), stride))
    if not starts:
        starts = [0]

    scores = []
    for i in range(0, len(starts), batch_size):
        batch_starts = starts[i : i + batch_size]

        ctx_list, tgt_list = [], []
        for s in batch_starts:
            ctx_end = s + ctx_frame_count
            tgt_end = ctx_end + tgt_frame_count

            ctx_idx = _sample_indices(s, min(ctx_end, num_frames), context_frames)
            tgt_idx = _sample_indices(
                min(ctx_end, num_frames), min(tgt_end, num_frames), target_frames
            )

            ctx_frames = vr.get_frames_at(indices=ctx_idx).data
            tgt_frames = vr.get_frames_at(indices=tgt_idx).data

            ctx_list.append(
                processor(ctx_frames, return_tensors="pt")["pixel_values_videos"]
            )
            tgt_list.append(
                processor(tgt_frames, return_tensors="pt")["pixel_values_videos"]
            )

        ctx_batch = torch.cat(ctx_list, dim=0).to(model_device)
        tgt_batch = torch.cat(tgt_list, dim=0).to(model_device)
        del ctx_list, tgt_list

        ctx_emb = _encode(model, ctx_batch)
        tgt_emb_true = _encode(model, tgt_batch)
        tgt_emb_pred = _predict_target(model, ctx_emb, tgt_emb_true.shape[1])

        per_window = (tgt_emb_pred.float() - tgt_emb_true.float()).abs().mean(dim=(1, 2))
        scores.extend(per_window.cpu().tolist())

        del ctx_batch, tgt_batch, ctx_emb, tgt_emb_true, tgt_emb_pred, per_window
        if model_device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    del vr

    if return_curve:
        return scores, [s / fps for s in starts]
    return max(scores) if agg == "max" else float(np.mean(scores))
