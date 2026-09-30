"""Standalone diagnostic: print every shape the surprise pipeline depends on.

Shapes do not depend on pixel content, so a real video is optional.

Usage:
    python check_shapes.py --model-size base
    python check_shapes.py --model-size base --video /path/to/video.mp4
"""

import argparse

import torch
from transformers import AutoModel, AutoVideoProcessor

MODELS = {
    "base": "apiantonio/vjepa2.1-vit-base-384",
    "large": "apiantonio/vjepa2.1-vit-large-384",
    "giant": "apiantonio/vjepa2.1-vit-giant-384",
    "gigantic": "apiantonio/vjepa2.1-vit-gigantic-384",
}


def get_frames(video, n, height, width):
    """Decode n frames as (T, C, H, W) uint8, or synthesise them if that fails."""
    if video is not None:
        try:
            from torchcodec.decoders import VideoDecoder

            vr = VideoDecoder(video)
            idx = list(range(min(n, len(vr))))
            frames = vr.get_frames_at(indices=idx).data
            print(f"  source: decoded from {video}")
            return frames
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"  decode failed ({type(exc).__name__}), falling back to synthetic frames")
            print(f"  {str(exc).splitlines()[0]}")

    print(f"  source: synthetic {height}x{width} noise")
    return torch.randint(0, 256, (n, 3, height, width), dtype=torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="optional; synthetic frames are used otherwise")
    ap.add_argument("--model-size", default="base", choices=list(MODELS))
    ap.add_argument("--context-frames", type=int, default=32)
    ap.add_argument("--target-frames", type=int, default=16)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--width", type=int, default=960)
    args = ap.parse_args()

    name = MODELS[args.model_size]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(dev).eval()
    processor = AutoVideoProcessor.from_pretrained(name, trust_remote_code=True)
    cfg = model.config

    print("=== config ===")
    for k in (
        "hidden_size",
        "pred_hidden_size",
        "pred_teacher_embed_dim",
        "n_output_distillation",
        "pred_return_all_tokens",
        "crop_size",
        "patch_size",
        "tubelet_size",
        "img_temporal_dim_size",
    ):
        print(f"  {k:24s} {getattr(cfg, k, '<missing>')}")
    for k in (
        "encoder_hierarchical_layers",
        "encoder_distillation_layers",
        "predictor_hierarchical_layers",
        "pretrained_grid_size",
    ):
        print(f"  {k:24s} {getattr(cfg, k, '<missing>')}")

    print("\n=== predictor heads ===")
    pe = model.predictor.embeddings.predictor_embed
    first = pe if isinstance(pe, torch.nn.Linear) else pe[0]
    print(f"  predictor_embed expects in_features = {first.in_features}")
    print(f"  proj:         {tuple(model.predictor.proj.weight.shape)}  (out, in)")
    pc = model.predictor.proj_context
    print(f"  proj_context: {tuple(pc.weight.shape) if pc is not None else None}")

    # --- frames --------------------------------------------------------
    print("\n=== frames ===")
    n = args.context_frames + args.target_frames
    frames = get_frames(args.video, n, args.height, args.width)
    ctx_frames = frames[: args.context_frames]
    tgt_frames = frames[args.context_frames :]
    print(f"  raw frames (T, C, H, W): {tuple(frames.shape)} {frames.dtype}")

    # --- A: exactly what pipeline.py does today ------------------------
    print("\n=== A: processor(tensor)  <- current pipeline.py ===")
    a_ctx = processor(ctx_frames, return_tensors="pt")["pixel_values_videos"]
    a_tgt = processor(tgt_frames, return_tensors="pt")["pixel_values_videos"]
    print(f"  ctx pixel_values_videos: {tuple(a_ctx.shape)}")
    print(f"  tgt pixel_values_videos: {tuple(a_tgt.shape)}")

    # --- B: the layout the model card documents ------------------------
    print("\n=== B: processor([list of HWC uint8 frames]) <- documented ===")
    to_hwc = lambda f: [x.permute(1, 2, 0).cpu().numpy() for x in f]
    b_ctx = processor([to_hwc(ctx_frames)], return_tensors="pt")["pixel_values_videos"]
    b_tgt = processor([to_hwc(tgt_frames)], return_tensors="pt")["pixel_values_videos"]
    print(f"  ctx pixel_values_videos: {tuple(b_ctx.shape)}")
    print(f"  tgt pixel_values_videos: {tuple(b_tgt.shape)}")
    print(f"  A and B identical: {a_ctx.shape == b_ctx.shape and torch.allclose(a_ctx, b_ctx)}")

    # --- encoder -------------------------------------------------------
    @torch.no_grad()
    def enc(x):
        return model(pixel_values_videos=x.to(dev), skip_predictor=True).last_hidden_state

    for tag, c, t in (("A", a_ctx, a_tgt), ("B", b_ctx, b_tgt)):
        ce, te = enc(c), enc(t)
        print(f"\n=== encoder, layout {tag} ===")
        print(f"  encode(ctx): {tuple(ce.shape)}")
        print(f"  encode(tgt): {tuple(te.shape)}")
        if tag == "B":
            ctx_emb, tgt_emb = ce, te

    # --- predictor, exactly as model.predict_target calls it -----------
    B, n_ctx, _ = ctx_emb.shape
    n_tgt = tgt_emb.shape[1]
    ctx_mask = torch.arange(n_ctx, device=dev).unsqueeze(0).repeat(B, 1)
    tgt_mask = torch.arange(n_ctx, n_ctx + n_tgt, device=dev).unsqueeze(0).repeat(B, 1)

    print("\n=== predictor (layout B) ===")
    with torch.no_grad():
        out = model.predictor(
            encoder_hidden_states=ctx_emb,
            context_mask=[ctx_mask],
            target_mask=[tgt_mask],
        )
    print(f"  last_hidden_state (predicted target): {tuple(out.last_hidden_state.shape)}")
    cxh = out.context_hidden_state
    print(f"  context_hidden_state:                 {tuple(cxh.shape) if cxh is not None else None}")
    print(f"  encoder target for comparison:        {tuple(tgt_emb.shape)}")
    print(
        "\n  >>> widths: pred="
        f"{out.last_hidden_state.shape[-1]}  encoder={tgt_emb.shape[-1]}  "
        f"match={out.last_hidden_state.shape[-1] == tgt_emb.shape[-1]}"
    )


if __name__ == "__main__":
    main()
