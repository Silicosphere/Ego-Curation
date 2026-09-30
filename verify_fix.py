"""Verify the V-JEPA 2.1 surprise pipeline after the teacher-space fix.

Runs without ffmpeg/torchcodec by default (synthetic frames); pass --video to
additionally exercise the real streaming pipeline.

Usage:
    python verify_fix.py --model-size gigantic
    python verify_fix.py --model-size giant --video /path/to/video.mp4
"""

import argparse
import sys

import torch

from ego_curation.config import VJEPA_MODELS, DEFAULT_MODEL_SIZE, SurpriseConfig
from ego_curation.model import (
    encode,
    predict_target,
    load_jepa2,
    _encoder_output_dim,
    _predictor_output_dim,
)

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(condition)


def make_clip(kind, n, h, w, seed=0):
    """Synthetic (T, C, H, W) uint8 clips with controlled predictability."""
    g = torch.Generator().manual_seed(seed)
    if kind == "static":
        frame = torch.randint(0, 256, (1, 3, h, w), generator=g, dtype=torch.uint8)
        return frame.repeat(n, 1, 1, 1)
    if kind == "motion":
        # A single frame translated by a constant offset each step: highly predictable.
        wide = torch.randint(0, 256, (3, h, w * 2), generator=g, dtype=torch.uint8)
        return torch.stack([wide[:, :, i * 4 : i * 4 + w] for i in range(n)])
    if kind == "random":
        return torch.randint(0, 256, (n, 3, h, w), generator=g, dtype=torch.uint8)
    raise ValueError(kind)


def score_window(model, processor, dev, ctx_frames, tgt_frames):
    """Reproduce exactly what pipeline.calc_surprise_streaming does per window."""
    ctx = processor(ctx_frames, return_tensors="pt")["pixel_values_videos"].to(dev)
    tgt = processor(tgt_frames, return_tensors="pt")["pixel_values_videos"].to(dev)
    ctx_emb = encode(model, ctx)
    tgt_emb_true = encode(model, tgt)
    tgt_emb_pred = predict_target(model, ctx_emb, tgt_emb_true.shape[1])
    score = (tgt_emb_pred.float() - tgt_emb_true.float()).abs().mean()
    return score.item(), ctx_emb, tgt_emb_true, tgt_emb_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", default=DEFAULT_MODEL_SIZE, choices=list(VJEPA_MODELS))
    ap.add_argument("--context-frames", type=int, default=32)
    ap.add_argument("--target-frames", type=int, default=16)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--video", default=None, help="also run the real streaming pipeline")
    ap.add_argument(
        "--skip-guard-check",
        action="store_true",
        help="skip downloading the base port to confirm it is rejected",
    )
    args = ap.parse_args()

    name = VJEPA_MODELS[args.model_size]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}\nModel:  {name} ({args.model_size})\n")

    # --- 1. loading ----------------------------------------------------
    print("1. Loading")
    model, processor = load_jepa2(name, dev)
    cfg = model.config
    dtypes = {p.dtype for p in model.parameters()}
    check("weights are float16", dtypes == {torch.float16}, str(dtypes))
    check(
        "attention implementation is sdpa",
        getattr(cfg, "_attn_implementation", None) == "sdpa",
        str(getattr(cfg, "_attn_implementation", None)),
    )

    # --- 2. config self-consistency ------------------------------------
    print("\n2. Config self-consistency")
    enc_dim = _encoder_output_dim(cfg)
    pred_dim = _predictor_output_dim(cfg)
    check(
        "predictor width == encoder hierarchical width",
        enc_dim == pred_dim,
        f"encoder={enc_dim}, predictor={pred_dim}",
    )
    check(
        "pred_teacher_embed_dim is null",
        cfg.pred_teacher_embed_dim is None,
        f"pred_teacher_embed_dim={cfg.pred_teacher_embed_dim}",
    )

    # --- 3. shapes on one window ---------------------------------------
    print("\n3. Shapes and the previously failing subtraction")
    ctx_frames = make_clip("motion", args.context_frames, args.height, args.width, seed=1)
    tgt_frames = make_clip("motion", args.target_frames, args.height, args.width, seed=1)
    score, ctx_emb, tgt_true, tgt_pred = score_window(
        model, processor, dev, ctx_frames, tgt_frames
    )

    grid = cfg.crop_size // cfg.patch_size
    want_ctx = (args.context_frames // cfg.tubelet_size) * grid * grid
    want_tgt = (args.target_frames // cfg.tubelet_size) * grid * grid
    print(f"     ctx_emb {tuple(ctx_emb.shape)}  tgt_true {tuple(tgt_true.shape)}"
          f"  tgt_pred {tuple(tgt_pred.shape)}")
    check("context token count", ctx_emb.shape[1] == want_ctx, f"want {want_ctx}")
    check("target token count", tgt_true.shape[1] == want_tgt, f"want {want_tgt}")
    check("encoder emits hierarchical width", ctx_emb.shape[-1] == enc_dim, f"want {enc_dim}")
    check(
        "prediction and truth have identical shape",
        tgt_pred.shape == tgt_true.shape,
        f"{tuple(tgt_pred.shape)} vs {tuple(tgt_true.shape)}",
    )

    # --- 4. float16 numerical health -----------------------------------
    print("\n4. float16 numerical health")
    check("encoder output finite", torch.isfinite(ctx_emb).all().item())
    check("predictor output finite", torch.isfinite(tgt_pred).all().item())
    check("score is finite and positive", score > 0 and score == score, f"score={score:.6f}")

    # --- 5. determinism -------------------------------------------------
    print("\n5. Determinism")
    score2, *_ = score_window(model, processor, dev, ctx_frames, tgt_frames)
    check("same input gives same score", score == score2, f"{score:.6f} vs {score2:.6f}")

    # --- 6. does the score actually measure surprise? -------------------
    print("\n6. Discrimination (the score must respond to predictability)")
    scores = {}
    for kind in ("static", "motion", "random"):
        c = make_clip(kind, args.context_frames, args.height, args.width, seed=2)
        t = make_clip(kind, args.target_frames, args.height, args.width, seed=3 if kind == "random" else 2)
        if kind == "motion":
            # continue the same translating pattern into the target window
            full = make_clip("motion", args.context_frames + args.target_frames,
                             args.height, args.width, seed=2)
            c, t = full[: args.context_frames], full[args.context_frames :]
        scores[kind], *_ = score_window(model, processor, dev, c, t)
        print(f"     {kind:7s} -> {scores[kind]:.6f}")
    check(
        "static clip scores below random clip",
        scores["static"] < scores["random"],
        f"static={scores['static']:.6f} random={scores['random']:.6f}",
    )
    check(
        "predictable motion scores below random",
        scores["motion"] < scores["random"],
        f"motion={scores['motion']:.6f} random={scores['random']:.6f}",
    )

    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # --- 7. the guard rejects an incompatible checkpoint ----------------
    if not args.skip_guard_check:
        print("\n7. Guard rejects a teacher-space checkpoint (downloads ~0.4 GB)")
        try:
            load_jepa2("apiantonio/vjepa2.1-vit-base-384", torch.device("cpu"))
            check("base port rejected by load_jepa2", False, "it loaded without error")
        except ValueError as exc:
            check("base port rejected by load_jepa2", True, str(exc)[:90] + "...")
    else:
        print("\n7. Guard check skipped")

    # --- 8. optional end-to-end run -------------------------------------
    if args.video:
        print("\n8. Real pipeline on a video")
        from ego_curation.pipeline import calc_surprise_streaming

        model, processor = load_jepa2(name, dev)
        cfg_s = SurpriseConfig(
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            return_curve=True,
        )
        curve, times = calc_surprise_streaming(model, processor, args.video, cfg_s)
        print(f"     {len(curve)} windows, first 5: {[round(s, 4) for s in curve[:5]]}")
        check("all window scores finite", all(s == s and s != float("inf") for s in curve))
        check("scores vary across windows", len(set(round(s, 6) for s in curve)) > 1)

    # --- summary --------------------------------------------------------
    failed = [n for n, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
