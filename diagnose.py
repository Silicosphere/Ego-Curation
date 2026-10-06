"""Diagnostics for the surprise metric.

collect   : run the pipeline's windows and dump per-token, per-level errors (GPU).
summarize : turn the dumps into a text report (numpy only, runs anywhere).

    python diagnose.py collect VIDEO... --out-dir DIR --model-size giant \
        --sample-fps 4 --context-frames 16 --target-frames 8 --stride-duration 6
    python diagnose.py summarize DIR
"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np

ARRAYS = ("err_pred", "err_copy", "err_zero", "pred_rms", "cos")


def collect(args) -> None:
    import torch
    import torch.nn.functional as F
    from torchcodec.decoders import VideoDecoder
    from tqdm.auto import tqdm

    from ego_curation.config import SurpriseConfig
    from ego_curation.model import (
        CROP_SIZE, PATCH_SIZE, encode, load_jepa2, normalize_levels, predict_target,
    )
    from ego_curation.pipeline import window_grid
    from ego_curation.sampling import get_fps, sample_indices

    config = SurpriseConfig(
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        sample_fps=args.sample_fps,
        stride_duration=args.stride_duration,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, processor = load_jepa2(
        args.model_size, device, num_frames=args.context_frames + args.target_frames
    )
    L = len(model.encoder.out_layers_distillation)
    d = model.encoder.embed_dim
    g = CROP_SIZE // PATCH_SIZE

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in vars(args).items() if k not in ("func", "videos")}
    (out_dir / "meta.json").write_text(json.dumps({**meta, "levels": L, "grid": g}))

    videos = args.videos[: args.n_videos] if args.n_videos else args.videos
    for video_file in tqdm(videos, desc="Videos"):
        out_path = out_dir / f"{Path(video_file).stem}.npz"
        if out_path.exists():
            continue

        vr = VideoDecoder(video_file)
        num_frames = len(vr)
        fps = get_fps(vr)
        step, n_total, starts = window_grid(video_file, num_frames, fps, config)

        rows = {k: [] for k in ARRAYS + ("ctx_shift", "score")}
        for s in starts:
            idx = sample_indices(s, n_total, step, num_frames)
            frames = vr.get_frames_at(indices=idx.tolist()).data
            window_proc = processor(frames.to(device))
            del frames

            ctx_emb = encode(model, window_proc[:, :, : config.context_frames])
            window_emb = encode(model, window_proc)
            n_ctx = ctx_emb.shape[1]
            true = normalize_levels(model, window_emb[:, n_ctx:])
            pred = predict_target(model, ctx_emb, true.shape[1]).float()
            T = true.shape[1] // (g * g)
            # Baseline: "nothing changes", i.e. the last context time slot repeated.
            copy = normalize_levels(model, window_emb[:, n_ctx - g * g : n_ctx])
            copy = copy.repeat(1, T, 1)

            P, Y, C = (x.view(T, g, g, L, d) for x in (pred, true, copy))
            err_pred = (P - Y).abs().mean(-1)
            per_token = {
                "err_pred": err_pred,
                "err_copy": (C - Y).abs().mean(-1),
                "err_zero": Y.abs().mean(-1),
                "pred_rms": P.pow(2).mean(-1).sqrt(),
                "cos": F.cosine_similarity(P, Y, dim=-1),
            }
            for k, v in per_token.items():
                rows[k].append(v.half().cpu().numpy())
            # How much the context features change when the target frames are visible.
            shift = normalize_levels(model, ctx_emb) - normalize_levels(model, window_emb[:, :n_ctx])
            rows["ctx_shift"].append(shift.abs().view(-1, L, d).mean((0, 2)).cpu().numpy())
            rows["score"].append(err_pred.mean().item())

            del window_proc, ctx_emb, window_emb, true, pred, copy, P, Y, C, per_token, shift
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        del vr

        np.savez_compressed(
            out_path,
            video=video_file,
            start_sec=np.array([s / fps for s in starts]),
            **{k: np.stack(v) for k, v in rows.items()},
        )


# ── summarize ────────────────────────────────────────────────────────────────


def _rank(x):
    return np.argsort(np.argsort(x)).astype(np.float64)


def _pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 else float("nan")


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def _top_mean(x, q):
    """Mean of the top fraction q along axis 1."""
    k = max(1, int(round(x.shape[1] * q)))
    return np.sort(x, axis=1)[:, -k:].mean(1)


def _eta2(values, groups):
    """Share of variance explained by the video (between-video / total)."""
    total = ((values - values.mean()) ** 2).sum()
    between = sum(
        (groups == gid).sum() * (values[groups == gid].mean() - values.mean()) ** 2
        for gid in np.unique(groups)
    )
    return float(between / total) if total > 0 else float("nan")


def _window_metrics(v, L):
    W = v["err_pred"].shape[0]
    ep = v["err_pred"].reshape(W, -1, L)
    ec = v["err_copy"].reshape(W, -1, L)
    m = {"mean (current)": ep.mean((1, 2))}
    for k in range(L):
        m[f"level {k + 1} mean"] = ep[..., k].mean(1)
    m["top-1% tokens"] = _top_mean(ep.mean(2), 0.01)
    m[f"top-1% tokens, level {L}"] = _top_mean(ep[..., -1], 0.01)
    m["pred/copy ratio"] = ep.mean((1, 2)) / ec.mean((1, 2))
    m[f"pred/copy ratio, level {L}"] = ep[..., -1].mean(1) / ec[..., -1].mean(1)
    m["1 - cos"] = 1 - v["cos"].reshape(W, -1).mean(1)
    return m


def summarize(args) -> None:
    out_dir = Path(args.dump_dir)
    files = sorted(out_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz dumps in {out_dir}")
    meta = json.loads((out_dir / "meta.json").read_text())
    L, g = meta["levels"], meta["grid"]

    vids = []
    for f in files:
        with np.load(f) as z:
            v = {k: z[k].astype(np.float32) for k in ARRAYS + ("ctx_shift", "score", "start_sec")}
            v["name"] = f.stem
            vids.append(v)

    cat = {k: np.concatenate([v[k] for v in vids]) for k in ARRAYS + ("ctx_shift", "score")}
    W, T = cat["err_pred"].shape[:2]
    lines = []
    p = lines.append

    p(f"model={meta['model_size']} fps={meta['sample_fps']} ctx={meta['context_frames']} "
      f"tgt={meta['target_frames']} stride={meta['stride_duration']}")
    p(f"videos={len(vids)} windows={W} target_slots={T} grid={g}x{g} levels={L}")

    p("\n[1] Per level, pooled over all tokens")
    p(f"{'level':>6} {'pred':>7} {'copy':>7} {'zero':>7} {'p/copy':>7} {'p/zero':>7} "
      f"{'cos':>6} {'predRMS':>8} {'ctxShift':>9}")
    for k in range(L):
        ep, ec, ez = (cat[a][..., k].mean() for a in ("err_pred", "err_copy", "err_zero"))
        p(f"{k + 1:>6} {ep:7.4f} {ec:7.4f} {ez:7.4f} {ep / ec:7.3f} {ep / ez:7.3f} "
          f"{cat['cos'][..., k].mean():6.3f} {cat['pred_rms'][..., k].mean():8.3f} "
          f"{cat['ctx_shift'][:, k].mean():9.4f}")

    p("\n[2] Per target time slot (pred error / copy error per level)")
    p(f"{'slot':>5} " + " ".join(f"{'L' + str(k + 1):>15}" for k in range(L)))
    for t in range(T):
        cells = [
            f"{cat['err_pred'][:, t, ..., k].mean():.4f}/{cat['err_copy'][:, t, ..., k].mean():.4f}"
            for k in range(L)
        ]
        p(f"{t:>5} " + " ".join(f"{c:>15}" for c in cells))

    p("\n[3] Token error distribution (pred error)")
    p(f"{'level':>6} {'p1':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'p99.9':>7} {'max':>7}")
    for k in range(L):
        q = np.percentile(cat["err_pred"][..., k], [1, 50, 90, 99, 99.9, 100])
        p(f"{k + 1:>6} " + " ".join(f"{x:7.4f}" for x in q))

    p("\n[4] Spatial profile, rings from centre (pred error / copy error)")
    c = (g - 1) / 2
    ring = np.maximum(*np.abs(np.mgrid[:g, :g] - c))
    bins = np.minimum((ring / (g / 2) * 4).astype(int), 3)
    p(f"{'ring':>5} " + " ".join(f"{'L' + str(k + 1):>15}" for k in range(L)))
    for b in range(4):
        sel = bins == b
        cells = [
            f"{cat['err_pred'][:, :, sel, k].mean():.4f}/{cat['err_copy'][:, :, sel, k].mean():.4f}"
            for k in range(L)
        ]
        p(f"{b:>5} " + " ".join(f"{x:>15}" for x in cells))

    per_vid = [_window_metrics(v, L) for v in vids]
    groups = np.concatenate([np.full(len(v["score"]), i) for i, v in enumerate(vids)])
    metrics = {k: np.concatenate([m[k] for m in per_vid]) for k in per_vid[0]}
    motion = np.concatenate([v["err_copy"].reshape(len(v["score"]), -1).mean(1) for v in vids])
    current = metrics["mean (current)"]
    p(f"\n[5] Window-level metrics ({W} windows)")
    p("  CV = std/mean across windows; eta2 = share of variance between videos;")
    p("  r(copy) = Pearson with copy-baseline error (motion proxy); rho(cur) = Spearman with current")
    p(f"{'metric':<28} {'mean':>8} {'std':>8} {'CV':>6} {'eta2':>6} {'r(copy)':>8} {'rho(cur)':>9}")
    for name, x in metrics.items():
        p(f"{name:<28} {x.mean():8.4f} {x.std():8.4f} {x.std() / x.mean():6.3f} "
          f"{_eta2(x, groups):6.3f} {_pearson(x, motion):8.3f} {_spearman(x, current):9.3f}")
    p(f"  check: stored pipeline score vs recomputed mean, max abs diff = "
      f"{np.abs(cat['score'] - current).max():.2e}")

    p("\n[6] Per video (window mean of each metric; 'max' = max window of current)")
    shown = ("mean (current)", "pred/copy ratio", f"level {L} mean", "top-1% tokens")
    p(f"{'video':<40} {'win':>4} " + " ".join(f"{s[:14]:>14}" for s in shown) + f" {'max':>8}")
    for v, m in sorted(zip(vids, per_vid), key=lambda vm: -vm[1]["mean (current)"].mean())[: args.max_videos]:
        p(f"{v['name'][:40]:<40} {len(v['score']):>4} "
          + " ".join(f"{m[s].mean():14.4f}" for s in shown)
          + f" {m['mean (current)'].max():8.4f}")
    if len(vids) > 2:
        vmean = np.array([m["mean (current)"].mean() for m in per_vid])
        vmax = np.array([m["mean (current)"].max() for m in per_vid])
        vratio = np.array([m["pred/copy ratio"].mean() for m in per_vid])
        p(f"  video ranking Spearman: mean vs max agg = {_spearman(vmean, vmax):.3f}, "
          f"current vs ratio = {_spearman(vmean, vratio):.3f}")

    starts = np.concatenate([v["start_sec"] for v in vids])
    names = [vids[i]["name"] for i in groups]
    for name in ("mean (current)", "pred/copy ratio"):
        order = np.argsort(-metrics[name])
        p(f"\n[7] Top / bottom windows by {name} (video, start_sec, value, copy error)")
        for tag, sel in (("top", order[: args.top]), ("bottom", order[-args.top:][::-1])):
            for i in sel:
                p(f"  {tag:<6} {names[i][:40]:<40} {starts[i]:8.1f}s "
                  f"{metrics[name][i]:8.4f} {motion[i]:8.4f}")

    report = "\n".join(lines)
    print(report)
    (out_dir / "summary.txt").write_text(report + "\n")


def main() -> None:
    # No ego_curation imports here: the package pulls in torch, which summarize must not need.
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="dump per-token errors (needs GPU + checkpoint)")
    c.add_argument("videos", nargs="+")
    c.add_argument("--out-dir", required=True)
    c.add_argument("--model-size", choices=("giant", "gigantic"), required=True)
    c.add_argument("--sample-fps", type=float, default=4.0)
    c.add_argument("--context-frames", type=int, default=32)
    c.add_argument("--target-frames", type=int, default=16)
    c.add_argument("--stride-duration", type=float, default=None)
    c.add_argument("--n-videos", type=int, default=None)
    c.add_argument("--device", default=None)
    c.set_defaults(func=collect)

    s = sub.add_parser("summarize", help="text report from the dumps")
    s.add_argument("dump_dir")
    s.add_argument("--top", type=int, default=8, help="windows listed in [7]")
    s.add_argument("--max-videos", type=int, default=40, help="videos listed in [6]")
    s.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
