"""Diagnostics for the surprise metric.

collect   : run the pipeline's windows and dump per-token, per-level errors (GPU).
summarize : turn the dumps into a text report (numpy only, runs anywhere).
pairs     : IntPhys2 possible/impossible pair analysis of the dumps (numpy only).

    python diagnose.py collect VIDEO... --out-dir DIR --model-size giant \
        --sample-fps 4 --context-frames 16 --target-frames 8 --stride-duration 6
    python diagnose.py summarize DIR
    python diagnose.py pairs DIR --metadata IntPhys2_Data/Main/metadata.csv
"""
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

ARRAYS = ("err_pred", "err_copy", "err_zero", "pred_rms", "cos")
FROZEN_COPY_ERR = 1e-3


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
        args.model_size,
        device,
        num_frames=args.context_frames + args.target_frames,
        online_encoder=args.context_encoder == "online",
    )
    # The official IntPhys2 eval feeds the predictor from the online encoder;
    # the pipeline uses the EMA target encoder for both context and targets.
    ctx_net = model.online_encoder if args.context_encoder == "online" else model.encoder
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

            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16):
                ctx_emb = ctx_net(window_proc[:, :, : config.context_frames])
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
            # How much the context features change when the target frames are visible
            # (with the online context encoder this also includes the encoder difference).
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
            fps=fps,
            n_frames=num_frames,
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
    # Floor the denominator: frozen windows would give inf.
    m["pred/copy ratio"] = ep.mean((1, 2)) / np.maximum(ec.mean((1, 2)), FROZEN_COPY_ERR)
    m[f"pred/copy ratio, level {L}"] = ep[..., -1].mean(1) / np.maximum(
        ec[..., -1].mean(1), FROZEN_COPY_ERR
    )
    m["1 - cos"] = 1 - v["cos"].reshape(W, -1).mean(1)
    return m


def summarize(args) -> None:
    out_dir = Path(args.dump_dir)
    files = sorted(out_dir.glob("*.npz"))
    if not files:
        raise SystemExit(f"no .npz dumps in {out_dir}")
    meta = json.loads((out_dir / "meta.json").read_text())
    L, g = meta["levels"], meta["grid"]

    keys = ARRAYS + ("ctx_shift", "score", "start_sec")
    vids, frozen = [], {}
    for f in files:
        with np.load(f) as z:
            v = {k: z[k].astype(np.float32) for k in keys}
        # Identical frames (black/frozen video) make the copy baseline exact.
        still = v["err_copy"].reshape(len(v["score"]), -1).mean(1) < FROZEN_COPY_ERR
        if still.any():
            frozen[f.stem] = (v["start_sec"][still], len(still))
        v = {k: x[~still] for k, x in v.items()}
        v["name"] = f.stem
        if len(v["score"]):
            vids.append(v)

    cat = {k: np.concatenate([v[k] for v in vids]) for k in ARRAYS + ("ctx_shift", "score")}
    W, T = cat["err_pred"].shape[:2]
    lines = []
    p = lines.append

    p(f"model={meta['model_size']} fps={meta['sample_fps']} ctx={meta['context_frames']} "
      f"tgt={meta['target_frames']} stride={meta['stride_duration']}")
    p(f"videos={len(vids)} windows={W} target_slots={T} grid={g}x{g} levels={L}")

    n_frozen = sum(len(s) for s, _ in frozen.values())
    p(f"\n[0] Frozen windows (copy error < {FROZEN_COPY_ERR:g}), excluded below: {n_frozen}")
    for name, (s, n) in frozen.items():
        shown_s = ", ".join(f"{x:g}" for x in s[:12]) + (" ..." if len(s) > 12 else "")
        p(f"  {name[:40]:<40} {len(s):>4}/{n:<4} at s = {shown_s}")

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
    slope, icpt = np.polyfit(np.log(motion), np.log(current), 1)
    # exp keeps it positive (about 1 on average), so the CV column stays meaningful.
    resid = np.log(current) - (slope * np.log(motion) + icpt)
    metrics["motion-adjusted"] = np.exp(resid)
    p(f"\n[5] Window-level metrics ({W} windows)")
    p("  CV = std/mean across windows; eta2 = share of variance between videos;")
    p("  r(copy) = Pearson with copy-baseline error (motion proxy); rho(cur) = Spearman with current")
    p(f"  motion-adjusted = current / fitted(copy); fit: log cur = {slope:.3f} log copy + {icpt:.3f}")
    p(f"{'metric':<28} {'mean':>8} {'std':>8} {'CV':>6} {'eta2':>6} {'r(copy)':>8} {'rho(cur)':>9}")
    for name, x in metrics.items():
        p(f"{name:<28} {x.mean():8.4f} {x.std():8.4f} {x.std() / x.mean():6.3f} "
          f"{_eta2(x, groups):6.3f} {_pearson(x, motion):8.3f} {_spearman(x, current):9.3f}")
    p(f"  check: stored pipeline score vs recomputed mean, max abs diff = "
      f"{np.abs(cat['score'] - current).max():.2e}")

    p("\n[6] Per video (window means; 'max' = max window of current; 'copy' = mean copy error)")
    shown = ("mean (current)", "motion-adjusted", "pred/copy ratio", f"level {L} mean")
    vm = {s: np.array([metrics[s][groups == i].mean() for i in range(len(vids))]) for s in shown}
    vmax = np.array([current[groups == i].max() for i in range(len(vids))])
    vmotion = np.array([motion[groups == i].mean() for i in range(len(vids))])
    p(f"{'video':<40} {'win':>4} " + " ".join(f"{s[:14]:>14}" for s in shown)
      + f" {'max':>8} {'copy':>8}")
    for i in np.argsort(-vm["mean (current)"])[: args.max_videos]:
        p(f"{vids[i]['name'][:40]:<40} {len(vids[i]['score']):>4} "
          + " ".join(f"{vm[s][i]:14.4f}" for s in shown)
          + f" {vmax[i]:8.4f} {vmotion[i]:8.4f}")
    if len(vids) > 2:
        cur = vm["mean (current)"]
        p(f"  video ranking Spearman with current: max agg = {_spearman(cur, vmax):.3f}, "
          f"motion-adjusted = {_spearman(cur, vm['motion-adjusted']):.3f}, "
          f"ratio = {_spearman(cur, vm['pred/copy ratio']):.3f}, "
          f"copy (motion) = {_spearman(cur, vmotion):.3f}")

    starts = np.concatenate([v["start_sec"] for v in vids])
    names = [vids[i]["name"] for i in groups]
    for name in ("mean (current)", "motion-adjusted"):
        order = np.argsort(-metrics[name])
        p(f"\n[7] Top / bottom windows by {name} (video, start_sec, value, copy error)")
        for tag, sel in (("top", order[: args.top]), ("bottom", order[-args.top:][::-1])):
            for i in sel:
                p(f"  {tag:<6} {names[i][:40]:<40} {starts[i]:8.1f}s "
                  f"{metrics[name][i]:8.4f} {motion[i]:8.4f}")

    report = "\n".join(lines)
    print(report)
    (out_dir / "summary.txt").write_text(report + "\n")


# ── pairs ────────────────────────────────────────────────────────────────────

GROUP_BY = ("condition", "Camera", "Difficulty")
EVENT_OFFSETS = range(-4, 9)


def _acc(diff):
    """Share of pairs with impossible > possible; ties count half."""
    diff = np.asarray(diff)
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def _auroc(imp, pos):
    """P(an impossible video scores above a possible one); ties count half."""
    return _acc(np.asarray(imp)[:, None] - np.asarray(pos)[None, :])


def _load_curves(path, L):
    """Per-window metric curves, the flat copy-error maps and window start times."""
    with np.load(path) as z:
        v = {k: z[k].astype(np.float32) for k in ("err_pred", "err_copy", "cos", "start_sec")}
    W = len(v["start_sec"])
    m = _window_metrics(v, L)
    m["copy error (motion)"] = v["err_copy"].reshape(W, -1).mean(1)
    m["pred - copy"] = m["mean (current)"] - m["copy error (motion)"]
    return m, v["err_copy"].reshape(W, -1), v["start_sec"]


def pairs(args) -> None:
    out_dir = Path(args.dump_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    L = meta["levels"]
    ctx_sec = meta["context_frames"] / meta["sample_fps"]
    with open(args.metadata, newline="") as f:
        rows = list(csv.DictReader(f))

    # A matched pair: same SceneIndex and type prefix ("1_" / "2_"), one of each label.
    groups = {}
    for r in rows:
        prefix, label = r["type"].split("_", 1)
        groups.setdefault((r["SceneIndex"], prefix), {})[label.lower()] = r

    recs, missing, incomplete = [], 0, 0
    for (scene, prefix), g in sorted(groups.items()):
        if set(g) != {"possible", "impossible"}:
            incomplete += 1
            continue
        paths = {k: out_dir / f"{Path(r['file_name']).stem}.npz" for k, r in g.items()}
        if not all(x.exists() for x in paths.values()):
            missing += 1
            continue
        (mp, cp, sp), (mi, ci, _) = (_load_curves(paths[k], L) for k in ("possible", "impossible"))
        W = min(len(cp), len(ci))
        # The two videos share the scene, so their copy-error maps match until the
        # event enters a target; the first window past half the peak marks the onset.
        div = np.abs(cp[:W] - ci[:W]).mean(1)
        onset = int(np.argmax(div >= 0.5 * div.max()))
        recs.append({
            "scene": scene, "prefix": prefix, "row": g["possible"], "W": W,
            "onset": onset, "onset_sec": float(sp[onset]) + ctx_sec, "div": div,
            "p": {k: x[:W] for k, x in mp.items()},
            "i": {k: x[:W] for k, x in mi.items()},
        })
    if not recs:
        raise SystemExit(f"no complete pairs with dumps in {out_dir}")

    n = len(recs)
    names = list(recs[0]["p"])
    lines = []
    p = lines.append
    p(f"model={meta['model_size']} context_encoder={meta.get('context_encoder', 'target')} "
      f"fps={meta['sample_fps']} ctx={meta['context_frames']} tgt={meta['target_frames']} "
      f"stride={meta['stride_duration']}")
    Ws = np.array([r["W"] for r in recs])
    p(f"pairs={n} (missing dumps: {missing}, scenes without both labels: {incomplete}); "
      f"windows per pair: median {np.median(Ws):g}, min {Ws.min()}, max {Ws.max()}")

    p("\n[1] Pairwise accuracy = share of pairs where the impossible video scores higher")
    p("  max / mean = aggregate over aligned windows; onset = the window where the pair")
    p("  first diverges (event in target); AUROC = all impossible vs all possible videos,")
    p(f"  max aggregate, no pairing. Chance 0.5, 95% CI half-width ~{0.98 / np.sqrt(n):.3f}")
    p(f"{'metric':<28} {'max':>6} {'mean':>6} {'onset':>6} {'AUROC':>6}")
    for s in names:
        d_max = [r["i"][s].max() - r["p"][s].max() for r in recs]
        d_mean = [r["i"][s].mean() - r["p"][s].mean() for r in recs]
        d_on = [r["i"][s][r["onset"]] - r["p"][s][r["onset"]] for r in recs]
        auc = _auroc([r["i"][s].max() for r in recs], [r["p"][s].max() for r in recs])
        p(f"{s:<28} {_acc(d_max):6.3f} {_acc(d_mean):6.3f} {_acc(d_on):6.3f} {auc:6.3f}")

    shown = ("mean (current)", f"level {L} mean", "pred/copy ratio", "pred - copy",
             "copy error (motion)")
    for key in GROUP_BY:
        p(f"\n[2] Pairwise accuracy by {key} (max aggregate)")
        p(f"{key:<16} {'n':>4} " + " ".join(f"{s[:14]:>14}" for s in shown))
        for val in sorted({r["row"].get(key, "?") for r in recs}):
            sel = [r for r in recs if r["row"].get(key, "?") == val]
            cells = [_acc([r["i"][s].max() - r["p"][s].max() for r in sel]) for s in shown]
            p(f"{val[:16]:<16} {len(sel):>4} " + " ".join(f"{c:14.3f}" for c in cells))

    locked = ("mean (current)", f"level {L} mean", "pred - copy", "copy error (motion)")
    p("\n[3] Event-locked difference, impossible - possible: mean / share > 0")
    p(f"  offset in windows ({meta['stride_duration']} s each) from the onset window")
    p(f"{'offset':>7} {'n':>5} " + " ".join(f"{s[:16]:>16}" for s in locked))
    for k in EVENT_OFFSETS:
        sel = [r for r in recs if 0 <= r["onset"] + k < r["W"]]
        if not sel:
            continue
        cells = []
        for s in locked:
            d = np.array([r["i"][s][r["onset"] + k] - r["p"][s][r["onset"] + k] for r in sel])
            cells.append(f"{d.mean():+.4f}/{(d > 0).mean():.2f}")
        p(f"{k:>+7} {len(sel):>5} " + " ".join(f"{c:>16}" for c in cells))

    onset_sec = np.array([r["onset_sec"] for r in recs])
    at_start = sum(r["onset"] == 0 for r in recs)
    flat = sum(r["div"].max() < FROZEN_COPY_ERR for r in recs)
    pre = [r["div"][: r["onset"]].mean() / r["div"].max() for r in recs if r["onset"] > 0]
    p("\n[4] Divergence onset (start of the first target that differs within the pair)")
    p(f"  onset sec percentiles 10/50/90: "
      + "/".join(f"{x:.1f}" for x in np.percentile(onset_sec, [10, 50, 90])))
    p(f"  onset in the first window (event may precede the first target): {at_start}/{n}")
    p(f"  pairs whose dumps never diverge (max div < {FROZEN_COPY_ERR:g}): {flat}/{n}")
    if pre:
        p(f"  pre-onset divergence / peak, median: {np.median(pre):.3f} (near 0 = sharp onset)")

    p(f"\n[5] Per pair (cur = mean (current); d = impossible - possible at the onset window)")
    p(f"{'scene':<8} {'pfx':>3} {'condition':<12} {'camera':<7} {'diff':<7} {'W':>3} "
      f"{'onset s':>7} {'cur max p':>9} {'cur max i':>9} {'d cur':>8} {'d p-c':>8}")
    for r in recs[: args.max_pairs]:
        row, o = r["row"], r["onset"]
        d_cur = r["i"]["mean (current)"][o] - r["p"]["mean (current)"][o]
        d_pc = r["i"]["pred - copy"][o] - r["p"]["pred - copy"][o]
        p(f"{r['scene'][:8]:<8} {r['prefix']:>3} {row.get('condition', '?')[:12]:<12} "
          f"{row.get('Camera', '?')[:7]:<7} {row.get('Difficulty', '?')[:7]:<7} {r['W']:>3} "
          f"{r['onset_sec']:7.1f} {r['p']['mean (current)'].max():9.4f} "
          f"{r['i']['mean (current)'].max():9.4f} {d_cur:+8.4f} {d_pc:+8.4f}")

    report = "\n".join(lines)
    print(report)
    (out_dir / "pairs.txt").write_text(report + "\n")


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
    c.add_argument(
        "--context-encoder", choices=("target", "online"), default="target",
        help="encoder for the predictor's context: EMA target encoder (pipeline) or "
        "online encoder (official IntPhys2 eval); targets always use the target encoder",
    )
    c.set_defaults(func=collect)

    s = sub.add_parser("summarize", help="text report from the dumps")
    s.add_argument("dump_dir")
    s.add_argument("--top", type=int, default=8, help="windows listed in [7]")
    s.add_argument("--max-videos", type=int, default=40, help="videos listed in [6]")
    s.set_defaults(func=summarize)

    q = sub.add_parser("pairs", help="IntPhys2 possible/impossible pair analysis")
    q.add_argument("dump_dir")
    q.add_argument("--metadata", required=True, help="IntPhys2 split metadata.csv")
    q.add_argument("--max-pairs", type=int, default=40, help="pairs listed in [5]")
    q.set_defaults(func=pairs)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
