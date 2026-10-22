"""PCA view of V-JEPA 2.1 patch features next to the frames they encode.

The three leading principal components of the patch tokens are mapped to RGB, so
patches the model represents alike get similar colours. One basis is fit on the
encoder features of the whole window and reused for every time slot and for the
predictor's output, so colours are comparable across time and between the
encoded and the predicted target. A last row shows the per-patch prediction
error, whose mean is the window's surprise score.

    python -m ego_curation.visualization video.mp4 --start-sec 12 30.5 \
        --out-dir pca_plots
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

from ego_curation.config import (
    DEFAULT_MODEL_SIZE,
    DEFAULT_SAMPLE_FPS,
    VJEPA_MODELS,
    SurpriseConfig,
)
from ego_curation.model import (
    CROP_SIZE,
    PATCH_SIZE,
    TUBELET,
    _MEAN,
    _STD,
    encode,
    load_jepa2,
    normalize_levels,
    predict_target,
)
from ego_curation.sampling import get_fps, sample_indices

GRID = CROP_SIZE // PATCH_SIZE

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"


def to_grid(tokens: torch.Tensor, n_slots: int) -> torch.Tensor:
    """(N, D) tokens in (time, row, col) order -> (n_slots, GRID, GRID, D)."""
    return tokens.reshape(n_slots, GRID, GRID, tokens.shape[-1])


def select_level(model, features: torch.Tensor, level: int | None) -> torch.Tensor:
    """One distillation level of the concatenated features, or all of them (None)."""
    if level is None:
        return features
    return features.split(model.encoder.embed_dim, dim=-1)[level]


def display_frames(window_proc: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalisation: (1, C, T, H, W) -> (T, H, W, C) in [0, 1].

    These are the cropped frames the model saw, so they line up with the patches.
    """
    x = window_proc[0].float().permute(1, 0, 2, 3).cpu()
    return (x * _STD + _MEAN).clamp(0, 1).permute(0, 2, 3, 1)


@dataclass
class RGBProjection:
    """Projection of features onto 3 principal components, rescaled to [0, 1]."""

    mean: torch.Tensor
    components: torch.Tensor  # (D, 3)
    lo: torch.Tensor
    hi: torch.Tensor

    @classmethod
    def fit(cls, features: torch.Tensor, clip: float = 0.01) -> "RGBProjection":
        """Fit on (..., D) features. Each component is scaled so that its
        `clip` and `1 - clip` quantiles map to 0 and 1, which keeps a few outlier
        patches from washing out the colours."""
        x = features.reshape(-1, features.shape[-1]).float().cpu()
        mean = x.mean(0)
        # Randomized PCA: only 3 of up to 6656 directions are needed. Seeded
        # locally so reruns give the same colours.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            _, _, v = torch.pca_lowrank(x - mean, q=min(8, *x.shape), niter=4, center=False)
        components = v[:, :3]
        # Component signs are arbitrary; fix them by the largest loading.
        peak = components.abs().argmax(0)
        components = components * components[peak, torch.arange(3)].sign()

        proj = (x - mean) @ components
        lo = torch.quantile(proj, clip, dim=0)
        hi = torch.quantile(proj, 1 - clip, dim=0)
        return cls(mean, components, lo, hi)

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """(..., D) features -> (..., 3) colours in [0, 1]."""
        proj = (features.float().cpu() - self.mean) @ self.components
        return ((proj - self.lo) / (self.hi - self.lo).clamp_min(1e-6)).clamp(0, 1)


def plot_pca(
    frames: torch.Tensor,
    maps: dict[str, torch.Tensor],
    slots: list[int],
    n_context_slots: int,
    slot_times: list[float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Frames on top, one row per entry of `maps` below, one column per slot.

    `frames` is (T, H, W, 3); slot s is shown with frame TUBELET * s, the first
    frame of its tubelet. A map is (n, GRID, GRID, 3) for colours or
    (n, GRID, GRID) for a scalar heatmap. A map with fewer slots than the window
    covers its last slots (the target), and is left blank for the others.
    """
    n_slots = frames.shape[0] // TUBELET
    n_rows = 1 + len(maps)
    fig, axes = plt.subplots(
        n_rows, len(slots), figsize=(2.4 * len(slots), 2.5 * n_rows),
        facecolor=SURFACE, squeeze=False,
    )
    vmax = {k: m.max().item() for k, m in maps.items() if m.ndim == 3}

    for col, s in enumerate(slots):
        part = "context" if s < n_context_slots else "target"
        label = f"{part} · slot {s}"
        if slot_times is not None:
            label += f"\n{slot_times[s]:.2f} s"
        axes[0, col].imshow(frames[TUBELET * s].numpy())
        axes[0, col].set_title(label, fontsize=9, color=INK if part == "target" else MUTED)

        for row, (name, m) in enumerate(maps.items(), start=1):
            i = s - (n_slots - m.shape[0])
            if i < 0:
                continue
            if m.ndim == 3:
                axes[row, col].imshow(m[i].numpy(), cmap="magma", vmin=0,
                                      vmax=vmax[name], interpolation="nearest")
            else:
                axes[row, col].imshow(m[i].numpy(), interpolation="nearest")

    for row, name in enumerate(["frame", *maps], start=0):
        axes[row, 0].set_ylabel(name, fontsize=10, color=INK)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def _pick_slots(n_context_slots: int, n_slots: int, context_cols: int, target_cols: int):
    """Evenly spaced slots from each part; the context always ends on its last slot."""
    ctx = np.linspace(0, n_context_slots - 1, context_cols) if context_cols else []
    tgt = np.linspace(n_context_slots, n_slots - 1, target_cols) if target_cols else []
    return sorted({int(round(s)) for s in (*ctx, *tgt)})


@torch.no_grad()
def visualize_window(
    model,
    processor,
    video_file: str,
    start_sec: float,
    config: SurpriseConfig,
    level: int | None = None,
    context_cols: int = 2,
    target_cols: int = 4,
) -> plt.Figure:
    """PCA figure for the window starting at `start_sec`, sampled as the pipeline does.

    `level` picks one distillation level (0..3, -1 for the last and most
    semantic one); None uses all of them, the space the surprise is scored in.
    """
    vr = VideoDecoder(video_file)
    fps = get_fps(vr)
    n_total = config.context_frames + config.target_frames
    idx = sample_indices(
        int(round(start_sec * fps)), n_total, fps / config.sample_fps, len(vr)
    )
    frames = vr.get_frames_at(indices=idx.tolist()).data
    device = next(model.parameters()).device
    window_proc = processor(frames.to(device))
    del frames, vr

    ctx_emb = encode(model, window_proc[:, :, : config.context_frames])
    true = normalize_levels(model, encode(model, window_proc))[0]
    n_ctx = ctx_emb.shape[1]
    pred = predict_target(model, ctx_emb, true.shape[0] - n_ctx)[0].float()

    n_slots = n_total // TUBELET
    n_context_slots = config.context_frames // TUBELET
    true = to_grid(select_level(model, true, level).cpu(), n_slots)
    pred = to_grid(select_level(model, pred, level).cpu(), n_slots - n_context_slots)
    error = (pred - true[n_context_slots:]).abs().mean(-1)

    project = RGBProjection.fit(true)
    maps = {"encoder PCA": project(true), "predicted PCA": project(pred), "|error|": error}
    slots = _pick_slots(n_context_slots, n_slots, context_cols, target_cols)
    slot_times = [start_sec + TUBELET * s / config.sample_fps for s in range(n_slots)]
    level_name = "all levels" if level is None else f"level {level}"
    title = (f"{Path(video_file).name} @ {start_sec:g} s  ·  {level_name}  ·  "
             f"mean |error| {error.mean().item():.4f}")
    fig = plot_pca(display_frames(window_proc), maps, slots, n_context_slots,
                   slot_times, title)

    del window_proc, ctx_emb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video", help="Video file")
    parser.add_argument(
        "--start-sec", type=float, nargs="+", required=True,
        help="Window start time(s) in seconds, e.g. start_sec from a segment CSV",
    )
    parser.add_argument("--out-dir", required=True, help="Directory for the PNGs")
    parser.add_argument(
        "--model-size", choices=list(VJEPA_MODELS.keys()), default=DEFAULT_MODEL_SIZE,
        help="V-JEPA 2.1 model size (default: %(default)s)",
    )
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS,
                        help="(default: %(default)s)")
    parser.add_argument("--context-frames", type=int, default=32,
                        help="(default: %(default)s)")
    parser.add_argument("--target-frames", type=int, default=16,
                        help="(default: %(default)s)")
    parser.add_argument(
        "--level", type=int, default=None,
        help="Distillation level to show (0-3, -1 = last); default: all levels",
    )
    parser.add_argument("--context-cols", type=int, default=2, help="(default: %(default)s)")
    parser.add_argument("--target-cols", type=int, default=4, help="(default: %(default)s)")
    parser.add_argument("--dpi", type=int, default=150, help="(default: %(default)s)")
    parser.add_argument("--device", default=None,
                        help="Device override (default: cuda if available else cpu)")
    return parser


def main(args_list: list[str] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")

    args = build_parser().parse_args(args_list)
    config = SurpriseConfig(
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        sample_fps=args.sample_fps,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model, processor = load_jepa2(
        args.model_size, device, num_frames=args.context_frames + args.target_frames
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for start in args.start_sec:
        fig = visualize_window(
            model, processor, args.video, start, config,
            level=args.level, context_cols=args.context_cols, target_cols=args.target_cols,
        )
        out_path = out_dir / f"{Path(args.video).stem}_{start:07.2f}s_pca.png"
        fig.savefig(out_path, dpi=args.dpi, facecolor=SURFACE)
        plt.close(fig)
        print(f"{start:g} s → {out_path}")


if __name__ == "__main__":
    main()
