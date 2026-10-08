"""Plot the surprise curve of each per-video segment CSV.

One PNG per CSV, named after the first --name-chars characters of the CSV's
file name (also the plot title), with the window centre time on the x axis and the surprise
score on the y axis. Windows missing from the CSV (e.g. when it was written
with --top-segments) leave a gap in the line instead of being bridged.

    python visualize_results.py segments/*.csv --out-dir plots
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED = ("start_sec", "end_sec", "surprise")
MAX_MARKERS = 100

SURFACE = "#fcfcfb"
LINE = "#2a78d6"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("csv", nargs="+", help="Segment CSV(s) written by --segments-dir")
    parser.add_argument("--out-dir", required=True, help="Directory for the PNG plots")
    parser.add_argument(
        "--name-chars",
        type=int,
        default=7,
        help="Characters of the CSV file name used to name and title each plot "
        "(default: %(default)s)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="(default: %(default)s)")
    return parser


def _with_gaps(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Window centres and scores in time order, with NaN where windows are missing."""
    df = df.sort_values("start_sec")
    t = ((df["start_sec"] + df["end_sec"]) / 2).to_numpy()
    y = df["surprise"].to_numpy(dtype=float)
    if len(t) < 2:
        return t, y
    stride = np.diff(t).min()
    gap = np.flatnonzero(np.diff(t) > 1.5 * stride) + 1
    return np.insert(t, gap, np.nan), np.insert(y, gap, np.nan)


def plot_csv(csv_path: Path, name: str, out_dir: Path, dpi: int) -> Path:
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing column(s) {', '.join(missing)}")
    if df.empty:
        raise ValueError("no rows")

    t, y = _with_gaps(df)
    peak = df.loc[df["surprise"].idxmax()]

    fig, ax = plt.subplots(figsize=(10, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Markers only while they stay readable; long videos have hundreds of windows.
    marker = "o" if len(df) <= MAX_MARKERS else None
    ax.plot(t, y, color=LINE, linewidth=2, marker=marker, markersize=4, solid_capstyle="round")
    # Anchor the label on the side that keeps it inside the axes.
    peak_t = (peak["start_sec"] + peak["end_sec"]) / 2
    pos = (peak_t - np.nanmin(t)) / max(np.nanmax(t) - np.nanmin(t), 1e-9)
    ha = "left" if pos < 0.15 else "right" if pos > 0.85 else "center"
    ax.annotate(
        f"max {peak['surprise']:.4f} at {peak['start_sec']:g}–{peak['end_sec']:g} s",
        xy=(peak_t, peak["surprise"]),
        xytext=(0, 8), textcoords="offset points", ha=ha, fontsize=9, color=INK,
    )

    ax.set_title(name, loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_title(csv_path.name, loc="right", fontsize=8, color=MUTED)
    ax.set_xlabel("time (s, window centre)", color=MUTED)
    ax.set_ylabel("surprise", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.margins(x=0.01, y=0.15)
    fig.tight_layout()

    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out_path


def main(args_list: list[str] | None = None) -> None:
    args = build_parser().parse_args(args_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written, used = 0, set()
    for csv_path in map(Path, args.csv):
        name = csv_path.stem[: args.name_chars]
        if name in used:
            print(f"{csv_path}: '{name}' is already taken, using the full name")
            name = csv_path.stem
        used.add(name)
        try:
            print(f"{csv_path} → {plot_csv(csv_path, name, out_dir, args.dpi)}")
            written += 1
        except (ValueError, OSError, pd.errors.ParserError) as e:
            print(f"skipped {csv_path}: {e}")
    print(f"\n{written}/{len(args.csv)} plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
