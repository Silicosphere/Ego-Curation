"""Write top / low / random training lists of packed clips, in the V-JEPA CSV format.

Reads the surprise scores from the segment CSVs of a scoring run (one folder per fold,
segments<N>/) and points every selected segment at its clip inside ego4d_clips/clips<N>.h5,
the file written by pack_hdf5.sh. The clip names follow extract.py:
    <video>/<video>_seg<rank:03d>_<start_sec:07.2f>s

Output (space-separated, no header, as the official V-JEPA loader expects):
    /data/.../ego4d_clips/clips1.h5::<video>/<clip> 0

  ego4d_top.csv     the most surprising fraction of segments
  ego4d_low.csv     the least surprising fraction
  ego4d_random.csv  a random fraction (the baseline), same size

By default the fraction is taken within each video, so every video contributes equally and
the three lists differ only in which moments they keep. --global ranks all segments together.
Only folds whose .h5 exists are used, and only clips present in it.

Usage (from the repository root, with h5py installed):
    python scripts/train/make_splits.py runs_results/run_1184 \\
        --clips-dir /data/asem.a.abdelaziz/gp-2027a/ego4d_clips --fraction 0.25 --out-dir lists/
"""
import argparse
import re
from pathlib import Path

import h5py
import pandas as pd

FOLD_DIR = re.compile(r"segments(\d+)$")


def load_segments(run_dir: Path, clips_dir: Path) -> pd.DataFrame:
    frames = []
    for seg_dir in sorted(run_dir.glob("segments*")):
        match = FOLD_DIR.search(seg_dir.name)
        if not match:
            continue
        fold = int(match.group(1))
        h5_path = clips_dir / f"clips{fold}.h5"
        if not h5_path.exists():
            print(f"fold {fold}: {h5_path} not found, skipped")
            continue

        df = pd.concat(pd.read_csv(f) for f in sorted(seg_dir.glob("*_segments.csv")))
        stem = df["video"].map(lambda v: Path(v).stem)
        df["key"] = (stem + "/" + stem + "_seg" + df["rank"].map("{:03d}".format)
                     + "_" + df["start_sec"].map("{:07.2f}".format) + "s")
        # Keep only clips that were actually extracted and packed.
        with h5py.File(h5_path, "r") as h5:
            present = df["key"].map(lambda k: k in h5)
        if (~present).any():
            print(f"fold {fold}: {(~present).sum()} segments have no clip in {h5_path.name}")
        df = df[present]
        df["path"] = f"{h5_path}::" + df["key"]
        print(f"fold {fold}: {len(df)} clips from {df['video'].nunique()} videos")
        frames.append(df)

    if not frames:
        raise SystemExit(f"No packed folds found in {clips_dir}")
    return pd.concat(frames, ignore_index=True)


def select(df: pd.DataFrame, fraction: float, how: str, global_: bool, seed: int) -> pd.DataFrame:
    """Keep `fraction` of the segments (at least one per group), per video unless global_."""
    if how == "random":
        ordered = df.sample(frac=1, random_state=seed)
    else:
        ordered = df.sort_values("surprise", ascending=(how == "low"), kind="stable")
    groups = ordered.assign(_all=0).groupby("_all" if global_ else "video", sort=False)
    position = groups.cumcount()
    quota = groups["surprise"].transform("size").mul(fraction).round().clip(lower=1)
    return ordered[position < quota]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="Scoring run with segments<N>/ folders")
    parser.add_argument("--clips-dir", type=Path, required=True,
                        help="Folder holding clips<N>.h5")
    parser.add_argument("--fraction", type=float, default=0.33,
                        help="Fraction of segments in each list (default: 0.33)")
    parser.add_argument("--global", dest="global_", action="store_true",
                        help="Rank all segments together instead of within each video")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random list")
    parser.add_argument("--out-dir", type=Path, default=Path("."),
                        help="Where to write the CSVs")
    parser.add_argument("--prefix", default="ego4d", help="File name prefix (default: ego4d)")
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in (0, 1]")

    df = load_segments(args.run_dir, args.clips_dir.resolve())
    if df["path"].str.contains(" ").any():
        raise SystemExit("Some clip paths contain spaces, which break the space-separated "
                         "CSV; rename those videos first.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for how in ("top", "low", "random"):
        chosen = select(df, args.fraction, how, args.global_, args.seed)
        out = args.out_dir / f"{args.prefix}_{how}.csv"
        chosen.assign(label=0)[["path", "label"]].to_csv(out, sep=" ", header=False, index=False)
        print(f"{out}: {len(chosen)} clips, mean surprise {chosen['surprise'].mean():.4f}")


if __name__ == "__main__":
    main()
