"""Extract the most surprising segments from per-video segment CSVs into clip files via ffmpeg.

Each CSV gets its own subdirectory of the output directory, named after the CSV's video.
By default every segment in each CSV is extracted, re-encoded for frame-accurate cuts.

Usage:
    python -m ego_curation.extract segments/*.csv --output clips/
    python -m ego_curation.extract segments/*.csv --top-segments 20 --video-root /data/videos
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REQUIRED = ("video", "start_sec", "end_sec", "surprise")


def _positive_int(value: str) -> int:
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return n


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract top segments from segment CSVs into clip files"
    )
    parser.add_argument(
        "csv", nargs="+", help="Per-video segment CSVs produced by --segments-dir"
    )
    parser.add_argument(
        "--top-segments",
        type=_positive_int,
        default=None,
        help="Extract only the top N segments per CSV (default: all segments)",
    )
    parser.add_argument(
        "--output", "-o", default="clips", help="Output directory for clips"
    )
    video_src = parser.add_mutually_exclusive_group()
    video_src.add_argument(
        "--video",
        default=None,
        help="Override video path (only with a single CSV; default: CSV 'video' column)",
    )
    video_src.add_argument(
        "--video-root",
        default=None,
        help="Look up each video by file name in this directory instead of the CSV's "
        "absolute path (for CSVs written on another machine)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Stream-copy instead of re-encoding (much faster, but cuts snap to the "
        "previous keyframe, so clips can start seconds early)",
    )
    parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Skip segments that overlap a higher-ranked segment already selected "
        "(useful when --stride-duration < window duration)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not re-extract clips whose output file already exists",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="ffmpeg executable path (default: ffmpeg)"
    )
    return parser


def _select_segments(
    df: pd.DataFrame, top_segments: int | None, no_overlap: bool
) -> pd.DataFrame:
    """Sort by surprise (descending), optionally drop overlapping segments, keep top N."""
    df = df.sort_values("surprise", ascending=False, na_position="last")
    if "rank" not in df.columns:
        df.insert(0, "rank", range(1, len(df) + 1))
    if no_overlap:
        kept: list[tuple[float, float]] = []
        keep_idx = []
        for idx, start, end in zip(df.index, df["start_sec"], df["end_sec"]):
            if all(end <= s or start >= e for s, e in kept):
                kept.append((start, end))
                keep_idx.append(idx)
                if top_segments and len(keep_idx) >= top_segments:
                    break
        df = df.loc[keep_idx]
    if top_segments:
        df = df.head(top_segments)
    return df.reset_index(drop=True)


def _resolve_video(
    df: pd.DataFrame, video: str | None, video_root: str | None
) -> Path:
    if video:
        return Path(video)
    csv_video = Path(str(df["video"].iloc[0]))
    if video_root:
        return Path(video_root) / csv_video.name
    return csv_video


def _ffmpeg_cmd(
    ffmpeg: str, video_path: Path, start: float, duration: float, out_file: Path, fast: bool
) -> list[str]:
    # -ss before -i seeks on input. With re-encoding this is frame-accurate (FFmpeg >= 2.1)
    # and avoids decoding from the start of the video for every clip.
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video_path),
           "-t", f"{duration:.3f}"]
    if fast:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac"]
    return cmd + [str(out_file)]


def extract(
    csv_paths: list[str],
    output_dir: str = "clips",
    top_segments: int | None = None,
    video: str | None = None,
    video_root: str | None = None,
    fast: bool = False,
    no_overlap: bool = False,
    skip_existing: bool = False,
    ffmpeg: str = "ffmpeg",
) -> None:
    if shutil.which(ffmpeg) is None:
        raise SystemExit(
            f"ffmpeg not found on PATH ('{ffmpeg}'). Install it or pass --ffmpeg.")
    if video and len(csv_paths) > 1:
        raise SystemExit(
            "--video can only be used with a single CSV; use --video-root for several.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_failed = n_skipped_csv = 0
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if csv_path.suffix.lower() != ".csv":
            print(f"Skipping non-CSV file: {csv_path}")
            n_skipped_csv += 1
            continue

        df = pd.read_csv(csv_path)
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            print(f"Skipping {csv_path}: missing columns {missing}")
            n_skipped_csv += 1
            continue
        if df.empty:
            print(f"Skipping {csv_path}: no segments")
            n_skipped_csv += 1
            continue

        df = _select_segments(df, top_segments, no_overlap)
        video_path = _resolve_video(df, video, video_root)
        if not video_path.exists():
            print(f"Skipping {csv_path}: video not found: {video_path}")
            n_skipped_csv += 1
            continue

        csv_out_dir = out_dir / video_path.stem
        csv_out_dir.mkdir(parents=True, exist_ok=True)

        n_csv_ok = 0
        for row in tqdm(df.itertuples(index=False), desc=video_path.stem[:16], total=len(df)):
            start = float(row.start_sec)
            duration = float(row.end_sec) - start
            out_file = csv_out_dir / \
                f"{video_path.stem}_seg{int(row.rank):03d}_{start:07.2f}s.mp4"
            if skip_existing and out_file.exists():
                n_csv_ok += 1
                continue

            cmd = _ffmpeg_cmd(ffmpeg, video_path, start, duration, out_file, fast)
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                n_csv_ok += 1
            except subprocess.CalledProcessError as e:
                n_failed += 1
                tqdm.write(f"ffmpeg failed for {out_file.name}:\n{e.stderr.strip()}")

        n_ok += n_csv_ok
        print(f"Extracted {n_csv_ok}/{len(df)} clips to {csv_out_dir}")

    print(f"\nDone: {n_ok} clips extracted, {n_failed} failed, "
          f"{n_skipped_csv} CSV(s) skipped.")


def main(args_list: list[str] | None = None) -> None:
    args = build_parser().parse_args(args_list)
    extract(
        csv_paths=args.csv,
        output_dir=args.output,
        top_segments=args.top_segments,
        video=args.video,
        video_root=args.video_root,
        fast=args.fast,
        no_overlap=args.no_overlap,
        skip_existing=args.skip_existing,
        ffmpeg=args.ffmpeg,
    )


if __name__ == "__main__":
    main()
