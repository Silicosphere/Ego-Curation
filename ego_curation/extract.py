"""Extract the most surprising segments from per-video segment CSVs into clip files via ffmpeg.

Each CSV gets its own subdirectory of the output directory, named after the CSV's video.
By default every segment in each CSV is extracted, and the cut method is picked per video:
stream copy (fast) cuts can only start at a keyframe, so they are used only when the video's
keyframes are at most --max-keyframe-gap seconds apart (e.g. intra-only video). Otherwise the
clip is re-encoded for a frame-accurate cut.

Usage:
    python -m ego_curation.extract segments/*.csv --output clips/
    python -m ego_curation.extract segments/*.csv --top-segments 20 --video-root /data/videos --jobs 8
    python -m ego_curation.extract segments/*.csv --cut-mode reencode --video-codec h264_nvenc
"""
import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REQUIRED = ("video", "start_sec", "end_sec", "surprise")
PROBE_SECONDS = 60


def _positive_float(value: str) -> float:
    x = float(value)
    if x <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return x


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
        "--cut-mode",
        choices=("auto", "copy", "reencode"),
        default="auto",
        help="auto (default): stream copy if the video's keyframes are dense enough, "
        "otherwise re-encode. copy: always stream copy (fast; cuts snap to the previous "
        "keyframe). reencode: always re-encode (frame-accurate, slower)",
    )
    parser.add_argument(
        "--max-keyframe-gap",
        type=_positive_float,
        default=0.5,
        help="In auto mode, the largest keyframe spacing (s) at which stream copy is "
        "still used (default: 0.5)",
    )
    parser.add_argument(
        "--video-codec",
        default="libx264",
        help="Video encoder for re-encoding (default: libx264; h264_nvenc uses the GPU "
        "if your ffmpeg build supports it)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Drop the audio stream from the clips",
    )
    parser.add_argument(
        "--jobs", "-j",
        type=_positive_int,
        default=1,
        help="Number of ffmpeg processes to run in parallel (default: 1)",
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
    parser.add_argument(
        "--ffprobe", default="ffprobe", help="ffprobe executable path (default: ffprobe)"
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


def _max_keyframe_gap(ffprobe: str, video_path: Path) -> float | None:
    """Largest gap (s) between video keyframes in the first PROBE_SECONDS of the video.

    Reads packet flags only (no decoding), so it is fast. Returns None if it cannot tell.
    """
    cmd = [ffprobe, "-v", "error", "-select_streams", "v:0",
           "-read_intervals", f"%+{PROBE_SECONDS}",
           "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", str(video_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    key_times = []
    for line in proc.stdout.splitlines():
        pts, _, flags = line.partition(",")
        if "K" in flags and pts not in ("", "N/A"):
            key_times.append(float(pts))
    key_times.sort()
    if len(key_times) < 2:
        return None
    return max(b - a for a, b in zip(key_times, key_times[1:]))


def _use_reencode(cut_mode: str, ffprobe: str, video_path: Path, max_gap: float) -> bool:
    if cut_mode != "auto":
        return cut_mode == "reencode"
    gap = _max_keyframe_gap(ffprobe, video_path)
    if gap is None:
        print(f"  could not read keyframes of {video_path.name}; re-encoding")
        return True
    reencode = gap > max_gap
    print(f"  {video_path.name}: keyframes up to {gap:.2f} s apart -> "
          f"{'re-encode' if reencode else 'stream copy'}")
    return reencode


def _ffmpeg_cmd(
    ffmpeg: str, video_path: Path, start: float, duration: float, out_file: Path,
    reencode: bool, video_codec: str, no_audio: bool,
) -> list[str]:
    # -ss before -i seeks on input, so ffmpeg never decodes from the start of the video.
    # With re-encoding this is still frame-accurate (FFmpeg >= 2.1).
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(video_path),
           "-t", f"{duration:.3f}"]
    if reencode:
        cmd += ["-c:v", video_codec]
        if video_codec == "libx264":
            cmd += ["-preset", "veryfast", "-crf", "18"]
        cmd += ["-an"] if no_audio else ["-c:a", "aac"]
    else:
        cmd += ["-c", "copy", "-avoid_negative_ts", "make_zero"]
        if no_audio:
            cmd += ["-an"]
    return cmd + [str(out_file)]


def extract(
    csv_paths: list[str],
    output_dir: str = "clips",
    top_segments: int | None = None,
    video: str | None = None,
    video_root: str | None = None,
    cut_mode: str = "auto",
    max_keyframe_gap: float = 0.5,
    video_codec: str = "libx264",
    no_audio: bool = False,
    jobs: int = 1,
    no_overlap: bool = False,
    skip_existing: bool = False,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
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

        reencode = _use_reencode(cut_mode, ffprobe, video_path, max_keyframe_gap)
        csv_out_dir = out_dir / video_path.stem
        csv_out_dir.mkdir(parents=True, exist_ok=True)

        cmds = []
        n_csv_ok = 0
        for row in df.itertuples(index=False):
            start = float(row.start_sec)
            duration = float(row.end_sec) - start
            out_file = csv_out_dir / \
                f"{video_path.stem}_seg{int(row.rank):03d}_{start:07.2f}s.mp4"
            if skip_existing and out_file.exists():
                n_csv_ok += 1
                continue
            cmds.append(_ffmpeg_cmd(ffmpeg, video_path, start, duration, out_file,
                                    reencode, video_codec, no_audio))

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = pool.map(
                lambda cmd: subprocess.run(cmd, capture_output=True, text=True), cmds)
            for proc in tqdm(results, desc=video_path.stem[:16], total=len(cmds)):
                if proc.returncode == 0:
                    n_csv_ok += 1
                else:
                    n_failed += 1
                    tqdm.write(f"ffmpeg failed for {Path(proc.args[-1]).name}:\n"
                               f"{proc.stderr.strip()}")

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
        cut_mode=args.cut_mode,
        max_keyframe_gap=args.max_keyframe_gap,
        video_codec=args.video_codec,
        no_audio=args.no_audio,
        jobs=args.jobs,
        no_overlap=args.no_overlap,
        skip_existing=args.skip_existing,
        ffmpeg=args.ffmpeg,
        ffprobe=args.ffprobe,
    )


if __name__ == "__main__":
    main()
