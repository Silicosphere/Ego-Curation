"""Extract top surprising segments from a segment CSV into clip files via ffmpeg for each CSV file in the segments folder.

Usage:
    python -m ego_curation.extract segments/* \
        --top-segments 20 --output clips/
"""
import argparse
import shutil
import subprocess
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract top segments from a segment CSV into clip files"
    )
    parser.add_argument(
        "csv", nargs="+", help="Per-video segment CSVs produced by --segments-dir")
    parser.add_argument(
        "--top-segments",
        type=int,
        default=None,
        help="Extract only the top N segments (default: all)",
    )
    parser.add_argument(
        "--output", "-o", default="clips", help="Output directory for clips"
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Override video path (default: from CSV 'video' column)",
    )
    parser.add_argument(
        "--accurate",
        action="store_true",
        help="Re-encode for frame-accurate cuts (slower; default is fast -c copy)",
    )
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="ffmpeg executable path (default: ffmpeg)"
    )
    return parser


def extract(
    csv_paths: list[str],
    output_dir: str = "clips",
    top_segments: int | None = None,
    video: str | None = None,
    accurate: bool = False,
    ffmpeg: str = "ffmpeg",
) -> None:
    if shutil.which(ffmpeg) is None:
        raise SystemExit(
            f"ffmpeg not found on PATH ('{ffmpeg}'). Install it or pass --ffmpeg.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in csv_paths:

        df = pd.read_csv(csv_path).sort_values(
            "surprise", ascending=False).reset_index(drop=True)
        if top_segments:
            df = df.head(top_segments)

        cur_video = video if video else str(df["video"].iloc[0])
        video_path = Path(cur_video)
        csv_out_dir = out_dir / video_path.stem[:8]
        csv_out_dir.mkdir(parents=True, exist_ok=True)
        if not video_path.exists():
            raise SystemExit(f"Video not found: {video_path}")

        for row in tqdm(df.itertuples(index=False), desc="Extracting", total=len(df)):
            start = float(row.start_sec)
            duration = float(row.end_sec) - start
            out_file = csv_out_dir / \
                f"{video_path.stem}_seg{row.rank:03d}_{start:07.2f}s.mp4"

            cmd = [ffmpeg, "-y"]
            if not accurate:
                cmd += ["-ss", f"{start:.3f}"]
            cmd += ["-i", str(video_path)]
            if accurate:
                cmd += ["-ss", f"{start:.3f}",
                        "-c:v", "libx264", "-c:a", "aac"]
            else:
                cmd += ["-c", "copy"]
            cmd += ["-t", f"{duration:.3f}", str(out_file)]

            subprocess.run(cmd, check=True, capture_output=True)

        print(f"Extracted {len(df)} clips to {csv_out_dir}")


def main(args_list: list[str] | None = None) -> None:
    args = build_parser().parse_args(args_list)
    extract(
        csv_paths=args.csv,
        output_dir=args.output,
        top_segments=args.top_segments,
        video=args.video,
        accurate=args.accurate,
        ffmpeg=args.ffmpeg,
    )


if __name__ == "__main__":
    main()
