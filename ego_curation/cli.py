import argparse
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import torch

from ego_curation.config import SurpriseConfig
from ego_curation.pipeline import calc_surprise_streaming
from ego_curation.model import load_jepa2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ego-Curation — rank egocentric videos by V-JEPA prediction surprise"
    )

    parser.add_argument("videos", nargs="+", help="Video file(s) to process")

    parser.add_argument(
        "--model",
        default="facebook/vjepa2-vitl-fpc64-256",
        help="V-JEPA model name (default: %(default)s)",
    )
    parser.add_argument(
        "--output", "-o", default="results.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--context-duration",
        type=float,
        default=2.0,
        help="Context window in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=1.0,
        help="Target window in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--context-frames",
        type=int,
        default=32,
        help="Frames sampled from context window (default: %(default)s)",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=16,
        help="Frames sampled from target window (default: %(default)s)",
    )
    parser.add_argument(
        "--stride-duration",
        type=float,
        default=None,
        help="Stride between windows in seconds (default: non-overlapping)",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "max"],
        default="mean",
        help="Aggregate per-window scores (default: %(default)s)",
    )
    parser.add_argument(
        "--segments-dir",
        default=None,
        help="Write a per-video CSV of all scored segments to this directory",
    )
    parser.add_argument(
        "--top-segments",
        type=int,
        default=None,
        help="Keep only the top N segments per video (requires --segments-dir)",
    )
    parser.add_argument(
        "--n-videos",
        type=int,
        default=None,
        help="Only process first N videos (default: all)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (default: cuda if available else cpu)",
    )
    return parser


def _build_segments_df(
    video_file: str,
    scores: list[float],
    starts: list[float],
    config: SurpriseConfig,
    top_segments: int | None,
) -> pd.DataFrame:
    window_duration = config.context_duration + config.target_duration
    df = pd.DataFrame(
        {
            "video": video_file,
            "start_sec": starts,
            "end_sec": [s + window_duration for s in starts],
            "surprise": scores,
        }
    ).sort_values("surprise", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    if top_segments:
        df = df.head(top_segments)
    return df


def main(args_list: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(args_list)

    if args.top_segments is not None and args.segments_dir is None:
        parser.error("--top-segments requires --segments-dir")

    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")

    model, processor = load_jepa2(args.model, device)

    config = SurpriseConfig(
        context_frames=args.context_frames,
        target_frames=args.target_frames,
        context_duration=args.context_duration,
        target_duration=args.target_duration,
        stride_duration=args.stride_duration,
        agg=args.agg,
        return_curve=args.segments_dir is not None,
    )

    if args.segments_dir:
        Path(args.segments_dir).mkdir(parents=True, exist_ok=True)

    video_files = args.videos[: args.n_videos] if args.n_videos else args.videos

    results = []
    for video_file in tqdm(video_files, desc="Videos"):
        out = calc_surprise_streaming(model, processor, video_file, config)

        if args.segments_dir:
            scores, starts = out
            s = max(scores) if config.agg == "max" else sum(scores) / len(scores)
            segments_path = Path(args.segments_dir) / f"{Path(video_file).stem}_segments.csv"
            _build_segments_df(
                video_file, scores, starts, config, args.top_segments
            ).to_csv(segments_path, index=False)
            print(f"  \u2192 surprise = {s:.4f}  segments \u2192 {segments_path}")
        else:
            s = out
            print(f"  \u2192 surprise = {s:.4f}")

        results.append({"video": video_file, "surprise": s})
        df = pd.DataFrame(results).sort_values("surprise", ascending=False)
        df.to_csv(args.output, index=False)

    print(f"\nDone! Video ranking saved to {args.output}")
    if args.segments_dir:
        print(f"Segment CSVs saved to {args.segments_dir}/")


if __name__ == "__main__":
    main()
