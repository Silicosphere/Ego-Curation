import argparse
import pandas as pd
from tqdm.auto import tqdm
import torch

from pipeline import calc_surprise_streaming
from load_model import load_jepa2


def main():
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
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for model forward passes (default: %(default)s)",
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

    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    )
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")

    model, processor = load_jepa2(args.model, device)
    model.eval()

    video_files = args.videos[: args.n_videos] if args.n_videos else args.videos

    results = []
    for video_file in tqdm(video_files, desc="Videos"):
        s = calc_surprise_streaming(
            model,
            processor,
            video_file,
            context_frames=args.context_frames,
            target_frames=args.target_frames,
            context_duration=args.context_duration,
            target_duration=args.target_duration,
            stride_duration=args.stride_duration,
            batch_size=args.batch_size,
            agg=args.agg,
        )
        results.append({"video": video_file, "surprise": s})
        df = pd.DataFrame(results).sort_values("surprise", ascending=False)
        df.to_csv(args.output, index=False)
        print(f"  \u2192 surprise = {s:.4f}")

    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()
