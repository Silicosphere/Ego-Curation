"""Pack a folder of extracted clips into a single HDF5 file, to stay under the file-count quota.

Input layout (written by extract_clips.sh):
    clips1/<video>/<video>_seg001_0123.00s.mp4 ...

Output layout:
    clips1.h5
      /<video>/<clip file stem>   uint8 dataset holding the clip's mp4 bytes, unchanged
                                  attrs: rank, start_sec

The clips are stored byte for byte (no re-encoding, no compression: mp4 is already compressed).
The file is written as <name>.h5.part and renamed only after every clip has been read back
and checked, so a killed job never leaves a half-written .h5 behind. With --delete, each
video folder is removed only after that check.

Usage (from the repository root):
    python scripts/extract/pack_hdf5.py /data/.../ego4d_clips/clips1 [--delete]

Read a clip back:
    with h5py.File("clips1.h5") as f:
        data = f["<video>/<clip stem>"][()].tobytes()   # the original .mp4 bytes
    decord.VideoReader(io.BytesIO(data))               # or write them to a file
"""
import argparse
import hashlib
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

CLIP_NAME = re.compile(r"_seg(\d+)_(\d+(?:\.\d+)?)s$")


def pack(clips_dir: Path, out_file: Path, delete: bool) -> None:
    video_dirs = sorted(d for d in clips_dir.iterdir() if d.is_dir())
    if not video_dirs:
        raise SystemExit(f"No video folders in {clips_dir}")
    if out_file.exists():
        raise SystemExit(f"{out_file} already exists; move or delete it first.")

    tmp_file = out_file.with_name(out_file.name + ".part")
    hashes = {}
    n_clips = 0
    with h5py.File(tmp_file, "w") as h5:
        h5.attrs["source_dir"] = str(clips_dir)
        for video_dir in tqdm(video_dirs, desc="Packing"):
            group = h5.create_group(video_dir.name)
            # Skip unfinished clips left by a killed extraction job.
            for clip in sorted(p for p in video_dir.glob("*.mp4")
                               if not p.name.endswith(".part.mp4")):
                data = clip.read_bytes()
                key = f"{video_dir.name}/{clip.stem}"
                ds = group.create_dataset(clip.stem, data=np.frombuffer(data, dtype=np.uint8))
                match = CLIP_NAME.search(clip.stem)
                if match:
                    ds.attrs["rank"] = int(match.group(1))
                    ds.attrs["start_sec"] = float(match.group(2))
                hashes[key] = hashlib.sha1(data).hexdigest()
                n_clips += 1

    # Read everything back and compare with the original files before trusting the file.
    with h5py.File(tmp_file, "r") as h5:
        for key, digest in tqdm(hashes.items(), desc="Verifying"):
            if hashlib.sha1(h5[key][()].tobytes()).hexdigest() != digest:
                raise SystemExit(f"Verification failed for {key}; {tmp_file} kept, "
                                 f"nothing deleted.")
    tmp_file.replace(out_file)
    size_gb = out_file.stat().st_size / 1e9
    print(f"Packed and verified {n_clips} clips from {len(video_dirs)} videos "
          f"into {out_file} ({size_gb:.1f} GB)")

    if delete:
        for video_dir in video_dirs:
            shutil.rmtree(video_dir)
        print(f"Deleted the {len(video_dirs)} packed video folders from {clips_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("clips_dir", type=Path, help="Folder of clips, e.g. ego4d_clips/clips1")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output .h5 file (default: <clips_dir>.h5 next to the folder)")
    parser.add_argument("--delete", action="store_true",
                        help="Delete the video folders after the .h5 file is verified")
    args = parser.parse_args()
    clips_dir = args.clips_dir.resolve()
    pack(clips_dir, args.output or clips_dir.with_suffix(".h5"), args.delete)


if __name__ == "__main__":
    main()
