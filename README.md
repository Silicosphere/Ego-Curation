<div align="center">

# Egocentric Curation

</div>

## Description

A pipeline for curating egocentric datasets (e.g., Ego4D) to extract the most surprising segments of the videos—the parts that hold the most valuable information.

>Note: This project is part of our team's BSc Thesis, and will be used for subsequent work.

## How It Works

This pipeline consists of two main scripts:

1. [pipeline.py](ego_curation/pipeline.py):
   Holds the data curation logic. It outputs a `segments` directory containing one CSV file per video with its top segments (e.g., the top 20 segments of the video).
2. [extract.py](ego_curation/extract.py):
   Holds the segment extraction logic, using ffmpeg to slice the videos. It outputs a `clips` directory with a subdirectory for each video's extracted segments.

We use [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) as the backbone for this pipeline to find the most surprising frames in a video. We use the **official release** from [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2): the model code is loaded through `torch.hub` (pinned to a commit) and the released checkpoints are downloaded from `dl.fbaipublicfiles.com`. Select a size via `--model-size`:

| Size       | Checkpoint                     | Parameters | Download size |
|------------|--------------------------------|------------|---------------|
| `giant`    | `vjepa2_1_vitg_384.pt`         | 1B         | 15.7 GB       |
| `gigantic` *(default)* | `vjepa2_1_vitG_384.pt` | 2B     | 28.2 GB       |

Checkpoints are downloaded once on first use into the torch hub cache (`~/.cache/torch/hub/checkpoints`, override with `TORCH_HOME`). They include training optimizer state, hence their size; they are memory-mapped at load time so only the model weights are read into RAM.

The `base` and `large` models are deliberately **not** offered. They are distilled from ViT-G, meaning their predictor outputs features in the *teacher's* 1664-dim space rather than in their own encoder space. A prediction error computed between those two spaces is meaningless, and scoring them would require keeping the 2B teacher resident anyway. The `giant` and `gigantic` checkpoints have no teacher, so their predictor reproduces their own hierarchical encoder features (5632 and 6656 dims) and the comparison is self-consistent. `load_jepa2` verifies this at load time and refuses any model where the two widths disagree. See [docs/vjepa21-representation-spaces.md](docs/vjepa21-representation-spaces.md).

### Methodology

`pipeline.py` takes a list of video paths along with various arguments (e.g., `--sample-fps`, `--context-frames`, `--target-frames`). The idea is:

1. Slide a window over the video. The window holds `context-frames + target-frames` frames sampled at `sample-fps`.
2. Split the sampled frames into a context clip (the first `context-frames`) and a target clip (the rest, which follows the context without a gap).
3. Feed them to the model and obtain the predicted target embeddings.
4. Compute the prediction error between the true target embeddings and the predicted ones, and save the scores.
5. Write the top surprising windows (highest prediction error) to a CSV file in the `segments` directory.

Once the pipeline produces the CSV files for all videos, the extractor script reads them and cuts the corresponding segments from the original videos.

## How to Run

### 1. Install dependencies

```bash
# clone project
git clone https://github.com/Silicosphere/Ego-Curation.git
cd Ego-Curation

# [RECOMMENDED] create virtual environment
conda create -n myenv
conda activate myenv

# Or
python3 -m venv myenv
source myenv/bin/activate   # For Linux
myenv/Scripts/activate      # For Windows

# install requirements
pip install -r requirements.txt
```

### 2. Download an egocentric video dataset

### 3. Run the curation pipeline

```bash
python3 main.py \
  <YOUR DATASET PATH>/* \
  --model-size gigantic \
  --sample-fps 4 \
  --context-frames 16 \
  --target-frames 8 \
  --stride-duration 6.0 \
  --segments-dir segments \
  --top-segments 20
```

This scores 6 s windows (4 s context + 2 s target) that tile the video without overlap. See [Choosing the parameters](#choosing-the-parameters) below.

#### Running on the cluster (Slurm)

The pipeline runs on the GPU cluster. A run takes a long time, so start it inside `tmux`: the job keeps running if your SSH connection drops.

1. On the login node, start a `tmux` session:

   ```bash
   tmux new -s ego
   ```

2. Inside `tmux`, start an interactive GPU job. **Always request memory:** the cluster default is 1 CPU and 2000 MB RAM, and a run that exceeds it is silently `Killed`.

   ```bash
   srun --gres=gpu:rtx5090 --cpus-per-task=8 --mem=32G --time=12:00:00 --pty bash
   ```

3. On the GPU node, run the pipeline script from the repository root:

   ```bash
   ./run.sh
   ```

   [run.sh](run.sh) loads the FFmpeg module that `torchcodec` needs (`module load FFmpeg/7.0.2-GCCcore-13.3.0`), exports `TORCH_HOME` so the checkpoints are stored under `/data` rather than your home directory, activates the virtual environment and calls `main.py`.

To leave the run in the background, detach with `Ctrl-b` then `d`. Reattach later with `tmux attach -t ego`.

### 4. Extract the top segments from the videos

```bash
python3 -m ego_curation.extract segments/*.csv --top-segments 20 --output clips/
```

## Choosing the parameters

Three numbers define a window, and the durations follow from them:

$$
\text{context duration} = \frac{\texttt{context-frames}}{\texttt{sample-fps}}, \qquad
\text{target duration} = \frac{\texttt{target-frames}}{\texttt{sample-fps}}
$$

Context and target always share one sampling rate. The predictor places the target right after the context and assumes the same time spacing, so a target sampled at a different rate would be compared against a prediction for the wrong span of time.

The model groups every **2 frames** into one time step, and each time step is **576 tokens** (a 24 × 24 grid at 384 px). Compute and memory grow with the number of tokens.

### Recommended order

1. **`--sample-fps`: time resolution.**
   - V-JEPA 2.1 was pretrained at **4 fps** (the default). Rates near it are the safest choice. Higher rates capture faster motion but need more frames to cover the same duration.
   - Prefer a rate that divides the video's frame rate evenly, so sampled frames are equally spaced. For 30 fps video (Ego4D), 5, 6, 7.5, 10 and 15 fps give an exact step. 4 fps gives a step of 7.5 source frames, so gaps alternate between 7 and 8 frames.
   - Do not exceed the video's own frame rate. The pipeline warns, because frames would be repeated.
2. **`--context-frames`: how much history the model sees.**
   - Must be even. Pick the context duration first, then set `context-frames = duration × sample-fps` (rounded to an even number).
   - Pretraining used clips of 16 frames (4 s at 4 fps) and later 64 frames (16 s at 4 fps).
   - Context tokens = `context-frames / 2 × 576`. For example, 16 frames is 4,608 tokens and 64 frames is 18,432 tokens.
3. **`--target-frames`: how far ahead the model is scored.**
   - Must be even. Same rule: `target-frames = duration × sample-fps`.
   - Predictions far from the context are harder, so scores rise with target length regardless of content. Only compare scores produced with the same settings.
   - The predictor processes context and target tokens together: `(context-frames + target-frames) / 2 × 576` tokens. This is usually what limits GPU memory.
4. **`--stride-duration`: seconds between window starts.**
   - Default: the window duration, so windows tile the video without overlap.
   - A smaller stride gives overlapping windows. Surprising moments are located more precisely, but runtime grows in proportion (half the stride, twice the windows) and top segments can overlap each other.
   - A larger stride skips part of the video.
5. **`--top-segments`: segments kept per video.** Each segment is one window long, so the extracted footage per video is `top-segments × window duration`.
6. **`--agg`: video-level score.** `mean` ranks videos by overall surprise; `max` ranks them by their single most surprising window.

### Examples for 30 fps video (4 s context + 2 s target, 6 s stride)

| Goal | `--sample-fps` | `--context-frames` | `--target-frames` | Predictor tokens |
|---|---|---|---|---|
| Match pretraining rate | 4 | 16 | 8 | 6,912 |
| Equal frame spacing | 5 | 20 | 10 | 8,640 |
| Finer motion | 15 | 60 | 30 | 25,920 |

### Checklist

- Both frame counts are even (the CLI rejects odd values).
- `sample-fps` is at most the video's frame rate.
- `(context-frames + target-frames) / 2 × 576` tokens fit in GPU memory. If a run runs out of memory, lower `sample-fps` or shorten the durations.
- Runs you want to compare use the same `sample-fps`, frame counts and stride.

## Expected Output

```
segments/
├── <video1>_segments.csv
├── <video2>_segments.csv
└── ... (one CSV per video)

clips/
├── <video1[:8]>/                      # subfolder per video (first 8 chars of filename)
│   ├── <video1>_seg001_0012.00s.mp4     # one clip per top segment
│   ├── <video1>_seg002_0186.00s.mp4
│   └── ...
└── ...
```

## Code Structure

```
Ego-Curation/
├── .gitignore
├── LICENSE
├── README.md
├── main.py                      # entrypoint
├── pyproject.toml               # package config
├── requirements.txt
├── ego_curation/
│   ├── __init__.py
│   ├── cli.py                   # CLI parsing
│   ├── config.py                # config/settings
│   ├── extract.py               # segment extraction
│   ├── model.py                 # model definitions
│   ├── pipeline.py              # pipeline orchestration
│   └── sampling.py              # sampling logic
└── (runtime dirs: segments/, clips/, not tracked)
```

## License

This project is licensed under the [MIT License](LICENSE).