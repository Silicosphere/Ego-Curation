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

We use the [V-JEPA 2](https://arxiv.org/abs/2506.09985) model as the base model for this pipeline to find the most surprising frames in a video. Specifically, we use the [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) model (0.3 billion parameters), hosted on [Hugging Face](https://huggingface.co/).

### Methodology

`pipeline.py` takes a list of video paths along with various arguments (e.g., `context-duration`, `target-duration`). The idea is:

1. Build a sliding window of size `context-duration + target-duration`.
2. Randomly sample context frames and target frames from within the window.
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
  --context-duration 4.0 \
  --target-duration 2.0 \
  --context-frames 64 \
  --target-frames 16 \
  --stride-duration 6.0 \
  --segments-dir segments \
  --top-segments 20
```

### 4. Extract the top segments from the videos

```bash
python3 -m ego_curation.extract segments/*.csv --top-segments 20 --output clips/
```

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