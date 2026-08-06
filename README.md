# Sequence Memory RNN

## What is this project?

Rats were trained to sniff a sequence of five odors, one at a time, in a specific order (like A, B, C,
D, E). Sometimes the sequence was correct ("InSeq"), and sometimes one odor was swapped out of place
("OutSeq"), and the rat had to notice the difference. While this happened, researchers recorded
electrical activity from the rat's hippocampus (a brain region central to memory), using tiny
electrodes.

Our goal is to build a model that looks at that raw brain activity and predicts, for each trial, two things:
1. Was this an InSeq or OutSeq trial?
2. Which odor (A, B, C, D, or E) was it?

A prior paper (MaGNet) already tried something similar using a graph-based model and got it right
roughly 66-74% of the time depending on the rat. Our goal is to try a different approach (a recurrent
neural network, RNN, a model designed for data that unfolds over time) and see if we can do better,
while also being able to explain *why* the model makes the predictions it does, not just report a
number.

## What's in this repo

- Code that loads and inspects the raw brain-recording files
- Code that turns raw recordings into clean, labeled chunks of data
- The RNN model itself
- Code that explains the model's predictions (which electrodes/timepoints mattered most)
- A fully reproducible environment, so this runs identically on any computer

## Setup

### Using Docker

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (free)
2. Open a terminal in this project folder and run:
   ```bash
   make build   # one-time, builds the environment, a few minutes
   make up      # starts Jupyter
   ```
3. Open `http://localhost:8888` in your browser

Note: on Mac, Docker can't use the Apple GPU, so training runs on CPU inside Docker. That's fine for
this project's scale.

### Using Terminal

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Getting the data

Raw `.npz` recording files go in `data/raw/<session_folder>/`. This folder is intentionally excluded from the git (see `.gitignore`). The raw data is shared separately (currently via the lab's Google Drive).

## Understanding the data

Open `notebooks/01_data_audit.ipynb`. It's written to be read top to bottom, every code cell has a plain-language explanation before it (what it does and why) and a findings summary after it (what we learned). No prior knowledge of the data assumed.

## Reproducibility notes

- All hyperparameters live in `configs/*.yaml`
- `src/utils.set_seed()` seeds Python, NumPy, and PyTorch (CPU/CUDA/MPS) so runs are consistent
- Train/val/test splits are done by recording session, not by individual trial, since trials from the same session aren't independent (same rat, same day), splitting at the trial level would leak information between train and test
- `requirements.txt` pins exact package versions so results reproduce identically across machines.

## Project structure

```
├── src/                  importable pipeline code
│   ├── data_loading.py   loading and inspecting raw .npz files
│   ├── preprocessing.py  trial segmentation, label extraction
│   ├── models.py         the MultiTaskRNN architecture
│   ├── train.py          training loop
│   ├── explain.py        interpretability (Integrated Gradients)
│   └── utils.py          seeding, config loading, device selection
├── notebooks/            exploration and analysis, e.g. the data audit
├── configs/              one YAML file per experiment's hyperparameters
├── data/                 raw (gitignored) and processed data
├── outputs/              model checkpoints and logs (gitignored)
├── .devcontainer/        GitHub Codespaces configuration
├── Dockerfile / docker-compose.yml
└── Makefile              shortcuts: make build / make up / make audit / make train
```

## Background reading

- The benchmark paper: [MaGNet, arXiv:2309.13459](https://arxiv.org/abs/2309.13459)
- The original behavioral task this data comes from: Fortin lab hippocampal sequence memory studies
