# Sequence Memory RNN

## What is this project?

Rats were trained to sniff a sequence of five odors, one at a time, in a specific order (like A, B, C,
D, E). Sometimes the sequence was correct ("InSeq"), and sometimes one odor was swapped out of place
("OutSeq"), and the rat had to notice the difference. While this happened, researchers recorded
electrical activity from the rat's hippocampus (a brain region central to memory), using tiny
electrodes, giving us local field potential (LFP) recordings for five rats (Mitt, Barat, Stella,
Superchris, Buchanan).

Our goal is to build a model that looks at that raw brain activity and predicts, for each trial, two
things:
1. Was this an InSeq or OutSeq trial?
2. Which odor (A, B, C, D, or E) was it?

A prior paper (MaGNet) already tried something similar using a graph-based model and got it right
roughly 66-74% of the time depending on the rat. We originally set out to try a different approach (a
recurrent neural network, RNN) to see if we could beat that benchmark. That RNN work is still in this
repo, but it isn't the final story — see below.

## What we actually ended up doing

The RNN could hit competitive accuracy, but it was hard to explain *why* it made a given prediction,
and being able to explain that was the actual point of this project. So the final, reported models are
simpler on purpose: a logistic regression (GLM) fit on band-power features (delta, theta, beta,
low-gamma, high-gamma), trained separately per rat, per task:

- **InSeq/OutSeq** — actually two final models, not one: a **Poke-In model** (trained on a window right
  after the rat pokes into the odor port) and a **Poke-Out model** (trained on a window ending right
  before the rat pokes back out). Neither one is consistently more accurate than the other across rats
  — they answer slightly different questions (early-building signal vs. right-before-the-decision
  signal) — so both are reported side by side rather than picking one as "the" result.
- **Odor Identity** — a single model, using a window after Poke-In, predicting which of the five odors
  (A-E) was presented, restricted to InSeq trials.

Because it's a GLM, the fitted coefficients are directly readable as "how much does this frequency
band push the prediction, and in which direction" — that's the centerpiece of the interpretability work,
including a time-resolved (sliding-window) view of which band matters *when*, not just overall.

## What's in this repo

- Code that loads and inspects the raw brain-recording files (`.lfp` and `.bvr` format)
- Code that turns raw recordings into clean, labeled chunks of data
- The band-power feature extraction used by the final GLM models
- The original multi-task RNN model and training code (early-phase work, kept for reference/comparison)
- Code that fits and interprets the final GLM models (per-band coefficients, time-resolved importance)
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

Raw `.lfp` and `.bvr` recording files go in `data/raw/<session_folder>/`, one folder per rat. This
folder is intentionally excluded from the git (see `.gitignore`). The raw data is shared separately
(currently via the lab's Google Drive).

## Understanding the data and results

Open `notebooks/01_data_audit.ipynb` first — it's written to be read top to bottom, every code cell has
a plain-language explanation before it (what it does and why) and a findings summary after it (what we
learned). No prior knowledge of the data assumed.

The final results and interpretability writeup live in `notebooks/024_final_models_presentation.ipynb`.
It walks through: selecting each model's time window (Sections 2-4), the final GLM results for both
InSeq/OutSeq models and the Odor Identity model (Sections 2-5), and the interpretability analysis —
which frequency bands matter, and when — for all three models (Section 6). Notebooks 019-023 are the
supporting/exploratory work (window search, model comparisons) behind that final notebook.

## Reproducibility notes

- All hyperparameters live in `configs/*.yaml`
- `src/utils.set_seed()` seeds Python, NumPy, and PyTorch (CPU/CUDA/MPS) so runs are consistent
- Train/val/test splits are done by recording session, not by individual trial, since trials from the
  same session aren't independent (same rat, same day) — splitting at the trial level would leak
  information between train and test
- `requirements.txt` pins exact package versions so results reproduce identically across machines

## Project structure

```
├── src/                  importable pipeline code
│   ├── data_loading.py   loading and inspecting raw .lfp/.bvr files
│   ├── preprocessing.py  trial segmentation, label extraction, band-power feature extraction
│   ├── models.py         the MultiTaskRNN architecture (early-phase; not the final reported model)
│   ├── train.py          RNN training loop
│   ├── explain.py        interpretability (Integrated Gradients, for the RNN)
│   └── utils.py          seeding, config loading, device selection
├── notebooks/            exploration and analysis
│   ├── 01_data_audit.ipynb              start here
│   ├── 019-023_*.ipynb                  window search / model comparisons (supporting work)
│   └── 024_final_models_presentation.ipynb   final GLM results + interpretability
├── configs/              one YAML file per experiment's hyperparameters
├── data/                 raw (gitignored) and processed data
├── outputs/              model checkpoints, logs, and figures (gitignored)
├── .devcontainer/        GitHub Codespaces configuration
├── Dockerfile / docker-compose.yml
└── Makefile              shortcuts: make build / make up / make audit / make train
```

## Background reading

- The benchmark paper: [MaGNet, arXiv:2309.13459](https://arxiv.org/abs/2309.13459)
- The original behavioral task this data comes from: Fortin lab hippocampal sequence memory studies
- Experiment/paper this data is drawn from: "Hippocampal ensembles represent sequential relationships
  among an extended sequence of nonspatial events" (Babak Shabhaha et al.)
