# Sequence Memory RNN

Multi-task RNN predicting InSeq/OutSeq and odor identity (A-E) from hippocampal
LFP activity, benchmarked against the graph-based baseline in
[MaGNet (Zhou et al.)](https://arxiv.org/abs/2309.13459).

## Setup

### Option A: GitHub Codespaces (zero local setup, great for non-technical collaborators)

On the repo page, click the green **Code** button → **Codespaces** tab →
**Create codespace on main**. This builds the same Docker environment used
locally, entirely in your browser, no install required. Takes about a
minute. Once it opens, run `jupyter notebook --ip=0.0.0.0` in the terminal
and click the forwarded port 8888 link when it pops up.

Requires the person to have a free GitHub account and be added as a
collaborator on the repo (Settings → Collaborators).

### Option B: Docker locally (recommended, guarantees reproducibility)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/)
installed and running.

```bash
# Build the image
make build

# Start Jupyter (visit http://localhost:8888 in your browser)
make up
```

Note: containers on Mac can't access the Apple GPU (MPS), so training inside
Docker runs on CPU. Fine for this project's scale, but if you want GPU
speedup while iterating, use Option B instead and save Docker for final,
shareable runs.

### Option C: Native venv (faster iteration on Apple Silicon)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Data

Place raw `.npz` files under `data/raw/<rat>/<session>/`. This directory is
gitignored, raw data never gets committed or pushed.

## Workflow

1. **Data audit first.** Open `notebooks/01_data_audit.ipynb` and run it
   against one session. Confirm real key names, shapes, trial markers, and
   label locations before writing any preprocessing logic.
2. Fill in `src/preprocessing.py` (`segment_trials`, `build_labels`) based on
   what the audit reveals.
3. Fill in the `Dataset`/`DataLoader` section of `src/train.py`.
4. Run training: `make train` (Docker) or `python -m src.train` (venv).
5. Interpretability via `src/explain.py`, Integrated Gradients per
   channel/timestep, directly comparable to the paper's own IntGradients
   baseline in Section 6.

## Reproducibility notes

- All hyperparameters live in `configs/*.yaml`, not hardcoded in scripts.
- `src/utils.set_seed()` seeds Python, NumPy, and PyTorch (CPU/CUDA/MPS).
- Splits are done by session, not by trial, to avoid leaking data from the
  same recording session across train/val/test.
- `requirements.txt` pins exact versions so results are reproducible across
  your machine and Wonjae's.

## Project structure

```
├── src/                  importable pipeline code
│   ├── data_loading.py   npz inspection + loading
│   ├── preprocessing.py  trial segmentation, label extraction
│   ├── models.py         MultiTaskRNN architecture
│   ├── train.py          training loop
│   ├── explain.py        interpretability (Integrated Gradients)
│   └── utils.py          seeding, config loading, device selection
├── notebooks/            exploration only, not the real pipeline
├── configs/               hyperparameters, one YAML per experiment
├── data/                 raw (gitignored) and processed data
├── outputs/              checkpoints and logs (gitignored)
├── Dockerfile / docker-compose.yml
└── Makefile              make build / make up / make audit / make train
```
