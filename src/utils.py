"""Shared utilities: reproducibility, config loading."""
import random
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    """Seed all RNGs we touch. Note: exact bit-for-bit reproducibility
    isn't guaranteed on GPU/MPS even with seeding, but this keeps runs
    consistent within normal noise.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Pick the best available device. Prints what it picked so it's
    always obvious from the logs what actually ran.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device
