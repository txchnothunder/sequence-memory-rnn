"""Trial segmentation and windowing.

Left as a stub with the shape of what's needed. Fill in once the data
audit (src/data_loading.py) confirms real key names and trial marker
locations, so this isn't guessed ahead of time.
"""
import numpy as np


def segment_trials(lfp: np.ndarray, trial_markers, window_ms: int, fs: float):
    """Cut continuous LFP into per-trial windows.

    Args:
        lfp: array of shape (n_channels, n_timepoints) or similar,
            confirm real layout during the data audit.
        trial_markers: indices or timestamps marking trial onsets,
            source TBD from the bvr file.
        window_ms: window length in milliseconds.
        fs: sampling rate in Hz.

    Returns:
        windows: array of shape (n_trials, n_channels, window_samples)

    TODO: implement once trial marker source and LFP layout are
    confirmed via inspect_session().
    """
    raise NotImplementedError("Fill in after the data audit.")


def build_labels(bvr_data) -> dict:
    """Extract InSeq/OutSeq and odor-identity labels per trial.

    TODO: implement once bvr's actual fields are known.
    Returns something like:
        {"inseq_outseq": np.ndarray, "odor_id": np.ndarray}
    """
    raise NotImplementedError("Fill in after the data audit.")


def train_val_test_split(n_sessions: int, train_frac: float, val_frac: float, seed: int):
    """Split by session (not by trial) to avoid leaking trials from
    the same recording session across train/val/test.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_sessions)
    n_train = int(train_frac * n_sessions)
    n_val = int(val_frac * n_sessions)
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:],
    )
