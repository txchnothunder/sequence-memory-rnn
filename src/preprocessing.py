"""Trial segmentation and label extraction.

Implemented based on findings from the data audits (notebooks/01 and
notebooks/02): bvr, lfp, and spk share one time axis; InSeqLog marks
odor onset (not a later decision point, offset is 0 samples); every
trial has exactly one active odor channel; and critically, sampling
rate is NOT constant across rats (714-1000 Hz depending on session),
so it must always be computed per-session, never hardcoded.
"""
import numpy as np


def get_sampling_rate(bvr_data: np.ndarray, keys: list) -> float:
    """Compute sampling rate (Hz) from a session's own TimeBin channel.

    Never hardcode a sampling rate. The light EDA across all 5 rats
    showed it ranges from ~714 Hz to 1000 Hz depending on the session.
    """
    timebin = bvr_data[keys.index("TimeBin")]
    return 1 / np.diff(timebin).mean()


def build_labels(bvr_data: np.ndarray, keys: list, odor_search_window: int = 200) -> dict:
    """Extract trial indices and both labels (InSeq/OutSeq, odor id) from
    one session's bvr data.

    Args:
        bvr_data: the (n_channels, n_timepoints) array from a bvr npz's
            'data' key.
        keys: the list of channel names from the same npz's 'keys' key.
        odor_search_window: how many samples before each trial marker to
            search for the active odor channel. 200 was validated
            against Mitt's data (offset was consistently 0, i.e. odor
            onset and trial marker are the same sample), kept as a
            search window rather than an exact index for a small safety
            margin.

    Returns:
        dict with:
            trial_idx: array of sample indices, one per trial
            inseq_outseq: binary array, 1 = InSeq, 0 = OutSeq
            odor_id: int array, 0-4 corresponding to Odor1-Odor5 (A-E)
            ambiguous: list of trial positions where odor detection
                failed (0 or >1 active channels), should be empty based
                on audit findings, but checked rather than assumed
    """
    inseq_idx = keys.index("InSeqLog")
    trial_idx = np.where(bvr_data[inseq_idx] != 0)[0]
    raw_inseq_vals = bvr_data[inseq_idx][trial_idx]
    # audit found values are -1 (OutSeq) / 1 (InSeq); convert to standard 0/1
    inseq_outseq = (raw_inseq_vals == 1).astype(np.int64)

    odor_names = [k for k in keys if k.startswith("Odor")]
    odor_rows = [bvr_data[keys.index(name)] for name in odor_names]

    odor_id = np.full(len(trial_idx), -1, dtype=np.int64)
    ambiguous = []
    for i, t in enumerate(trial_idx):
        start = max(0, t - odor_search_window)
        active = [j for j, row in enumerate(odor_rows) if row[start:t + 1].any()]
        if len(active) == 1:
            odor_id[i] = active[0]
        else:
            ambiguous.append(i)

    return {
        "trial_idx": trial_idx,
        "inseq_outseq": inseq_outseq,
        "odor_id": odor_id,
        "ambiguous": ambiguous,
    }


def segment_trials(lfp_data: np.ndarray, trial_idx: np.ndarray, window_ms: float, fs: float) -> np.ndarray:
    """Cut continuous LFP into per-trial windows.

    Per the audit, InSeqLog fires at the exact moment of odor onset
    (0 sample offset), so windows extend FORWARD from the trial index,
    not backward or centered, matching what the rat is actually
    experiencing during the window.

    Args:
        lfp_data: array of shape (n_channels, n_timepoints), the 'data'
            key from an lfp npz.
        trial_idx: array of trial onset sample indices, from build_labels.
        window_ms: window length in milliseconds.
        fs: this session's sampling rate in Hz (from get_sampling_rate,
            never a hardcoded constant, see module docstring).

    Returns:
        windows: array of shape (n_trials, n_channels, window_samples).
            Trials where the window would run past the end of the
            recording are dropped (rare, only possible for the very
            last trial or two in a session).
    """
    window_samples = int(round(window_ms / 1000 * fs))
    n_channels, n_timepoints = lfp_data.shape

    windows = []
    kept_idx = []
    for t in trial_idx:
        end = t + window_samples
        if end > n_timepoints:
            continue  # trial too close to the end of the recording, drop it
        windows.append(lfp_data[:, t:end])
        kept_idx.append(t)

    return np.stack(windows, axis=0), np.array(kept_idx)


def train_val_test_split(n_sessions: int, train_frac: float, val_frac: float, seed: int):
    """Split by session (not by trial) to avoid leaking trials from
    the same recording session across train/val/test.

    Use this once multiple rats/sessions are pooled together.
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


def trial_level_split(n_trials: int, train_frac: float, seed: int):
    """Split at the trial level within a SINGLE session.

    WARNING: this is only appropriate as a sanity check while working
    with one rat's data alone (like Mitt, this week). It's not a valid
    evaluation once multiple sessions are pooled, at that point use
    train_val_test_split() instead, splitting by session to avoid
    leaking correlated trials (same rat, same day) across the split.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_trials)
    n_train = int(train_frac * n_trials)
    return indices[:n_train], indices[n_train:]
