"""Data loading and inspection.

Start here. Before writing any preprocessing or model code, run
`inspect_npz` on one session's files so you know the real key names,
shapes, and dtypes rather than assuming they match the README.
"""
from pathlib import Path
import numpy as np


def inspect_npz(filepath: str) -> dict:
    """Print and return a summary of everything inside an .npz file.

    Usage:
        from src.data_loading import inspect_npz
        info = inspect_npz("data/raw/<session>/<file>_lfp.npz")
    """
    filepath = Path(filepath)
    data = np.load(filepath, allow_pickle=True)

    summary = {}
    print(f"\n{'=' * 60}")
    print(f"File: {filepath.name}")
    print(f"{'=' * 60}")
    for key in data.files:
        arr = data[key]
        info = {
            "shape": arr.shape,
            "dtype": arr.dtype,
        }
        # For small arrays, show actual values, useful for spotting
        # label arrays vs. raw signal arrays at a glance.
        if arr.size <= 20:
            info["values"] = arr.tolist()
        summary[key] = info
        print(f"  {key:20s} shape={str(arr.shape):15s} dtype={arr.dtype}")
        if "values" in info:
            print(f"  {'':20s} values={info['values']}")
    return summary


def inspect_session(session_dir: str) -> dict:
    """Run inspect_npz on every .npz file in a session directory.

    Usage:
        from src.data_loading import inspect_session
        inspect_session("data/raw/<rat>/<session>")
    """
    session_dir = Path(session_dir)
    results = {}
    for npz_file in sorted(session_dir.glob("*.npz")):
        results[npz_file.stem] = inspect_npz(str(npz_file))
    return results


def load_session(session_dir: str) -> dict:
    """Load all npz files in a session into a dict of {name: NpzFile}.

    TODO: once inspect_session tells us the real structure, replace
    this with a function that returns a clean, typed structure, e.g.
    {"lfp": np.ndarray, "trial_labels": pd.DataFrame, ...} instead of
    raw NpzFile handles.
    """
    session_dir = Path(session_dir)
    return {
        npz_file.stem: np.load(str(npz_file), allow_pickle=True)
        for npz_file in sorted(session_dir.glob("*.npz"))
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.data_loading <session_dir>")
        sys.exit(1)
    inspect_session(sys.argv[1])
