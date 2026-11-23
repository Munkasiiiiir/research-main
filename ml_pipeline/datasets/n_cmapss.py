"""N-CMAPSS loader with cycle-level aggregation (memory-safe)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .sequence_utils import build_windows_from_units, fit_scale_windows


def _decode_vars(arr) -> list[str]:
    out = []
    for x in np.array(arr):
        if isinstance(x, (bytes, np.bytes_)):
            out.append(x.decode())
        else:
            out.append(str(x))
    return out


def _cycle_aggregate(h5_path: Path, split: str = "dev") -> pd.DataFrame:
    """Mean-aggregate W + X_s per (unit, cycle); RUL is cycle-constant."""
    with h5py.File(h5_path, "r") as hdf:
        A = np.array(hdf[f"A_{split}"])
        Y = np.array(hdf[f"Y_{split}"]).ravel()
        W = np.array(hdf[f"W_{split}"])
        Xs = np.array(hdf[f"X_s_{split}"])
        W_var = _decode_vars(hdf["W_var"])
        Xs_var = _decode_vars(hdf["X_s_var"])

    unit = A[:, 0].astype(np.int32)
    cycle = A[:, 1].astype(np.int32)
    # vectorized group keys
    key = unit.astype(np.int64) * 100_000 + cycle.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    breaks = np.flatnonzero(np.r_[True, key_s[1:] != key_s[:-1], True])

    rows = []
    feat_names = W_var + Xs_var
    for i in range(len(breaks) - 1):
        sl = order[breaks[i] : breaks[i + 1]]
        u = int(unit[sl[0]])
        c = int(cycle[sl[0]])
        feats = np.concatenate([W[sl].mean(0), Xs[sl].mean(0)])
        rows.append((u, c, float(Y[sl[0]]), *feats.tolist()))

    cols = ["unit", "cycle", "RUL"] + feat_names
    return pd.DataFrame(rows, columns=cols)


def load_n_cmapss(
    root: str | Path,
    filename: str = "N-CMAPSS_DS01-005.h5",
    seq_len: int = 20,
    rul_cap: float = 100.0,
):
    root = Path(root)
    path = root / filename
    if not path.exists():
        # fall back to first available h5
        candidates = sorted(root.glob("N-CMAPSS_DS*.h5"))
        if not candidates:
            raise FileNotFoundError(f"No N-CMAPSS h5 files in {root}")
        path = candidates[0]
        filename = path.name

    train_df = _cycle_aggregate(path, "dev")
    test_df = _cycle_aggregate(path, "test")
    if rul_cap is not None:
        train_df["RUL"] = np.minimum(train_df["RUL"], rul_cap)
        test_df["RUL"] = np.minimum(test_df["RUL"], rul_cap)

    feature_cols = [c for c in train_df.columns if c not in ("unit", "cycle", "RUL")]

    units = train_df["unit"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(units)
    n_val = max(1, int(0.2 * len(units)))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])

    X_tr, y_tr, _ = build_windows_from_units(
        train_df.loc[train_df["unit"].isin(train_units)],
        feature_cols,
        "RUL",
        seq_len=seq_len,
        last_only=False,
    )
    X_va, y_va, _ = build_windows_from_units(
        train_df.loc[train_df["unit"].isin(val_units)],
        feature_cols,
        "RUL",
        seq_len=seq_len,
        last_only=False,
    )
    X_te, y_te, _ = build_windows_from_units(
        test_df, feature_cols, "RUL", seq_len=seq_len, last_only=True
    )

    X_tr, (X_va, X_te), _ = fit_scale_windows(X_tr, [X_va, X_te])
    return {
        "name": f"N-CMAPSS-{filename.replace('N-CMAPSS_', '').replace('.h5', '')}",
        "task": "regression",
        "tol": 10.0,
        "X_train": X_tr,
        "y_train": y_tr.astype(np.float32),
        "X_val": X_va,
        "y_val": y_va.astype(np.float32),
        "X_test": X_te,
        "y_test": y_te.astype(np.float32),
    }
