"""Helpers for building sliding windows from multivariate time series."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def add_rul_labels(df, unit_col="unit", cycle_col="cycle", cap: float | None = 125.0):
    """Add piece-wise linear RUL (optionally capped) for each unit."""
    max_cycle = df.groupby(unit_col)[cycle_col].transform("max")
    rul = max_cycle - df[cycle_col]
    if cap is not None:
        rul = np.minimum(rul, cap)
    out = df.copy()
    out["RUL"] = rul.astype(float)
    return out


def build_windows_from_units(
    df,
    feature_cols,
    target_col,
    seq_len: int = 30,
    unit_col: str = "unit",
    cycle_col: str = "cycle",
    last_only: bool = False,
):
    """
    Build (N, seq_len, F) windows.
    If last_only=True, keep only the final window per unit (typical for test engines).
    """
    Xs, ys, units = [], [], []
    for unit, g in df.groupby(unit_col, sort=True):
        g = g.sort_values(cycle_col)
        feats = g[feature_cols].to_numpy(dtype=np.float32, copy=True)
        targets = g[target_col].to_numpy(dtype=np.float32, copy=True)
        n = len(g)
        if n == 0:
            continue

        # Match classic sliding windows when history is long enough; pad only for short series.
        if last_only:
            ends = [n - 1]
        elif n >= seq_len:
            ends = list(range(seq_len - 1, n))
        else:
            ends = list(range(n))
        for end_idx in ends:
            hist = feats[: end_idx + 1]
            if hist.shape[0] < seq_len:
                # Left-pad with the earliest available timestep (no negative-slice bugs)
                pad = np.repeat(hist[:1], seq_len - hist.shape[0], axis=0)
                window = np.vstack([pad, hist])
            else:
                window = hist[-seq_len:]
            if window.shape != (seq_len, feats.shape[1]):
                raise RuntimeError(
                    f"Bad window shape {window.shape} for unit={unit} end={end_idx}"
                )
            Xs.append(window)
            ys.append(targets[end_idx])
            units.append(unit)

    if not Xs:
        fdim = len(feature_cols)
        return (
            np.zeros((0, seq_len, fdim), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(Xs, axis=0).astype(np.float32, copy=False),
        np.asarray(ys, dtype=np.float32),
        np.asarray(units),
    )


def fit_scale_windows(X_train, X_other_list):
    """Scale features using training windows; apply to other splits."""
    n, t, f = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, f))
    out = [scaler.transform(X_train.reshape(-1, f)).reshape(n, t, f).astype(np.float32)]
    for X in X_other_list:
        nn, tt, ff = X.shape
        out.append(scaler.transform(X.reshape(-1, ff)).reshape(nn, tt, ff).astype(np.float32))
    return out[0], out[1:], scaler
