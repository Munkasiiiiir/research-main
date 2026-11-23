"""PHM08 challenge dataset. Test RUL is hidden, so we evaluate on a held-out train split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sequence_utils import add_rul_labels, build_windows_from_units, fit_scale_windows

COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

FEATURE_COLS = [
    "op1",
    "op2",
    "op3",
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]


def load_phm08(root: str | Path, seq_len: int = 30, rul_cap: float = 125.0, stride: int = 2):
    root = Path(root)
    train = pd.read_csv(root / "train.txt", sep=r"\s+", header=None, names=COLS)
    train = add_rul_labels(train, cap=rul_cap)

    units = train["unit"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(units)
    n = len(units)
    n_test = max(1, int(0.20 * n))
    n_val = max(1, int(0.15 * n))
    test_units = set(units[:n_test])
    val_units = set(units[n_test : n_test + n_val])
    train_units = set(units[n_test + n_val :])

    def subset(u_set, last_only=False):
        return build_windows_from_units(
            train.loc[train["unit"].isin(u_set)],
            FEATURE_COLS,
            "RUL",
            seq_len=seq_len,
            last_only=last_only,
        )

    X_tr, y_tr, _ = subset(train_units, last_only=False)
    X_va, y_va, _ = subset(val_units, last_only=False)
    # Evaluate like the challenge: predict RUL at end of each held-out unit trajectory
    X_te, y_te, _ = subset(test_units, last_only=True)

    if stride > 1:
        X_tr, y_tr = X_tr[::stride], y_tr[::stride]
        X_va, y_va = X_va[::stride], y_va[::stride]

    X_tr, (X_va, X_te), _ = fit_scale_windows(X_tr, [X_va, X_te])
    return {
        "name": "PHM08",
        "task": "regression",
        "tol": 10.0,
        "note": "Official test RUL is hidden; metrics use a held-out 20% of training units.",
        "X_train": X_tr,
        "y_train": y_tr.astype(np.float32),
        "X_val": X_va,
        "y_val": y_va.astype(np.float32),
        "X_test": X_te,
        "y_test": y_te.astype(np.float32),
    }
