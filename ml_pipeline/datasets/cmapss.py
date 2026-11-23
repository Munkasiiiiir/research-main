"""NASA CMAPSS turbofan RUL dataset loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sequence_utils import add_rul_labels, build_windows_from_units, fit_scale_windows

COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors that are typically constant / low-info on FD001; keep ops + informative sensors
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


def _read_cmapss(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS)
    return df


def load_cmapss(
    root: str | Path,
    fd: str = "FD001",
    seq_len: int = 30,
    rul_cap: float = 125.0,
    stride_train: int = 1,
):
    root = Path(root)
    train = _read_cmapss(root / f"train_{fd}.txt")
    test = _read_cmapss(root / f"test_{fd}.txt")
    rul_true = pd.read_csv(root / f"RUL_{fd}.txt", sep=r"\s+", header=None, names=["RUL"])

    train = add_rul_labels(train, cap=rul_cap)

    # Attach true RUL for test (RUL at last cycle of each unit)
    test_max = test.groupby("unit")["cycle"].transform("max")
    # placeholder; fill only last cycle then forward for windows ending earlier via piecewise
    last_rul = rul_true["RUL"].to_numpy()
    unit_last_rul = {u: float(last_rul[i]) for i, u in enumerate(sorted(test["unit"].unique()))}
    test = test.copy()
    test["RUL"] = test["unit"].map(unit_last_rul) + (test_max - test["cycle"])
    if rul_cap is not None:
        test["RUL"] = np.minimum(test["RUL"], rul_cap)

    X_tr, y_tr, _ = build_windows_from_units(
        train, FEATURE_COLS, "RUL", seq_len=seq_len, last_only=False
    )
    if stride_train > 1:
        X_tr, y_tr = X_tr[::stride_train], y_tr[::stride_train]

    X_te, y_te, _ = build_windows_from_units(
        test, FEATURE_COLS, "RUL", seq_len=seq_len, last_only=True
    )

    # hold out 15% of train units for validation
    units = train["unit"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(units)
    n_val = max(1, int(0.15 * len(units)))
    val_units = set(units[:n_val])
    train_units = set(units[n_val:])

    train_mask = train["unit"].isin(train_units)
    val_mask = train["unit"].isin(val_units)
    X_tr2, y_tr2, _ = build_windows_from_units(
        train.loc[train_mask], FEATURE_COLS, "RUL", seq_len=seq_len, last_only=False
    )
    X_va, y_va, _ = build_windows_from_units(
        train.loc[val_mask], FEATURE_COLS, "RUL", seq_len=seq_len, last_only=False
    )
    if stride_train > 1:
        X_tr2, y_tr2 = X_tr2[::stride_train], y_tr2[::stride_train]
        X_va, y_va = X_va[::stride_train], y_va[::stride_train]

    X_tr2, (X_va, X_te), _ = fit_scale_windows(X_tr2, [X_va, X_te])

    return {
        "name": f"CMAPSS-{fd}",
        "task": "regression",
        "tol": 10.0,
        "X_train": X_tr2,
        "y_train": y_tr2.astype(np.float32),
        "X_val": X_va,
        "y_val": y_va.astype(np.float32),
        "X_test": X_te,
        "y_test": y_te.astype(np.float32),
    }
