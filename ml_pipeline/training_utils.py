"""Shared training helpers for within-family improvements."""

from __future__ import annotations

import numpy as np


def balance_resample(
    X: np.ndarray,
    y: np.ndarray,
    max_per_class: int | None = 2500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Upsample minority classes (with replacement) for imbalanced classification."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).ravel()
    X = np.asarray(X)
    classes = np.unique(y)
    target = max(int(np.bincount(y).max()), 1)
    if max_per_class is not None:
        target = min(target, max_per_class)

    Xs, ys = [], []
    for c in classes:
        idx = np.flatnonzero(y == c)
        if len(idx) == 0:
            continue
        take = rng.choice(idx, size=target, replace=len(idx) < target)
        Xs.append(X[take])
        ys.append(y[take])
    order = rng.permutation(sum(len(a) for a in ys))
    Xb = np.concatenate(Xs, axis=0)[order]
    yb = np.concatenate(ys, axis=0)[order]
    return Xb, yb
