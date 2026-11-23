"""Classical baselines that consume flattened sequence windows."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape(X.shape[0], -1)


class FlatModel:
    """Wraps an sklearn estimator that expects flattened (N, T*F) inputs."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, X_val=None, y_val=None):
        self.estimator.fit(flatten_sequences(X), np.asarray(y).ravel())
        return self

    def predict(self, X):
        return self.estimator.predict(flatten_sequences(X))


def make_random_forest(task: str, seed: int = 42):
    if task == "classification":
        est = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample",
        )
    else:
        est = RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            n_jobs=-1,
            random_state=seed,
        )
    return FlatModel(est)


def make_gradient_boosting(task: str, seed: int = 42):
    if task == "classification":
        est = HistGradientBoostingClassifier(
            max_depth=8,
            max_iter=100,
            learning_rate=0.08,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        est = HistGradientBoostingRegressor(
            max_depth=8,
            max_iter=100,
            learning_rate=0.08,
            random_state=seed,
        )
    return FlatModel(est)


def make_linear(task: str, seed: int = 42):
    if task == "classification":
        est = LogisticRegression(
            max_iter=500,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced",
        )
    else:
        est = Ridge(alpha=1.0)
    return FlatModel(est)
