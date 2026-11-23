#!/usr/bin/env python3
"""
Within-family model comparison using each family's native metrics.

Families
--------
RUL  (CMAPSS FD001–FD004, PHM08, N-CMAPSS):
  primary = RMSE (lower better); also MAE, R², PHM Score
SCANIA:
  primary = Total_cost (lower better); also macro-F1, per-class recall, accuracy

No cross-family leaderboard is produced.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import (  # noqa: E402
    load_raw_cmapss,
    load_raw_n_cmapss,
    load_raw_phm08,
)
from datasets.scania import load_scania  # noqa: E402
from datasets.sequence_utils import build_windows_from_units  # noqa: E402
from metrics import rul_family_metrics, scania_metrics  # noqa: E402
from models import (  # noqa: E402
    LSTMModel,
    make_gradient_boosting,
    make_linear,
    make_random_forest,
)
from training_utils import balance_resample  # noqa: E402

SEED = 42
EPOCHS = 20
EPOCHS_HARD = 30  # multi-condition / hard RUL sets
PATIENCE = 5

# Multi-condition sets get stronger LSTM + all features
HARD_RUL = {"CMAPSS-FD002", "CMAPSS-FD004", "PHM08"}


def _scale_train_only(X_tr, X_va, X_te):
    n, t, f = X_tr.shape
    scaler = StandardScaler().fit(X_tr.reshape(-1, f))
    X_tr = scaler.transform(X_tr.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
    X_va = scaler.transform(X_va.reshape(-1, f)).reshape(X_va.shape).astype(np.float32)
    X_te = scaler.transform(X_te.reshape(-1, f)).reshape(X_te.shape).astype(np.float32)
    return X_tr, X_va, X_te


def pack_to_arrays(pack: dict):
    """Build scaled windows from a raw RUL pack (ablation_data format)."""
    hard = pack["name"] in HARD_RUL
    feat_key = "all" if hard and "all" in pack["feature_presets"] else pack["default_features"]
    feats = pack["feature_presets"][feat_key]
    seq = int(pack["default_seq_len"])
    stride = int(pack.get("stride_train", 1))

    X_tr, y_tr, _ = build_windows_from_units(
        pack["train_df"], feats, "RUL", seq_len=seq, last_only=False
    )
    X_va, y_va, _ = build_windows_from_units(
        pack["val_df"], feats, "RUL", seq_len=seq, last_only=False
    )
    X_te, y_te, _ = build_windows_from_units(
        pack["test_df"], feats, "RUL", seq_len=seq, last_only=True
    )
    if stride > 1:
        X_tr, y_tr = X_tr[::stride], y_tr[::stride]
        X_va, y_va = X_va[::stride], y_va[::stride]
    X_tr, X_va, X_te = _scale_train_only(X_tr, X_va, X_te)
    return {
        "name": pack["name"],
        "family": "RUL",
        "hard": hard,
        "X_train": X_tr,
        "y_train": y_tr.astype(np.float32),
        "X_val": X_va,
        "y_val": y_va.astype(np.float32),
        "X_test": X_te,
        "y_test": y_te.astype(np.float32),
        "note": pack.get("note"),
        "feature_set": feat_key,
    }


def make_models(task: str, n_classes: int | None = None, hard: bool = False):
    epochs = EPOCHS_HARD if hard and task == "regression" else EPOCHS
    hidden = 128 if hard and task == "regression" else 64
    lstm_kw = dict(
        task=task,
        hidden_dim=hidden,
        num_layers=2,
        dropout=0.2,
        epochs=epochs,
        batch_size=64,
        patience=PATIENCE,
        n_classes=n_classes,
        seed=SEED,
    )
    if task == "classification":
        lstm_kw["class_weight"] = "balanced"
    return {
        "LSTM": LSTMModel(**lstm_kw),
        "RandomForest": make_random_forest(task, seed=SEED),
        "GradientBoosting": make_gradient_boosting(task, seed=SEED),
        "Linear": make_linear(task, seed=SEED),
    }


def evaluate_rul_dataset(bundle: dict) -> list[dict]:
    rows = []
    print(f"\n===== RUL family :: {bundle['name']} =====")
    if bundle.get("note"):
        print(f"note: {bundle['note']}")
    print(
        f"shapes: {bundle['X_train'].shape} / {bundle['X_val'].shape} / {bundle['X_test'].shape}"
    )
    hard = bool(bundle.get("hard"))
    if hard:
        print(f"hard-set mode: features={bundle.get('feature_set')} epochs={EPOCHS_HARD} hidden=128")
    for name, model in make_models("regression", hard=hard).items():
        print(f"  -> {name} ...", flush=True)
        t0 = time.time()
        model.fit(bundle["X_train"], bundle["y_train"], bundle["X_val"], bundle["y_val"])
        pred = np.maximum(model.predict(bundle["X_test"]), 0.0)
        m = rul_family_metrics(bundle["y_test"], pred)
        m.update(
            {
                "dataset": bundle["name"],
                "model": name,
                "seconds": round(time.time() - t0, 2),
                "status": "ok",
                "hard_mode": hard,
            }
        )
        print(
            f"     RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}  R²={m['r2']:.3f}  "
            f"PHM={m['phm_score']:.1f}  (±10={m['tol_acc_10']:.3f})"
        )
        rows.append(m)
    return rows


def evaluate_scania() -> list[dict]:
    print("\n===== SCANIA family :: Component-X classification =====")
    print("primary metric: Total_cost (IDA challenge cost matrix; lower is better)")
    bundle = load_scania(
        WORKSPACE / "SCANIA-Component-X" / "data",
        seq_len=8,
        max_train_vehicles=3500,
        max_windows_per_vehicle=5,
    )
    print(
        f"shapes: {bundle['X_train'].shape} / {bundle['X_val'].shape} / {bundle['X_test'].shape}"
    )
    # Class-balanced train resample (cost-aware setup); val unchanged for early stopping
    X_tr_b, y_tr_b = balance_resample(
        bundle["X_train"], bundle["y_train"], max_per_class=2000, seed=SEED
    )
    print(
        f"balanced train: {X_tr_b.shape} class counts={dict(zip(*np.unique(y_tr_b, return_counts=True)))}"
    )
    rows = []
    for name, model in make_models("classification", n_classes=5).items():
        print(f"  -> {name} ...", flush=True)
        t0 = time.time()
        model.fit(X_tr_b, y_tr_b, bundle["X_val"], bundle["y_val"])
        pred = model.predict(bundle["X_test"])
        m = scania_metrics(bundle["y_test"], pred)
        m.update(
            {
                "dataset": "SCANIA-Component-X",
                "model": name,
                "seconds": round(time.time() - t0, 2),
                "status": "ok",
                "train_balanced": True,
            }
        )
        print(
            f"     Cost={m['total_cost']:.0f}  F1_macro={m['f1_macro']:.4f}  "
            f"Acc={m['accuracy']:.4f}  "
            f"R1-4=[{m['recall_1']:.2f},{m['recall_2']:.2f},{m['recall_3']:.2f},{m['recall_4']:.2f}]"
        )
        rows.append(m)
    return rows


def rank_family(df: pd.DataFrame, primary: str, lower_is_better: bool) -> pd.DataFrame:
    out = df.copy()
    ascending = lower_is_better
    out["rank"] = out.groupby("dataset")[primary].rank(method="min", ascending=ascending)
    return out.sort_values(["dataset", "rank", primary], ascending=[True, True, ascending])


def main():
    out_dir = ROOT / "results" / "family_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    rul_rows = []
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        rul_rows.extend(evaluate_rul_dataset(pack_to_arrays(load_raw_cmapss(fd))))
    rul_rows.extend(evaluate_rul_dataset(pack_to_arrays(load_raw_phm08())))
    rul_rows.extend(evaluate_rul_dataset(pack_to_arrays(load_raw_n_cmapss())))

    scania_rows = evaluate_scania()

    rul_df = pd.DataFrame(rul_rows)
    scania_df = pd.DataFrame(scania_rows)

    rul_ranked = rank_family(rul_df, "rmse", lower_is_better=True)
    scania_ranked = rank_family(scania_df, "total_cost", lower_is_better=True)

    rul_ranked.to_csv(out_dir / "rul_family_metrics.csv", index=False)
    scania_ranked.to_csv(out_dir / "scania_family_metrics.csv", index=False)

    # Winners per dataset (within family only)
    rul_winners = (
        rul_ranked.sort_values(["dataset", "rank"])
        .groupby("dataset", as_index=False)
        .first()[["dataset", "model", "rmse", "mae", "r2", "phm_score", "tol_acc_10"]]
    )
    scania_winners = (
        scania_ranked.sort_values(["dataset", "rank"])
        .groupby("dataset", as_index=False)
        .first()[
            [
                "dataset",
                "model",
                "total_cost",
                "f1_macro",
                "accuracy",
                "recall_1",
                "recall_2",
                "recall_3",
                "recall_4",
            ]
        ]
    )

    summary = {
        "plan": "Compare models within each dataset/family using that family's metrics only.",
        "families": {
            "RUL": {
                "datasets": sorted(rul_df["dataset"].unique().tolist()),
                "primary_metric": "rmse",
                "also": ["mae", "r2", "phm_score", "tol_acc_10"],
                "winners": rul_winners.to_dict(orient="records"),
            },
            "SCANIA": {
                "datasets": ["SCANIA-Component-X"],
                "primary_metric": "total_cost",
                "also": ["f1_macro", "accuracy", "recall_1..4"],
                "winners": scania_winners.to_dict(orient="records"),
            },
        },
        "note": "No cross-family ranking is performed.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n========== RUL FAMILY WINNERS (lowest RMSE) ==========")
    print(rul_winners.to_string(index=False))
    print("\n========== SCANIA FAMILY WINNER (lowest Total_cost) ==========")
    print(scania_winners.to_string(index=False))
    print(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()
