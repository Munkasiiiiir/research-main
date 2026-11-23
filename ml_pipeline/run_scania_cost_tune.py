#!/usr/bin/env python3
"""
SCANIA val-cost tuning: fit on balanced train, tune decision rule on validation
Total_cost, then evaluate once on test (family metric = Total_cost).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from datasets.scania import load_scania  # noqa: E402
from metrics import scania_metrics, scania_total_cost  # noqa: E402
from models import LSTMModel, make_gradient_boosting, make_random_forest  # noqa: E402
from models.baselines import flatten_sequences  # noqa: E402
from training_utils import balance_resample  # noqa: E402

SEED = 42


def _predict_proba(model, X):
    if isinstance(model, LSTMModel):
        import torch

        model.model.eval()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = model.model(torch.from_numpy(X).to(model.device))
            prob = torch.softmax(logits, dim=1).cpu().numpy()
        return prob
    # FlatModel wrappers
    est = model.estimator
    return est.predict_proba(flatten_sequences(X))


def _cost_decode(proba: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Bayes decision under SCANIA cost matrix: argmin_j E[cost | pred=j]."""
    # expected cost for predicting class j: sum_i P(i) * cost[i,j]
    # proba: (N,C), cost: (C,C) with [actual, predicted]
    exp = proba @ cost  # (N,C)
    return exp.argmin(axis=1).astype(int)


def main():
    from metrics import SCANIA_COST

    out = ROOT / "results" / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_scania(
        WORKSPACE / "SCANIA-Component-X" / "data",
        seq_len=8,
        max_train_vehicles=3500,
        max_windows_per_vehicle=5,
    )
    X_tr, y_tr = balance_resample(
        bundle["X_train"], bundle["y_train"], max_per_class=2000, seed=SEED
    )
    X_va, y_va = bundle["X_val"], bundle["y_val"]
    X_te, y_te = bundle["X_test"], bundle["y_test"]

    models = {
        "LSTM": LSTMModel(
            task="classification",
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            epochs=20,
            patience=5,
            n_classes=5,
            class_weight="balanced",
            seed=SEED,
        ),
        "RandomForest": make_random_forest("classification", seed=SEED),
        "GradientBoosting": make_gradient_boosting("classification", seed=SEED),
    }

    rows = []
    for name, model in models.items():
        print(f"-> {name}: fit ...", flush=True)
        model.fit(X_tr, y_tr, X_va, y_va)

        # baseline argmax
        if isinstance(model, LSTMModel):
            pred_te_base = model.predict(X_te)
            pred_va_base = model.predict(X_va)
        else:
            pred_te_base = model.predict(X_te)
            pred_va_base = model.predict(X_va)

        base_va = scania_total_cost(y_va, pred_va_base)
        base_te = scania_metrics(y_te, pred_te_base)

        # cost-sensitive decode from probabilities
        p_va = _predict_proba(model, X_va)
        p_te = _predict_proba(model, X_te)
        pred_va_cs = _cost_decode(p_va, SCANIA_COST)
        pred_te_cs = _cost_decode(p_te, SCANIA_COST)
        cs_va = scania_total_cost(y_va, pred_va_cs)
        cs_te = scania_metrics(y_te, pred_te_cs)

        chosen = "cost_decode" if cs_va <= base_va else "argmax"
        final = cs_te if chosen == "cost_decode" else base_te
        print(
            f"   val cost argmax={base_va:.0f} cost_decode={cs_va:.0f} -> use {chosen}; "
            f"test cost={final['total_cost']:.0f} F1={final['f1_macro']:.4f}"
        )
        rows.append(
            {
                "dataset": "SCANIA-Component-X",
                "model": name,
                "decision_rule": chosen,
                "val_cost_argmax": float(base_va),
                "val_cost_decode": float(cs_va),
                **{f"test_{k}": final[k] for k in ("total_cost", "f1_macro", "accuracy", "recall_1", "recall_2", "recall_3", "recall_4")},
            }
        )

    df = pd.DataFrame(rows).sort_values("test_total_cost")
    df.to_csv(out / "scania_cost_tune.csv", index=False)
    (out / "scania_cost_tune.json").write_text(json.dumps(rows, indent=2))
    print("\n=== Ranked by test Total_cost ===")
    print(df[["model", "decision_rule", "test_total_cost", "test_f1_macro", "test_accuracy"]].to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
