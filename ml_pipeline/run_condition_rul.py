#!/usr/bin/env python3
"""
Multi-regime RUL: cluster operating conditions (op1/op2/op3) on TRAIN only,
then z-score sensors within each cluster before windowing/modeling.
Compare vs plain scaling on FD002/FD004 (family metric = RMSE).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import FEAT_ALL, FEAT_SELECTED, load_raw_cmapss  # noqa: E402
from datasets.sequence_utils import build_windows_from_units  # noqa: E402
from metrics import rul_family_metrics  # noqa: E402
from models import LSTMModel, make_gradient_boosting, make_linear, make_random_forest  # noqa: E402

SEED = 42
OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [c for c in FEAT_ALL if c.startswith("s")]


def _condition_normalize(train_df, val_df, test_df, n_clusters: int = 6):
    """Fit KMeans + per-cluster sensor scalers on TRAIN units only."""
    rng_state = SEED
    km = KMeans(n_clusters=n_clusters, random_state=rng_state, n_init=10)
    op_tr = train_df[OP_COLS].to_numpy(dtype=np.float64)
    # standardize ops before clustering (train-fit)
    op_scaler = StandardScaler().fit(op_tr)
    km.fit(op_scaler.transform(op_tr))

    def _assign(df):
        labs = km.predict(op_scaler.transform(df[OP_COLS].to_numpy(dtype=np.float64)))
        out = df.copy()
        out["_cond"] = labs
        return out

    train_df = _assign(train_df)
    val_df = _assign(val_df)
    test_df = _assign(test_df)

    scalers = {}
    for c in range(n_clusters):
        mask = train_df["_cond"].to_numpy() == c
        if mask.sum() < 10:
            continue
        sc = StandardScaler().fit(train_df.loc[mask, SENSOR_COLS].to_numpy(dtype=np.float64))
        scalers[c] = sc

    # fallback global sensor scaler
    global_sc = StandardScaler().fit(train_df[SENSOR_COLS].to_numpy(dtype=np.float64))

    def _apply(df):
        out = df.copy()
        arr = out[SENSOR_COLS].to_numpy(dtype=np.float64)
        for c, sc in scalers.items():
            m = out["_cond"].to_numpy() == c
            if m.any():
                arr[m] = sc.transform(arr[m])
        # any cluster without scaler
        missing = ~np.isin(out["_cond"].to_numpy(), list(scalers.keys()))
        if missing.any():
            arr[missing] = global_sc.transform(arr[missing])
        out[SENSOR_COLS] = arr
        # also scale ops with train op_scaler for model input
        out[OP_COLS] = op_scaler.transform(out[OP_COLS].to_numpy(dtype=np.float64))
        return out

    return _apply(train_df), _apply(val_df), _apply(test_df)


def _plain_scale_windows(train_df, val_df, test_df, feats, seq_len=30, stride=2):
    X_tr, y_tr, _ = build_windows_from_units(train_df, feats, "RUL", seq_len=seq_len)
    X_va, y_va, _ = build_windows_from_units(val_df, feats, "RUL", seq_len=seq_len)
    X_te, y_te, _ = build_windows_from_units(test_df, feats, "RUL", seq_len=seq_len, last_only=True)
    if stride > 1:
        X_tr, y_tr = X_tr[::stride], y_tr[::stride]
        X_va, y_va = X_va[::stride], y_va[::stride]
    n, t, f = X_tr.shape
    sc = StandardScaler().fit(X_tr.reshape(-1, f))
    X_tr = sc.transform(X_tr.reshape(-1, f)).reshape(n, t, f).astype(np.float32)
    X_va = sc.transform(X_va.reshape(-1, f)).reshape(X_va.shape).astype(np.float32)
    X_te = sc.transform(X_te.reshape(-1, f)).reshape(X_te.shape).astype(np.float32)
    return X_tr, y_tr.astype(np.float32), X_va, y_va.astype(np.float32), X_te, y_te.astype(np.float32)


def _cond_windows(train_df, val_df, test_df, feats, seq_len=30, stride=2):
    tr, va, te = _condition_normalize(train_df, val_df, test_df, n_clusters=6)
    # features already scaled in-dataframe; identity scale via windows
    X_tr, y_tr, _ = build_windows_from_units(tr, feats, "RUL", seq_len=seq_len)
    X_va, y_va, _ = build_windows_from_units(va, feats, "RUL", seq_len=seq_len)
    X_te, y_te, _ = build_windows_from_units(te, feats, "RUL", seq_len=seq_len, last_only=True)
    if stride > 1:
        X_tr, y_tr = X_tr[::stride], y_tr[::stride]
        X_va, y_va = X_va[::stride], y_va[::stride]
    return (
        X_tr.astype(np.float32),
        y_tr.astype(np.float32),
        X_va.astype(np.float32),
        y_va.astype(np.float32),
        X_te.astype(np.float32),
        y_te.astype(np.float32),
    )


def _models():
    return {
        "LSTM": LSTMModel(
            task="regression",
            hidden_dim=128,
            num_layers=2,
            dropout=0.2,
            epochs=25,
            patience=5,
            seed=SEED,
        ),
        "RandomForest": make_random_forest("regression", seed=SEED),
        "GradientBoosting": make_gradient_boosting("regression", seed=SEED),
        "Linear": make_linear("regression", seed=SEED),
    }


def run_fd(fd: str):
    pack = load_raw_cmapss(fd)
    feats = FEAT_ALL
    rows = []
    for mode, builder in (
        ("plain", _plain_scale_windows),
        ("cond_cluster", _cond_windows),
    ):
        print(f"\n=== {pack['name']} :: {mode} ===", flush=True)
        X_tr, y_tr, X_va, y_va, X_te, y_te = builder(
            pack["train_df"], pack["val_df"], pack["test_df"], feats, seq_len=30, stride=2
        )
        for name, model in _models().items():
            print(f"  -> {name} ...", flush=True)
            t0 = time.time()
            model.fit(X_tr, y_tr, X_va, y_va)
            pred = np.maximum(model.predict(X_te), 0.0)
            m = rul_family_metrics(y_te, pred)
            m.update(
                {
                    "dataset": pack["name"],
                    "mode": mode,
                    "model": name,
                    "seconds": round(time.time() - t0, 2),
                }
            )
            print(f"     RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} R2={m['r2']:.3f} PHM={m['phm_score']:.1f}")
            rows.append(m)
    return rows


def main():
    out = ROOT / "results" / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fd in ("FD002", "FD004"):
        rows.extend(run_fd(fd))
    df = pd.DataFrame(rows)
    df.to_csv(out / "condition_rul_metrics.csv", index=False)

    # best per dataset by RMSE
    best = df.sort_values("rmse").groupby("dataset", as_index=False).first()[
        ["dataset", "mode", "model", "rmse", "mae", "r2", "phm_score"]
    ]
    # delta: best cond vs best plain per dataset
    deltas = []
    for ds, g in df.groupby("dataset"):
        plain = g[g["mode"] == "plain"].sort_values("rmse").iloc[0]
        cond = g[g["mode"] == "cond_cluster"].sort_values("rmse").iloc[0]
        deltas.append(
            {
                "dataset": ds,
                "plain_best": f"{plain.model} RMSE={plain.rmse:.3f}",
                "cond_best": f"{cond.model} RMSE={cond.rmse:.3f}",
                "delta_rmse_cond_minus_plain": float(cond.rmse - plain.rmse),
            }
        )
    summary = {"winners": best.to_dict(orient="records"), "cond_vs_plain": deltas}
    (out / "condition_rul_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Best overall ===")
    print(best.to_string(index=False))
    print("\n=== Cond vs plain ===")
    print(pd.DataFrame(deltas).to_string(index=False))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
