#!/usr/bin/env python3
"""Expand N-CMAPSS coverage: compare models across multiple DS files (RUL metrics)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from datasets.ablation_data import load_raw_n_cmapss  # noqa: E402
from run_family_comparison import make_models, pack_to_arrays  # noqa: E402
from metrics import rul_family_metrics  # noqa: E402

# Prefer intact files; skip known truncated DS08d
DATASETS = [
    "N-CMAPSS_DS01-005.h5",
    "N-CMAPSS_DS02-006.h5",
    "N-CMAPSS_DS03-012.h5",
    "N-CMAPSS_DS05.h5",
    "N-CMAPSS_DS07.h5",
]


def main():
    out = ROOT / "results" / "family_comparison"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname in DATASETS:
        print(f"\n===== {fname} =====", flush=True)
        try:
            pack = load_raw_n_cmapss(filename=fname)
            bundle = pack_to_arrays(pack)
        except Exception as e:
            print(f"  skip: {e}")
            continue
        print(
            f"  units train/val/test="
            f"{pack['train_df']['unit'].nunique()}/"
            f"{pack['val_df']['unit'].nunique()}/"
            f"{pack['test_df']['unit'].nunique()}  "
            f"n_test_windows={len(bundle['y_test'])}"
        )
        for name, model in make_models("regression", hard=False).items():
            t0 = time.time()
            print(f"  -> {name} ...", flush=True)
            model.fit(bundle["X_train"], bundle["y_train"], bundle["X_val"], bundle["y_val"])
            pred = np.maximum(model.predict(bundle["X_test"]), 0.0)
            m = rul_family_metrics(bundle["y_test"], pred)
            m.update(
                {
                    "dataset": bundle["name"],
                    "source_file": fname,
                    "model": name,
                    "seconds": round(time.time() - t0, 2),
                    "status": "ok",
                }
            )
            print(f"     RMSE={m['rmse']:.3f} MAE={m['mae']:.3f} PHM={m['phm_score']:.1f}")
            rows.append(m)

    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("dataset")["rmse"].rank(method="min", ascending=True)
    df = df.sort_values(["dataset", "rank"])
    df.to_csv(out / "n_cmapss_multi_metrics.csv", index=False)
    winners = (
        df.sort_values(["dataset", "rank"]).groupby("dataset", as_index=False).first()[
            ["dataset", "source_file", "model", "rmse", "mae", "phm_score", "n_test"]
        ]
    )
    summary = {
        "primary_metric": "rmse",
        "files": DATASETS,
        "winners": winners.to_dict(orient="records"),
        "pooled_mean_rmse_by_model": df.groupby("model")["rmse"]
        .mean()
        .sort_values()
        .to_dict(),
    }
    (out / "n_cmapss_multi_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== Winners per DS file ===")
    print(winners.to_string(index=False))
    print("\n=== Mean RMSE across files ===")
    print(df.groupby("model")["rmse"].mean().sort_values())
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
