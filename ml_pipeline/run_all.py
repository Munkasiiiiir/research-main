#!/usr/bin/env python3
"""End-to-end ML pipeline across available research datasets."""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
sys.path.insert(0, str(ROOT))

from datasets import load_cmapss, load_n_cmapss, load_phm08, load_scania  # noqa: E402
from metrics import classification_report_metrics, regression_report  # noqa: E402
from models import (  # noqa: E402
    LSTMModel,
    make_gradient_boosting,
    make_linear,
    make_random_forest,
)


def build_models(task: str, n_classes: int | None = None, seed: int = 42):
    models = {
        "LSTM": LSTMModel(
            task=task,
            hidden_dim=64,
            num_layers=2,
            dropout=0.2,
            epochs=12,
            batch_size=64,
            patience=3,
            n_classes=n_classes,
            seed=seed,
        ),
        "RandomForest": make_random_forest(task, seed=seed),
        "GradientBoosting": make_gradient_boosting(task, seed=seed),
        "Linear": make_linear(task, seed=seed),
    }
    return models


def evaluate_bundle(bundle: dict) -> list[dict]:
    task = bundle["task"]
    name = bundle["name"]
    n_classes = bundle.get("n_classes")
    tol = bundle.get("tol", 10.0)
    results = []

    print(f"\n=== Dataset: {name} ({task}) ===")
    print(
        f"shapes train/val/test: "
        f"{bundle['X_train'].shape} / {bundle['X_val'].shape} / {bundle['X_test'].shape}"
    )
    if bundle.get("note"):
        print(f"note: {bundle['note']}")

    models = build_models(task, n_classes=n_classes)
    for model_name, model in models.items():
        t0 = time.time()
        print(f"  -> Training {model_name} ...", flush=True)
        try:
            model.fit(
                bundle["X_train"],
                bundle["y_train"],
                X_val=bundle["X_val"],
                y_val=bundle["y_val"],
            )
            pred = model.predict(bundle["X_test"])
            if task == "classification":
                metrics = classification_report_metrics(bundle["y_test"], pred)
            else:
                metrics = regression_report(bundle["y_test"], pred, tol=float(tol))
            metrics.update(
                {
                    "dataset": name,
                    "model": model_name,
                    "seconds": round(time.time() - t0, 2),
                    "n_train": int(len(bundle["y_train"])),
                    "n_test": int(len(bundle["y_test"])),
                    "status": "ok",
                }
            )
            acc = metrics["accuracy"]
            extra = (
                f"F1={metrics['f1_macro']:.4f}"
                if task == "classification"
                else f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f}"
            )
            print(f"     {model_name}: accuracy={acc:.4f}  {extra}  ({metrics['seconds']}s)")
        except Exception as exc:
            traceback.print_exc()
            metrics = {
                "dataset": name,
                "model": model_name,
                "task": task,
                "accuracy": None,
                "status": f"error: {exc}",
                "seconds": round(time.time() - t0, 2),
            }
            print(f"     {model_name}: FAILED ({exc})")
        results.append(metrics)
    return results


def load_all_datasets():
    bundles = []

    # CMAPSS — all four sub-datasets
    cmapss_root = WORKSPACE / "CMAPSS"
    for fd in ("FD001", "FD002", "FD003", "FD004"):
        print(f"Loading CMAPSS {fd} ...")
        stride = 1 if fd in ("FD001", "FD003") else 2
        bundles.append(load_cmapss(cmapss_root, fd=fd, seq_len=30, stride_train=stride))

    print("Loading PHM08 ...")
    bundles.append(load_phm08(WORKSPACE / "PHM08", seq_len=30, stride=2))

    print("Loading N-CMAPSS (DS01 cycle-aggregated) ...")
    bundles.append(
        load_n_cmapss(
            WORKSPACE / "N-CMAPSS" / "data_set",
            filename="N-CMAPSS_DS01-005.h5",
            seq_len=20,
        )
    )

    print("Loading SCANIA Component X ...")
    bundles.append(
        load_scania(
            WORKSPACE / "SCANIA-Component-X" / "data",
            seq_len=8,
            max_train_vehicles=3500,
            max_windows_per_vehicle=5,
        )
    )

    # ALFA tools repo has no flight data files locally
    alfa_dir = WORKSPACE / "alfa-dataset-tools"
    if alfa_dir.exists():
        print(
            "Skipping ALFA: only tooling is present locally "
            "(no flight/sequence data files to train on)."
        )

    return bundles


def main():
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundles = load_all_datasets()
    all_results = []
    for bundle in bundles:
        all_results.extend(evaluate_bundle(bundle))

    df = pd.DataFrame(all_results)
    csv_path = out_dir / "metrics.csv"
    json_path = out_dir / "metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(all_results, indent=2, default=str))

    ok = df[df["status"] == "ok"].copy()
    if len(ok):
        best = ok.loc[ok["accuracy"].astype(float).idxmax()]
        summary = {
            "best_dataset": best["dataset"],
            "best_model": best["model"],
            "best_accuracy": float(best["accuracy"]),
            "best_task": best["task"],
            "leaderboard": ok.sort_values("accuracy", ascending=False)[
                ["dataset", "model", "accuracy", "task"]
            ].to_dict(orient="records"),
        }
    else:
        summary = {"error": "no successful runs"}

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n========== SUMMARY ==========")
    if "best_dataset" in summary:
        print(
            f"Highest accuracy: {summary['best_accuracy']:.4f} "
            f"by {summary['best_model']} on {summary['best_dataset']}"
        )
        print("\nLeaderboard (by accuracy):")
        for row in summary["leaderboard"]:
            print(f"  {row['accuracy']:.4f}  {row['dataset']:28s}  {row['model']}")
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
