"""SCANIA Component X multi-class failure-window classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _failure_class(time_to_fail: float) -> int:
    """Map time-to-failure to challenge classes (windows before failure)."""
    if time_to_fail < 0:
        return 0
    if time_to_fail < 6:
        return 4
    if time_to_fail < 12:
        return 3
    if time_to_fail < 24:
        return 2
    if time_to_fail < 48:
        return 1
    return 0


def _encode_specs(specs: pd.DataFrame, ref_columns: list[str] | None = None):
    cat = pd.get_dummies(specs.drop(columns=["vehicle_id"]), prefix_sep="=")
    if ref_columns is not None:
        cat = cat.reindex(columns=ref_columns, fill_value=0)
    return cat.to_numpy(dtype=np.float32), list(cat.columns)


def _build_vehicle_sequences(
    ops: pd.DataFrame,
    labels: pd.Series,
    specs_map: dict[int, np.ndarray],
    seq_len: int,
    feature_cols: list[str],
    max_per_vehicle: int | None = None,
    rng: np.random.Generator | None = None,
):
    Xs, ys = [], []
    for vid, g in ops.groupby("vehicle_id", sort=False):
        if vid not in labels.index:
            continue
        g = g.sort_values("time_step")
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        # labels aligned by index of g
        y_all = labels.loc[g.index].to_numpy()
        spec = specs_map.get(int(vid))
        if spec is None:
            continue
        n = len(g)
        ends = list(range(n))
        if max_per_vehicle is not None and n > max_per_vehicle:
            assert rng is not None
            ends = sorted(rng.choice(n, size=max_per_vehicle, replace=False).tolist())

        for end in ends:
            start = max(0, end - seq_len + 1)
            window = feats[start : end + 1]
            if len(window) < seq_len:
                pad = np.repeat(window[:1], seq_len - len(window), axis=0)
                window = np.vstack([pad, window])
            # append static specs to every timestep
            spec_tile = np.repeat(spec.reshape(1, -1), seq_len, axis=0)
            Xs.append(np.hstack([window, spec_tile]))
            ys.append(int(y_all[end]))
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _last_readout_sequences(
    ops: pd.DataFrame,
    labels: pd.DataFrame,
    specs_map: dict[int, np.ndarray],
    seq_len: int,
    feature_cols: list[str],
):
    label_map = labels.set_index("vehicle_id")["class_label"]
    Xs, ys, vids = [], [], []
    for vid, g in ops.groupby("vehicle_id", sort=False):
        if vid not in label_map.index:
            continue
        g = g.sort_values("time_step")
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        spec = specs_map.get(int(vid))
        if spec is None:
            continue
        window = feats[-seq_len:]
        if len(window) < seq_len:
            pad = np.repeat(window[:1], seq_len - len(window), axis=0)
            window = np.vstack([pad, window])
        spec_tile = np.repeat(spec.reshape(1, -1), seq_len, axis=0)
        Xs.append(np.hstack([window, spec_tile]))
        ys.append(int(label_map.loc[vid]))
        vids.append(int(vid))
    return np.asarray(Xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(vids)


def load_scania(
    root: str | Path,
    seq_len: int = 8,
    max_train_vehicles: int = 4000,
    max_windows_per_vehicle: int = 6,
):
    root = Path(root)
    tte = pd.read_csv(root / "train_tte.csv")
    train_specs = pd.read_csv(root / "train_specifications.csv")
    val_specs = pd.read_csv(root / "validation_specifications.csv")
    test_specs = pd.read_csv(root / "test_specifications.csv")
    val_labels = pd.read_csv(root / "validation_labels.csv")
    test_labels = pd.read_csv(root / "test_labels.csv")

    # Spec encoding fitted on all splits for stable columns
    all_specs = pd.concat([train_specs, val_specs, test_specs], ignore_index=True)
    _, spec_cols = _encode_specs(all_specs)
    train_spec_m, _ = _encode_specs(train_specs, spec_cols)
    val_spec_m, _ = _encode_specs(val_specs, spec_cols)
    test_spec_m, _ = _encode_specs(test_specs, spec_cols)
    train_specs_map = {
        int(v): train_spec_m[i] for i, v in enumerate(train_specs["vehicle_id"].to_numpy())
    }
    val_specs_map = {
        int(v): val_spec_m[i] for i, v in enumerate(val_specs["vehicle_id"].to_numpy())
    }
    test_specs_map = {
        int(v): test_spec_m[i] for i, v in enumerate(test_specs["vehicle_id"].to_numpy())
    }

    # Sample train vehicles for tractable runtime
    rng = np.random.default_rng(42)
    all_train_ids = tte["vehicle_id"].to_numpy()
    if len(all_train_ids) > max_train_vehicles:
        # keep all repaired vehicles + random healthy ones
        repaired = tte.loc[tte["in_study_repair"] == 1, "vehicle_id"].to_numpy()
        healthy = tte.loc[tte["in_study_repair"] == 0, "vehicle_id"].to_numpy()
        n_healthy = max(0, max_train_vehicles - len(repaired))
        pick_h = rng.choice(healthy, size=min(n_healthy, len(healthy)), replace=False)
        selected = set(repaired.tolist() + pick_h.tolist())
    else:
        selected = set(all_train_ids.tolist())

    usecols = None  # load all; filter vehicles after
    print(f"[SCANIA] Loading operational readouts (selected {len(selected)} train vehicles)...")
    train_ops = pd.read_csv(root / "train_operational_readouts.csv")
    train_ops = train_ops[train_ops["vehicle_id"].isin(selected)].copy()
    val_ops = pd.read_csv(root / "validation_operational_readouts.csv")
    test_ops = pd.read_csv(root / "test_operational_readouts.csv")

    feature_cols = [c for c in train_ops.columns if c not in ("vehicle_id", "time_step")]
    # fill NaNs
    for df in (train_ops, val_ops, test_ops):
        df[feature_cols] = df[feature_cols].fillna(0.0)

    # Label every train readout from TTE / failure windows
    tte_map = tte.set_index("vehicle_id")
    repair_flag = train_ops["vehicle_id"].map(tte_map["in_study_repair"]).astype(int)
    fail_time = train_ops["vehicle_id"].map(tte_map["length_of_study_time_step"]).astype(float)
    ttf = fail_time - train_ops["time_step"].astype(float)
    train_y = [
        0 if rf == 0 else _failure_class(float(dt))
        for rf, dt in zip(repair_flag.to_numpy(), ttf.to_numpy())
    ]
    train_ops = train_ops.copy()
    train_ops["_y"] = train_y
    train_label_series = train_ops["_y"]
    train_label_series.index = train_ops.index

    X_tr, y_tr = _build_vehicle_sequences(
        train_ops,
        train_label_series,
        train_specs_map,
        seq_len=seq_len,
        feature_cols=feature_cols,
        max_per_vehicle=max_windows_per_vehicle,
        rng=rng,
    )

    X_va, y_va, _ = _last_readout_sequences(
        val_ops, val_labels, val_specs_map, seq_len, feature_cols
    )
    X_te, y_te, _ = _last_readout_sequences(
        test_ops, test_labels, test_specs_map, seq_len, feature_cols
    )

    # Scale only operational part (first len(feature_cols) dims)
    n_ops = len(feature_cols)
    scaler = StandardScaler()
    flat = X_tr[:, :, :n_ops].reshape(-1, n_ops)
    scaler.fit(flat)

    def scale(X):
        ops = X[:, :, :n_ops]
        rest = X[:, :, n_ops:]
        ops_s = scaler.transform(ops.reshape(-1, n_ops)).reshape(ops.shape).astype(np.float32)
        return np.concatenate([ops_s, rest], axis=-1)

    X_tr, X_va, X_te = scale(X_tr), scale(X_va), scale(X_te)

    return {
        "name": "SCANIA-Component-X",
        "task": "classification",
        "n_classes": 5,
        "tol": None,
        "X_train": X_tr,
        "y_train": y_tr,
        "X_val": X_va,
        "y_val": y_va,
        "X_test": X_te,
        "y_test": y_te,
    }
