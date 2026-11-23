# LSTM Ablation Study — Findings

## Protocol (fair / leakage-safe)
- Unit/vehicle-level splits only (seed=42), identical across ablations
- `StandardScaler` fit on **train windows only** when normalization is enabled
- One-factor-at-a-time vs fixed baseline: normalize=on, selected features, dropout=0.2, 2×64 LSTM
- Metrics: RMSE, MAE, R², NASA PHM Score
- SCANIA evaluated as TTE regression on held-out train vehicles (true event times)

## Component importance (mean |ΔRMSE| vs baseline, all datasets)

| Rank | Component | mean \|ΔRMSE\| | Interpretation |
|------|-----------|----------------|----------------|
| 1 | **Normalization** | **8.07** | Largest effect |
| 2 | Sliding windows | 3.68 | Temporal context matters |
| 3 | Architecture | 3.18 | Depth/width matter moderately |
| 4 | Features | 2.42 | Sensor content critical when model fits |
| 5 | Dropout | 0.44 | Smallest effect |

On well-fit sets only (**CMAPSS-FD001 / FD003**, baseline R² > 0.85), ranking sharpens:

| Rank | Component | mean \|ΔRMSE\| |
|------|-----------|----------------|
| 1 | Normalization | **27.8** |
| 2 | Features | 8.0 |
| 3 | Sliding windows | 4.3 |
| 4 | Architecture | 2.0 |
| 5 | Dropout | 1.3 |

## What hurts most
- **No normalization:** FD001 RMSE 14.1 → 40.2; FD003 11.7 → 41.2; R² collapses to ~0
- **Ops-only features:** same collapse — operating settings alone do not predict RUL
- **Window length 1:** loses degradation trajectory (FD003 +8.5 RMSE)

## What helps / is mild
- Sensors-only ≈ or slightly better than selected features on FD001
- Longer windows ≈ baseline; short windows worse
- Dropout 0 ↔ 0.5: small deltas
- Wider LSTM (128) slightly helps FD001/FD003; on hard sets a smaller net sometimes fails less badly

## Caveat
Baselines on FD002/FD004/PHM08/N-CMAPSS/SCANIA did not fit well (R² ≤ 0). Ablation deltas there are less trustworthy; primary conclusions come from FD001/FD003.
