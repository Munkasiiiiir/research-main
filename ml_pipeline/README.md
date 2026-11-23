# End-to-End ML Pipeline

Compare models **within each dataset/family** using that family’s native metrics (no cross-family accuracy ranking).

## Families

| Family | Datasets | Primary metric | Also report |
|--------|----------|----------------|-------------|
| **RUL** | CMAPSS FD001–FD004, PHM08, N-CMAPSS | RMSE (↓) | MAE, R², PHM Score |
| **SCANIA** | Component X | Total_cost (↓) | macro-F1, per-class recall, accuracy |

## Run

```bash
cd "/home/munkasiiiiir/Documents/Research(Main)"
.venv/bin/python ml_pipeline/run_family_comparison.py
```

Results: `ml_pipeline/results/family_comparison/`

Other entry points: `run_all.py` (legacy), `run_ablation.py` (LSTM ablations).
