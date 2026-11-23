# Continuous follow-up plan (within-family metrics)

Active loop continues this plan until stopped.

## Rules
1. Compare models **within** a dataset/family only.
2. RUL primary = RMSE; SCANIA primary = Total_cost.
3. No cross-family accuracy leaderboard.
4. Keep splits leakage-safe (unit/vehicle hold-out, train-only scaling).

## Iteration backlog
1. ~~SCANIA: class-balanced training + weighted LSTM (cost/F1)~~ done
2. ~~Hard RUL (FD002/FD004/PHM08): all features, longer/wider LSTM~~ done (LSTM still weak)
3. ~~Re-run `run_family_comparison.py`~~ done (improved run)
4. ~~Expand N-CMAPSS coverage~~ done (`n_cmapss_multi_*.csv`) — Linear 0.0 RMSE on tiny tests is unreliable; prefer GradBoost mean RMSE
5. ~~SCANIA val-cost Bayes decode~~ done — RF cost_decode test cost ≈ 49039 (was ~55k+)
6. ~~Condition clustering FD002/FD004~~ done — LSTM RMSE FD002 43.3→14.7, FD004 44.5→15.3
7. ~~Backlog complete~~ — continuous follow-up loop stopped (2026-11-22). Restart if new work is requested.

## Commands
```bash
.venv/bin/python ml_pipeline/run_family_comparison.py
```
