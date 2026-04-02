# Synced model artifacts

Populated by **GitHub Actions → Sync model artifacts to repo** from GCS `structured/model_runs/`.

| Subfolder | Contents |
|-----------|----------|
| `predictions/` | `<run_id>-predictions.csv` |
| `metrics/` | `<run_id>-metrics.json`, `-metrics.csv`, `-model_info.json` |
| `permutation_importance/` | `<run_id>-permutation_importance.csv` |
| `pdp/` | `<run_id>_pdp_top*.png` |
| `history/` | `metrics_history.csv` (cumulative) |
