# Archive

## `legacy_predictions/`

Contains **historical hourly prediction CSVs** from the pre-upgrade pipeline (`results/<YYYYMMDDHH>-preds.csv` from the old decision-tree path writing to `structured/preds/`).

They are preserved for continuity and grading evidence but are **not** used by the current notebook. The active layout is under `results/predictions/`, `results/metrics/`, etc., populated by the **Sync model artifacts** workflow from `structured/model_runs/`.
