# Craigslist car listings — ETL, LLM enrichment, and price model (Mid Term Project)

Mid term project pipeline that scrapes Craigslist car listings (scraper **unchanged** in this repo), extracts structured fields with **regex + Vertex Gemini**, materializes a master CSV, trains a **time-aware** price model (**RandomForest vs ExtraTrees**, **log1p** target, dollar metrics) in GCP, and syncs **metrics, predictions, and interpretability artifacts** back to GitHub for analysis in a notebook.

## Project overview

1. **Scrape**: raw `.txt` per listing in GCS under `scrapes/`.
2. **Regex ETL**: `extractor-per-listing` → `structured/run_id=<run>/jsonl/<post_id>.jsonl`.
3. **LLM ETL**: `extractor-llm-poc` reads each JSONL, downloads `source_txt`, merges regex hints with Gemini JSON → `jsonl_llm/<post_id>_llm.jsonl`.
4. **Materialize**: `materialize-master-llm` dedupes by `post_id` (newest run wins) → **`structured/datasets/listings_master_llm.csv`** (canonical training table).
5. **Train**: `train-dt` reads the LLM master CSV, trains on all **America/New_York** local dates before the latest day, holds out the latest day, tunes **RandomForest** and **ExtraTrees** (selects the better on the tune-validation day), fits on **log1p(price)** with **metrics and `predictions.csv` on original USD** (`expm1`), writes artifacts under **`structured/model_runs/<YYYYMMDDHH>/`**.
6. **Sync**: GitHub Actions copies the latest run’s files into **`results/`** subfolders for grading and notebooks.

## What changed from the baseline

| Area | Before | After |
|------|--------|--------|
| Regex extract | price, year, make, model, mileage | + transmission, color, drive, fuel, condition, title_status, type, cylinders, seller_type, city, state, zip (when explicit) |
| LLM extract | Few fields; `body_type` / `exterior_color` | Full rubric schema; `type` / `color`; location + zip rules; hybrid merge with regex |
| Master CSV | Narrow columns | Full enriched schema (see below) |
| Model | Single decision tree, MAE only | Tuned **RandomForest vs ExtraTrees** (best wins), **log1p(price)** training with dollar-scale metrics, MAE / MAPE / RMSE / Bias, permutation importance, PDPs |
| Artifacts | `structured/preds/.../preds.csv` only | `structured/model_runs/<id>/` (+ cumulative `metrics_history.csv`) |
| Sync | Flat `results/*-preds.csv` (legacy) | `results/{predictions,metrics,...}` + legacy files in `archive/legacy_predictions/` |

## Final ETL schema (canonical CSV)

Column order in **`listings_master_llm.csv`**:

`post_id`, `run_id`, `scraped_at`, `source_txt`, `price`, `year`, `make`, `model`, `mileage`, `transmission`, `color`, `city`, `state`, `zip_code`, `drive`, `fuel`, `condition`, `title_status`, `type`, `cylinders`, `seller_type`, `llm_provider`, `llm_model`, `llm_ts`

## Modeling approach

- **Split**: `scraped_at` → UTC → **America/New_York** local date. Train on all rows with `date_local < max(date_local)`; evaluate on the latest local date (“today” in the dataset).
- **Target**: Train on **`log1p(price_num)`**; predictions in `predictions.csv` and error metrics (MAE, MAPE, RMSE, bias) use **original dollars** after **`numpy.expm1`** on model outputs. Documented in `model_info.json` (`target`, `target_training`, `target_transform`).
- **Row filters (conservative)**: Before splitting, rows are dropped for missing/non-positive price, extreme price (defaults **$500–$500,000**), implausible model year vs scrape year, impossible mileage, non-standard **title_status** (e.g. salvage, parts-only, missing/blank), and **condition == project**. Counts per rule are in `metrics.json` / `model_info.json` under **`filtering`**. Optional overrides: `MIN_PRICE_USD`, `MAX_PRICE_USD`, `MIN_MODEL_YEAR`, `MAX_MILEAGE`.
- **Features**: categoricals (make, model, transmission, fuel, drive, condition, title_status, type, seller_type, state, city, `zip_prefix`) with imputation + one-hot encoding; numeric **`vehicle_age` (clipped), `log_mileage`, `cylinders`** (raw `year_num` / `mileage_num` omitted as redundant with age / log mileage); rare levels bucketed for sparse categories.
- **Tuning**: When ≥3 distinct dates exist, **ParameterSampler** over a small hyperparameter grid for **both** `RandomForestRegressor` and `ExtraTreesRegressor`, validated on the **second-to-last** local date using **dollar MAE** (not random K-fold).
- **Baselines**: Optional decision tree MAE on the same tune split in dollar space (logged in `metrics.json`).
- **Final model**: The ensemble with **lower tune-validation dollar MAE**; tie-break **Random Forest**. Stored as `chosen_model` / `model` in `metrics.json` and `metrics_history.csv`.

## Artifact outputs (GCS)

Each training run (not dry run) writes to `gs://<bucket>/structured/model_runs/<YYYYMMDDHH>/`:

| File | Description |
|------|-------------|
| `predictions.csv` | `post_id`, `scraped_at`, `actual_price`, `pred_price`, `residual`, make/model/year/mileage |
| `metrics.json` | MAE, MAPE, RMSE, bias, row counts, tuning notes, RF + ExtraTrees params, `chosen_model`, `filtering`, `target_transform`, `model_comparison` |
| `metrics.csv` | Same metrics in one row |
| `permutation_importance.csv` | All input features, mean/std importance on holdout |
| `pdp/pdp_top*.png` | Partial dependence plots for top 3 features by importance |
| `model_info.json` | Feature lists, chosen + both ensembles’ params, data key, log-target notes, filtering summary |
| `metrics_history.csv` | Snapshot of cumulative history (also appended to `structured/model_runs/metrics_history.csv`) |

## Repo `results/` layout 

Synced artifacts land **only** in these subfolders (no new files committed to `results/` root):

```
results/
  predictions/              <run_id>-predictions.csv
  metrics/                    <run_id>-metrics.json, <run_id>-metrics.csv, <run_id>-model_info.json
  permutation_importance/     <run_id>-permutation_importance.csv
  pdp/                        <run_id>_pdp_top*.png  (may be empty until a run produces PDPs)
  history/                    metrics_history.csv    (cumulative, overwritten each sync)
```

**Legacy hourly decision-tree predictions** (`*-preds.csv`) are preserved under **`archive/legacy_predictions/`** so the GitHub tree stays browsable. They are not used by the current notebook.

## Notebook (Colab / local)

- Path: [`notebooks/model_trending.ipynb`](notebooks/model_trending.ipynb)
- **No GCP credentials** and **no retraining** — only reads synced files under `results/{history,permutation_importance,pdp}/` and related paths.
- **Colab**: defaults clone **`https://github.com/OPIM5512-mjb24001/myscrapers-mjb24001-v3.git`**. For a fork, set `NOTEBOOK_REPO_URL` (optional: `NOTEBOOK_REPO_BRANCH`, default `main`).

## How to run / validate

1. Deploy Cloud Functions via existing workflows (`deploy-extractor`, `deploy-extractor-llm`, `deploy-materialize-master-llm`, `deploy-train-dt`).
2. Ensure scheduler / manual POST runs materialize so **`listings_master_llm.csv`** exists.
3. **Training writes artifacts by default:** Cloud Scheduler uses `dry_run: false` unless repo variable **`TRAIN_DT_BODY`** overrides it (set to `{"dry_run":true}` for safe testing). Post-deploy workflow smoke tests use `dry_run: true` only so pushes do not trigger a full train.
4. Run **`Sync model artifacts to repo`** workflow (hourly or `workflow_dispatch`) to refresh `results/`.
5. Open the notebook and run all cells.

## Cloud Function entrypoints

| Function | Entry | Purpose |
|----------|--------|---------|
| `extractor-per-listing` | `extract_http` | Regex JSONL |
| `extractor-llm-poc` | `llm_extract_http` | LLM JSONL |
| `materialize-master-llm` | `materialize_http` | Master CSV |
| `train-dt` | `train_dt_http` | Train + artifacts |

## Requirements

Python dependencies are pinned per function under each `cloud_function/*/requirements.txt`. Training adds **matplotlib** for PDP export.
