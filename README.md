# Craigslist car listings — ETL, LLM enrichment, and price model (Mid Term Project)

Mid term project pipeline that scrapes Craigslist car listings (scraper **unchanged** in this repo), extracts structured fields with **regex + Vertex Gemini**, materializes a master CSV, trains a **time-aware** price model in GCP (**sklearn benchmark**: DecisionTree, RandomForest, ExtraTrees, HistGradientBoosting; **raw vs log** target; winner tuned on a calendar validation day), and syncs **metrics, predictions, and interpretability artifacts** back to GitHub for analysis in a notebook.

## Project overview

1. **Scrape**: raw `.txt` per listing in GCS under `scrapes/`.
2. **Regex ETL**: `extractor-per-listing` → `structured/run_id=<run>/jsonl/<post_id>.jsonl`.
3. **LLM ETL**: `extractor-llm-poc` reads each JSONL, downloads `source_txt`, merges regex hints with Gemini JSON → `jsonl_llm/<post_id>_llm.jsonl`.
4. **Materialize**: `materialize-master-llm` dedupes by `post_id` (newest run wins) → **`structured/datasets/listings_master_llm.csv`** (canonical training table).
5. **Train**: `train-dt` reads the LLM master CSV, uses **America/New_York** local dates, **holds out the latest day**, and when ≥3 days exist benchmarks **DecisionTree / RandomForest / ExtraTrees / HistGradientBoosting** on **raw `price_num` vs `log1p(price_num)`** using the **second-latest day** as validation (no random split). The **top two (model, target)** pairs are **ParameterSampler-tuned** on that same validation day; the winner is refit on all pre-holdout rows. **`predictions.csv` and reported MAE/MAPE/RMSE/bias stay on original USD** (log models use `expm1` on the holdout). Evidence lives in **`metrics.json` / `model_info.json`** under **`benchmark`**. Artifacts: **`structured/model_runs/<YYYYMMDDHH>/`**.
6. **Sync**: GitHub Actions copies the latest run’s files into **`results/`** subfolders for grading and notebooks.

## What changed from the baseline

| Area | Before | After |
|------|--------|--------|
| Regex extract | price, year, make, model, mileage | + transmission, color, drive, fuel, condition, title_status, type, cylinders, seller_type, city, state, zip (when explicit) |
| LLM extract | Few fields; `body_type` / `exterior_color` | Full rubric schema; `type` / `color`; location + zip rules; hybrid merge with regex |
| Master CSV | Narrow columns | Full enriched schema (see below) |
| Model | Single decision tree, MAE only | **Time-aware benchmark** of tree / forest / boosting models on **raw vs log** target; **tune top 2** finalists; dollar-scale holdout metrics, permutation importance, PDPs |
| Artifacts | `structured/preds/.../preds.csv` only | `structured/model_runs/<id>/` (+ cumulative `metrics_history.csv`) |
| Sync | Flat `results/*-preds.csv` (legacy) | `results/{predictions,metrics,...}` + legacy files in `archive/legacy_predictions/` |

## Final ETL schema (canonical CSV)

Column order in **`listings_master_llm.csv`**:

`post_id`, `run_id`, `scraped_at`, `source_txt`, `price`, `year`, `make`, `model`, `mileage`, `transmission`, `color`, `city`, `state`, `zip_code`, `drive`, `fuel`, `condition`, `title_status`, `type`, `cylinders`, `seller_type`, `llm_provider`, `llm_model`, `llm_ts`

**Enriched-field policy (submission-oriented):** `zip_code` is kept as a **5-digit string** when valid (leading zeros preserved end-to-end; invalid or geography-inconsistent values are nulled — e.g. CT listings only keep ZIPs in the **06xxx** USPS range). LLM and post-process steps **prefer null over guessed labels**: ambiguous fuel, title, condition, seller type, etc. are not mapped to a generic “other”. Training reads `zip_code` as text, builds **`zip_prefix` from the cleaned 5-digit string** only, and never relies on integer ZIPs.

## Modeling approach

- **Split (time-aware only)**: `scraped_at` → UTC → **America/New_York** local date. **Train** on all rows with `date_local < max(date_local)`; **final evaluation** on the **latest** local date. When **≥3** distinct dates exist, **validation** for benchmark/tuning uses the **second-latest** date (train slice = all earlier dates in the training window). **No random train/test split and no ordinary K-fold** for model selection.
- **Target**: Each run compares **`price_num` (raw)** and **`log1p(price_num)`** on the validation day; the **winning target strategy** is stored as **`target_strategy`** / **`target_training`** in `metrics.json` and `model_info.json`. **`predictions.csv` and holdout MAE / MAPE / RMSE / bias are always in original dollars** (log models: **`numpy.expm1`** on predictions).
- **Row filters (conservative)**: Same as before: missing/non-positive price, extreme price (defaults **$500–$500,000**), implausible model year vs scrape year, impossible mileage, non-standard **title_status**, **condition == project**. Counts under **`filtering`** in `metrics.json` / `model_info.json` / **`benchmark.filtering_applied`**.
- **Features**: Categoricals as before; numeric **variant A** = `vehicle_age`, `log_mileage`, `cylinders`. If **variant B** (adds `year_num`, `mileage_num`) beats A by **>1% validation MAE** on a quick RF+log check, that variant is used for the run (**`feature_variant`** in artifacts).
- **Benchmark**: **DecisionTreeRegressor**, **RandomForestRegressor**, **ExtraTreesRegressor**, **HistGradientBoostingRegressor** × **(raw | log)** — ranked on validation **dollar MAE**, then **RMSE**, then **|bias|**.
- **Tuning**: Only the **top two** distinct **(model, target)** pairs from the benchmark; **ParameterSampler** with model-specific grids (forests vs HGB vs tree). **Final model** = best post-tune validation score with the same tie order; refit on **all** pre-holdout data. **`chosen_model`** / **`model`** in `metrics.json` and `metrics_history.csv` (plus optional **`target_strategy`**, **`tune_validation_mae`** columns in history).

## Artifact outputs (GCS)

Each training run (not dry run) writes to `gs://<bucket>/structured/model_runs/<YYYYMMDDHH>/`:

| File | Description |
|------|-------------|
| `predictions.csv` | `post_id`, `scraped_at`, `actual_price`, `pred_price`, `residual`, make/model/year/mileage |
| `metrics.json` | MAE, MAPE, RMSE, bias, row counts, `chosen_model`, `target_strategy`, `benchmark` (candidates + results), `filtering`, `target_transform`, `model_comparison` |
| `metrics.csv` | Holdout metrics in one row + `model`, `target_strategy` |
| `permutation_importance.csv` | All input features, mean/std importance on holdout |
| `pdp/pdp_top*.png` | Partial dependence plots for top 3 features by importance |
| `model_info.json` | Feature lists, `target_strategy`, `benchmark` block, chosen params, filtering summary |
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
