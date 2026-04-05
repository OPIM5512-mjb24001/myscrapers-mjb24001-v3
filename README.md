# Craigslist Car Price Prediction Pipeline

**Midterm Project**

## Project Summary

This project extends the Craigslist car-price pipeline used in class while keeping the **scraper unchanged**. Downstream, **RegEx + Gemini LLM ETL** builds an enriched dataset; **GCP** hosts training that **benchmarks, tunes, and selects** time-aware price models; and **GitHub Actions** syncs the latest run into this repo for the grading notebook.

The aim is a more complete, production-style ML workflow: stronger **structured extraction**, **model benchmarking and tuning**, **automated artifacts**, and **performance tracking over time**.

---

## What This Project Does

This pipeline:

- Extracts structured vehicle information from Craigslist listings.
- Enriches data with a **RegEx + LLM** hybrid.
- Materializes a **canonical training dataset** for modeling.
- **Benchmarks** multiple regressors for **listing price** prediction.
- Compares **raw price** vs **`log1p(price)`** targets.
- **Tunes** the strongest candidate configurations.
- Reports **MAE**, **MAPE**, **RMSE**, and **Bias** on the holdout.
- Saves **predictions** and **interpretability** outputs (permutation importance, PDPs).
- Surfaces trends and comparisons in **`notebooks/model_trending.ipynb`**.

**Compared to the earlier baseline:** richer ETL schema, stronger model-selection logic, **time-aware** validation (not a minimal baseline-only split), and **synced** predictions and interpretability under `results/` for grading.

---

## Pipeline Overview

| Step | What happens |
|------|----------------|
| **1. Scrape** | Raw listing text is stored in GCS under `scrapes/`. |
| **2. Regex ETL** | `extractor-per-listing` converts each listing to per-listing JSONL under `structured/`. |
| **3. LLM ETL** | `extractor-llm-poc` merges regex hints with Gemini extraction and writes enriched records to `jsonl_llm/`. |
| **4. Materialize** | `materialize-master-llm` deduplicates by `post_id` and writes the canonical dataset: **`structured/datasets/listings_master_llm.csv`**. |
| **5. Train + Sync** | `train-dt` benchmarks, tunes, and selects the final model, writes artifacts to GCS under `structured/model_runs/<YYYYMMDDHH>/`, and a workflow syncs the latest outputs into **`results/`**. |

---

## Enriched Fields

Beyond core fields such as **price**, **year**, **make**, **model**, **mileage**, and **transmission**, the pipeline adds attributes including:

- **color**, **city**, **state**, **zip_code**
- **drive**, **fuel**, **condition**, **title_status**, **type**, **cylinders**, **seller_type**

The master CSV also retains **LLM provenance** columns.

**Data quality (high level)**

- `zip_code` is kept as a **5-digit string** so leading zeros are not lost.
- Invalid or suspicious ZIP values are set to **null**.
- Ambiguous enrichment is handled **conservatively** (null preferred over guessing).

---

## Modeling Approach

**Target:** Craigslist **listing price** (reported in **dollars** on the holdout, including when the model was trained on `log1p(price)`).

### Time-aware evaluation

`scraped_at` is converted to **America/New_York** local dates. The split is **not** a random train/test split:

- **Training:** all local dates **before** the latest date in the data.
- **Holdout:** the **latest** local date.

That setup approximates using history to predict **newer** listings.

### Model benchmarking

The training workflow benchmarks several **sklearn** regressors, including:

- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `HistGradientBoostingRegressor`

It compares two **target** strategies: **raw price** and **`log1p(price)`**.

When enough distinct dates exist, the **second-latest** local date acts as a **calendar validation** day for comparison. The **top two** model/target combinations are **tuned**; the winner is **retrained** on all rows before the holdout day.

### Final metrics

Holdout performance is reported as **MAE**, **MAPE**, **RMSE**, and **Bias**. Log-trained models are mapped back to the **original dollar scale** for predictions and metrics.

---

## Artifacts Produced

Each run writes to **`structured/model_runs/<YYYYMMDDHH>/`** in GCS (mirrored into `results/` after sync), including:

| Artifact | Role |
|----------|------|
| `predictions.csv` | Holdout actual vs predicted price |
| `metrics.json`, `metrics.csv` | Final metrics and run metadata |
| `model_benchmark.csv` | Validation benchmark rows; marks the selected configuration |
| `tuning_trials.csv` | Every `ParameterSampler` trial on the validation day (MAE/RMSE/MAPE/bias/composite) |
| `permutation_importance.csv` | Feature importance on the holdout |
| `pdp/pdp_top*.png` | **Top 3** partial dependence plots |
| `model_info.json` | Model and preprocessing summary |
| `metrics_history.csv` | Append row for this run (also aggregated over time) |

Together these support final holdout scores, **model comparison**, **interpretability**, and **trend tracking**.

---

## Synced Results in This Repo

```
results/
  predictions/
  metrics/
  permutation_importance/
  pdp/
  history/
```

Older flat prediction files are kept under **`archive/legacy_predictions/`** for historical reference only; the current notebook does **not** use them.

---

## Notebook

**[`notebooks/model_trending.ipynb`](notebooks/model_trending.ipynb)** is the main summary.

- It does **not** retrain the pipeline and does **not** require **GCP** credentials.
- It reads **synced repo artifacts only** and presents performance over time, **benchmark** comparison, final holdout **metrics**, **permutation importance**, the **top 3** PDPs, and a short written **interpretation**.

In short: the **README** explains how the pipeline works; the **notebook** explains what the results mean.

---

## How to Run

1. Deploy the Cloud Functions using workflows under **`.github/workflows/`**.
2. Ensure **`structured/datasets/listings_master_llm.csv`** exists (ETL + materialize).
3. Run **`train-dt`** with **`dry_run: false`** so real artifacts are written to GCS.
4. Run the **sync** workflow to refresh **`results/`**.
5. Open **`notebooks/model_trending.ipynb`** and run all cells.

For safe testing, **`TRAIN_DT_BODY`** can be set to `{"dry_run": true}`.

---

## Cloud Functions

| Function | Role |
|----------|------|
| `extractor-per-listing` | Regex ETL → JSONL |
| `extractor-llm-poc` | LLM enrichment → `jsonl_llm` |
| `materialize-master-llm` | Builds the enriched master dataset |
| `train-dt` | Benchmarks, tunes, selects the final model, writes artifacts |

---

## Requirements

Dependencies are **pinned** in each **`cloud_function/*/requirements.txt`**.

Training uses **pandas**, **numpy**, **scikit-learn**, and **matplotlib** (including PDP export).
