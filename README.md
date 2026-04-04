# Craigslist Car Price Prediction Pipeline
**Midterm Project**

## Project summary

The **scraper is unchanged**. Downstream, the pipeline runs **RegEx + Vertex Gemini (LLM) ETL**, builds an **enriched master dataset**, trains a **time-aware** listing-price model in GCP, and **syncs artifacts to GitHub** for the grading notebook.

---

## What this project does

- Extract structured vehicle fields from listings (regex pass, then LLM merge).
- Enrich rows with additional attributes and conservative location handling.
- Materialize a **single canonical training table** for modeling.
- Train and **benchmark multiple regression models**, tune finalists, and predict **price**.
- Track **MAE, MAPE, RMSE, and bias** over time and save **predictions** plus **interpretability** outputs (permutation importance, PDPs).

**Compared to the baseline:** narrow regex-only features and a simpler tree setup → full enriched schema, time-aware model comparison (raw vs log target), and graded artifacts under `results/`.

---

## Pipeline overview

| Step | What happens |
|------|----------------|
| **1. Scrape** | Raw listing text in GCS (`scrapes/`). |
| **2. Regex ETL** | `extractor-per-listing` → per-listing JSONL under `structured/`. |
| **3. LLM ETL** | `extractor-llm-poc` merges hints + Gemini JSON → `jsonl_llm/`. |
| **4. Materialize** | `materialize-master-llm` dedupes by `post_id` → **`structured/datasets/listings_master_llm.csv`**. |
| **5. Train + Sync** | `train-dt` writes run artifacts to GCS; **Sync model artifacts** copies the latest run into `results/`. |

---

## Enriched fields (high level)

Beyond core listing fields (e.g. price, year, make, model, mileage, transmission), the pipeline adds attributes such as **color**, **city**, **state**, **zip_code**, **drive**, **fuel**, **condition**, **title_status**, **type**, **cylinders**, and **seller_type** (plus LLM provenance columns in the master CSV). ZIPs stay **string-safe** for training; ambiguous enrichment is nulled rather than guessed.

---

## Modeling approach

- **Target:** Craigslist **listing price** (dollars on the holdout).
- **Split:** Local dates from `scraped_at` (**America/New_York**). The model trains on **all days before the latest day** in the data; the **latest day is the holdout**. This is the main evaluation—**not** a random train/test split.
- **Selection:** When enough days exist, a **calendar validation day** (second-latest) is used to **benchmark** several sklearn regressors and **raw vs `log1p(price)`** training; the **top two** configurations are tuned, then the winner is refit on all pre-holdout data.
- **Reported metrics (holdout, USD):** **MAE**, **MAPE**, **RMSE**, **Bias**. Log-trained models are reported in dollars via **`expm1`** on predictions.

---

## Artifacts (each training run)

Written under `structured/model_runs/<YYYYMMDDHH>/` in GCS (and mirrored into `results/` after sync):

- **Predictions** — holdout `actual_price` / `pred_price` (USD).
- **Metrics** — `metrics.json`, `metrics.csv` (holdout errors + chosen model / target).
- **Model benchmark** — `model_benchmark.csv` (validation rows + `selected` for the winner); full detail also in `metrics.json` / `model_info.json`.
- **Permutation importance** — feature rankings on the holdout.
- **PDPs** — top **3** partial dependence plots (PNG).
- **Metrics history** — cumulative CSV appended over runs.

---

## Synced `results/` layout

```
results/
  predictions/
  metrics/
  permutation_importance/
  pdp/
  history/
```

Older flat prediction files live under **`archive/legacy_predictions/`** and are not used by the current notebook.

---

## Notebook

**[`notebooks/model_trending.ipynb`](notebooks/model_trending.ipynb)** is the main **grader-facing** summary:

- Reads **synced repo artifacts only** (no retraining, **no GCP credentials**).
- Shows **metric trends**, **benchmark comparison**, final **holdout metrics**, **permutation importance**, **PDPs**, and short **interpretation** text.

For **Colab**, the default clone URL is the course repo; use **`NOTEBOOK_REPO_URL`** (and optionally **`NOTEBOOK_REPO_BRANCH`**) if you use a fork.

---

## How to run (high level)

1. Deploy the Cloud Functions (workflows in `.github/workflows/`).
2. Run ETL + **materialize** so **`structured/datasets/listings_master_llm.csv`** exists.
3. Run **`train-dt`** with **`dry_run: false`** so artifacts are written (see repo notes on `TRAIN_DT_BODY` for testing).
4. Run **Sync model artifacts to repo** to refresh **`results/`**.
5. Open **`notebooks/model_trending.ipynb`** and run all cells.

---

## Cloud Functions

| Function | Role |
|----------|------|
| `extractor-per-listing` | Regex ETL → JSONL |
| `extractor-llm-poc` | LLM enrichment → `jsonl_llm` |
| `materialize-master-llm` | Master dataset CSV |
| `train-dt` | Train, benchmark, tune, write artifacts |

---

## Requirements

Dependencies are **pinned** in each **`cloud_function/*/requirements.txt`**. Training uses the usual **pandas / numpy / scikit-learn** stack plus **matplotlib** for PDP export.
