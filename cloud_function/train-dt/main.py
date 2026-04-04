# Time-aware training: train on all local dates before latest; hold out latest day.
# Benchmarks multiple sklearn regressors on raw vs log1p(price); tunes top-2 on time-aware val day.
# Artifacts under structured/model_runs/<run_id>/; metrics on original USD.
# HTTP entrypoint: train_dt_http

from __future__ import annotations

import io
import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# ---- ENV ----
PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
DATA_KEY = os.getenv("DATA_KEY", "structured/datasets/listings_master_llm.csv")
MODEL_RUN_PREFIX = os.getenv("MODEL_RUN_PREFIX", "structured/model_runs")
METRICS_HISTORY_KEY = os.getenv("METRICS_HISTORY_KEY", "structured/model_runs/metrics_history.csv")
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RARE_CAT_MIN_COUNT = int(os.getenv("RARE_CAT_MIN_COUNT", "5"))

# Conservative listing filters (dollar scale)
MIN_PRICE = float(os.getenv("MIN_PRICE_USD", "500"))
MAX_PRICE = float(os.getenv("MAX_PRICE_USD", "500000"))
MIN_MODEL_YEAR = int(os.getenv("MIN_MODEL_YEAR", "1980"))
MAX_MILEAGE = float(os.getenv("MAX_MILEAGE", "999999"))

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

TARGET_ORIGINAL = "price_num"
TARGET_LOG = "log_price_num"


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    buf = io.BytesIO(blob.download_as_bytes())
    df = pd.read_csv(buf, low_memory=False)
    if "zip_code" in df.columns:
        df["zip_code"] = df["zip_code"].astype("string")
    return df


def _clean_zip_series(zip_s: pd.Series, state_s: pd.Series | None = None) -> pd.Series:
    """
    ZIP as 5-digit string; invalid / suspicious → NA. Never coerce via float/int.
    CT listings: USPS ZIPs are 06xxx — drop mismatch (conservative geography).
    """
    if zip_s is None:
        return pd.Series(pd.NA, dtype="string")
    z = zip_s.astype("string")
    z = z.replace({"<NA>": pd.NA})
    stripped = z.str.strip()
    stripped = stripped.mask(stripped.isin(["", "nan", "None", "<NA>"]), pd.NA)
    stripped = stripped.str.replace(r"^(\d+)\.0$", r"\1", regex=True)
    digits = stripped.str.replace(r"\D", "", regex=True)
    long = digits.notna() & (digits.str.len() >= 9)
    digits = digits.where(~long, digits.str.slice(0, 5))
    ok = (
        digits.notna()
        & (digits.str.len() == 5)
        & digits.str.fullmatch(r"\d{5}", na=False)
    )
    out = pd.Series(pd.NA, index=zip_s.index, dtype="string")
    out = out.mask(ok, digits)
    if state_s is not None and len(state_s) == len(out):
        st = state_s.astype("string").str.upper().str.strip()
        bad_ct = (st == "CT") & out.notna() & ~out.str.startswith("06", na=False)
        out = out.mask(bad_ct, pd.NA)
    return out


def _zip_prefix_from_clean_zip(zip5: pd.Series) -> pd.Series:
    """First three characters of validated 5-digit ZIP; missing ZIP → 'na' sentinel for the model."""
    pref = pd.Series("na", index=zip5.index, dtype="object")
    m = zip5.notna() & (zip5.str.len() == 5)
    pref.loc[m] = zip5.loc[m].str.slice(0, 3)
    return pref


def _write_bytes_to_gcs(client: storage.Client, bucket: str, key: str, data: bytes, content_type: str):
    b = client.bucket(bucket)
    blob = b.blob(key)
    blob.upload_from_string(data, content_type=content_type)


def _write_text_to_gcs(client: storage.Client, bucket: str, key: str, text: str, content_type: str = "text/plain"):
    _write_bytes_to_gcs(client, bucket, key, text.encode("utf-8"), content_type)


def _read_text_from_gcs(client: storage.Client, bucket: str, key: str) -> str | None:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        return None
    return blob.download_as_text()


def _clean_numeric(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.replace(r"[^\d.]+", "", regex=True).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _bucket_rare_categories(df: pd.DataFrame, cols: list[str], min_count: int) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        vc = out[c].astype(str).replace({"nan": np.nan}).value_counts(dropna=True)
        rare_vals = set(vc[vc < min_count].index.astype(str))
        if not rare_vals:
            continue
        mask = out[c].astype(str).isin(rare_vals)
        out.loc[mask, c] = "rare_bucket"
    return out


def _apply_training_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Conservative row filters; logs counts per rule in returned stats."""
    stats: dict[str, Any] = {"input_rows": int(len(df))}
    d = df.copy()
    n0 = len(d)

    m = d[TARGET_ORIGINAL].notna() & (d[TARGET_ORIGINAL] > 0)
    stats["dropped_missing_or_nonpositive_price"] = int(n0 - m.sum())
    d = d.loc[m]
    n1 = len(d)

    m = (d[TARGET_ORIGINAL] >= MIN_PRICE) & (d[TARGET_ORIGINAL] <= MAX_PRICE)
    stats["dropped_extreme_price"] = int(n1 - m.sum())
    d = d.loc[m]
    n2 = len(d)

    ref_year = d["scraped_at_local"].dt.year
    yn = d["year_num"]
    m = yn.notna() & (yn >= MIN_MODEL_YEAR) & (yn <= ref_year + 1)
    stats["dropped_bad_year"] = int(n2 - m.sum())
    d = d.loc[m]
    n3 = len(d)

    mi = d["mileage_num"]
    m_ok = mi.isna() | ((mi >= 0) & (mi <= MAX_MILEAGE))
    stats["dropped_bad_mileage"] = int(n3 - m_ok.sum())
    d = d.loc[m_ok]
    n4 = len(d)

    if "title_status" in d.columns:
        ts = d["title_status"].astype(str).str.lower().str.strip()
        bad_titles = {"salvage", "parts only", "parts-only", "missing"}
        bad = (
            ts.isin(bad_titles)
            | ts.isin(["", "nan", "none"])
            | ts.str.contains("salvage", na=False)
            | (ts.str.contains("parts", na=False) & ts.str.contains("only", na=False))
        )
        stats["dropped_title_status_nonstandard"] = int(bad.sum())
        d = d.loc[~bad]
    else:
        stats["dropped_title_status_nonstandard"] = 0
    n5 = len(d)

    if "condition" in d.columns:
        cond = d["condition"].astype(str).str.lower().str.strip()
        bad = cond == "project"
        stats["dropped_condition_project"] = int(bad.sum())
        d = d.loc[~bad]
    else:
        stats["dropped_condition_project"] = 0

    stats["output_rows"] = int(len(d))
    stats["rows_removed_total"] = int(stats["input_rows"] - stats["output_rows"])
    pct = (100.0 * stats["rows_removed_total"] / stats["input_rows"]) if stats["input_rows"] else 0.0
    stats["pct_rows_removed"] = round(pct, 2)
    logging.info(
        "Row filtering: in=%d out=%d removed=%d (%.2f%%) detail=%s",
        stats["input_rows"],
        stats["output_rows"],
        stats["rows_removed_total"],
        pct,
        {k: stats[k] for k in stats if k.startswith("dropped_")},
    )
    return d, stats


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


dollar_mae_scorer = make_scorer(
    lambda y_true_log, y_pred_log: -mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log)),
    greater_is_better=True,
)

raw_dollar_mae_scorer = make_scorer(
    lambda y_true, y_pred: -mean_absolute_error(y_true, y_pred),
    greater_is_better=True,
)

TargetMode = Literal["raw", "log"]

# Tuning search sizes (time-aware validation only; no K-fold)
TUNE_N_ITER_RF_ET = 16
TUNE_N_ITER_HGB = 16

PARAM_GRID_RF_ET: dict[str, list] = {
    "n_estimators": [100, 200, 400],
    "max_depth": [10, 20, 30, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "max_features": ["sqrt", 0.3, 0.5, 0.7],
}

PARAM_GRID_HGB: dict[str, list] = {
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [6, 10, None],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [10, 20, 40],
    "max_iter": [200],
}

DEFAULT_RF = {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 4, "max_features": "sqrt"}


def _dollar_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": _mape(y_true, y_pred),
        "rmse": _rmse(y_true, y_pred),
        "bias": _bias(y_true, y_pred),
    }


def _preds_to_dollars(y_pred: np.ndarray, target_mode: TargetMode) -> np.ndarray:
    if target_mode == "log":
        return np.expm1(y_pred)
    return y_pred


def _y_for_target(df: pd.DataFrame, target_mode: TargetMode) -> pd.Series:
    if target_mode == "raw":
        return df[TARGET_ORIGINAL]
    return df[TARGET_LOG]


def _num_cols_variant(df: pd.DataFrame, variant: Literal["A", "B"]) -> list[str]:
    base = ["vehicle_age", "log_mileage", "cylinders"]
    cols = [c for c in base if c in df.columns]
    if variant == "B":
        for extra in ("year_num", "mileage_num"):
            if extra in df.columns and extra not in cols:
                cols.append(extra)
    return cols


def _select_feature_variant(
    tune_train: pd.DataFrame,
    tune_val: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[list[str], str]:
    """Variant A = age + log_mileage + cylinders; B adds year_num + mileage_num if >1% val MAE gain (RF+log)."""
    num_a = _num_cols_variant(tune_train, "A")
    num_b = _num_cols_variant(tune_train, "B")
    if len(num_b) <= len(num_a):
        return num_a, "A"
    feats_a = cat_cols + num_a
    feats_b = cat_cols + num_b
    reg = RandomForestRegressor(random_state=42, n_jobs=-1, **DEFAULT_RF)
    mae_a = _quick_val_mae_dollars(
        reg, feats_a, cat_cols, num_a, tune_train, tune_val, "log"
    )
    mae_b = _quick_val_mae_dollars(
        reg, feats_b, cat_cols, num_b, tune_train, tune_val, "log"
    )
    if np.isfinite(mae_b) and mae_b < mae_a * 0.99:
        logging.info("Feature variant B selected (val MAE %.2f vs A %.2f)", mae_b, mae_a)
        return num_b, "B"
    logging.info("Feature variant A selected (val MAE A=%.2f B=%.2f)", mae_a, mae_b)
    return num_a, "A"


def _quick_val_mae_dollars(
    regressor,
    feats: list[str],
    cat_cols: list[str],
    num_cols: list[str],
    tune_train: pd.DataFrame,
    tune_val: pd.DataFrame,
    target_mode: TargetMode,
) -> float:
    y_tr = _y_for_target(tune_train, target_mode)
    y_va = _y_for_target(tune_val, target_mode)
    y_dollar_va = tune_val[TARGET_ORIGINAL]
    m_tr = y_tr.notna()
    m_va = y_dollar_va.notna() & y_va.notna()
    if not m_tr.any() or not m_va.any():
        return float("nan")
    pipe = _build_preprocess_and_model(cat_cols, num_cols, regressor)
    try:
        pipe.fit(tune_train.loc[m_tr, feats], y_tr.loc[m_tr])
        raw_p = pipe.predict(tune_val.loc[m_va, feats])
        pred_d = _preds_to_dollars(raw_p, target_mode)
        return float(mean_absolute_error(y_dollar_va.loc[m_va].values.astype(float), pred_d))
    except Exception:
        return float("nan")


def _benchmark_one(
    name: str,
    regressor,
    target_mode: TargetMode,
    cat_cols: list[str],
    num_cols: list[str],
    feats: list[str],
    tune_train: pd.DataFrame,
    tune_val: pd.DataFrame,
) -> dict[str, Any] | None:
    y_tr = _y_for_target(tune_train, target_mode)
    y_va_space = _y_for_target(tune_val, target_mode)
    y_dollar_va = tune_val[TARGET_ORIGINAL]
    m_tr = y_tr.notna()
    m_va = y_dollar_va.notna() & y_va_space.notna()
    if not m_tr.any() or not m_va.any():
        return None
    pipe = _build_preprocess_and_model(cat_cols, num_cols, regressor)
    try:
        pipe.fit(tune_train.loc[m_tr, feats], y_tr.loc[m_tr])
        raw_p = pipe.predict(tune_val.loc[m_va, feats])
        pred_d = _preds_to_dollars(raw_p, target_mode)
        yt = y_dollar_va.loc[m_va].values.astype(float)
        m = _dollar_metrics_dict(yt, pred_d)
        return {
            "model": name,
            "target_strategy": target_mode,
            "val_mae": m["mae"],
            "val_rmse": m["rmse"],
            "val_mape": m["mape"],
            "val_bias": m["bias"],
        }
    except Exception as ex:
        logging.warning("Benchmark failed %s %s: %s", name, target_mode, ex)
        return None


def _default_regressor_builders() -> list[tuple[str, Callable[[], Any]]]:
    return [
        (
            "DecisionTreeRegressor",
            lambda: DecisionTreeRegressor(
                max_depth=12, min_samples_leaf=15, random_state=42
            ),
        ),
        (
            "RandomForestRegressor",
            lambda: RandomForestRegressor(random_state=42, n_jobs=-1, **DEFAULT_RF),
        ),
        (
            "ExtraTreesRegressor",
            lambda: ExtraTreesRegressor(
                random_state=42, n_jobs=-1, bootstrap=False, **DEFAULT_RF
            ),
        ),
        (
            "HistGradientBoostingRegressor",
            lambda: HistGradientBoostingRegressor(
                random_state=42,
                max_iter=200,
                learning_rate=0.08,
                max_depth=10,
                max_leaf_nodes=31,
                min_samples_leaf=20,
            ),
        ),
    ]


def _benchmark_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["val_mae"]),
        float(row["val_rmse"]),
        abs(float(row["val_bias"])),
    )


def _tune_on_val(
    model_name: str,
    build_reg: Callable[[dict[str, Any]], Any],
    param_grid: dict[str, list],
    target_mode: TargetMode,
    cat_cols: list[str],
    num_cols: list[str],
    feats: list[str],
    tune_train: pd.DataFrame,
    tune_val: pd.DataFrame,
    n_iter: int,
    default_params: dict[str, Any],
) -> tuple[dict[str, Any], float, dict[str, float]]:
    y_tr = _y_for_target(tune_train, target_mode)
    y_va_space = _y_for_target(tune_val, target_mode)
    y_dollar_va = tune_val[TARGET_ORIGINAL]
    best_params: dict[str, Any] = {}
    best_mae = float("inf")
    best_metrics: dict[str, float] = {}
    for params in ParameterSampler(param_grid, n_iter=n_iter, random_state=42):
        p = dict(params)
        try:
            reg = build_reg(p)
            row = _benchmark_one(
                model_name, reg, target_mode, cat_cols, num_cols, feats, tune_train, tune_val
            )
            if row is None:
                continue
            mae = row["val_mae"]
            if mae < best_mae:
                best_mae = mae
                best_params = p
                best_metrics = {
                    "val_mae": row["val_mae"],
                    "val_rmse": row["val_rmse"],
                    "val_mape": row["val_mape"],
                    "val_bias": row["val_bias"],
                }
        except Exception as ex:
            logging.warning("Tune iteration failed (%s): %s", model_name, ex)
    if not best_params or not np.isfinite(best_mae):
        reg0 = build_reg(default_params)
        row = _benchmark_one(
            model_name, reg0, target_mode, cat_cols, num_cols, feats, tune_train, tune_val
        )
        return dict(default_params), float(row["val_mae"]) if row else float("nan"), (
            {k: float(row[k]) for k in ("val_mae", "val_rmse", "val_mape", "val_bias")} if row else {}
        )
    return best_params, best_mae, best_metrics


def _build_reg_for_model(model_name: str, params: dict[str, Any]) -> Any:
    if model_name == "DecisionTreeRegressor":
        return DecisionTreeRegressor(random_state=42, **params)
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(random_state=42, n_jobs=-1, **params)
    if model_name == "ExtraTreesRegressor":
        return ExtraTreesRegressor(
            random_state=42, n_jobs=-1, bootstrap=False, **params
        )
    if model_name == "HistGradientBoostingRegressor":
        return HistGradientBoostingRegressor(random_state=42, **params)
    raise ValueError(f"unknown model {model_name}")


def _param_grid_for_model(model_name: str) -> dict[str, list]:
    if model_name in ("RandomForestRegressor", "ExtraTreesRegressor"):
        return PARAM_GRID_RF_ET
    if model_name == "HistGradientBoostingRegressor":
        return PARAM_GRID_HGB
    if model_name == "DecisionTreeRegressor":
        return {
            "max_depth": [8, 12, 16, 24, None],
            "min_samples_leaf": [5, 10, 20, 40],
            "min_samples_split": [2, 10, 20],
        }
    return {}


def _default_params_for_model(model_name: str) -> dict[str, Any]:
    if model_name == "DecisionTreeRegressor":
        return {"max_depth": 12, "min_samples_leaf": 15, "min_samples_split": 2}
    if model_name in ("RandomForestRegressor", "ExtraTreesRegressor"):
        return dict(DEFAULT_RF)
    if model_name == "HistGradientBoostingRegressor":
        return {
            "max_iter": 200,
            "learning_rate": 0.08,
            "max_depth": 10,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
        }
    return {}


def _build_preprocess_and_model(
    cat_cols: list[str],
    num_cols: list[str],
    regressor,
) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=50, sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", regressor)])


def _tune_n_iter_for_model(model_name: str) -> int:
    if model_name == "HistGradientBoostingRegressor":
        return TUNE_N_ITER_HGB
    if model_name == "DecisionTreeRegressor":
        return 18
    return TUNE_N_ITER_RF_ET


def _pick_top_two_finalists(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distinct (model, target_strategy) pairs in benchmark order."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for r in sorted(rows, key=_benchmark_sort_key):
        key = (r["model"], r["target_strategy"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= 2:
            break
    return out


def _build_model_benchmark_df(
    run_id: str,
    chosen_name: str,
    chosen_target_mode: TargetMode,
    candidate_results: list[dict[str, Any]],
    finalists_tuned: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Flat table for notebooks / graders: default-param benchmark rows + tuned finalist rows.
    `selected` = yes only for the tuned finalist row that matches the final refit model.
    If no benchmark rows exist (e.g. insufficient dates), emit a single fallback row.
    """
    rows: list[dict[str, Any]] = []
    for r in candidate_results:
        rows.append(
            {
                "run_id": run_id,
                "stage": "benchmark_default_params",
                "model": r["model"],
                "target_strategy": r["target_strategy"],
                "val_mae": r.get("val_mae"),
                "val_rmse": r.get("val_rmse"),
                "val_mape": r.get("val_mape"),
                "val_bias": r.get("val_bias"),
                "selected": "no",
            }
        )
    for f in finalists_tuned:
        is_winner = f["model"] == chosen_name and f["target_strategy"] == chosen_target_mode
        m = f.get("validation_after_tune") or {}
        rows.append(
            {
                "run_id": run_id,
                "stage": "tuned_finalist",
                "model": f["model"],
                "target_strategy": f["target_strategy"],
                "val_mae": m.get("val_mae"),
                "val_rmse": m.get("val_rmse"),
                "val_mape": m.get("val_mape"),
                "val_bias": m.get("val_bias"),
                "selected": "yes" if is_winner else "no",
            }
        )
    if not rows:
        rows.append(
            {
                "run_id": run_id,
                "stage": "fallback_no_time_aware_benchmark",
                "model": chosen_name,
                "target_strategy": chosen_target_mode,
                "val_mae": None,
                "val_rmse": None,
                "val_mape": None,
                "val_bias": None,
                "selected": "yes",
            }
        )
    return pd.DataFrame(rows)


def run_once(dry_run: bool = False) -> dict[str, Any]:
    client = storage.Client(project=PROJECT_ID or None)
    df = _read_csv_from_gcs(client, GCS_BUCKET, DATA_KEY)

    required = {"scraped_at", "price", "year", "make", "model"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    dt = pd.to_datetime(df["scraped_at"], errors="coerce", utc=True)
    df["scraped_at_dt_utc"] = dt
    try:
        df["scraped_at_local"] = df["scraped_at_dt_utc"].dt.tz_convert(TIMEZONE)
    except Exception:
        df["scraped_at_local"] = df["scraped_at_dt_utc"]
    df["date_local"] = df["scraped_at_local"].dt.date

    df[TARGET_ORIGINAL] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df.get("mileage", pd.Series(index=df.index)))

    ref_year = df["scraped_at_local"].dt.year
    df["vehicle_age"] = (ref_year - df["year_num"]).clip(lower=0, upper=80)
    df["log_mileage"] = np.log1p(df["mileage_num"].clip(lower=0).fillna(0))

    zip5 = _clean_zip_series(
        df.get("zip_code", pd.Series(index=df.index, dtype="string")),
        df["state"] if "state" in df.columns else None,
    )
    df["zip_prefix"] = _zip_prefix_from_clean_zip(zip5)

    for c in [
        "make",
        "model",
        "transmission",
        "fuel",
        "drive",
        "condition",
        "title_status",
        "type",
        "seller_type",
        "state",
        "city",
        "zip_prefix",
    ]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].replace({np.nan: None}).astype(object)

    df = _bucket_rare_categories(df, ["make", "model", "city"], RARE_CAT_MIN_COUNT)

    if "cylinders" in df.columns:
        df["cylinders"] = _clean_numeric(df["cylinders"])
    else:
        df["cylinders"] = np.nan

    df, filter_stats = _apply_training_filters(df)

    df[TARGET_LOG] = np.log1p(df[TARGET_ORIGINAL].astype(float))

    orig_rows = len(df)
    valid_price_rows = int(df[TARGET_ORIGINAL].notna().sum())
    logging.info("Rows after filters total=%d | valid price=%d", orig_rows, valid_price_rows)

    unique_dates = sorted(d for d in df["date_local"].dropna().unique())
    if len(unique_dates) < 2:
        return {
            "status": "noop",
            "reason": "need at least two distinct local dates",
            "dates": [str(d) for d in unique_dates],
        }

    today_local = unique_dates[-1]
    train_df = df[df["date_local"] < today_local].copy()
    holdout_df = df[df["date_local"] == today_local].copy()
    train_df = train_df[train_df[TARGET_ORIGINAL].notna()]

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    cat_cols = [
        "make",
        "model",
        "transmission",
        "fuel",
        "drive",
        "condition",
        "title_status",
        "type",
        "seller_type",
        "state",
        "city",
        "zip_prefix",
    ]
    cat_cols = [c for c in cat_cols if c in train_df.columns]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    run_prefix = f"{MODEL_RUN_PREFIX}/{run_id}"

    benchmark_skip_reason: str | None = None
    val_local: Any = None
    tune_train = pd.DataFrame()
    tune_val = pd.DataFrame()
    candidate_results: list[dict[str, Any]] = []
    finalists_tuned: list[dict[str, Any]] = []
    feature_variant_label = "A"
    num_cols: list[str] = []
    feats: list[str] = []

    can_benchmark = len(unique_dates) >= 3
    if can_benchmark:
        val_local = unique_dates[-2]
        tune_train = train_df[train_df["date_local"] < val_local].copy()
        tune_val = train_df[train_df["date_local"] == val_local].copy()
        if len(tune_train) < 40 or len(tune_val) < 5:
            can_benchmark = False
            benchmark_skip_reason = "insufficient rows on tune_train or tune_val for time-aware benchmark"

    chosen_name: str
    chosen_target_mode: TargetMode
    chosen_params: dict[str, Any]

    if can_benchmark:
        num_cols, feature_variant_label = _select_feature_variant(tune_train, tune_val, cat_cols)
        feats = cat_cols + num_cols
        for model_name, builder in _default_regressor_builders():
            for tm in ("raw", "log"):
                reg_inst = builder()
                row = _benchmark_one(
                    model_name,
                    reg_inst,
                    tm,
                    cat_cols,
                    num_cols,
                    feats,
                    tune_train,
                    tune_val,
                )
                if row:
                    row["validation_date_local"] = str(val_local)
                    row["feature_variant"] = feature_variant_label
                    candidate_results.append(row)
        candidate_results.sort(key=_benchmark_sort_key)
        top_finalists = _pick_top_two_finalists(candidate_results)
        if not top_finalists:
            benchmark_skip_reason = "all benchmark fits failed; fallback to RandomForest+log defaults"
            chosen_name = "RandomForestRegressor"
            chosen_target_mode = "log"
            chosen_params = dict(DEFAULT_RF)
            num_cols = _num_cols_variant(train_df, "A")
            feats = cat_cols + num_cols
        else:
            for fin in top_finalists:
                mn = fin["model"]
                tm: TargetMode = fin["target_strategy"]
                grid = _param_grid_for_model(mn)
                defs = _default_params_for_model(mn)
                n_it = _tune_n_iter_for_model(mn)
                bp, _bmae, bmet = _tune_on_val(
                    mn,
                    lambda p, model_name=mn: _build_reg_for_model(model_name, p),
                    grid,
                    tm,
                    cat_cols,
                    num_cols,
                    feats,
                    tune_train,
                    tune_val,
                    n_it,
                    defs,
                )
                finalists_tuned.append(
                    {
                        "model": mn,
                        "target_strategy": tm,
                        "tuned_params": bp,
                        "validation_after_tune": bmet,
                    }
                )

            def _finalist_sort_key(f: dict[str, Any]) -> tuple[float, float, float]:
                m = f.get("validation_after_tune") or {}
                return (
                    float(m.get("val_mae", float("inf"))),
                    float(m.get("val_rmse", float("inf"))),
                    abs(float(m.get("val_bias", 0.0))),
                )

            finalists_tuned.sort(key=_finalist_sort_key)
            winner = finalists_tuned[0]
            chosen_name = winner["model"]
            chosen_target_mode = winner["target_strategy"]
            chosen_params = winner["tuned_params"]
    else:
        if benchmark_skip_reason is None:
            benchmark_skip_reason = (
                "fewer_than_3_distinct_local_dates"
                if len(unique_dates) < 3
                else "benchmark_disabled"
            )
        chosen_name = "RandomForestRegressor"
        chosen_target_mode = "log"
        chosen_params = dict(DEFAULT_RF)
        num_cols = _num_cols_variant(train_df, "A")
        feats = cat_cols + num_cols

    reg_final = _build_reg_for_model(chosen_name, chosen_params)
    pipe = _build_preprocess_and_model(cat_cols, num_cols, reg_final)
    y_train_fit = _y_for_target(train_df, chosen_target_mode)
    m_fit = y_train_fit.notna()
    pipe.fit(train_df.loc[m_fit, feats], y_train_fit.loc[m_fit])

    baseline_mae = None
    for row in candidate_results:
        if row.get("model") == "DecisionTreeRegressor" and row.get("target_strategy") == "log":
            baseline_mae = row.get("val_mae")
            break

    rf_tune_val_mae = next(
        (f["validation_after_tune"]["val_mae"] for f in finalists_tuned if f["model"] == "RandomForestRegressor"),
        float("nan"),
    )
    et_tune_val_mae = next(
        (f["validation_after_tune"]["val_mae"] for f in finalists_tuned if f["model"] == "ExtraTreesRegressor"),
        float("nan"),
    )
    best_rf_params = next(
        (f["tuned_params"] for f in finalists_tuned if f["model"] == "RandomForestRegressor"),
        dict(DEFAULT_RF),
    )
    best_et_params = next(
        (f["tuned_params"] for f in finalists_tuned if f["model"] == "ExtraTreesRegressor"),
        dict(DEFAULT_RF),
    )
    tune_note = (
        f"benchmark_val_date={val_local} winner={chosen_name} target={chosen_target_mode} "
        f"feature_variant={feature_variant_label}"
    )
    if benchmark_skip_reason:
        tune_note += f" | note={benchmark_skip_reason}"

    mae_today = mape_today = rmse_today = bias_today = None
    preds_df = pd.DataFrame()
    perm_df = pd.DataFrame()
    top_features: list[str] = []

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat_raw_space = pipe.predict(X_h)
        y_hat = _preds_to_dollars(y_hat_raw_space, chosen_target_mode)

        preds_df = pd.DataFrame(
            {
                "post_id": holdout_df.get("post_id", pd.Series(index=holdout_df.index)),
                "scraped_at": holdout_df["scraped_at"],
                "actual_price": holdout_df[TARGET_ORIGINAL],
                "pred_price": np.round(y_hat, 2),
                "residual": np.round(y_hat - holdout_df[TARGET_ORIGINAL].values, 2),
                "make": holdout_df.get("make"),
                "model": holdout_df.get("model"),
                "year": holdout_df.get("year"),
                "mileage": holdout_df.get("mileage"),
            }
        )

        mask = holdout_df[TARGET_ORIGINAL].notna()
        if mask.any():
            yt = holdout_df.loc[mask, TARGET_ORIGINAL].values.astype(float)
            yp = y_hat[mask.to_numpy()]
            mae_today = float(mean_absolute_error(yt, yp))
            mape_today = _mape(yt, yp)
            rmse_today = _rmse(yt, yp)
            bias_today = _bias(yt, yp)

        try:
            m_perm = holdout_df[TARGET_ORIGINAL].notna()
            if m_perm.any():
                if chosen_target_mode == "log":
                    y_perm = holdout_df.loc[m_perm, TARGET_LOG].values
                    perm_scoring = dollar_mae_scorer
                else:
                    y_perm = holdout_df.loc[m_perm, TARGET_ORIGINAL].values.astype(float)
                    perm_scoring = raw_dollar_mae_scorer
                r = permutation_importance(
                    pipe,
                    X_h.loc[m_perm],
                    y_perm,
                    n_repeats=15,
                    random_state=42,
                    scoring=perm_scoring,
                    n_jobs=-1,
                )
                names = np.array(feats)
                perm_df = pd.DataFrame(
                    {
                        "feature": names,
                        "importance_mean": r.importances_mean,
                        "importance_std": r.importances_std,
                    }
                ).sort_values("importance_mean", ascending=False)
                top_features = perm_df["feature"].head(3).tolist()
        except Exception as ex:
            logging.warning("permutation_importance failed: %s", ex)

    model_comparison = {
        "random_forest": {
            "tune_val_mae_dollars": rf_tune_val_mae if np.isfinite(rf_tune_val_mae) else None,
            "best_params": best_rf_params,
        },
        "extra_trees": {
            "tune_val_mae_dollars": et_tune_val_mae if np.isfinite(et_tune_val_mae) else None,
            "best_params": best_et_params,
        },
        "selected": chosen_name,
        "selected_target_strategy": chosen_target_mode,
        "selection_rule": (
            "time-aware validation on second-latest local date (dollar MAE/RMSE/bias); "
            "benchmark 4 models x (raw|log) targets; tune top 2 (model,target) pairs; "
            "winner = lowest post-tune val MAE then RMSE then |bias|; final holdout = latest local date"
        ),
    }

    target_transform_desc = (
        f"log1p({TARGET_ORIGINAL})" if chosen_target_mode == "log" else f"{TARGET_ORIGINAL} raw USD"
    )
    metrics_scale_desc = (
        "original_usd (expm1 predictions vs actual price_num)"
        if chosen_target_mode == "log"
        else "original_usd (direct predictions vs actual price_num)"
    )
    target_training_col = TARGET_LOG if chosen_target_mode == "log" else TARGET_ORIGINAL
    feature_notes_txt = (
        "variant B: vehicle_age, log_mileage, cylinders, year_num, mileage_num (+ categoricals)"
        if feature_variant_label == "B"
        else (
            "variant A: vehicle_age, log_mileage, cylinders (+ categoricals); "
            "year_num/mileage_num omitted as redundant with age/log mileage"
        )
    )

    benchmark_block = {
        "model_candidates": [
            "DecisionTreeRegressor",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "HistGradientBoostingRegressor",
        ],
        "target_strategies_compared": ["raw", "log"],
        "validation_date_local": str(val_local) if val_local is not None else None,
        "holdout_date_local": str(today_local),
        "feature_variant": feature_variant_label,
        "skip_reason": benchmark_skip_reason,
        "candidate_results": candidate_results,
        "tuned_finalists": finalists_tuned,
        "filtering_applied": filter_stats,
    }

    metrics = {
        "run_id": run_id,
        "timezone": TIMEZONE,
        "eval_date_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae": mae_today,
        "mape": mape_today,
        "rmse": rmse_today,
        "bias": bias_today,
        "baseline_dt_mae_tune_val": baseline_mae,
        "tuning": tune_note,
        "best_rf_params": best_rf_params,
        "best_extra_trees_params": best_et_params,
        "chosen_model": chosen_name,
        "chosen_model_params": chosen_params,
        "model_comparison": model_comparison,
        "target_strategy": chosen_target_mode,
        "target_training": target_training_col,
        "target_transform": target_transform_desc,
        "metrics_scale": metrics_scale_desc,
        "benchmark": benchmark_block,
        "filtering": filter_stats,
        "data_key": DATA_KEY,
        "model": chosen_name,
    }

    model_info = {
        "run_id": run_id,
        "data_key": DATA_KEY,
        "target": TARGET_ORIGINAL,
        "target_training": target_training_col,
        "target_transform": (
            "log1p(price_num); holdout predictions back-transformed with numpy.expm1"
            if chosen_target_mode == "log"
            else "price_num in USD; no log transform"
        ),
        "target_strategy": chosen_target_mode,
        "features": feats,
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "feature_variant": feature_variant_label,
        "feature_notes": feature_notes_txt,
        "timezone": TIMEZONE,
        "eval_date_local": str(today_local),
        "random_forest_params": best_rf_params,
        "extra_trees_params": best_et_params,
        "chosen_model": chosen_name,
        "chosen_model_params": chosen_params,
        "model_comparison": model_comparison,
        "benchmark": benchmark_block,
        "filtering": filter_stats,
        "pipeline": f"ColumnTransformer(SimpleImputer+OneHotEncoder) -> {chosen_name}",
        "tuning_note": tune_note,
    }

    if not dry_run and not preds_df.empty:
        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            f"{run_prefix}/predictions.csv",
            preds_df.to_csv(index=False),
            "text/csv",
        )
        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            f"{run_prefix}/metrics.json",
            json.dumps(metrics, indent=2, default=str),
            "application/json",
        )
        mcsv = pd.DataFrame(
            [
                {
                    "run_id": run_id,
                    "eval_date_local": str(today_local),
                    "mae": mae_today,
                    "mape": mape_today,
                    "rmse": rmse_today,
                    "bias": bias_today,
                    "model": chosen_name,
                    "target_strategy": chosen_target_mode,
                }
            ]
        )
        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            f"{run_prefix}/metrics.csv",
            mcsv.to_csv(index=False),
            "text/csv",
        )
        if not perm_df.empty:
            _write_text_to_gcs(
                client,
                GCS_BUCKET,
                f"{run_prefix}/permutation_importance.csv",
                perm_df.to_csv(index=False),
                "text/csv",
            )

        for i, fname in enumerate(top_features[:3], start=1):
            try:
                if fname not in X_h.columns:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                PartialDependenceDisplay.from_estimator(
                    pipe,
                    X_h,
                    features=[fname],
                    ax=ax,
                )
                ax.set_title(f"PDP: {fname} ({chosen_target_mode} target)")
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120)
                plt.close(fig)
                safe = re.sub(r"[^\w\-]+", "_", str(fname))[:40]
                _write_bytes_to_gcs(
                    client,
                    GCS_BUCKET,
                    f"{run_prefix}/pdp/pdp_top{i}_{safe}.png",
                    buf.getvalue(),
                    "image/png",
                )
            except Exception as ex:
                logging.warning("PDP failed for %s: %s", fname, ex)

        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            f"{run_prefix}/model_info.json",
            json.dumps(model_info, indent=2, default=str),
        )

        bench_df = _build_model_benchmark_df(
            run_id,
            chosen_name,
            chosen_target_mode,
            candidate_results,
            finalists_tuned,
        )
        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            f"{run_prefix}/model_benchmark.csv",
            bench_df.to_csv(index=False),
            "text/csv",
        )

        tune_val_mae_hist = None
        if finalists_tuned:
            tune_val_mae_hist = finalists_tuned[0].get("validation_after_tune", {}).get("val_mae")
        hist_row = {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "eval_date_local": str(today_local),
            "mae": mae_today,
            "mape": mape_today,
            "rmse": rmse_today,
            "bias": bias_today,
            "train_rows": len(train_df),
            "holdout_rows": len(holdout_df),
            "data_key": DATA_KEY,
            "model": chosen_name,
            "target_strategy": chosen_target_mode,
            "tune_validation_mae": tune_val_mae_hist,
        }
        prev = _read_text_from_gcs(client, GCS_BUCKET, METRICS_HISTORY_KEY)
        if prev:
            try:
                old = pd.read_csv(io.StringIO(prev))
                hist = pd.concat([old, pd.DataFrame([hist_row])], ignore_index=True)
            except Exception:
                hist = pd.DataFrame([hist_row])
        else:
            hist = pd.DataFrame([hist_row])
        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            METRICS_HISTORY_KEY,
            hist.to_csv(index=False),
            "text/csv",
        )
        _write_text_to_gcs(
            client,
            GCS_BUCKET,
            f"{run_prefix}/metrics_history.csv",
            hist.to_csv(index=False),
            "text/csv",
        )

        logging.info("Wrote model run to gs://%s/%s/", GCS_BUCKET, run_prefix)
    else:
        logging.info("Dry run or empty holdout; skip artifact writes. run_id=%s", run_id)

    return {
        "status": "ok",
        "run_id": run_id,
        "today_local": str(today_local),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "mae": mae_today,
        "mape": mape_today,
        "rmse": rmse_today,
        "bias": bias_today,
        "chosen_model": chosen_name,
        "chosen_target_strategy": chosen_target_mode,
        "feature_variant": feature_variant_label,
        "benchmark_rows": len(candidate_results),
        "run_prefix": f"gs://{GCS_BUCKET}/{run_prefix}/",
        "dry_run": dry_run,
        "timezone": TIMEZONE,
    }


def train_dt_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        code = 200 if result.get("status") == "ok" else 204
        return (json.dumps(result, default=str), code, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s", e)
        logging.error("Trace:\n%s", traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
