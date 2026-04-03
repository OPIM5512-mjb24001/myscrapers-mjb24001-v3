# Time-aware training: train on all local dates before latest; hold out latest day.
# Tuned tree ensemble (RandomForest vs ExtraTrees) + artifacts under structured/model_runs/<run_id>/
# Target: log1p(price); metrics and predictions on original dollar scale.
# HTTP entrypoint: train_dt_http

from __future__ import annotations

import io
import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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


def _dollar_mae_from_log(y_log_true: np.ndarray, y_log_pred: np.ndarray) -> float:
    return float(mean_absolute_error(np.expm1(y_log_true), np.expm1(y_log_pred)))


dollar_mae_scorer = make_scorer(
    lambda y_true_log, y_pred_log: -mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log)),
    greater_is_better=True,
)


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


def _dt_baseline_mae_dollars(
    X_train: pd.DataFrame,
    y_log_train: pd.Series,
    X_val: pd.DataFrame,
    y_log_val: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
) -> float | None:
    if X_val.empty or not y_log_val.notna().any():
        return None
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
        ],
    )
    pipe = Pipeline(
        [("pre", pre), ("model", DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=42))]
    )
    try:
        m_tr = y_log_train.notna()
        m_va = y_log_val.notna()
        pipe.fit(X_train.loc[m_tr], y_log_train.loc[m_tr])
        pred_log = pipe.predict(X_val.loc[m_va])
        return _dollar_mae_from_log(y_log_val.loc[m_va].values.astype(float), pred_log)
    except Exception:
        return None


def _tune_ensemble(
    name: str,
    build_reg: Callable[[dict[str, Any]], Any],
    param_grid: dict[str, list],
    X_tr: pd.DataFrame,
    y_log_tr: pd.Series,
    X_va: pd.DataFrame,
    y_log_va: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
    default_params: dict[str, Any],
    n_iter: int = 12,
) -> tuple[dict[str, Any], float]:
    best_mae = float("inf")
    best_params: dict[str, Any] = {}
    for params in ParameterSampler(param_grid, n_iter=n_iter, random_state=42):
        p = dict(params)
        try:
            reg = build_reg(p)
            pipe = _build_preprocess_and_model(cat_cols, num_cols, reg)
            pipe.fit(X_tr, y_log_tr)
            pred_log = pipe.predict(X_va)
            m_va = y_log_va.notna()
            if not m_va.any():
                continue
            mae = _dollar_mae_from_log(y_log_va.loc[m_va].values.astype(float), pred_log[m_va.to_numpy()])
            if mae < best_mae:
                best_mae = mae
                best_params = {k: v for k, v in p.items()}
        except Exception as ex:
            logging.warning("Tune iteration failed (%s): %s", name, ex)
    if not best_params or not np.isfinite(best_mae):
        return dict(default_params), float("nan")
    return best_params, best_mae


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

    # Simplified numerics: vehicle_age + log_mileage (+ cylinders); drop redundant year_num / mileage_num
    num_cols = ["vehicle_age", "log_mileage", "cylinders"]
    num_cols = [c for c in num_cols if c in train_df.columns]
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
    feats = cat_cols + num_cols

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H")
    run_prefix = f"{MODEL_RUN_PREFIX}/{run_id}"

    param_grid_shared: dict[str, list] = {
        "n_estimators": [100, 200, 300],
        "max_depth": [12, 20, 28, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", 0.3, 0.5],
    }

    default_rf = {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 4, "max_features": "sqrt"}
    best_rf_params = dict(default_rf)
    best_et_params = dict(default_rf)
    tune_note = "defaults"
    rf_tune_val_mae = float("nan")
    et_tune_val_mae = float("nan")

    if len(unique_dates) >= 3:
        val_local = unique_dates[-2]
        tune_train = train_df[train_df["date_local"] < val_local]
        tune_val = train_df[train_df["date_local"] == val_local]
        if len(tune_train) >= 40 and len(tune_val) >= 5:
            X_tr, y_tr = tune_train[feats], tune_train[TARGET_LOG]
            X_va, y_va = tune_val[feats], tune_val[TARGET_LOG]

            def build_rf(p: dict) -> RandomForestRegressor:
                return RandomForestRegressor(random_state=42, n_jobs=-1, **p)

            def build_et(p: dict) -> ExtraTreesRegressor:
                return ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=False, **p)

            best_rf_params, rf_tune_val_mae = _tune_ensemble(
                "RandomForest",
                build_rf,
                param_grid_shared,
                X_tr,
                y_tr,
                X_va,
                y_va,
                cat_cols,
                num_cols,
                default_rf,
            )
            best_et_params, et_tune_val_mae = _tune_ensemble(
                "ExtraTrees",
                build_et,
                param_grid_shared,
                X_tr,
                y_tr,
                X_va,
                y_va,
                cat_cols,
                num_cols,
                default_rf,
            )
            tune_note = (
                f"val_date={val_local} rf_tune_mae_dollars={rf_tune_val_mae:.2f} "
                f"et_tune_mae_dollars={et_tune_val_mae:.2f}"
            )
            logging.info("Tuning done: %s rf_params=%s et_params=%s", tune_note, best_rf_params, best_et_params)

    # Select model by tune validation dollar MAE (lower wins); tie-break RF
    if np.isfinite(rf_tune_val_mae) and np.isfinite(et_tune_val_mae):
        if et_tune_val_mae < rf_tune_val_mae - 1e-9:
            chosen_name = "ExtraTreesRegressor"
            chosen_params = best_et_params
            chosen_build = lambda p: ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=False, **p)
        else:
            chosen_name = "RandomForestRegressor"
            chosen_params = best_rf_params
            chosen_build = lambda p: RandomForestRegressor(random_state=42, n_jobs=-1, **p)
    elif np.isfinite(et_tune_val_mae):
        chosen_name = "ExtraTreesRegressor"
        chosen_params = best_et_params
        chosen_build = lambda p: ExtraTreesRegressor(random_state=42, n_jobs=-1, bootstrap=False, **p)
    elif np.isfinite(rf_tune_val_mae):
        chosen_name = "RandomForestRegressor"
        chosen_params = best_rf_params
        chosen_build = lambda p: RandomForestRegressor(random_state=42, n_jobs=-1, **p)
    else:
        chosen_name = "RandomForestRegressor"
        chosen_params = best_rf_params
        chosen_build = lambda p: RandomForestRegressor(random_state=42, n_jobs=-1, **p)

    reg_final = chosen_build(chosen_params)
    pipe = _build_preprocess_and_model(cat_cols, num_cols, reg_final)
    X_train = train_df[feats]
    y_train_log = train_df[TARGET_LOG]
    pipe.fit(X_train, y_train_log)

    baseline_mae = None
    if len(unique_dates) >= 3:
        val_local = unique_dates[-2]
        tune_train = train_df[train_df["date_local"] < val_local]
        tune_val = train_df[train_df["date_local"] == val_local]
        baseline_mae = _dt_baseline_mae_dollars(
            tune_train[feats],
            tune_train[TARGET_LOG],
            tune_val[feats],
            tune_val[TARGET_LOG],
            cat_cols,
            num_cols,
        )

    mae_today = mape_today = rmse_today = bias_today = None
    preds_df = pd.DataFrame()
    perm_df = pd.DataFrame()
    top_features: list[str] = []

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat_log = pipe.predict(X_h)
        y_hat = np.expm1(y_hat_log)

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
                y_perm_log = holdout_df.loc[m_perm, TARGET_LOG].values
                r = permutation_importance(
                    pipe,
                    X_h.loc[m_perm],
                    y_perm_log,
                    n_repeats=15,
                    random_state=42,
                    scoring=dollar_mae_scorer,
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
        "selection_rule": "lower tune validation MAE on original dollar scale (second-to-last local date)",
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
        "target_transform": f"log1p({TARGET_ORIGINAL})",
        "metrics_scale": "original_usd (expm1 predictions vs actual price_num)",
        "filtering": filter_stats,
        "data_key": DATA_KEY,
        "model": chosen_name,
    }

    model_info = {
        "run_id": run_id,
        "data_key": DATA_KEY,
        "target": TARGET_ORIGINAL,
        "target_training": TARGET_LOG,
        "target_transform": "log1p(price_num); predictions back-transformed with numpy.expm1",
        "features": feats,
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "feature_notes": "vehicle_age and log_mileage only (year_num and mileage_num omitted as redundant)",
        "timezone": TIMEZONE,
        "eval_date_local": str(today_local),
        "random_forest_params": best_rf_params,
        "extra_trees_params": best_et_params,
        "chosen_model": chosen_name,
        "chosen_model_params": chosen_params,
        "model_comparison": model_comparison,
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
                ax.set_title(f"PDP: {fname} (log target)")
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
