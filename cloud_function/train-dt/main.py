# Time-aware training: train on all local dates before latest; hold out latest day.
# Tuned RandomForestRegressor + artifacts under structured/model_runs/<run_id>/
# HTTP entrypoint: train_dt_http

from __future__ import annotations

import io
import json
import logging
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _read_csv_from_gcs(client: storage.Client, bucket: str, key: str) -> pd.DataFrame:
    b = client.bucket(bucket)
    blob = b.blob(key)
    if not blob.exists():
        raise FileNotFoundError(f"gs://{bucket}/{key} not found")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


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


def _build_preprocess_and_model(
    cat_cols: list[str],
    num_cols: list[str],
    rf_params: dict[str, Any],
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
    rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
        **rf_params,
    )
    return Pipeline([("pre", pre), ("model", rf)])


def _dt_baseline_mae(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cat_cols: list[str],
    num_cols: list[str],
) -> float | None:
    if X_val.empty or not y_val.notna().any():
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
        m_tr = y_train.notna()
        m_va = y_val.notna()
        pipe.fit(X_train.loc[m_tr], y_train.loc[m_tr])
        pred = pipe.predict(X_val.loc[m_va])
        return float(mean_absolute_error(y_val.loc[m_va], pred))
    except Exception:
        return None


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

    df["price_num"] = _clean_numeric(df["price"])
    df["year_num"] = _clean_numeric(df["year"])
    df["mileage_num"] = _clean_numeric(df.get("mileage", pd.Series(index=df.index)))

    ref_year = df["scraped_at_local"].dt.year
    df["vehicle_age"] = (ref_year - df["year_num"]).clip(lower=0, upper=80)
    df["log_mileage"] = np.log1p(df["mileage_num"].clip(lower=0).fillna(0))

    z = df.get("zip_code", pd.Series(index=df.index)).astype(str).str.replace(r"\D", "", regex=True)
    df["zip_prefix"] = z.str[:3].replace({"": np.nan, "nan": np.nan})
    df["zip_prefix"] = df["zip_prefix"].fillna("na")

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

    orig_rows = len(df)
    valid_price_rows = int(df["price_num"].notna().sum())
    logging.info("Rows total=%d | valid price=%d", orig_rows, valid_price_rows)

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
    train_df = train_df[train_df["price_num"].notna()]

    if len(train_df) < 40:
        return {"status": "noop", "reason": "too few training rows", "train_rows": int(len(train_df))}

    target = "price_num"
    num_cols = ["vehicle_age", "log_mileage", "year_num", "mileage_num", "cylinders"]
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

    # --- Date-aware tuning: train < val_day < today_local ---
    best_params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
    }
    tune_note = "defaults"

    if len(unique_dates) >= 3:
        val_local = unique_dates[-2]
        tune_train = train_df[train_df["date_local"] < val_local]
        tune_val = train_df[train_df["date_local"] == val_local]
        if len(tune_train) >= 40 and len(tune_val) >= 5:
            X_tr, y_tr = tune_train[feats], tune_train[target]
            X_va, y_va = tune_val[feats], tune_val[target]
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [12, 20, 28, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", 0.3, 0.5],
            }
            best_mae = float("inf")
            for params in ParameterSampler(param_grid, n_iter=12, random_state=42):
                pipe = _build_preprocess_and_model(cat_cols, num_cols, params)
                try:
                    pipe.fit(X_tr, y_tr)
                    pred = pipe.predict(X_va)
                    m_va = y_va.notna()
                    if not m_va.any():
                        continue
                    m = mean_absolute_error(y_va.loc[m_va], pred[m_va.to_numpy()])
                    if m < best_mae:
                        best_mae = m
                        best_params = dict(params)
                except Exception as ex:
                    logging.warning("Tune iteration failed: %s", ex)
            tune_note = f"val_date={val_local} best_mae={best_mae:.2f}"
            logging.info("Tuning done: %s params=%s", tune_note, best_params)

    X_train = train_df[feats]
    y_train = train_df[target]
    pipe = _build_preprocess_and_model(cat_cols, num_cols, best_params)
    pipe.fit(X_train, y_train)

    baseline_mae = None
    if len(unique_dates) >= 3:
        val_local = unique_dates[-2]
        tune_train = train_df[train_df["date_local"] < val_local]
        tune_val = train_df[train_df["date_local"] == val_local]
        baseline_mae = _dt_baseline_mae(
            tune_train[feats], tune_train[target], tune_val[feats], tune_val[target], cat_cols, num_cols
        )

    mae_today = mape_today = rmse_today = bias_today = None
    preds_df = pd.DataFrame()
    perm_df = pd.DataFrame()
    top_features: list[str] = []

    if not holdout_df.empty:
        X_h = holdout_df[feats]
        y_hat = pipe.predict(X_h)

        preds_df = pd.DataFrame(
            {
                "post_id": holdout_df.get("post_id", pd.Series(index=holdout_df.index)),
                "scraped_at": holdout_df["scraped_at"],
                "actual_price": holdout_df["price_num"],
                "pred_price": np.round(y_hat, 2),
                "residual": np.round(y_hat - holdout_df["price_num"].values, 2),
                "make": holdout_df.get("make"),
                "model": holdout_df.get("model"),
                "year": holdout_df.get("year"),
                "mileage": holdout_df.get("mileage"),
            }
        )

        mask = holdout_df["price_num"].notna()
        if mask.any():
            yt = holdout_df.loc[mask, "price_num"].values.astype(float)
            yp = y_hat[mask.to_numpy()]
            mae_today = float(mean_absolute_error(yt, yp))
            mape_today = _mape(yt, yp)
            rmse_today = _rmse(yt, yp)
            bias_today = _bias(yt, yp)

        # Permutation importance on holdout rows with known price
        try:
            m_perm = holdout_df["price_num"].notna()
            if m_perm.any():
                r = permutation_importance(
                    pipe,
                    X_h.loc[m_perm],
                    holdout_df.loc[m_perm, target],
                    n_repeats=15,
                    random_state=42,
                    scoring="neg_mean_absolute_error",
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
        "best_rf_params": best_params,
        "data_key": DATA_KEY,
        "model": "RandomForestRegressor",
    }

    model_info = {
        "run_id": run_id,
        "data_key": DATA_KEY,
        "target": target,
        "features": feats,
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "timezone": TIMEZONE,
        "eval_date_local": str(today_local),
        "random_forest_params": best_params,
        "pipeline": "ColumnTransformer(SimpleImputer+OneHotEncoder) -> RandomForestRegressor",
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

        # PDPs for top 3 features (by permutation importance)
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
                ax.set_title(f"PDP: {fname}")
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
            "application/json",
        )

        # Append cumulative metrics history (single CSV blob)
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
            "model": "RandomForestRegressor",
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
        # Copy snapshot into this run for sync convenience
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
