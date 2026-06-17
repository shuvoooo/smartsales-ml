"""
src/predict.py
Loads trained models and provides inference functions for the API.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta

import xgboost as xgb

log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

#  Model loader (cached at startup) 

# Keep a module-level cache so API requests reuse loaded model objects instead
# of hitting disk on every call. This keeps inference latency low and avoids
# repeated deserialization work inside a long-lived web process.
_cache = {}

def _load(key: str, loader):
    """Lazy-load and memoize a model artifact or metadata blob.

    Args:
        key: Stable cache key identifying the artifact.
        loader: Zero-argument callable that loads the artifact from disk.

    Returns:
        Any: The cached or newly loaded artifact.
    """
    if key not in _cache:
        _cache[key] = loader()
    return _cache[key]

def get_seg_model():
    """Return the cached K-Means segmentation model."""
    return _load("seg_model",  lambda: joblib.load(os.path.join(MODELS_DIR, "segmentation", "kmeans.pkl")))

def get_seg_scaler():
    """Return the cached scaler used for RFM standardization."""
    return _load("seg_scaler", lambda: joblib.load(os.path.join(MODELS_DIR, "segmentation", "scaler.pkl")))

def get_seg_labels():
    """Return the cached cluster-to-segment label mapping."""
    return _load("seg_labels", lambda: json.load(open(os.path.join(MODELS_DIR, "segmentation", "label_map.json"))))

def get_xgb_model():
    """Return the cached XGBoost forecasting model."""
    def _load_xgb():
        m = xgb.XGBRegressor()
        m.load_model(os.path.join(MODELS_DIR, "forecasting", "xgb_model.json"))
        return m
    return _load("xgb_model", _load_xgb)

def get_fc_info():
    """Return cached forecasting feature metadata."""
    return _load("fc_info",   lambda: json.load(open(os.path.join(MODELS_DIR, "forecasting", "feature_info.json"))))

def get_churn_model():
    """Return the cached LightGBM churn model."""
    return _load("churn_model", lambda: joblib.load(os.path.join(MODELS_DIR, "churn", "lgbm_model.pkl")))

def get_churn_info():
    """Return cached churn model metadata."""
    return _load("churn_info",  lambda: json.load(open(os.path.join(MODELS_DIR, "churn", "model_info.json"))))


#  1. Segment Prediction 

def predict_segment(recency: float, frequency: float, monetary: float) -> dict:
    """
    Predict the RFM-based customer segment.

    Args:
        recency: Days since last purchase; smaller values represent more recent
            engagement and therefore better recency.
        frequency: Number of unique invoices or orders associated with the
            customer.
        monetary: Total spend in pounds.

    Returns:
        dict: Predicted cluster ID, business-facing segment name, heuristic RFM
        scores, and a segment description.

    Notes:
        Cluster assignment comes from the trained K-Means model after applying
        the saved scaler. The returned R/F/M scores are lightweight 1-5 summary
        scores for API consumers: recency is bucketed into roughly five chunks
        across a year, while frequency and monetary use log scaling so a few
        extreme high-value customers do not compress everyone else into low
        scores.
    """
    model   = get_seg_model()
    scaler  = get_seg_scaler()
    labels  = get_seg_labels()

    X = scaler.transform([[recency, frequency, monetary]])
    cluster = int(model.predict(X)[0])
    segment = labels.get(str(cluster), "Unknown")

    # These scores are heuristics for explainability in the API response, not
    # the features used by K-Means. Log scaling keeps skewed business metrics
    # such as spend and order count more interpretable on a 1-5 scale.
    r_score = max(1, 6 - int(recency / 73))   # ~365/5 buckets
    f_score = min(5, max(1, int(np.log1p(frequency) / np.log1p(80) * 4) + 1))
    m_score = min(5, max(1, int(np.log1p(monetary)  / np.log1p(5000) * 4) + 1))

    return {
        "cluster_id":   cluster,
        "segment_name": segment,
        "rfm_scores":   {"R": r_score, "F": f_score, "M": m_score},
        "description":  _segment_description(segment),
    }


def _segment_description(segment: str) -> str:
    """Return business guidance associated with a predicted segment.

    Args:
        segment: Human-readable segment name.

    Returns:
        str: Short explanation used directly in API responses.
    """
    desc = {
        "Champions":        "Bought recently, buy often, and spend the most.",
        "Loyal Customers":  "Buy regularly and respond well to promotions.",
        "At Risk":          "Haven't bought recently. Need re-engagement campaigns.",
        "Lost / Inactive":  "Last purchase was long ago. Win-back offers recommended.",
    }
    return desc.get(segment, "")


#  2. Sales Forecast 

def predict_forecast(horizon_days: int = 30) -> dict:
    """
    Forecast daily revenue using recursive multi-step prediction.

    Args:
        horizon_days: Number of future days to predict.

    Returns:
        dict: Forecast dates, day-level predictions, and aggregate totals.

    Notes:
        The forecasting model is trained for one-step prediction from lag and
        rolling-window features. To generate multiple future days, each newly
        predicted value is appended to the rolling history and reused to build
        the next day's features. This recursive/autoregressive strategy lets a
        single supervised regressor forecast an arbitrary horizon.
    """
    model = get_xgb_model()
    info  = get_fc_info()

    feature_cols = info["feature_cols"]
    last_date    = datetime.strptime(info["last_date"], "%Y-%m-%d")
    history      = info["last_values"]   # last 60 days of y

    predictions = []
    rolling_window = list(history)

    for i in range(horizon_days):
        future_date = last_date + timedelta(days=i + 1)

        # Once we move beyond the first forecasted day, lag features may come
        # from earlier predictions rather than historical observations.
        lag_7  = rolling_window[-7]  if len(rolling_window) >= 7  else 0
        lag_14 = rolling_window[-14] if len(rolling_window) >= 14 else 0
        lag_30 = rolling_window[-30] if len(rolling_window) >= 30 else 0
        lag_60 = rolling_window[-60] if len(rolling_window) >= 60 else 0
        rolling_7  = np.mean(rolling_window[-7:])
        rolling_30 = np.mean(rolling_window[-30:]) if len(rolling_window) >= 30 else np.mean(rolling_window)

        row = {
            "lag_7":       lag_7,
            "lag_14":      lag_14,
            "lag_30":      lag_30,
            "lag_60":      lag_60,
            "rolling_7":   rolling_7,
            "rolling_30":  rolling_30,
            "dow":         future_date.weekday(),
            "month":       future_date.month,
            "week":        future_date.isocalendar()[1],
            "is_weekend":  int(future_date.weekday() >= 5),
            "is_december": int(future_date.month == 12),
        }

        X = pd.DataFrame([row])[feature_cols]
        pred = float(model.predict(X)[0])
        pred = max(0, pred)   # Clamp noise from the regressor to a valid KPI.

        rolling_window.append(pred)
        predictions.append(round(pred, 2))

    dates = [
        (last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon_days)
    ]

    return {
        "dates":          dates,
        "predicted":      predictions,
        "total_forecast": round(sum(predictions), 2),
        "avg_daily":      round(np.mean(predictions), 2),
        "horizon_days":   horizon_days,
        "forecast_from":  (last_date + timedelta(days=1)).strftime("%Y-%m-%d"),
    }


#  3. Churn Prediction 

def predict_churn(recency: float, frequency: float, monetary: float,
                  avg_order_value: float, purchase_span: float,
                  total_items: float, days_since_first: float,
                  purchase_rate: float) -> dict:
    """
    Predict churn probability for a single customer.

    Note: ``recency`` is accepted for API compatibility (and returned in the
    response context) but is NOT passed to the model — including it would leak
    the churn label because the label is defined as ``Recency >= 90 days``.

    Args:
        recency: Days since the customer's most recent purchase.
        frequency: Number of unique orders placed by the customer.
        monetary: Total revenue generated by the customer.
        avg_order_value: Average spend per order.
        purchase_span: Days between first and last recorded purchase.
        total_items: Total quantity of items purchased.
        days_since_first: Customer age in days since the first purchase.
        purchase_rate: Approximate purchases per month across the customer
            lifetime.

    Returns:
        dict: Churn probability, risk band, boolean hard prediction, and a
        recommended retention action.
    """
    model        = get_churn_model()
    info         = get_churn_info()
    feature_cols = info["feature_cols"]
    threshold    = info["threshold"]

    # Recency is intentionally excluded — it directly defines the churn label.
    # All other behavioural signals are safe predictors.
    row = {
        "Frequency":      frequency,
        "Monetary":       monetary,
        "AvgOrderValue":  avg_order_value,
        "PurchaseSpan":   purchase_span,
        "TotalItems":     total_items,
        "DaysSinceFirst": days_since_first,
        "PurchaseRate":   purchase_rate,
    }
    # Rebuild the exact training feature schema so column order and presence
    # match the persisted model's expectations.
    for col in feature_cols:
        if col not in row:
            row[col] = 0

    X    = pd.DataFrame([row])[feature_cols]
    prob = float(model.predict_proba(X)[0][1])

    risk = "High" if prob >= 0.6 else "Medium" if prob >= 0.3 else "Low"

    recommendations = {
        "High":   "Send win-back email with 20% discount immediately.",
        "Medium": "Add to loyalty re-engagement campaign next week.",
        "Low":    "Customer is healthy — keep up regular communications.",
    }

    return {
        "churn_probability": round(prob, 4),
        "risk_level":        risk,
        "churned_predicted": prob >= threshold,
        "recommendation":    recommendations[risk],
    }


#  4. Bulk loaders (for dashboard) 

def get_all_segments() -> list:
    """Return dashboard-ready segmentation rows.

    Returns:
        list: Customer segmentation records, or an empty list when the scored
        export does not exist yet because training has not been run.
    """
    path = os.path.join(DATA_DIR, "rfm_segmented.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df[["CustomerID", "Recency", "Frequency", "Monetary",
               "AvgOrderValue", "cluster", "segment"]].to_dict(orient="records")


def get_all_churn_scores() -> list:
    """Return dashboard-ready churn scores sorted by highest risk first.

    Returns:
        list: Churn scoring records, or an empty list when the training step
        has not produced the scored CSV yet.
    """
    path = os.path.join(DATA_DIR, "churn_scored.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    cols = ["CustomerID", "Recency", "Frequency", "Monetary",
            "churn_probability", "risk_level"]
    return df[cols].sort_values("churn_probability", ascending=False).to_dict(orient="records")


def get_historical_sales() -> list:
    """Return historical daily revenue for dashboard charting.

    Returns:
        list: Historical revenue records, or an empty list when the processed
        time-series file is unavailable because the data pipeline has not run.
    """
    path = os.path.join(DATA_DIR, "timeseries.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, parse_dates=["ds"])
    return df[["ds", "y"]].rename(columns={"ds": "date", "y": "revenue"}).assign(
        date=lambda x: x["date"].dt.strftime("%Y-%m-%d")
    ).to_dict(orient="records")
