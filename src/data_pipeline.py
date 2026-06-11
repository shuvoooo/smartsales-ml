"""
src/data_pipeline.py
Full data pipeline: load → clean → feature engineering → save.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── 1. Load ───────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV and parse dates."""
    log.info(f"Loading data from {path}")
    df = pd.read_csv(path, parse_dates=["InvoiceDate"], dtype={"CustomerID": str})
    log.info(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ── 2. Clean ──────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove known data quality issues:
      - Rows with missing CustomerID (guest checkouts — can't track behaviour)
      - Cancelled orders (InvoiceNo starts with 'C')
      - Non-positive quantities or unit prices
    """
    log.info("Cleaning data...")
    before = len(df)

    # Drop missing CustomerID
    df = df.dropna(subset=["CustomerID"])
    log.info(f"  After dropping null CustomerID: {len(df):,} rows")

    # Drop cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    log.info(f"  After dropping cancellations:   {len(df):,} rows")

    # Drop non-positive quantities / prices
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    log.info(f"  After dropping bad qty/price:   {len(df):,} rows")

    # Derived column
    df = df.copy()
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    log.info(f"  Removed {before - len(df):,} rows total ({(before - len(df)) / before:.1%})")
    return df.reset_index(drop=True)


# ── 3. RFM Features ───────────────────────────────────────────────────────────

def build_rfm(df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, Monetary per customer.

    Args:
        df:            Cleaned dataframe
        snapshot_date: Reference date (defaults to max InvoiceDate + 1 day)

    Returns:
        DataFrame indexed by CustomerID with columns [Recency, Frequency, Monetary,
        AvgOrderValue, PurchaseSpan, FirstPurchase, LastPurchase]
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max() + timedelta(days=1)

    log.info(f"Building RFM features (snapshot: {snapshot_date.date()})")

    rfm = df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        TotalItems=("Quantity", "sum"),
    ).reset_index()

    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["PurchaseSpan"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["AvgOrderValue"] = rfm["AvgOrderValue"].round(2)

    log.info(f"  RFM built for {len(rfm):,} customers")
    return rfm


# ── 4. Time-Series Features ───────────────────────────────────────────────────

def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily revenue time series with lag and rolling features for forecasting.

    Returns:
        DataFrame with columns [ds, y, lag_7, lag_14, lag_30, lag_60,
        rolling_7, rolling_30, dow, month, week, is_weekend, is_december]
    """
    log.info("Building time-series features...")

    daily = (
        df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"]
        .sum()
        .reset_index()
        .rename(columns={"InvoiceDate": "ds", "TotalPrice": "y"})
    )
    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.set_index("ds").asfreq("D").fillna(0).reset_index()

    # Lag features
    for lag in [7, 14, 30, 60]:
        daily[f"lag_{lag}"] = daily["y"].shift(lag)

    # Rolling averages
    daily["rolling_7"] = daily["y"].shift(1).rolling(7).mean()
    daily["rolling_30"] = daily["y"].shift(1).rolling(30).mean()

    # Calendar features
    daily["dow"] = daily["ds"].dt.dayofweek  # 0=Mon
    daily["month"] = daily["ds"].dt.month
    daily["week"] = daily["ds"].dt.isocalendar().week.astype(int)
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["is_december"] = (daily["month"] == 12).astype(int)

    # Drop rows where lags are NaN
    daily = daily.dropna().reset_index(drop=True)

    log.info(f"  Time series: {len(daily):,} days  |  "
             f"{daily['ds'].min().date()} → {daily['ds'].max().date()}")
    return daily


# ── 5. Churn Features ─────────────────────────────────────────────────────────

def build_churn_features(df: pd.DataFrame, rfm: pd.DataFrame,
                         churn_days: int = 90) -> pd.DataFrame:
    """
    Build features for churn classification.
    Churn label = 1 if customer did NOT purchase in the last `churn_days` days.

    Args:
        df:         Cleaned dataframe
        rfm:        RFM dataframe
        churn_days: Inactivity threshold to define churn

    Returns:
        DataFrame with features and 'churned' label column
    """
    log.info(f"Building churn features (churn threshold: {churn_days} days)...")

    snapshot = df["InvoiceDate"].max()

    # Country one-hot (top 5 + Other)
    top_countries = df.groupby("CustomerID")["Country"].first().reset_index()
    top5 = df["Country"].value_counts().head(5).index.tolist()
    top_countries["Country"] = top_countries["Country"].where(
        top_countries["Country"].isin(top5), other="Other"
    )

    features = rfm[[
        "CustomerID", "Recency", "Frequency", "Monetary",
        "AvgOrderValue", "PurchaseSpan", "TotalItems"
    ]].copy()
    features["DaysSinceFirst"] = (snapshot - rfm["FirstPurchase"]).dt.days
    features["PurchaseRate"] = (features["Frequency"] /
                                (features["DaysSinceFirst"] + 1) * 30).round(4)

    # Churn label
    features["churned"] = (features["Recency"] >= churn_days).astype(int)

    features = features.merge(top_countries, on="CustomerID", how="left")
    features = pd.get_dummies(features, columns=["Country"], drop_first=False)

    churn_rate = features["churned"].mean()
    log.info(f"  Churn features: {len(features):,} customers  |  "
             f"Churn rate: {churn_rate:.1%}")
    return features


# ── 6. Save processed data ────────────────────────────────────────────────────

def save_processed(rfm: pd.DataFrame, ts: pd.DataFrame,
                   churn: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rfm.to_csv(os.path.join(out_dir, "rfm.csv"), index=False)
    ts.to_csv(os.path.join(out_dir, "timeseries.csv"), index=False)
    churn.to_csv(os.path.join(out_dir, "churn_features.csv"), index=False)
    log.info(f"  Saved processed data to {out_dir}/")


# ── 7. Main ───────────────────────────────────────────────────────────────────

def run_pipeline(raw_path: str, processed_dir: str) -> dict:
    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)
    rfm = build_rfm(df_clean)
    ts = build_time_series(df_clean)
    churn = build_churn_features(df_clean, rfm)
    save_processed(rfm, ts, churn, processed_dir)
    log.info("Pipeline complete ✓")
    return {"rfm": rfm, "timeseries": ts, "churn": churn}


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(__file__))
    run_pipeline(
        raw_path=os.path.join(base, "data", "raw", "online_retail_II.csv"),
        processed_dir=os.path.join(base, "data", "processed"),
    )
