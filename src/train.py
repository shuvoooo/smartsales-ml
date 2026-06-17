"""Train the SmartSales machine learning models and register results.

This module orchestrates the full offline training workflow for the project:
customer segmentation, revenue forecasting, and churn prediction. It reads the
processed feature tables produced by ``src/data_pipeline.py``, fits one model
per use case, persists the trained artifacts under ``models/``, writes scored
datasets back to ``data/processed/``, and logs comparable metrics to MLflow.

The pipeline depends on scikit-learn for clustering and evaluation utilities,
XGBoost for time-series regression, LightGBM for churn classification, and
MLflow for experiment tracking.

Run:
    python src/train.py
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, \
    mean_squared_error, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Resolve paths from the repository root so the script behaves the same whether
# it is launched from the repo root, src/, or another working directory.
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
# Allow an external MLflow server when provided, but default to a local SQLite
# store so training remains self-contained for local development.
MLFLOW_URI  = os.environ.get("MLFLOW_TRACKING_URI",
                              f"sqlite:///{BASE_DIR}/mlruns.db")

# These names define the business-facing vocabulary for segments. The actual
# cluster IDs produced by K-Means are arbitrary, so training remaps them to
# labels after ranking clusters by customer value.
SEGMENT_LABELS = {
    0: "Champions",
    1: "Loyal Customers",
    2: "At Risk",
    3: "Lost / Inactive",
}

#  1. Customer Segmentation (K-Means on RFM) 

def train_segmentation(rfm: pd.DataFrame) -> dict:
    """Train the customer segmentation model on RFM features.

    Args:
        rfm: Customer-level RFM table containing at least ``Recency``,
            ``Frequency``, and ``Monetary`` columns.

    Returns:
        dict: Training artifacts, evaluation scores, the scored RFM table, and
        per-segment summary statistics.

    Notes:
        StandardScaler is applied because K-Means relies on Euclidean distance;
        without scaling, the largest-magnitude feature would dominate cluster
        assignment. The function evaluates ``k`` from 2 through 6 and selects
        the best silhouette score to balance cohesion within clusters against
        separation between clusters without hard-coding the number of segments.
    """
    log.info("" * 50)
    log.info("Training: Customer Segmentation (K-Means)")

    features = rfm[["Recency", "Frequency", "Monetary"]].copy()

    # RFM columns live on very different numeric scales, so standardization
    # keeps distance-based clustering from over-weighting Monetary.
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # Search a small, business-friendly range of cluster counts and use the
    # silhouette score to pick the option with the cleanest separation.
    best_k, best_score, best_model = 4, -1, None
    scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        s = silhouette_score(X, labels)
        scores[k] = round(s, 4)
        if s > best_score:
            best_k, best_score, best_model = k, s, km

    log.info(f"  Best k={best_k}  silhouette={best_score:.4f}")

    rfm = rfm.copy()
    rfm["cluster"] = best_model.predict(X)

    # K-Means cluster IDs are not ordered, so rank clusters by median spend to
    # attach stable business labels from highest- to lowest-value customers.
    cluster_monetary = rfm.groupby("cluster")["Monetary"].median().sort_values(ascending=False)
    label_map = {}
    label_names = ["Champions", "Loyal Customers", "At Risk", "Lost / Inactive"]
    for i, (cluster_id, _) in enumerate(cluster_monetary.items()):
        label_map[int(cluster_id)] = label_names[min(i, len(label_names)-1)]

    rfm["segment"] = rfm["cluster"].map(label_map)

    # Cluster summary stats
    summary = rfm.groupby("segment")[["Recency","Frequency","Monetary"]].median().round(2)

    # Save
    out_dir = os.path.join(MODELS_DIR, "segmentation")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(out_dir, "kmeans.pkl"))
    joblib.dump(scaler,     os.path.join(out_dir, "scaler.pkl"))
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    # Save scored customer table
    rfm.to_csv(os.path.join(DATA_DIR, "rfm_segmented.csv"), index=False)

    return {
        "model":       best_model,
        "scaler":      scaler,
        "label_map":   label_map,
        "silhouette":  best_score,
        "k":           best_k,
        "k_scores":    scores,
        "rfm_scored":  rfm,
        "summary":     summary,
    }


#  2. Sales Forecasting (XGBoost Regression) 

def train_forecasting(ts: pd.DataFrame) -> dict:
    """Train the daily sales forecasting model.

    Args:
        ts: Time-series feature table containing target column ``y`` and the
            lag/calendar features used for supervised forecasting.

    Returns:
        dict: The fitted XGBoost model, regression metrics, held-out test data,
        predictions, model parameters, and feature metadata.

    Notes:
        The train/test split is chronological rather than random so evaluation
        mirrors real forecasting, where the model only sees past data when
        predicting the future. MAPE uses ``y_test + 1`` in the denominator to
        avoid division-by-zero explosions on zero-revenue days while still
        penalizing relative error on small values.
    """
    log.info("" * 50)
    log.info("Training: Sales Forecasting (XGBoost)")

    FEATURE_COLS = ["lag_7", "lag_14", "lag_30", "lag_60",
                    "rolling_7", "rolling_30",
                    "dow", "month", "week", "is_weekend", "is_december"]
    TARGET = "y"

    df = ts.dropna(subset=FEATURE_COLS).copy()
    X  = df[FEATURE_COLS]
    y  = df[TARGET]

    # Hold out the most recent period so metrics reflect forward-looking
    # generalization instead of a random mix of past and future observations.
    split_idx = len(df) - 30
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    params = {
        "n_estimators":      300,
        "max_depth":         5,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "random_state":      42,
        "early_stopping_rounds": 20,
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    # Adding 1 keeps zero-sales days from making percentage error undefined.
    mape  = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100

    log.info(f"  MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.2f}%")

    # Save
    out_dir = os.path.join(MODELS_DIR, "forecasting")
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, "xgb_model.json"))
    with open(os.path.join(out_dir, "feature_info.json"), "w") as f:
        json.dump({
            "feature_cols":  FEATURE_COLS,
            "last_date":     str(ts["ds"].max().date()),
            # Recursive inference needs recent actual/predicted values to
            # rebuild lag and rolling-window features day by day.
            "last_values":   ts.tail(60)["y"].tolist(),
        }, f)

    return {
        "model":  model,
        "mae":    mae,
        "rmse":   rmse,
        "mape":   mape,
        "params": params,
        "X_test": X_test,
        "y_test": y_test,
        "preds":  preds,
        "feature_cols": FEATURE_COLS,
    }


#  3. Churn Prediction (LightGBM Classifier) 

def train_churn(churn_df: pd.DataFrame) -> dict:
    """Train the churn classifier and score the full customer base.

    Args:
        churn_df: Customer-level churn feature table containing the binary
            ``churned`` target and engineered predictors.

    Returns:
        dict: The fitted LightGBM model, AUC and classification metrics,
        feature names used for training, and the scored churn table.

    Notes:
        ``select_dtypes(include=[np.number])`` ensures the training matrix only
        contains model-ready numeric columns after one-hot encoding. The
        ``scale_pos_weight`` parameter compensates for churn/non-churn class
        imbalance so the model pays more attention to the rarer positive class.
        A probability threshold of 0.5 is used for the default hard label
        because it is the standard neutral cutoff when no business-specific
        precision/recall trade-off has been configured.
    """
    log.info("" * 50)
    log.info("Training: Churn Prediction (LightGBM)")

    drop_cols = ["CustomerID", "churned"]
    feature_cols = [c for c in churn_df.columns if c not in drop_cols]

    # Keep only numeric features so the persisted model sees the same encoded
    # matrix shape that the inference layer will later reconstruct.
    X = churn_df[feature_cols].select_dtypes(include=[np.number])
    y = churn_df["churned"]

    feature_cols = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    params = {
        "n_estimators":    400,
        "max_depth":       6,
        "learning_rate":   0.05,
        "num_leaves":      31,
        "subsample":       0.8,
        "colsample_bytree":0.8,
        # Weight the minority churn class more heavily to reduce the tendency
        # of gradient boosting to optimize for the majority class only.
        "scale_pos_weight":(y_train == 0).sum() / (y_train == 1).sum(),
        "random_state":    42,
        "verbose":        -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])

    probs  = model.predict_proba(X_test)[:, 1]
    # Use a neutral 0.5 cutoff for binary labels while preserving probabilities
    # for ranking and downstream campaign decisions.
    preds  = (probs >= 0.5).astype(int)
    auc    = roc_auc_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)

    log.info(f"  AUC={auc:.4f}  F1={report['1']['f1-score']:.4f}  "
             f"Precision={report['1']['precision']:.4f}  "
             f"Recall={report['1']['recall']:.4f}")

    # Score all customers
    all_probs = model.predict_proba(X)[:, 1]
    churn_df  = churn_df.copy()
    churn_df["churn_probability"] = all_probs.round(4)
    churn_df["risk_level"] = pd.cut(
        all_probs,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"]
    )
    churn_df.to_csv(os.path.join(DATA_DIR, "churn_scored.csv"), index=False)

    # Save
    out_dir = os.path.join(MODELS_DIR, "churn")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "lgbm_model.pkl"))
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump({
            "feature_cols": feature_cols,
            "threshold":    0.5,
            "auc":          round(auc, 4),
        }, f)

    return {
        "model":        model,
        "auc":          auc,
        "report":       report,
        "feature_cols": feature_cols,
        "churn_scored": churn_df,
    }


#  4. MLflow run 

def log_to_mlflow(seg_results: dict, fc_results: dict, churn_results: dict):
    """Log model artifacts and headline metrics to MLflow.

    Args:
        seg_results: Output from :func:`train_segmentation`.
        fc_results: Output from :func:`train_forecasting`.
        churn_results: Output from :func:`train_churn`.

    Returns:
        None

    Notes:
        Segmentation logs the selected ``k``, silhouette metrics, K-Means
        model, and scaler. Forecasting logs XGBoost hyperparameters plus MAE,
        RMSE, and MAPE. Churn logs AUC, positive-class precision/recall/F1, and
        the trained LightGBM model.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("SmartSales-ML")

    #  Segmentation 
    with mlflow.start_run(run_name="customer-segmentation"):
        mlflow.log_param("k",                  seg_results["k"])
        mlflow.log_metric("silhouette_score",  seg_results["silhouette"])
        for k, s in seg_results["k_scores"].items():
            mlflow.log_metric(f"silhouette_k{k}", s)
        mlflow.sklearn.log_model(seg_results["model"],  "kmeans_model")
        mlflow.sklearn.log_model(seg_results["scaler"], "scaler")
        log.info("  MLflow: segmentation run logged")

    #  Forecasting 
    with mlflow.start_run(run_name="sales-forecasting"):
        mlflow.log_params(fc_results["params"])
        mlflow.log_metric("MAE",  fc_results["mae"])
        mlflow.log_metric("RMSE", fc_results["rmse"])
        mlflow.log_metric("MAPE", fc_results["mape"])
        mlflow.xgboost.log_model(fc_results["model"], "xgb_model")
        log.info("  MLflow: forecasting run logged")

    #  Churn 
    with mlflow.start_run(run_name="churn-prediction"):
        mlflow.log_params({k: v for k, v in churn_results["report"].items()
                           if isinstance(v, (int, float))})
        mlflow.log_metric("AUC",       churn_results["auc"])
        mlflow.log_metric("F1_churn",  churn_results["report"]["1"]["f1-score"])
        mlflow.log_metric("Precision", churn_results["report"]["1"]["precision"])
        mlflow.log_metric("Recall",    churn_results["report"]["1"]["recall"])
        mlflow.lightgbm.log_model(churn_results["model"], "lgbm_model")
        log.info("  MLflow: churn run logged")


#  5. Main 

def main():
    """Run the end-to-end training pipeline from processed feature tables.

    Raises:
        FileNotFoundError: If any required processed dataset is missing because
            the data pipeline has not been run yet.
    """
    log.info("=" * 50)
    log.info("SmartSales ML — Training Pipeline")
    log.info("=" * 50)

    rfm_path   = os.path.join(DATA_DIR, "rfm.csv")
    ts_path    = os.path.join(DATA_DIR, "timeseries.csv")
    churn_path = os.path.join(DATA_DIR, "churn_features.csv")

    for p in [rfm_path, ts_path, churn_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Run data pipeline first: {p}")

    rfm      = pd.read_csv(rfm_path,   parse_dates=["FirstPurchase", "LastPurchase"])
    ts       = pd.read_csv(ts_path,    parse_dates=["ds"])
    churn_df = pd.read_csv(churn_path)

    seg_results   = train_segmentation(rfm)
    fc_results    = train_forecasting(ts)
    churn_results = train_churn(churn_df)

    log.info("" * 50)
    log.info("Logging to MLflow...")
    log_to_mlflow(seg_results, fc_results, churn_results)

    log.info("=" * 50)
    log.info("Training complete ")
    log.info(f"  Segmentation silhouette : {seg_results['silhouette']:.4f}")
    log.info(f"  Forecast MAPE           : {fc_results['mape']:.2f}%")
    log.info(f"  Churn AUC               : {churn_results['auc']:.4f}")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
