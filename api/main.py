"""
api/main.py
SmartSales ML — FastAPI inference service.

Endpoints:
  POST /segment       — Customer segmentation (K-Means / RFM)
  POST /forecast      — Sales forecasting (XGBoost)
  POST /churn         — Churn prediction (LightGBM)
  GET  /segments/all  — All customers with segments (for dashboard)
  GET  /churn/all     — All customers with churn scores (for dashboard)
  GET  /sales/history — Historical daily revenue (for dashboard)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    SegmentRequest, SegmentResponse,
    ForecastRequest, ForecastResponse,
    ChurnRequest,   ChurnResponse,
    HealthResponse,
)
from src.predict import (
    predict_segment, predict_forecast, predict_churn,
    get_all_segments, get_all_churn_scores, get_historical_sales,
)

log = logging.getLogger(__name__)

#  App 

app = FastAPI(
    title       = "SmartSales ML API",
    description = "E-commerce intelligence: customer segmentation, sales forecasting, and churn prediction.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


#  Segmentation 

@app.post("/segment", response_model=SegmentResponse, tags=["Segmentation"],
          summary="Predict customer segment from RFM values")
def segment_customer(req: SegmentRequest):
    """
    Given Recency (days), Frequency (orders), and Monetary (£ spend),
    returns the customer segment and RFM quintile scores.
    """
    try:
        result = predict_segment(req.recency, req.frequency, req.monetary)
        return SegmentResponse(**result)
    except Exception as e:
        log.exception("Segmentation error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segments/all", tags=["Segmentation"],
         summary="All customers with segment labels (for dashboard)")
def all_segments():
    """Returns all customers with their RFM values and segment labels."""
    try:
        return get_all_segments()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Forecasting 

@app.post("/forecast", response_model=ForecastResponse, tags=["Forecasting"],
          summary="Forecast daily revenue for the next N days")
def forecast_sales(req: ForecastRequest):
    """
    Recursively predicts daily revenue for the next `horizon_days` days
    using lag features from the training data tail.
    """
    try:
        result = predict_forecast(req.horizon_days)
        return ForecastResponse(**result)
    except Exception as e:
        log.exception("Forecast error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sales/history", tags=["Forecasting"],
         summary="Historical daily revenue (for dashboard chart)")
def sales_history():
    """Returns historical daily revenue for the full training period."""
    try:
        return get_historical_sales()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Churn 

@app.post("/churn", response_model=ChurnResponse, tags=["Churn"],
          summary="Predict churn probability for a customer")
def churn_predict(req: ChurnRequest):
    """
    Predicts the probability that a customer will churn (no purchase in 90 days).
    Returns risk level (Low / Medium / High) and a recommended action.
    """
    try:
        result = predict_churn(
            recency          = req.recency,
            frequency        = req.frequency,
            monetary         = req.monetary,
            avg_order_value  = req.avg_order_value,
            purchase_span    = req.purchase_span,
            total_items      = req.total_items,
            days_since_first = req.days_since_first,
            purchase_rate    = req.purchase_rate,
        )
        return ChurnResponse(**result)
    except Exception as e:
        log.exception("Churn prediction error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/churn/all", tags=["Churn"],
         summary="All customers ranked by churn probability (for dashboard)")
def all_churn():
    """Returns all customers sorted by churn probability (highest risk first)."""
    try:
        return get_all_churn_scores()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#  Dev runner 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
