"""
api/schemas.py
Pydantic request / response schemas for all endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


#  Segmentation 

class SegmentRequest(BaseModel):
    """Request payload for single-customer segmentation inference."""

    recency:   float = Field(..., ge=0,   description="Days since last purchase")
    frequency: float = Field(..., ge=1,   description="Number of unique orders")
    monetary:  float = Field(..., ge=0.0, description="Total spend in £")

    model_config = {
        "json_schema_extra": {
            "example": {"recency": 15, "frequency": 12, "monetary": 450.0}
        }
    }


class RFMScores(BaseModel):
    """Compact 1-5 recency, frequency, and monetary scores for responses."""

    R: int
    F: int
    M: int


class SegmentResponse(BaseModel):
    """Response payload describing the predicted customer segment."""

    cluster_id:   int
    segment_name: str
    rfm_scores:   RFMScores
    description:  str


#  Forecasting 

class ForecastRequest(BaseModel):
    """Request payload for forward revenue forecasting."""

    horizon_days: int = Field(default=30, ge=1, le=90,
                              description="Number of days to forecast (1–90)")

    model_config = {
        "json_schema_extra": {"example": {"horizon_days": 30}}
    }


class ForecastResponse(BaseModel):
    """Response payload containing a multi-day revenue forecast."""

    forecast_from:  str
    horizon_days:   int
    dates:          List[str]
    predicted:      List[float]
    total_forecast: float
    avg_daily:      float


#  Churn 

class ChurnRequest(BaseModel):
    """Request payload for scoring churn risk for one customer profile."""

    recency:          float = Field(..., ge=0,   description="Days since last purchase")
    frequency:        float = Field(..., ge=1,   description="Number of unique orders")
    monetary:         float = Field(..., ge=0.0, description="Total spend in £")
    avg_order_value:  float = Field(default=0.0, ge=0.0, description="Average spend per order")
    purchase_span:    float = Field(default=0.0, ge=0.0, description="Days between first and last purchase")
    total_items:      float = Field(default=0.0, ge=0.0, description="Total quantity of purchased items")
    days_since_first: float = Field(default=0.0, ge=0.0, description="Days since the customer's first purchase")
    purchase_rate:    float = Field(default=0.0, ge=0.0, description="Purchases per month")

    model_config = {
        "json_schema_extra": {
            "example": {
                "recency": 95, "frequency": 3, "monetary": 120.0,
                "avg_order_value": 40.0, "purchase_span": 180.0,
                "total_items": 15.0, "days_since_first": 300.0,
                "purchase_rate": 0.3,
            }
        }
    }


class ChurnResponse(BaseModel):
    """Response payload for churn probability and retention guidance."""

    churn_probability: float
    risk_level:        str
    churned_predicted: bool
    recommendation:    str


#  Health 

class HealthResponse(BaseModel):
    """Health-check response summarizing API and model readiness."""

    status:  str
    version: str
    models:  List[str]
