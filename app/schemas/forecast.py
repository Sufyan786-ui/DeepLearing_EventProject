"""Pydantic schemas for power forecasting requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class HourlyForecastItem(BaseModel):
    """Prediction for a single future hour."""
    hour_ahead: int = Field(..., ge=1, le=24, description="Hours ahead (1-24)")
    timestamp: str = Field(..., description="Target timestamp (YYYY-MM-DD HH:MM:SS)")
    predicted_kw: float = Field(..., ge=0.0, description="Predicted power consumption in kW")


class ForecastSummary(BaseModel):
    """Aggregate statistics for the 24-hour forecasted period."""
    forecast_start: str = Field(..., description="Start timestamp of forecast")
    forecast_end: str = Field(..., description="End timestamp of forecast")
    daily_total_kwh: float = Field(..., ge=0.0, description="Estimated total energy in kWh (sum of 24h kW)")
    average_hourly_kw: float = Field(..., ge=0.0, description="Average hourly power demand in kW")
    peak_kw: float = Field(..., ge=0.0, description="Peak hourly power demand in kW")
    peak_timestamp: str = Field(..., description="Timestamp of highest consumption")
    min_kw: float = Field(..., ge=0.0, description="Lowest hourly power demand in kW")


class InputWindowInfo(BaseModel):
    """Information regarding the 48-hour historical window used for the prediction."""
    start: str = Field(..., description="Start timestamp of input history")
    end: str = Field(..., description="End timestamp of input history")
    hours_used: int = Field(..., description="Number of past hours fed to the LSTM")
    avg_input_kw: float = Field(..., description="Average power consumption over input window")


class ForecastResponse(BaseModel):
    """Full API response for a 24-hour forecast."""
    status: str = Field("success", description="Status of the prediction request")
    model_name: str = Field(..., description="Underlying Keras LSTM model name")
    input_window: InputWindowInfo
    forecast_summary: ForecastSummary
    hourly_predictions: List[HourlyForecastItem]


class HealthResponse(BaseModel):
    """Service health and model readiness status."""
    status: str = Field("ok", description="Service status")
    model_name: str
    lookback_hours: int
    forecast_horizon_hours: int
    features_count: int
    artifacts_ready: bool
