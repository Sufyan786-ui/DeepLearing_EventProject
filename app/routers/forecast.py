"""FastAPI router for household power forecasting endpoints."""

import io
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.forecast import ForecastResponse, HealthResponse
from app.services.predict import get_prediction_service

router = APIRouter(prefix="/forecast", tags=["Forecasting"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check model and service health status",
)
async def check_health() -> HealthResponse:
    """Return model status, configuration, and artifact readiness."""
    try:
        service = get_prediction_service()
        return HealthResponse(
            status="ok",
            model_name=service.model.name,
            lookback_hours=service.lookback,
            forecast_horizon_hours=service.horizon,
            features_count=len(service.feature_cols),
            artifacts_ready=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifacts not ready: {exc}",
        ) from exc


@router.post(
    "/predict/csv",
    response_model=ForecastResponse,
    summary="Predict next 24-hour power usage from uploaded CSV",
)
async def predict_from_csv_upload(
    file: UploadFile = File(..., description="CSV file containing at least 48 hours of power data"),
) -> ForecastResponse:
    """Upload a CSV with historical power data and receive a 24-hour forecast.

    - **file**: CSV file containing at least 48 hourly readings.
    - Required column: `datetime` (or `time`/`timestamp`) and `use [kW]` (or `use`/`power`).
    - Optional appliance columns: `gen [kW]`, `Dishwasher [kW]`, `Furnace 1 [kW]` (default to 0.0 if omitted).
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid .csv file.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        service = get_prediction_service()
        result = service.predict_from_csv(io.BytesIO(contents))
        return ForecastResponse(**result)

    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Data validation error: {val_err}",
        ) from val_err
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecasting error: {exc}",
        ) from exc
