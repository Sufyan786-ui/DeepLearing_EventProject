"""Prediction service for household power forecasting using the trained LSTM model.

Performs automated feature engineering on user-provided CSV data,
scales the input, runs LSTM inference, and formats the 24-hour forecast.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

# Default paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "saved_models" / "lstm_power_model.keras"
DEFAULT_SCALER_X_PATH = PROJECT_ROOT / "saved_models" / "scaler_X.joblib"
DEFAULT_SCALER_Y_PATH = PROJECT_ROOT / "saved_models" / "scaler_y.joblib"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "saved_models" / "model_config.json"


class PowerPredictionService:
    """Loads artifacts and handles end-to-end forecasting from user CSV data."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        scaler_x_path: Path = DEFAULT_SCALER_X_PATH,
        scaler_y_path: Path = DEFAULT_SCALER_Y_PATH,
        config_path: Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_x_path = Path(scaler_x_path)
        self.scaler_y_path = Path(scaler_y_path)
        self.config_path = Path(config_path)

        self._validate_artifacts_exist()

        # Load configuration
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.lookback: int = self.config.get("lookback", 48)
        self.horizon: int = self.config.get("horizon", 24)
        self.target_col: str = self.config.get("target_col", "use [kW]")
        self.feature_cols: List[str] = self.config.get(
            "feature_cols",
            [
                "use [kW]",
                "gen [kW]",
                "Dishwasher [kW]",
                "Furnace 1 [kW]",
                "hour_sin",
                "hour_cos",
                "dow_sin",
                "dow_cos",
                "month_sin",
                "month_cos",
                "is_weekend",
            ],
        )

        # Load model and scalers
        self.model = keras.models.load_model(self.model_path)
        self.scaler_x = joblib.load(self.scaler_x_path)
        self.scaler_y = joblib.load(self.scaler_y_path)

    def _validate_artifacts_exist(self) -> None:
        missing = [p for p in (self.model_path, self.scaler_x_path, self.scaler_y_path, self.config_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required model artifact(s): {[str(p) for p in missing]}. "
                "Ensure training has completed and saved_models/ contains the artifacts."
            )

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize datetime and compute cyclical and calendar time features.
        
        Accepts raw or processed CSV data, detects datetime/time column,
        fills missing optional sub-meters with zeros, and computes all 7 time features.
        """
        df = df.copy()

        # 1. Resolve datetime column
        dt_col = None
        for candidate in ["datetime", "time", "timestamp", "Date", "Datetime"]:
            if candidate in df.columns:
                dt_col = candidate
                break

        if dt_col is None:
            raise ValueError(
                "Uploaded CSV must contain a datetime column ('datetime', 'time', or 'timestamp')."
            )

        # Parse unix epoch (seconds) or string dates
        if pd.api.types.is_numeric_dtype(df[dt_col]):
            df["datetime"] = pd.to_datetime(df[dt_col], unit="s")
        else:
            df["datetime"] = pd.to_datetime(df[dt_col])

        # 2. Sort chronologically and reset index
        df = df.sort_values("datetime").reset_index(drop=True)

        # 3. Check target / power column
        if self.target_col not in df.columns:
            # Check common aliases
            aliases = ["use", "power", "total_power", "consumption", "power_usage"]
            found_alias = next((a for a in aliases if a in df.columns), None)
            if found_alias:
                df[self.target_col] = df[found_alias]
            else:
                raise ValueError(
                    f"Uploaded CSV must contain target column '{self.target_col}' (or 'power', 'use')."
                )

        # 4. Handle optional sub-meter appliance columns (default to 0.0 if not provided)
        optional_submeters = ["gen [kW]", "Dishwasher [kW]", "Furnace 1 [kW]"]
        for col in optional_submeters:
            if col not in df.columns:
                df[col] = 0.0

        # 5. Feature Engineering: Cyclical and calendar time features
        hour = df["datetime"].dt.hour
        dow = (
            df["datetime"].dt.day_of_week
            if hasattr(df["datetime"].dt, "day_of_week")
            else df["datetime"].dt.dayofweek
        )
        month = df["datetime"].dt.month

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)
        df["is_weekend"] = (dow >= 5).astype(int)

        return df

    def predict_from_csv(
        self,
        file_input: Union[str, Path, io.BytesIO, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Read CSV, engineer features, extract last 48 hours, and predict next 24 hours.

        Parameters
        ----------
        file_input : Union[str, Path, io.BytesIO, pd.DataFrame]
            Path to CSV file, bytes/file-like object, or existing DataFrame.

        Returns
        -------
        Dict[str, Any]
            Detailed forecast response containing 24h predictions and summary metrics.
        """
        # 1. Ingest input
        if isinstance(file_input, pd.DataFrame):
            df_raw = file_input.copy()
        elif isinstance(file_input, (str, Path)):
            df_raw = pd.read_csv(file_input)
        elif isinstance(file_input, io.BytesIO):
            file_input.seek(0)
            df_raw = pd.read_csv(file_input)
        else:
            df_raw = pd.read_csv(file_input)

        if len(df_raw) < self.lookback:
            raise ValueError(
                f"CSV must contain at least {self.lookback} hourly rows. "
                f"Received {len(df_raw)} rows."
            )

        # 2. Engineer features
        df_engineered = self.engineer_features(df_raw)

        # 3. Take the most recent lookback rows (e.g. 48 hours)
        recent_window = df_engineered.iloc[-self.lookback :].reset_index(drop=True)
        input_start = str(recent_window["datetime"].iloc[0])
        input_end = str(recent_window["datetime"].iloc[-1])

        # 4. Scale features
        X_scaled = self.scaler_x.transform(recent_window[self.feature_cols])  # shape: (48, 11)

        # 5. Reshape to 3D tensor: (1, lookback, num_features)
        X_tensor = np.expand_dims(X_scaled, axis=0)  # shape: (1, 48, 11)

        # 6. Model Inference
        pred_scaled = self.model.predict(X_tensor, verbose=0)  # shape: (1, 24)

        # 7. Inverse transform back to real kW
        pred_kw = self.scaler_y.inverse_transform(pred_scaled)[0]  # shape: (24,)
        # Consumption cannot be negative
        pred_kw = np.maximum(pred_kw, 0.0)

        # 8. Construct future timestamps
        last_dt = recent_window["datetime"].iloc[-1]
        future_timestamps = [
            (last_dt + pd.Timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(1, self.horizon + 1)
        ]

        # 9. Format response
        hourly_forecast = [
            {
                "hour_ahead": i + 1,
                "timestamp": future_timestamps[i],
                "predicted_kw": round(float(pred_kw[i]), 4),
            }
            for i in range(self.horizon)
        ]

        peak_idx = int(np.argmax(pred_kw))
        daily_kwh = float(np.sum(pred_kw))  # 1 hour intervals -> sum(kW) = kWh

        return {
            "status": "success",
            "model_name": self.model.name,
            "input_window": {
                "start": input_start,
                "end": input_end,
                "hours_used": self.lookback,
                "avg_input_kw": round(float(recent_window[self.target_col].mean()), 4),
            },
            "forecast_summary": {
                "forecast_start": future_timestamps[0],
                "forecast_end": future_timestamps[-1],
                "daily_total_kwh": round(daily_kwh, 3),
                "average_hourly_kw": round(float(np.mean(pred_kw)), 4),
                "peak_kw": round(float(pred_kw[peak_idx]), 4),
                "peak_timestamp": future_timestamps[peak_idx],
                "min_kw": round(float(np.min(pred_kw)), 4),
            },
            "hourly_predictions": hourly_forecast,
        }


# Global singleton instance for cached service loading in FastAPI
_service_instance: PowerPredictionService | None = None


def get_prediction_service() -> PowerPredictionService:
    """Return a cached singleton instance of PowerPredictionService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = PowerPredictionService()
    return _service_instance


if __name__ == "__main__":
    # Self-test using the cleaned dataset
    clean_csv = PROJECT_ROOT / "data" / "processed" / "home_power_clean.csv"
    print(f"Testing prediction service with {clean_csv}...")
    service = get_prediction_service()
    result = service.predict_from_csv(clean_csv)
    print("\n--- Prediction Service Output ---")
    print("Input Window:", result["input_window"])
    print("Forecast Summary:", result["forecast_summary"])
    print("First 3 Hourly Predictions:")
    for p in result["hourly_predictions"][:3]:
        print(f"  {p['timestamp']} (+{p['hour_ahead']}h): {p['predicted_kw']} kW")
