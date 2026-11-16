"""End-to-end integration and robustness tests for DL power prediction and FastAPI endpoints."""

import io
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi.testclient import TestClient

from app.services.predict import get_prediction_service
from main import app
CLEAN_CSV = PROJECT_ROOT / "data" / "processed" / "home_power_clean.csv"


class TestPowerPredictionIntegration(unittest.TestCase):
    """Integration test suite verifying both DL prediction service and FastAPI routes."""

    @classmethod
    def setUpClass(cls):
        cls.service = get_prediction_service()
        cls.client = TestClient(app)
        full_df = pd.read_csv(CLEAN_CSV)
        cls.sample_48h_df = full_df.tail(48).copy().reset_index(drop=True)

    def test_01_service_prediction_full_csv(self):
        """Test inference on 48h data with all 11 features."""
        result = self.service.predict_from_csv(self.sample_48h_df)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["input_window"]["hours_used"], 48)
        self.assertEqual(len(result["hourly_predictions"]), 24)
        self.assertGreater(result["forecast_summary"]["daily_total_kwh"], 0)
        self.assertGreaterEqual(
            result["forecast_summary"]["peak_kw"],
            result["forecast_summary"]["min_kw"],
        )

    def test_02_service_prediction_minimal_csv(self):
        """Test user CSV with ONLY datetime and use [kW] (missing sub-meters)."""
        minimal_df = self.sample_48h_df[["datetime", "use [kW]"]].copy()
        result = self.service.predict_from_csv(minimal_df)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["hourly_predictions"]), 24)
        self.assertTrue(all(p["predicted_kw"] >= 0.0 for p in result["hourly_predictions"]))

    def test_03_service_prediction_epoch_time(self):
        """Test CSV with unix epoch timestamp instead of ISO string."""
        epoch_df = self.sample_48h_df[["datetime", "use [kW]"]].copy()
        epoch_df["time"] = pd.to_datetime(epoch_df["datetime"]).astype("int64") // 10**9
        epoch_df = epoch_df.drop(columns=["datetime"])

        result = self.service.predict_from_csv(epoch_df)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["hourly_predictions"]), 24)

    def test_04_service_insufficient_rows_raises_error(self):
        """Test that fewer than 48 rows triggers ValueError."""
        short_df = self.sample_48h_df.head(30)
        with self.assertRaises(ValueError) as ctx:
            self.service.predict_from_csv(short_df)
        self.assertIn("must contain at least 48", str(ctx.exception))

    def test_05_service_missing_target_raises_error(self):
        """Test that missing use [kW] triggers ValueError."""
        no_target_df = self.sample_48h_df.drop(columns=["use [kW]"])
        with self.assertRaises(ValueError) as ctx:
            self.service.predict_from_csv(no_target_df)
        self.assertIn("must contain target column", str(ctx.exception))

    def test_06_fastapi_api_metadata(self):
        """Test discovery /api endpoint."""
        response = self.client.get("/api")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("endpoints", data)
        self.assertEqual(data["endpoints"]["health"], "/api/forecast/health")

    def test_07_fastapi_health_endpoint(self):
        """Test /api/forecast/health endpoint."""
        response = self.client.get("/api/forecast/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["artifacts_ready"])
        self.assertEqual(data["lookback_hours"], 48)
        self.assertEqual(data["forecast_horizon_hours"], 24)

    def test_08_fastapi_predict_csv_upload_success(self):
        """Test uploading a CSV to /api/forecast/predict/csv."""
        csv_bytes = self.sample_48h_df.to_csv(index=False).encode("utf-8")
        files = {"file": ("test_power.csv", io.BytesIO(csv_bytes), "text/csv")}

        response = self.client.post("/api/forecast/predict/csv", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(len(data["hourly_predictions"]), 24)
        self.assertIn("forecast_summary", data)
        self.assertGreater(data["forecast_summary"]["daily_total_kwh"], 0)

    def test_09_fastapi_predict_csv_upload_invalid_filetype(self):
        """Test uploading a non-CSV file returns 400 Bad Request."""
        files = {"file": ("test.txt", io.BytesIO(b"dummy text"), "text/plain")}
        response = self.client.post("/api/forecast/predict/csv", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid file format", response.json()["detail"])

    def test_10_fastapi_predict_csv_upload_too_few_rows(self):
        """Test uploading a CSV with too few rows returns 400 Bad Request."""
        short_csv = self.sample_48h_df.head(20).to_csv(index=False).encode("utf-8")
        files = {"file": ("short.csv", io.BytesIO(short_csv), "text/csv")}
        response = self.client.post("/api/forecast/predict/csv", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("must contain at least 48", response.json()["detail"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
