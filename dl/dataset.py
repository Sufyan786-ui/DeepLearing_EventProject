"""Data loading, chronological splitting, scaling, and sliding-window generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

DATETIME_COL = "datetime"
TARGET_COL = "use [kW]"
FEATURE_COLS = [
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
]


@dataclass
class PreparedSequences:
    """Container for windowed train/validation/test arrays and fitted scalers."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    baseline_test_kw: np.ndarray
    scaler_X: MinMaxScaler
    scaler_y: MinMaxScaler
    split_info: dict


def create_sequences(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert 2D scaled arrays into 3D LSTM lookback windows and multi-step targets."""
    if len(X_scaled) <= lookback + horizon:
        raise ValueError(
            f"Split length ({len(X_scaled)}) must exceed lookback + horizon ({lookback + horizon})."
        )

    X_seq, y_seq = [], []
    max_start = len(X_scaled) - lookback - horizon + 1
    for idx in range(max_start):
        X_seq.append(X_scaled[idx : idx + lookback])
        y_seq.append(y_scaled[idx + lookback : idx + lookback + horizon, 0])

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def _create_persistence_baseline(
    raw_target: np.ndarray,
    lookback: int,
    horizon: int,
) -> np.ndarray:
    """Use the immediately preceding 24 hours as a naive persistence baseline."""
    if lookback < horizon:
        raise ValueError("Lookback must be >= horizon to extract a 24h persistence baseline.")

    baselines = []
    max_start = len(raw_target) - lookback - horizon + 1
    for idx in range(max_start):
        window_end = idx + lookback
        baselines.append(raw_target[window_end - horizon : window_end])
    return np.asarray(baselines, dtype=np.float32)


def load_and_prepare_data(
    csv_path: Path,
    scaler_dir: Path,
    lookback: int = 48,
    horizon: int = 24,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> PreparedSequences:
    """Load hourly power CSV, split chronologically, fit scalers on train, and build windows."""
    df = pd.read_csv(csv_path, parse_dates=[DATETIME_COL])
    df = df.sort_values(DATETIME_COL).reset_index(drop=True)

    missing_cols = [c for c in [DATETIME_COL, *FEATURE_COLS] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in {csv_path}: {missing_cols}")

    n_rows = len(df)
    train_end = int(n_rows * train_ratio)
    val_end = int(n_rows * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    # Fit scalers strictly on training rows to avoid leaking validation/test ranges
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(train_df[FEATURE_COLS])
    X_val_scaled = scaler_X.transform(val_df[FEATURE_COLS])
    X_test_scaled = scaler_X.transform(test_df[FEATURE_COLS])

    y_train_scaled = scaler_y.fit_transform(train_df[[TARGET_COL]])
    y_val_scaled = scaler_y.transform(val_df[[TARGET_COL]])
    y_test_scaled = scaler_y.transform(test_df[[TARGET_COL]])

    scaler_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_X, scaler_dir / "scaler_X.joblib")
    joblib.dump(scaler_y, scaler_dir / "scaler_y.joblib")

    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, lookback, horizon)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, lookback, horizon)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, lookback, horizon)

    baseline_test_kw = _create_persistence_baseline(
        test_df[TARGET_COL].to_numpy(dtype=np.float32), lookback, horizon
    )

    split_info = {
        "total_rows": n_rows,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_range": [str(train_df[DATETIME_COL].iloc[0]), str(train_df[DATETIME_COL].iloc[-1])],
        "val_range": [str(val_df[DATETIME_COL].iloc[0]), str(val_df[DATETIME_COL].iloc[-1])],
        "test_range": [str(test_df[DATETIME_COL].iloc[0]), str(test_df[DATETIME_COL].iloc[-1])],
    }

    return PreparedSequences(
        X_train=X_train_seq,
        y_train=y_train_seq,
        X_val=X_val_seq,
        y_val=y_val_seq,
        X_test=X_test_seq,
        y_test=y_test_seq,
        baseline_test_kw=baseline_test_kw,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        split_info=split_info,
    )
