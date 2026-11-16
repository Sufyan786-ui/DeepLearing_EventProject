"""Training, evaluation, and artifact export script for the home power LSTM."""

import json
import os
from pathlib import Path
import sys

# Reduce TensorFlow info logs before importing TensorFlow via dl.model
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

from dl.dataset import DATETIME_COL, FEATURE_COLS, TARGET_COL, load_and_prepare_data
from dl.model import build_lstm_model

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "home_power_clean.csv"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
PLOTS_DIR = PROJECT_ROOT / "plots"

LOOKBACK = 48
HORIZON = 24
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """Set NumPy and TensorFlow random seeds for reproducible training."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def inverse_scale_targets(scaler_y, arr: np.ndarray) -> np.ndarray:
    """Convert scaled 2D predictions `(samples, horizon)` back to real kW values."""
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)


def compute_metrics(y_true_kw: np.ndarray, y_pred_kw: np.ndarray) -> dict:
    """Compute hourly MAE, hourly RMSE, and next-day total kWh MAE."""
    hourly_mae = float(mean_absolute_error(y_true_kw.ravel(), y_pred_kw.ravel()))
    hourly_rmse = float(np.sqrt(mean_squared_error(y_true_kw.ravel(), y_pred_kw.ravel())))
    daily_total_mae_kwh = float(np.mean(np.abs(y_true_kw.sum(axis=1) - y_pred_kw.sum(axis=1))))
    return {
        "hourly_mae_kw": round(hourly_mae, 4),
        "hourly_rmse_kw": round(hourly_rmse, 4),
        "daily_total_mae_kwh": round(daily_total_mae_kwh, 4),
    }


def save_training_plot(history: tf.keras.callbacks.History, output_path: Path) -> None:
    """Save training vs. validation Huber loss and MAE curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Huber Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train MAE")
    axes[1].plot(history.history["val_mae"], label="Val MAE")
    axes[1].set_title("Scaled MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_plot(
    y_true_kw: np.ndarray,
    y_pred_kw: np.ndarray,
    baseline_kw: np.ndarray,
    output_path: Path,
) -> None:
    """Plot 24-hour actual vs. LSTM predicted vs. persistence baseline for 3 test windows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_indices = np.linspace(0, len(y_true_kw) - 1, num=min(3, len(y_true_kw)), dtype=int)

    fig, axes = plt.subplots(len(sample_indices), 1, figsize=(10, 3.2 * len(sample_indices)), sharex=True)
    if len(sample_indices) == 1:
        axes = [axes]

    hours = np.arange(1, y_true_kw.shape[1] + 1)
    for ax, idx in zip(axes, sample_indices):
        ax.plot(hours, y_true_kw[idx], label="Actual (kW)", linewidth=2)
        ax.plot(hours, y_pred_kw[idx], label="LSTM Predicted (kW)", linewidth=2)
        ax.plot(hours, baseline_kw[idx], label="Yesterday Baseline (kW)", linestyle="--", alpha=0.75)
        ax.set_title(f"Test Window #{idx} (24-Hour Forecast)")
        ax.set_ylabel("Power (kW)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Forecast Hour Ahead")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_and_evaluate() -> dict:
    """Run the full data prep, LSTM training, evaluation, and artifact export pipeline."""
    set_seeds()
    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare_data(
        csv_path=DATA_PATH,
        scaler_dir=SAVED_MODELS_DIR,
        lookback=LOOKBACK,
        horizon=HORIZON,
    )
    print("Split summary:", data.split_info)
    print(
        f"Sequences -> Train: {data.X_train.shape}, Val: {data.X_val.shape}, Test: {data.X_test.shape}"
    )

    model = build_lstm_model(
        lookback=LOOKBACK,
        num_features=len(FEATURE_COLS),
        horizon=HORIZON,
        learning_rate=LEARNING_RATE,
    )
    model.summary()

    model_path = SAVED_MODELS_DIR / "lstm_power_model.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", save_best_only=True
        ),
    ]

    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on unseen test windows in original kW units
    best_model = tf.keras.models.load_model(model_path)
    y_pred_scaled = best_model.predict(data.X_test, verbose=0)
    y_pred_kw = np.clip(inverse_scale_targets(data.scaler_y, y_pred_scaled), a_min=0.0, a_max=None)
    y_true_kw = inverse_scale_targets(data.scaler_y, data.y_test)

    lstm_metrics = compute_metrics(y_true_kw, y_pred_kw)
    baseline_metrics = compute_metrics(y_true_kw, data.baseline_test_kw)

    print("\nTest Metrics (LSTM):", lstm_metrics)
    print("Test Metrics (Yesterday Baseline):", baseline_metrics)

    save_training_plot(history, PLOTS_DIR / "training_loss.png")
    save_prediction_plot(y_true_kw, y_pred_kw, data.baseline_test_kw, PLOTS_DIR / "actual_vs_predicted_24h.png")

    config = {
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "datetime_col": DATETIME_COL,
        "target_col": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "epochs_trained": len(history.history["loss"]),
        "split_info": data.split_info,
        "test_metrics": lstm_metrics,
        "baseline_metrics": baseline_metrics,
    }
    config_path = SAVED_MODELS_DIR / "model_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"\nSaved model to {model_path.relative_to(PROJECT_ROOT)} "
        f"and config to {config_path.relative_to(PROJECT_ROOT)}"
    )
    return config


if __name__ == "__main__":
    train_and_evaluate()
