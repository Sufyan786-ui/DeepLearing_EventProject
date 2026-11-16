"""LSTM model definition for multi-step hourly household power forecasting."""

from tensorflow import keras
from tensorflow.keras import layers


def build_lstm_model(
    lookback: int,
    num_features: int,
    horizon: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile a 2-layer LSTM predicting the next `horizon` hours of power usage."""
    model = keras.Sequential(
        [
            keras.Input(shape=(lookback, num_features), name="power_history"),
            layers.LSTM(64, return_sequences=True, name="lstm_1"),
            layers.Dropout(0.2, name="dropout_1"),
            layers.LSTM(32, name="lstm_2"),
            layers.Dropout(0.2, name="dropout_2"),
            layers.Dense(32, activation="relu", name="dense_hidden"),
            layers.Dense(horizon, name="forecast_24h"),
        ],
        name="home_power_lstm",
    )

    # Huber loss dampens occasional high-wattage appliance spikes better than pure MSE
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),
        metrics=["mae"],
    )
    return model
