# ⚡ Household Power Forecasting with Deep Learning (LSTM)

An end-to-end deep learning system and REST API for next-day (24-hour ahead) hourly household electricity consumption forecasting, built with **TensorFlow / Keras** and **FastAPI**.

---

## 📌 Overview

* **Problem**: Predict next-day hourly electricity usage ($\text{kW}$) given the previous 48 hours of smart-meter consumption and appliance data.
* **Model**: Stacked 2-Layer LSTM ($64 \to 32$ units) with Huber loss for outlier-resistant training and dense projection for multi-step 24-hour direct forecasting.
* **Feature Engineering**: Automated cyclical time encodings ($\sin/\cos$ for hour, day of week, and month) + calendar flags (`is_weekend`).
* **API Service**: FastAPI web application with interactive Swagger documentation and CSV batch inference endpoints.

---

## 📁 Project Structure

```text
├── app/
│   ├── routers/       # FastAPI route controllers (/forecast/predict/csv, /health)
│   ├── schemas/       # Pydantic request & response schemas
│   └── services/      # Inference service (predict.py) & feature engineering
├── dl/
│   ├── dataset.py     # Chronological splitting, scaling, sliding-window generation
│   ├── model.py       # 2-layer LSTM architecture definition
│   └── train.py       # Training loop, Huber loss, callbacks & evaluation
├── saved_models/      # Trained LSTM weights (.keras), scalers (.joblib), model config
├── plots/             # Loss curves and 24h actual vs. predicted evaluation plots
├── tests/             # Integration and API test suite
├── main.py            # FastAPI entry point
└── requirements.txt   # Project dependencies
```

---

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies in a virtual environment:
```bash
git clone https://github.com/Sufyan786-ui/DeepLearing_EventProject.git
cd DeepLearing_EventProject
python -m venv lstm_env
# Windows:
lstm_env\Scripts\activate
# Linux/macOS:
source lstm_env/bin/activate

pip install -r requirements.txt
```

### 2. Start the API Server
```bash
python main.py
```
Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to view the interactive Swagger documentation.

### 3. Run Inference (CSV Upload)
You can upload any CSV containing at least 48 hours of historical readings:
```bash
curl -X POST "http://127.0.0.1:8000/api/forecast/predict/csv" \
     -F "file=@data/test_samples/sample_test_minimal.csv"
```

### 4. (Optional) Re-train the Model
```bash
python dl/train.py
```

### 5. Run Integration Tests
```bash
python -m unittest tests/test_integration.py
```

---

## 📊 Model Performance

| Metric | LSTM Model | Persistence Baseline (Yesterday) |
| :--- | :--- | :--- |
| **Hourly MAE** | **0.3598 kW** | 0.3736 kW |
| **Hourly RMSE** | **0.4740 kW** | 0.5764 kW |
| **Daily Total MAE** | **4.3938 kWh** | 4.8920 kWh |

---

## 📄 License
MIT License
