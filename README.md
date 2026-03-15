# Stock Market Trend Prediction System

An end-to-end machine learning pipeline leveraging Advanced Deep Learning architectures (Standard LSTMs, Bidirectional LSTMs, and Custom Attention-enhanced LSTMs) to predict stock market trends and prices.

## 🚀 Features
- **Automated Data Pipeline**: Fetch historical data directly from Yahoo Finance.
- **Advanced Feature Engineering**: Employs `pandas-ta` for MACD, RSI, Bollinger Bands, EMAs, ATR, and OBV.
- **Deep Learning Architectures**: Built using TensorFlow/Keras with modular design.
- **MLflow Tracking**: Hyperparameters and metrics automatically logged.
- **VectorBT Backtesting**: Simulates trading strategies based on predictions.
- **Interactive Dashboard**: Built on Streamlit to visualize predictions interactively.

## 📁 Project Structure
```
stock_lstm/
├── data/               (Raw and processed time-series data)
├── dashboard/          (Streamlit interactive app)
├── models/             (Stored .keras weights and scalers)
├── notebooks/          (Jupyter experimentation env)
├── reports/            (Figures and metrics JSONs)
├── mlruns/             (Local MLflow tracking server)
├── src/                (Core pipeline modules)
│   ├── data_loader.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── tests/              (PyTest unit tests)
├── config.yaml         (Central configuration parameters)
├── Dockerfile          (Production Docker image)
├── docker-compose.yml  (Run Dashboard & MLflow)
├── main.py             (Orchestrator script)
└── requirements.txt    (Pinned dependencies)
```

## ⚙️ Installation & Usage

1. **Clone & Setup:**
```bash
git clone <repo_url>
cd stock_lstm
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Run Pipeline End-to-End:**
Easily trigger ingestion, feature engineering, training, and evaluation in one command:
```bash
python main.py --mode full
```

3. **Start Interactive Dashboard:**
```bash
streamlit run dashboard/app.py
```

4. **Launch MLflow UI:**
```bash
mlflow ui
```

## 🐳 Docker Deployment
To spin up both the Streamlit dashboard and the MLflow UI using Docker Compose:
```bash
docker-compose up --build
```
- Dashboard will be available at: `http://localhost:8501`
- MLflow UI will be available at: `http://localhost:5000`

## 🧪 Testing
We use `pytest` for all unit tests. Run them via:
```bash
pytest tests/
```
