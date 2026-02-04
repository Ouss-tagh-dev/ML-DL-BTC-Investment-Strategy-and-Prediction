# Bitcoin ML/DL Backend (FastAPI)

A high-performance algorithmic trading backend powered by an ensemble of 9 Machine Learning and Deep Learning models.

## 🚀 Features

- **Multi-Model Ensemble**: Orchestrates 9 models including LSTM, GRU, LSTM-CNN, XGBoost, SVM, and Random Forest.
- **Zero-Lag Inference**: Models are pre-loaded on startup for instantaneous real-time predictions.
- **AI Consensus Engine**: Aggregates model votes to generate Buy/Sell/Neutral signals.
- **FastAPI Architecture**: Asynchronous, high-throughput REST API with automatic documentation.
- **Live Data Pipeline**: Fetches real-time OHLCV data from Binance for inference.

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python)
- **ML Libraries**: TensorFlow/Keras, Scikit-learn, XGBoost
- **Data Handling**: Pandas, NumPy, CCXT (Crypto Exchange)
- **Server**: Uvicorn

## 📦 Installation

1.  **Clone the repository** (if not already done).
2.  **Navigate to the backend directory**:
    ```bash
    cd dl-ml-btc
    ```
3.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## ⚡ Usage

### Start the Server

```bash
python server/main.py
```
*The server will start on `http://localhost:8000`. You will see logs indicating that all 9 models are being loaded.*

### API Documentation

Once the server is running, access the interactive documentation:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Key Endpoints

- `GET /health`: System health and model loading status.
- `POST /api/models/batch-predict`: Run inference across all models simultaneously.
- `GET /api/models/list`: List all available models and their status.
- `GET /api/comparison/accuracy`: Get comparative accuracy metrics.

## 🧠 Model Zoo

The system utilizes the following models variants:
- **Deep Learning**: LSTM, GRU, MLP, LSTM-CNN
- **Machine Learning**: XGBoost, SVM, Random Forest, Logistic Regression, Naive Bayes

## 📂 Project Structure

```
dl-ml-btc/
├── models/             # Saved model artifacts (.pkl, .h5, .json)
├── notebooks/          # Jupyter notebooks for training and research
├── server/             # FastAPI application source code
│   ├── routers/        # API endpoints (models, data, metrics)
│   ├── services/       # Business logic (inference, extensive data processing)
│   ├── config.py       # Configuration settings
│   └── main.py         # Application entry point
└── data/               # Raw and processed datasets
```
