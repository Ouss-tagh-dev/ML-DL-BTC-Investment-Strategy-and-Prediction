# Bitcoin ML/DL Dashboard API

FastAPI backend API for Bitcoin prediction dashboard with 9 ML/DL models.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies from project root
cd dl-ml-btc
pip install -r requirements.txt
```

### Launch the server

```bash
# From project root
cd dl-ml-btc/server
python main.py
```

The server starts on `http://localhost:8000`

- **Swagger Documentation:** http://localhost:8000/docs
- **ReDoc Documentation:** http://localhost:8000/redoc

## 📡 API Endpoints

### Data Endpoints (`/api/data`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data/historical` | GET | Historical OHLCV data with filters |
| `/api/data/features` | GET | Specific engineered features |
| `/api/data/statistics` | GET | Dataset statistics |
| `/api/data/latest` | GET | Latest available data |
| `/api/data/feature-names` | GET | List of all features |

### Models Endpoints (`/api/models`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/models/list` | GET | List of all models |
| `/api/models/predict` | POST | Prediction with a model |
| `/api/models/{model_id}/info` | GET | Model metadata |
| `/api/models/batch-predict` | POST | Predictions with all models |
| `/api/models/load/{model_id}` | POST | Load a model into memory |

### Metrics Endpoints (`/api/metrics`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics/performance` | GET | Performance metrics |
| `/api/metrics/{model_id}/backtesting` | GET | Backtesting results |
| `/api/metrics/comparison` | GET | Complete comparison |

### Comparison Endpoints (`/api/comparison`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/comparison/accuracy` | GET | Accuracy comparison |
| `/api/comparison/sharpe-ratio` | GET | Sharpe Ratios comparison |
| `/api/comparison/returns` | GET | Returns comparison |
| `/api/comparison/all-metrics` | GET | All metrics |
| `/api/comparison/ml-vs-dl` | GET | ML vs DL comparison |

## 🤖 Available Models

### Machine Learning (5 models)
- `logistic_regression` - Logistic Regression
- `naive_bayes` - Naive Bayes with PCA
- `random_forest` - Random Forest ⭐
- `svm` - Support Vector Machine
- `xgboost` - XGBoost

### Deep Learning (4 models)
- `mlp` - Multi-Layer Perceptron
- `lstm` - Long Short-Term Memory 🔄
- `gru` - Gated Recurrent Unit 🔄
- `lstm_cnn` - LSTM-CNN Hybrid 🔄

## 📝 Usage Examples

### Get historical data

```bash
curl "http://localhost:8000/api/data/historical?limit=100"
```

### List all models

```bash
curl "http://localhost:8000/api/models/list"
```

### Make a prediction

```bash
curl -X POST "http://localhost:8000/api/models/predict" \
  -H "Content-Type": application/json" \
  -d '{
    "model_id": "random_forest",
    "use_latest": true
  }'
```

### Predictions with all models

```bash
curl -X POST "http://localhost:8000/api/models/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "use_latest": true
  }'
```

### Compare performances

```bash
curl "http://localhost:8000/api/comparison/all-metrics"
```

## 🏗️ Architecture

```
server/
├── main.py                 # FastAPI Application
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── services/
│   ├── data_service.py    # Data service
│   └── model_service.py   # Model service
└── routers/
    ├── data.py            # Data endpoints
    ├── models.py          # Model endpoints
    ├── metrics.py         # Metrics endpoints
    └── comparison.py      # Comparison endpoints
```

## ⚙️ Configuration

### CORS

The server is configured to accept requests from:
- `http://localhost:3000` (React dev server)
- `http://localhost:5173` (Vite dev server)

To add other origins, modify `config.py`:

```python
CORS_ORIGINS: list = [
    "http://localhost:3000",
    "http://your-frontend-url.com"
]
```

### Paths

Paths to data and models are configured in `config.py`:

```python
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
FEATURES_FILE = DATA_DIR / "features" / "btc_features_complete.csv"
```

## 🔧 Development

### Debug Mode

```bash
uvicorn main:app --reload --log-level debug
```

### Tests

```bash
# Test server health
curl http://localhost:8000/health

# Test an endpoint
curl http://localhost:8000/api/data/statistics
```

## 📊 React Integration

Usage example in React:

```javascript
// Get models
const response = await fetch('http://localhost:8000/api/models/list');
const models = await response.json();

// Make a prediction
const prediction = await fetch('http://localhost:8000/api/models/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model_id: 'random_forest',
    use_latest: true
  })
});
const result = await prediction.json();
```

## 🚨 Important Notes

- DL models (LSTM, GRU, LSTM-CNN) are large and may take 1-2 seconds to load
- DL predictions may take 100-500ms
- The complete dataset (~78MB) is loaded into memory on first call
- Use `use_latest: true` to predict with the latest available data

## 📝 License

This project is open-source for educational purposes.
