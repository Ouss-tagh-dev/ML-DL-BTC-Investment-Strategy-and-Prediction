# Investment Strategies Based on Machine Learning and Deep Learning for Bitcoin

This project aims to predict Bitcoin price movements using an ensemble of 9 Machine Learning and Deep Learning models. It combines historical data analysis with real-time inference to generate Buy, Sell, or Neutral signals.

The system is built with a Python (FastAPI) backend for model inference and a React frontend for visualizing the consensus of these models.

The system features a 3-layer pyramidal architecture:
1.  **Data Layer**: Real-time aggregation of OHLCV market data.
2.  **Model Layer (Backend)**: Fast async inference using LSTM, GRU, XGBoost, and more.
3.  **Presentation Layer (Frontend)**: interactive, glassmorphic React dashboard for real-time monitoring and "AI Consensus" visualization.

## 📂 Project Structure

This monorepo contains two main components:

### 1. [Backend & Models (`dl-ml-btc/`)](./dl-ml-btc/)
The core intelligence of the system.
- **Ensemble Engine**: Runs 9 concurrent models (LSTM, GRU, CNN-LSTM, XGBoost, SVM, etc.).
- **FastAPI Server**: High-performance REST API handling real-time inference requests.
- **Notebooks**: Research and training environments for all models.

➡️ [**Read the Backend Documentation**](./dl-ml-btc/README.md)

### 2. [Dashboard (`dashbord/`)](./dashbord/)
The command center for traders.
- **AI Consensus**: Visualizes the aggregate "Buy/Sell/Neutral" vote from all models.
- **Real-Time Control**: Trigger manual inference and view live market data.
- **Analytics**: Compare model performance metrics (Accuracy, Sharpe Ratio) via interactive charts.

➡️ [**Read the Dashboard Documentation**](./dashbord/README.md)

---

## 🚀 Quick Start

To launch the full stack environment, you will need two terminal windows.

### Step 1: Start the Backend (API)

```bash
cd dl-ml-btc
# Activate virtual environment (Windows)
.\venv\Scripts\activate
# Start the server
python server/main.py
```
*Server runs at `http://localhost:8000`*

### Step 2: Start the Frontend (Dashboard)

```bash
cd dashbord
# Start React development server
npm start
```
*Dashboard opens at `http://localhost:3000`*

---

## 🧠 Model Ensemble Strategy

CryptoOracle relies on a diversity of algorithms to minimize risk:
- **Deep Learning**: Captures complex temporal dependencies (LSTM, GRU, MLP).
- **Machine Learning**: Robust statistical prediction (XGBoost, Random Forest, SVM).
- **Consensus Logic**: A democratic voting system determines the final signal, filtering out noise from individual models.
