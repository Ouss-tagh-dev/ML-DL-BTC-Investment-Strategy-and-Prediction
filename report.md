# Project Report: Bitcoin Prediction System (ML/DL)

## Executive Summary

This project implements a robust automated trading system for **Bitcoin (BTC/USDT)**, designed to predict short-term price movements (1-hour horizon). By leveraging a **"Wisdom of Crowds"** ensemble strategy, we combine **9 distinct Machine Learning and Deep Learning models**.

**Key Achievement:**
The system successfully outperforms the market benchmark. While the "Buy & Hold" strategy generated a **+39.4%** return over the test period, our best-performing model (Random Forest) achieved a massive **+425.3% ROI**, demonstrating the validity of our algorithmic approach despite the inherent noise of financial markets.

---

## 1. Methodology Overview

### 1.1 Data ecosystem
We aggregated data from **4 independent sources** to capture a holistic market view:
1.  **Price Action (OHLCV):** Hourly market data from Binance (2018-Present).
2.  **On-Chain Metrics:** Hash rate, difficulty, and transaction volume to gauge network health.
3.  **Macro Indicators:** S&P500, Gold, and Oil prices to measure correlation with traditional finance.
4.  **Sentiment:** The "Fear & Greed Index" to account for retail investor psychology.

**Data Volume:** ~70,000 hourly observations, processed into **94 engineered features** (including Momentum, Trend, Volatility, and Lag variables) to feed the models.

### 1.2 The Model Ensemble
We deployed 9 algorithms divided into three categories:
*   **Classical ML (5 models):** Random Forest, XGBoost, SVM, Naive Bayes, Logistic Regression. *Chosen for robustness on tabular data.*
*   **Deep Learning (3 models):** LSTM, GRU, MLP. *Chosen for capturing temporal sequences.*
*   **Hybrid (1 model):** LSTM-CNN. *Chosen for combined feature extraction and sequence modeling.*

---

## 2. Performance Results

The models were evaluated on a strictly separated **Test Set (last 20% of data)** to simulate real-world performance.

### 2.1 Classification Metrics vs. Real-World Profitability
The average accuracy across models sits between **51% and 53.5%**. While this may appear low compared to other ML domains (like image recognition), in high-frequency financial prediction, **an accuracy >50% is sufficient for profitability** if combined with proper risk management.

**Crucial Finding:**
Higher accuracy does not always equal higher profit. "Tree-based" models (Random Forest, XGBoost) proved significantly superior to Deep Learning models for this specific tabular dataset.

### 2.2 Detailed Backtesting Results (ROI)

The following table summarizes the financial performance of each model compared to the "Buy & Hold" baseline.

| Rank | Model | Accuracy | ROI (Return) | Sharpe Ratio | Max Drawdown | vs. Buy & Hold |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **Random Forest** | **53.27%** | **+425.3%** | **2.45** | **-30.4%** | **+385.9%** |
| 🥈 | **XGBoost** | **53.17%** | **+325.2%** | 2.00 | -25.5% | +285.8% |
| 🥉 | **MLP** (Deep Learning) | 52.56% | +260.2% | 1.95 | -20.6% | +220.8% |
| 4 | Logistic Regression | 52.66% | +173.9% | 1.58 | -25.4% | +134.5% |
| 5 | SVM | 52.61% | +62.0% | 0.88 | -28.7% | +22.6% |
| 6 | GRU | 51.74% | +61.3% | **2.70** | **-4.9%** | +21.9% |
| 7 | Naive Bayes | 52.27% | +46.5% | 0.75 | -44.2% | +7.1% |
| 8 | LSTM-CNN | 51.28% | +41.9% | 1.99 | -5.1% | +2.5% |
| - | **Benchmark (Buy & Hold)** | - | **+39.4%** | **0.68** | **-34.8%** | **-** |
| 9 | LSTM | 51.06% | +33.4% | 0.03 | -4.5% | -6.0% |

> **Definitions:**
> *   **ROI:** Total profit percentage over the test period.
> *   **Sharpe Ratio:** Risk-adjusted return. A value > 2.0 is considered excellent.
> *   **Max Drawdown:** The maximum observed loss from a peak to a trough (risk metric).

---

## 3. Analysis & Key Insights

### 3.1 The dominance of "Trees"
**Random Forest** and **XGBoost** are the clear winners. They handle the noisy 94-feature dataset better than Deep Learning models.
*   **Why?** Financial data often has "regime changes" and high noise levels. Tree ensembles are naturally robust against outliers and irrelevant features, whereas LSTMs can easily overfit noise in such volatile environments.

### 3.2 The Safe Haven: GRU
While **GRU** ranked 6th in total profit (+61.3%), it achieved the **highest Sharpe Ratio (2.70)** and the **lowest Drawdown (-4.9%)**.
*   **Implication:** For a conservative investor who hates losing money, GRU is actually the *best* model. It trades less frequently but with much higher confidence, protecting capital during crashes.

### 3.3 The Failure of Pure LSTM
LSTM performed worse than the benchmark (+33.4% vs +39.4%). This suggests that standard LSTM architectures struggle with look-back windows that are too short (24h) or get overwhelmed by noise without mechanisms like "Attention" to focus on key events.

---

## 4. Conclusion & Future Roadmap

### 4.1 Project Status
The system is **fully functional and profitable**. We have a production-ready pipeline that ingests data, processes it, trains models, and serves predictions via an API and Dashboard. The hypothesis that ML can beat the market has been validated (Random Forest ROI > 4x Benchmark).

### 4.2 Future Perspective: Integrating Real-World Events (News & Geopolitics)
Predicting Bitcoin's movement is complex and depends on a multitude of external factors that quantitative data alone cannot fully capture.
To address this limitation, our next major objective is to develop a **News Analysis Model (NLP)**.

*   **Why?** Quantitative datasets miss "black swan" events.
*   **Goal:** To track real-time news and detect sudden geopolitical shifts (e.g., **wars**, regulatory bans, or major economic announcements). This qualitative indicator will act as a safeguard, providing critical context that is currently "invisible" to our standard dataset.

