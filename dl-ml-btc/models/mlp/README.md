# 🧠 MLP Deep Learning Model - Bitcoin Price Prediction

## 📋 Overview

This notebook implements a **Multi-Layer Perceptron (MLP)** - a deep neural network for Bitcoin directional prediction. It extends the traditional ML models with advanced deep learning techniques.

## 🎯 Key Objectives

1. **Advanced Neural Architecture**: 4-layer deep network with batch normalization and dropout
2. **Optimal Regularization**: L2 penalty + dropout + early stopping to prevent overfitting
3. **Adaptive Learning**: Learning rate scheduling with ReduceLROnPlateau
4. **Comprehensive Evaluation**: Classification metrics + financial backtesting
5. **Production-Ready Export**: H5 model + scaler + metadata

## 📊 Model Architecture

```
Input Layer (30 features)
    ↓
Dense(256) + ReLU + BatchNormalization + Dropout(0.3)
    ↓
Dense(128) + ReLU + BatchNormalization + Dropout(0.3)
    ↓
Dense(64) + ReLU + BatchNormalization + Dropout(0.25)
    ↓
Dense(32) + ReLU + BatchNormalization + Dropout(0.2)
    ↓
Output Layer: Dense(1, Sigmoid) → Binary Classification
```

### Architecture Statistics

- **Total Parameters**: ~45,000
- **Trainable Parameters**: ~45,000
- **Batch Normalization Layers**: 4
- **Regularization Layers**: 4 Dropout layers
- **Activation Function**: ReLU (hidden), Sigmoid (output)

## 🔧 Training Configuration

| Parameter               | Value                                  |
| ----------------------- | -------------------------------------- |
| Optimizer               | Adam                                   |
| Learning Rate           | 0.001 (dynamic with ReduceLROnPlateau) |
| Loss Function           | Binary Crossentropy                    |
| Batch Size              | 32                                     |
| Max Epochs              | 150                                    |
| Validation Split        | 20%                                    |
| Early Stopping Patience | 15 epochs                              |
| L2 Regularization       | 1e-4                                   |

## 📈 Expected Performance

| Metric         | Expected Value | vs Random Forest |
| -------------- | -------------- | ---------------- |
| Accuracy       | 53.5-54.5%     | +0.2-1.2%        |
| AUC-ROC        | 0.54-0.55      | +0.01-0.02       |
| Training Time  | 5-15 min       | ~5-10x slower    |
| Inference Time | Similar        | Similar          |

## ✅ Advantages Over Traditional ML

- ✅ **Complex Non-linearities**: Deep networks capture subtle patterns ML can miss
- ✅ **Batch Normalization**: Stabilizes training and accelerates convergence
- ✅ **Adaptive Learning**: ReduceLROnPlateau automatically adjusts learning rate
- ✅ **Efficient Regularization**: Multiple dropout layers prevent overfitting
- ✅ **Native Metrics**: AUC, accuracy calculated directly
- ✅ **GPU Acceleration**: 10-100x faster training on NVIDIA GPU

## ❌ Challenges

- ❌ **Training Time**: 5-15 minutes (vs 1-2 min for traditional ML)
- ❌ **No Interpretability**: Pure black-box model
- ❌ **Hyperparameter Sensitivity**: More tuning required
- ❌ **Data Normalization Critical**: StandardScaler essential for neural networks
- ❌ **Overfitting Risk**: High capacity = high risk without proper regularization
- ❌ **GPU Dependency**: CPU training is slow; GPU recommended

## 📂 Files Generated

```
models/mlp/
├── btc_mlp_hourly_model.h5          # Trained model (Keras format)
├── btc_mlp_hourly_scaler.pkl        # StandardScaler for preprocessing
└── btc_mlp_hourly_metadata.json     # Complete metadata
```

## 🚀 Usage - Inference

```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Load model and scaler
model = load_model('btc_mlp_hourly_model.h5')
scaler = joblib.load('btc_mlp_hourly_scaler.pkl')

# Prepare features (30 selected features)
X_new = df[top_features]
X_scaled = scaler.transform(X_new)

# Make predictions
predictions_proba = model.predict(X_scaled)
predictions = (predictions_proba > 0.5).astype(int).flatten()

# predictions[i] = 1 → Bullish (Hausse)
# predictions[i] = 0 → Bearish (Baisse)
```

## 📊 Backtesting Results

The model is evaluated using:

1. **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
2. **Financial Metrics**:
   - Total Return vs Benchmark
   - Annualized Volatility
   - Sharpe Ratio
   - Maximum Drawdown

## 🔍 Comparison with Other Models

| Aspect                    | MLP    | Random Forest | XGBoost |
| ------------------------- | ------ | ------------- | ------- |
| Accuracy                  | ~54%   | 53.27%        | 53.17%  |
| Training Speed            | Slow   | Fast          | Medium  |
| Interpretability          | None   | Low           | Low     |
| GPU Beneficial            | Yes    | No            | No      |
| Hyperparameter Complexity | High   | Medium        | High    |
| Production Ready          | Medium | High          | High    |

## 🎓 Next Steps

1. **Hyperparameter Optimization**: Use Bayesian Optimization (Optuna) to find best architecture
2. **Alternative Architectures**:
   - Deeper networks (5-6 layers)
   - LSTM/GRU for temporal sequences
   - Attention mechanisms (Transformer-based)
3. **Ensemble Methods**: Combine MLP + RF + XGB predictions
4. **Feature Engineering**: Test additional lags and indicators
5. **Walk-Forward Validation**: Out-of-sample performance testing

## ⚠️ Important Notes

- **Data Normalization**: StandardScaler MUST fit on training data only
- **No Data Leakage**: Features exclude 'target*' and 'future*' columns
- **Sequential Split**: 80/20 temporal split prevents look-ahead bias
- **Regularization Critical**: Dropout + L2 penalty prevent overfitting
- **GPU Recommended**: Training on CPU is slow (15+ minutes)

## 📚 References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Guide](https://www.tensorflow.org/guide)
- [Deep Learning for Time Series](https://www.deeplearningbook.org/)

## 📝 Metadata Example

```json
{
  "model_name": "MLP Bitcoin Directional Predictor",
  "model_type": "Deep Neural Network (Multi-Layer Perceptron)",
  "accuracy": 0.545,
  "auc_roc": 0.546,
  "architecture": {
    "layers": [
      "Input: 30 features",
      "Dense(256) + BatchNorm + Dropout(0.3)",
      "Dense(128) + BatchNorm + Dropout(0.3)",
      "Dense(64) + BatchNorm + Dropout(0.25)",
      "Dense(32) + BatchNorm + Dropout(0.2)",
      "Output: Dense(1, sigmoid)"
    ],
    "total_parameters": 45000
  },
  "export_timestamp": "2026-01-27 14:30:00"
}
```

---

**Created:** January 27, 2026  
**Status:** Development & Validation  
**GPU Recommended:** Yes  
**Estimated Training Time:** 5-15 minutes
