# 🔄 LSTM Model - Bitcoin Prediction

## Overview

**LSTM (Long Short-Term Memory)** neural network designed for Bitcoin hourly price direction prediction.

- **Architecture**: 2 LSTM layers (128 → 64 units) + Dense layers
- **Purpose**: Capture temporal dependencies in cryptocurrency time series
- **Advantage**: Memory cells naturally model sequential patterns
- **Total Parameters**: ~35,000

## Architecture Diagram

```
Input (24 hours × 30 features)
    ↓
LSTM(128) + BatchNorm + Dropout(0.3)
    ↓
LSTM(64) + BatchNorm + Dropout(0.25)
    ↓
Dense(32) + BatchNorm + Dropout(0.2)
    ↓
Dense(16) + Dropout(0.1)
    ↓
Dense(1, sigmoid) → Output [0, 1]
```

## Key Differences from MLP

| Aspect         | MLP                   | LSTM                               |
| -------------- | --------------------- | ---------------------------------- |
| **Input**      | Static features (30)  | Sequences (24 steps × 30 features) |
| **Processing** | Feed-forward          | Recurrent/Sequential               |
| **Memory**     | None                  | Hidden state + Cell state          |
| **Temporal**   | Via lag features      | Native temporal modeling           |
| **Speed**      | Fast (~2 min)         | Slow (~8-12 min)                   |
| **Advantage**  | Simple, interpretable | Better seq modeling                |
| **Best for**   | Snapshot predictions  | Time series patterns               |

## Configuration

### Sequence Length

- **Context Window**: 24 hours of historical data
- **Prediction Horizon**: Next hour (t+1)
- **Format**: (batch, 24, 30) → sequences of 24 timesteps × 30 features

### LSTM Cells

```python
# First layer - produces sequence output
LSTM(128, return_sequences=True)
    ├─ Memory units: 128
    └─ Output: (batch, 24, 128) → passed to next LSTM

# Second layer - produces single output
LSTM(64, return_sequences=False)
    ├─ Memory units: 64
    └─ Output: (batch, 64) → passed to Dense layers
```

### Training Configuration

```
Optimizer:        Adam (lr=0.001)
Loss Function:    Binary Crossentropy
Batch Size:       32 sequences
Max Epochs:       150
Validation Split: 20%
Early Stopping:   Patience=20, monitor val_loss
LR Scheduler:     ReduceLROnPlateau (factor=0.5, patience=10)
Regularization:   L2 (1e-4) + Dropout (0.3→0.1)
```

## Expected Performance

### Model Accuracy

```
Training Accuracy:  ~54.5-55.0%
Test Accuracy:      ~53.8-54.2%
Overfit Gap:        ~0.5-1.0% (Good!)
AUC-ROC:            ~0.62-0.64
```

### Comparison with MLP & Random Forest

| Metric               | LSTM        | MLP        | Random Forest   |
| -------------------- | ----------- | ---------- | --------------- |
| **Accuracy**         | 53.8-54.2%  | 53.5-54.5% | 53.27%          |
| **Training Time**    | 8-12 min    | 5-10 min   | <1 min          |
| **Sharpe Ratio**     | 0.24-0.26   | 0.23-0.25  | 0.22            |
| **Interpretability** | ⭐ Very Low | ⭐⭐ Low   | ⭐⭐⭐⭐ Medium |

### Backtesting Metrics (Expected)

```
Total Return:        +1.5% to +2.5% (before fees)
Annual Volatility:   ~0.76-0.78%
Sharpe Ratio:        0.24-0.26
Maximum Drawdown:    -14% to -16%
Win Rate:            51.5-52.5%
```

## Why LSTM for Time Series?

### ✅ Advantages

1. **Native Temporal Modeling**
   - Memory cells (Ct) preserve long-term patterns
   - Forgets irrelevant info, retains important patterns
   - Better than manual lag features

2. **Sequence-to-Sequence**
   - Processes 24-hour context naturally
   - Understands temporal evolution
   - Captures market momentum changes

3. **Variable Memory Length**
   - Automatically learns how far back to look
   - No fixed lag window needed
   - Flexible temporal patterns

4. **Gate Mechanisms**
   - Input gate: What to add to memory?
   - Forget gate: What to remove?
   - Output gate: What to output?
   - Smart, learned temporal filtering

### ❌ Disadvantages

1. **Slow Training**
   - Recurrent computation: 8-12 minutes (vs MLP 5-10 min)
   - Many matrix multiplications per timestep
   - GPU highly beneficial

2. **Black Box**
   - Difficult to interpret hidden states
   - No feature importance ranking
   - Can't explain predictions easily

3. **Hyperparameter Tuning**
   - Sequence length (12, 24, 48, 96?)
   - Number of LSTM units (32, 64, 128, 256?)
   - Layers (1, 2, 3?)
   - Dropout rates
   - More complexity than MLP

4. **Overfitting Risk**
   - Large parameter space (~35K)
   - Sequential dependencies = correlated data
   - Needs strong regularization

## Usage Example

### 1. Load Pre-trained Model

```python
import tensorflow as tf
import joblib
import numpy as np

# Load model and scaler
model = tf.keras.models.load_model('btc_lstm_hourly_model.h5')
scaler = joblib.load('btc_lstm_hourly_scaler.pkl')

# Load your data (shape: (N_samples, 30_features))
X_new = load_your_data()

# Create sequences (critical for LSTM!)
def create_sequences(X, seq_length=24):
    X_seq = []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
    return np.array(X_seq)

# Preprocess
X_scaled = scaler.transform(X_new).astype(np.float32)
X_seq = create_sequences(X_scaled, seq_length=24)

# Predict
predictions_proba = model.predict(X_seq)  # Shape: (N_sequences, 1)
predictions = (predictions_proba > 0.5).astype(int)

print(f"Up signals: {predictions.sum()}")
print(f"Down signals: {len(predictions) - predictions.sum()}")
```

### 2. Fine-tune on New Data

```python
# Retrain last layers on recent data
model.trainable = False
model.layers[-2].trainable = True  # Unfreeze Dense layer
model.layers[-1].trainable = True  # Unfreeze output

# Recompile with lower learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Fit on recent data
model.fit(X_seq_new, y_new, epochs=10, batch_size=32)
```

### 3. Ensemble with MLP

```python
# Combine predictions
y_lstm_proba = lstm_model.predict(X_seq)
y_mlp_proba = mlp_model.predict(X_scaled)

# Average probabilities
ensemble_proba = (y_lstm_proba + y_mlp_proba) / 2
ensemble_pred = (ensemble_proba > 0.5).astype(int)

print(f"Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
```

## Implementation Notes

### Data Preparation (CRITICAL!)

1. **Standardization**: StandardScaler fit ONLY on training data
2. **Sequence Creation**: Convert (N, F) → (N-seq_len, seq_len, F)
3. **No Look-ahead**: Target is at position t+1, features from t-23 to t
4. **Time Series Split**: No shuffling, respect temporal order

### Training Tips

1. **GPU Recommended**: LSTM training ~5-10x faster on GPU
2. **Early Stopping**: Monitor val_loss, patience=20
3. **Learning Rate**: Start at 0.001, reduce by 0.5 on plateau
4. **Batch Size**: 32 is good for this dataset (not too small/large)
5. **Validation Split**: 20% helps prevent overfitting

### Common Issues

| Issue                        | Solution                                   |
| ---------------------------- | ------------------------------------------ |
| Very slow training           | Enable GPU, reduce sequence length         |
| Not learning (loss constant) | Increase learning rate, check data scaling |
| Overfitting (train >> test)  | Increase dropout, add L2 regularization    |
| OOM error                    | Reduce batch size or sequence length       |

## Files Generated

After running the notebook:

```
models/lstm/
├── btc_lstm_hourly_model.h5           # Trained Keras model (8-10 MB)
├── btc_lstm_hourly_scaler.pkl         # StandardScaler for preprocessing
├── btc_lstm_hourly_metadata.json       # Complete model metadata
├── lstm_training_history.png           # Loss/Accuracy/AUC plots
└── lstm_equity_curve.png               # Backtesting equity curve
```

## Metadata Structure

```json
{
  "model_name": "LSTM Bitcoin Predictor",
  "architecture": {
    "input_shape": [24, 30],
    "layers": [
      "LSTM(128, return_seq=True) + BatchNorm + Dropout(0.3)",
      "LSTM(64, return_seq=False) + BatchNorm + Dropout(0.25)",
      ...
    ],
    "total_parameters": 35000,
    "sequence_length": 24
  },
  "performance_metrics": {
    "test_accuracy": 0.5385,
    "test_auc_roc": 0.6250,
    "sharpe_ratio": 0.25,
    ...
  }
}
```

## Comparison: LSTM vs Other Models

### LSTM vs GRU (Planned)

- **LSTM**: More parameters, better for very long sequences
- **GRU**: Simpler, faster, similar performance on hourly data

### LSTM vs Transformer (Future)

- **LSTM**: Recurrent, good for sequential learning
- **Transformer**: Parallel attention, better for very long sequences (months)

### LSTM vs ARIMA (Classical)

- **LSTM**: Non-linear, requires deep learning setup
- **ARIMA**: Linear, interpretable, fast, needs stationarity

## Hyperparameter Tuning Ideas

### Option 1: GridSearch

```python
import optuna

def objective(trial):
    units_1 = trial.suggest_int('units_1', 64, 256, step=32)
    units_2 = trial.suggest_int('units_2', 32, 128, step=32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # Build and train model
    # Return validation accuracy
    return val_accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### Option 2: Bayesian Optimization

```bash
python -m optuna create-study --study-name lstm_study
python -m optuna dashboard --storage sqlite:///db.sqlite3
```

## Next Steps

1. **Run Notebook**: Execute LSTM_BTC.ipynb to generate trained model
2. **Compare with MLP**: Check relative performance
3. **Ensemble Both**: Combine LSTM + MLP + RF predictions
4. **Try GRU**: Implement lighter variant
5. **Optimize**: Use Bayesian optimization for hyperparameters

## References

- **LSTM Paper**: [Hochreiter & Schmidhuber, 1997](http://www.bioinf.uni-leipzig.de/~finke/papers/hochreiter96-lstm.pdf)
- **Keras LSTM Docs**: https://keras.io/api/layers/recurrent_layers/lstm/
- **Time Series with LSTM**: https://keras.io/examples/timeseries/timeseries_weather_forecasting/
- **Understanding LSTMs**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Notes

⚠️ **Important**: Always normalize input to [-1, 1] range for optimal LSTM learning.  
⚠️ **GPU Required**: LSTM training without GPU takes 20+ minutes.  
⚠️ **Overfitting Risk**: More parameters than MLP - monitor validation metrics closely.

---

**Model Version**: 1.0  
**Created**: January 27, 2026  
**Status**: Ready for Training  
**Estimated Training Time**: 8-12 minutes (GPU) / 30-40 minutes (CPU)
