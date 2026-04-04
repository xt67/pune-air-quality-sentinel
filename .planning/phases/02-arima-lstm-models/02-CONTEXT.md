# Phase 2 Context: ARIMA & LSTM Models

## Phase Goal
Train baseline ARIMA and LSTM models with verified forward passes and initial MAE metrics.

## Requirements Covered
- ARIMA-01 to ARIMA-04 (ARIMA baseline)
- LSTM-01 to LSTM-07 (LSTM model)

## Environment
- **GPU:** NVIDIA GeForce RTX 4050 Laptop (6GB VRAM)
- **PyTorch:** 2.2.0+cu121 (CUDA enabled)
- **Data:** Phase 1 complete - IoT simulation ready

## Data Available
- `data/simulated/node_*.csv` - 10 Pune node CSVs (2 years hourly)
- `data/raw/*.csv` - Kaggle AQI data (459 files)
- `src/data/pipeline.py` - Unified data loading

## Architecture Decisions

### ARIMA
- **Auto-ARIMA** with seasonal=True, m=24 (daily seasonality)
- Fallback to ARIMA(2,1,2) on convergence failure
- One model per node (10 total)
- Forecasts: 24h and 48h ahead

### LSTM
- **Stacked LSTM**: 128 → 64 hidden units
- **Attention mechanism** for temporal focus
- **Input**: seq_len=72 (3 days), horizon=24/48
- **Loss**: HuberLoss (robust to outliers)
- **Optimizer**: Adam, lr=1e-3
- **Early stopping**: patience=10 on val MAE
- **Mixed precision (AMP)** to fit in 6GB VRAM

## File Structure (to create)
```
src/models/
├── dataset.py      # Sliding window Dataset
├── arima.py        # ARIMA wrapper
├── lstm.py         # LSTM with attention
└── train.py        # Training script

tests/
├── test_dataset.py
└── test_lstm.py

models/
├── arima_N01.pkl ... arima_N10.pkl
└── lstm_best.pth
```

## Success Criteria
- [ ] ARIMA models saved at models/arima_*.pkl
- [ ] LSTM checkpoint saved at models/lstm_best.pth
- [ ] ARIMA MAE ≤30 AQI units on test set
- [ ] LSTM forward pass produces correct shape (batch, horizon)
- [ ] No NaN in gradients during training
- [ ] Training completes without OOM on RTX 4050

## Dependencies
- Phase 1 complete ✓
- PyTorch with CUDA ✓
- pmdarima for auto-ARIMA (to install)
