# Model Comparison: Pune Air Quality Forecasting

## Project: Pune Air Quality Sentinel (PAQS)

Comparison of forecasting models for 24-hour AQI prediction.

## Summary

| Model | MAE | RMSE | Category Accuracy | Target Met |
|-------|-----|------|-------------------|------------|
| ARIMA | 87.36 | 112.45 | 58.0% | ❌ |
| LSTM | 59.93 | 78.21 | 67.0% | ❌ |
| ST-GNN | 52.18 | 68.94 | 72.0% | ❌ |

## Detailed Results

| Metric | ARIMA | LSTM | ST-GNN |
|--------|-------|------|--------|
| MAE | 87.36 | 59.93 | 52.18 |
| RMSE | 112.45 | 78.21 | 68.94 |
| MAPE (%) | 45.2 | 32.1 | 28.5 |
| R² | 0.42 | 0.65 | 0.71 |
| Category Accuracy | 58.0% | 67.0% | 72.0% |
| Adjacent Category Acc. | 82.0% | 89.0% | 91.0% |
| Forecast Horizon | 24 | 24 | 24 |
| Target MAE | 30 | 20 | 15 |
| Training Time | ~3 min | ~5 min | ~8 min |
| Parameters | ~50 (auto-selected) | 754,201 | 68,376 |

## Model Descriptions

### ARIMA (Baseline)
- **Type**: Statistical time series model
- **Architecture**: Auto-ARIMA with order selection
- **Strengths**: Interpretable, fast training, no GPU needed
- **Limitations**: Univariate, no spatial awareness

### LSTM (Deep Learning)
- **Type**: Recurrent Neural Network
- **Architecture**: 2-layer LSTM (256→128) with attention
- **Optimizer**: AdamW with weight decay + ReduceLROnPlateau
- **Strengths**: Captures long-term temporal patterns
- **Limitations**: Single station, no spatial modeling

### ST-GNN (Spatio-Temporal)
- **Type**: Graph Neural Network
- **Architecture**: GRU temporal + GCN spatial blocks
- **Graph**: 10 Pune stations, Gaussian kernel adjacency
- **Strengths**: Models both temporal and spatial dependencies
- **Limitations**: Requires multi-station data for full benefit

## Conclusions

1. **LSTM outperforms ARIMA** by 31% MAE reduction (59.93 vs 87.36)
2. **ST-GNN shows best potential** with spatial modeling capability
3. **Category accuracy** is reasonable (67-72%) for 6-class AQI prediction
4. **Adjacent category accuracy** (89-91%) shows predictions are usually close
5. **PRD targets** (MAE ≤30/20/15) are aggressive for 24h forecasting

## Data Source

- **Station**: Karve Road, Pune (MH020)
- **Records**: 33,538 hourly observations
- **Split**: 70% train / 15% validation / 15% test
- **Features**: PM2.5, PM10, NO₂, SO₂, O₃, temperature, humidity
