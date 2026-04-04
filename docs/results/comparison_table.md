# Model Comparison: Pune Air Quality Forecasting

## Project: Pune Air Quality Sentinel (PAQS)

Comparison of forecasting models for 24-hour AQI prediction.

## Summary (Updated with Clean Multi-Station Data)

| Model | MAE | RMSE | Category Accuracy | Improvement |
|-------|-----|------|-------------------|-------------|
| ARIMA | 87.36 | 112.45 | 58.0% | Baseline |
| **LSTM** | **55.22** | **71.95** | **50.7%** | **7.9% ↓ MAE** |
| ST-GNN | 56.08 | 72.12 | 48.1% | - |

**Best Model:** LSTM (MAE: 55.22 µg/m³)

## Detailed Results

| Metric | ARIMA | LSTM | ST-GNN |
|--------|-------|------|--------|
| MAE | 87.36 | **55.22** | 56.08 |
| RMSE | 112.45 | **71.95** | 72.12 |
| Category Accuracy | 58.0% | 50.7% | 48.1% |
| Forecast Horizon | 24h | 24h | 24h |
| Training Time | ~3 min | ~7 min | ~14 min |
| Parameters | ~50 | 754,201 | 267,800 |

## Training Data

- **Stations**: 6 Pune stations (MH020-MH025)
- **Records**: 150,574 hourly observations (cleaned)
- **Split**: 70% train / 15% validation / 15% test
- **Features**: PM2.5, PM10, NO₂, SO₂, O₃, temperature, humidity

### Data Cleaning Applied
1. Removed negative values
2. Removed extreme outliers (>99.9th percentile × 2)
3. Linear interpolation for gaps ≤6 hours
4. Forward/backward fill for remaining gaps
5. Median fill for any remaining NaN
6. **Result**: 0 NaN remaining

## Model Descriptions

### ARIMA (Baseline)
- **Type**: Statistical time series model
- **Architecture**: Auto-ARIMA with order selection
- **Strengths**: Interpretable, fast training, no GPU needed
- **Limitations**: Univariate, no spatial awareness

### LSTM (Best Performer)
- **Type**: Recurrent Neural Network
- **Architecture**: 2-layer LSTM (256→128) with attention
- **Optimizer**: AdamW + CosineAnnealingWarmRestarts
- **Loss**: SmoothL1Loss (robust to outliers)
- **Strengths**: Captures long-term temporal patterns

### ST-GNN (Spatio-Temporal)
- **Type**: Graph Neural Network
- **Architecture**: GRU temporal + GCN spatial blocks
- **Graph**: Features as nodes with correlation-based adjacency
- **Note**: Better suited for true multi-station spatial data

## Key Improvements Made

1. **More Data**: 150K rows vs 33K (6 stations combined)
2. **Clean Data**: 0 NaN (thorough cleaning pipeline)
3. **Better Normalization**: StandardScaler vs MinMaxScaler
4. **Better Scheduler**: CosineAnnealingWarmRestarts
5. **Robust Loss**: SmoothL1Loss instead of HuberLoss
6. **Shorter Sequences**: 48h input (less noise)

## Conclusions

1. **LSTM achieved best MAE** (55.22) after clean data training
2. **7.9% improvement** over previous LSTM (59.93 → 55.22)
3. **ST-GNN similar performance** (56.08) - spatial benefit limited without true multi-location data
4. **Data quality matters**: Clean 6-station data improved results significantly
