# Requirements: Pune Air Quality Sentinel (PAQS)

**Defined:** 2026-04-02
**Core Value:** Accurate, hyper-local AQI forecasting that answers: "Is it safe to take a morning run in my neighborhood tomorrow?"

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Pipeline

- [ ] **DATA-01**: System fetches historical AQI data from Kaggle Air Quality India dataset
- [ ] **DATA-02**: System fetches weather data from Open-Meteo API (temp, humidity, wind, rainfall)
- [ ] **DATA-03**: System fetches pollution data from OpenAQ API for Pune stations
- [ ] **DATA-04**: System cleans data (forward-fill gaps ≤3h, interpolate 3-12h, drop >12h gaps)
- [ ] **DATA-05**: System detects and clips outliers beyond 99.9th percentile
- [ ] **DATA-06**: System computes AQI from sub-indices using CPCB breakpoint formula
- [ ] **DATA-07**: System engineers lag features (AQI_lag_1h, 6h, 24h) without data leakage
- [ ] **DATA-08**: System engineers rolling features (24h mean, 72h mean, 24h std)
- [ ] **DATA-09**: System creates wind vectors (wind_u, wind_v from speed + direction)
- [ ] **DATA-10**: System normalizes features using MinMaxScaler fitted on train split only
- [ ] **DATA-11**: System saves processed data as Parquet for fast reload
- [ ] **DATA-12**: System saves train/val/test split indices (70/15/15 chronological)

### IoT Simulation

- [ ] **IOT-01**: System generates 2-year hourly time series for all 10 Pune nodes
- [ ] **IOT-02**: System applies node-specific biases (Pimpri +25, Kothrud -10, etc.)
- [ ] **IOT-03**: System injects rush-hour spikes (7-9 AM, 5-8 PM: +15 AQI)
- [ ] **IOT-04**: System injects Diwali week spike (+80 AQI)
- [ ] **IOT-05**: System injects monsoon dip (July-Sept: -20 AQI)
- [ ] **IOT-06**: System saves one CSV per node with validated schema

### ARIMA Model

- [ ] **ARIMA-01**: System trains auto-ARIMA with seasonal=True, m=24 per node
- [ ] **ARIMA-02**: System falls back to ARIMA(2,1,2) on convergence failure
- [ ] **ARIMA-03**: System produces 24h and 48h ahead forecasts
- [ ] **ARIMA-04**: System serializes trained models with joblib

### LSTM Model

- [ ] **LSTM-01**: System creates sliding window Dataset (seq_len=72, horizon=24/48)
- [ ] **LSTM-02**: System implements stacked LSTM (128→64) with attention mechanism
- [ ] **LSTM-03**: System trains with HuberLoss, Adam optimizer, lr=1e-3
- [ ] **LSTM-04**: System applies early stopping (patience=10 on val MAE)
- [ ] **LSTM-05**: System uses mixed precision (AMP) to fit in 6GB VRAM
- [ ] **LSTM-06**: System checkpoints best model by validation MAE
- [ ] **LSTM-07**: System completes training in ≤30 minutes on RTX 4050

### ST-GNN Model

- [ ] **STGNN-01**: System constructs graph with 10 nodes and distance-based edges
- [ ] **STGNN-02**: System implements A3TGCN architecture (or GRU+GCNConv fallback)
- [ ] **STGNN-03**: System trains with graph batching and mixed precision
- [ ] **STGNN-04**: System predicts all nodes simultaneously
- [ ] **STGNN-05**: System achieves MAE ≤15 AQI units on test set
- [ ] **STGNN-06**: System completes training in ≤2 hours on RTX 4050

### Evaluation

- [ ] **EVAL-01**: System computes MAE, RMSE, MAPE for all models
- [ ] **EVAL-02**: System computes category accuracy (Good/Moderate/Poor/Severe)
- [ ] **EVAL-03**: System computes skill score vs ARIMA baseline
- [ ] **EVAL-04**: System generates comparison table CSV
- [ ] **EVAL-05**: System generates actual vs predicted plots per node
- [ ] **EVAL-06**: System generates MAE comparison bar chart

### Visualization

- [ ] **VIZ-01**: System creates Folium heatmap with AQI color coding (green→maroon)
- [ ] **VIZ-02**: System adds CircleMarkers for each node with tooltips
- [ ] **VIZ-03**: System adds legend showing AQI category colors
- [ ] **VIZ-04**: System exports heatmaps as HTML files
- [ ] **VIZ-05**: System creates time-series forecast plots with category bands

### Streamlit Demo

- [ ] **DEMO-01**: Page 1 displays project overview and key stats
- [ ] **DEMO-02**: Page 2 provides location dropdown + date picker for prediction
- [ ] **DEMO-03**: Page 2 shows predicted AQI, category badge, health advisory
- [ ] **DEMO-04**: Page 3 renders interactive Folium heatmap
- [ ] **DEMO-05**: Page 4 displays model comparison charts (Plotly)
- [ ] **DEMO-06**: Page 5 shows architecture diagram and training curves
- [ ] **DEMO-07**: Demo handles errors gracefully (no raw tracebacks)
- [ ] **DEMO-08**: Demo achieves ≤2 second inference latency

### Infrastructure

- [ ] **INFRA-01**: Repository follows specified folder structure
- [ ] **INFRA-02**: All tests pass (pytest with mocked API responses)
- [ ] **INFRA-03**: All scripts accept --seed argument (default 42)
- [ ] **INFRA-04**: Configs stored in YAML (no hardcoded hyperparameters)
- [ ] **INFRA-05**: Structured JSON logging with level/module/message
- [ ] **INFRA-06**: README includes setup, data, train, and demo instructions

## v2 Requirements (Future)

### Real-time Features

- **RT-01**: Real-time CPCB API integration (requires registration)
- **RT-02**: WhatsApp alert bot via Twilio
- **RT-03**: Mobile PWA wrapper for Streamlit

### Extended Coverage

- **EXT-01**: Multi-city generalization (Mumbai, Delhi, Bangalore)
- **EXT-02**: Satellite AOD (Aerosol Optical Depth) as input feature
- **EXT-03**: District-level Punjab/Maharashtra predictions

### Advanced Models

- **ADV-01**: Transformer-based temporal model comparison
- **ADV-02**: Ensemble of LSTM + ST-GNN predictions
- **ADV-03**: Monte Carlo dropout for uncertainty quantification

## Out of Scope

| Feature | Reason |
|---------|--------|
| Real-time sensor hardware | No hardware available; simulated IoT sufficient for demo |
| CPCB API integration | Requires registration; public datasets adequate |
| Mobile app | Web-based Streamlit meets viva demo needs |
| Multi-city deployment | Pune focus adds authenticity; generalize in v2 |
| Commercial hosting | Streamlit Cloud free tier sufficient |
| Real traffic data | Traffic proxy via hour encoding; real data needs permissions |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 to DATA-12 | Phase 1 | Pending |
| IOT-01 to IOT-06 | Phase 1 | Pending |
| ARIMA-01 to ARIMA-04 | Phase 2 | Pending |
| LSTM-01 to LSTM-07 | Phase 2 | Pending |
| STGNN-01 to STGNN-06 | Phase 3 | Pending |
| EVAL-01 to EVAL-06 | Phase 3 | Pending |
| VIZ-01 to VIZ-05 | Phase 4 | Pending |
| DEMO-01 to DEMO-08 | Phase 5 | Pending |
| INFRA-01 to INFRA-06 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 53 total
- Mapped to phases: 53
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-02*
*Last updated: 2026-04-02 after initial definition*
