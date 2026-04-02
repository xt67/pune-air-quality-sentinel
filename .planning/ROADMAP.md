# Roadmap: Pune Air Quality Sentinel (PAQS)

**Created:** 2026-04-02
**Core Value:** Accurate, hyper-local AQI forecasting that answers: "Is it safe to take a morning run in my neighborhood tomorrow?"

## Milestone: v1.0 — MVP Demo

5-phase roadmap aligned with 5-week build plan. Each phase ends with testable deliverables.

---

## Phase 1: Data Pipeline & IoT Simulation

**Goal:** Build complete data ingestion, preprocessing, and IoT simulation pipelines with passing tests.

**Week:** Week 1

### Requirements Covered
- DATA-01 to DATA-12 (Data Pipeline)
- IOT-01 to IOT-06 (IoT Simulation)
- INFRA-01 (Repository structure)

### Deliverables
1. Conda environment with GPU verification
2. Data fetch modules (Kaggle, OpenAQ, Open-Meteo)
3. Preprocessing pipeline (clean, merge, normalize, feature engineering)
4. IoT simulation generating 10-node synthetic sensor CSVs
5. Processed Parquet file with all features
6. Test suite: test_fetch.py, test_preprocess.py, test_iot_sim.py (all green)

### Success Criteria
- [ ] Parquet file exists at data/processed/pune_aqi_processed.parquet
- [ ] 10 IoT CSVs exist at data/simulated/node_*.csv
- [ ] All 3 test files pass: `pytest tests/test_fetch.py tests/test_preprocess.py tests/test_iot_sim.py -v`
- [ ] No NaN in processed features (assert checked)
- [ ] AQI values in [0, 500] (assert checked)

### Dependencies
- None (first phase)

---

## Phase 2: ARIMA & LSTM Models

**Goal:** Train baseline ARIMA and LSTM models with verified forward passes and initial MAE metrics.

**Week:** Week 2

### Requirements Covered
- ARIMA-01 to ARIMA-04 (ARIMA baseline)
- LSTM-01 to LSTM-07 (LSTM model)

### Deliverables
1. PyTorch Dataset class for sliding windows
2. ARIMA wrapper with auto-order selection (10 models, one per node)
3. LSTM architecture with attention mechanism
4. Training script with early stopping and mixed precision
5. Trained LSTM checkpoint (50+ epochs verified)
6. Test suite: test_dataset.py, test_lstm.py (all green)
7. ARIMA baseline MAE recorded

### Success Criteria
- [ ] ARIMA models saved at models/arima_*.pkl
- [ ] LSTM checkpoint saved at models/lstm_best.pth
- [ ] ARIMA MAE ≤30 AQI units on test set
- [ ] LSTM forward pass produces correct shape (batch, horizon)
- [ ] No NaN in gradients during training
- [ ] Training completes without OOM on RTX 4050

### Dependencies
- Phase 1 (processed data, IoT CSVs)

---

## Phase 3: ST-GNN & Full Evaluation

**Goal:** Train ST-GNN model and generate comprehensive model comparison across all three architectures.

**Week:** Week 3

### Requirements Covered
- STGNN-01 to STGNN-06 (ST-GNN model)
- EVAL-01 to EVAL-06 (Evaluation metrics)

### Deliverables
1. Graph construction (node_coords.json, adjacency_matrix.npy, edge_index.pt)
2. ST-GNN architecture using A3TGCN (or GRU+GCNConv fallback)
3. ST-GNN training script with graph batching
4. Metrics module (MAE, RMSE, MAPE, category accuracy, skill score)
5. Comparison module generating side-by-side results
6. Full comparison table (outputs/comparison_table.csv)
7. Test suite: test_stgnn.py, test_metrics.py (all green)

### Success Criteria
- [ ] ST-GNN checkpoint saved at models/stgnn_best.pth
- [ ] ST-GNN MAE ≤15 AQI units (target ≤12)
- [ ] Comparison table shows all 3 models, both horizons
- [ ] ST-GNN skill score > 30% improvement over ARIMA
- [ ] test_stgnn.py and test_metrics.py pass

### Dependencies
- Phase 2 (ARIMA and LSTM models for comparison)

---

## Phase 4: Geospatial Visualization

**Goal:** Create interactive Folium heatmaps and forecast plots for all 10 Pune nodes.

**Week:** Week 4

### Requirements Covered
- VIZ-01 to VIZ-05 (Visualization)

### Deliverables
1. Folium heatmap builder with AQI color coding
2. CircleMarkers with tooltips (node name, predicted AQI, category, advisory)
3. AQI legend component
4. Time-series forecast plots (actual vs all 3 models, category bands)
5. 5 representative heatmaps (good/moderate/Diwali/monsoon/summer)
6. 10 forecast plot PNGs (one per node)
7. Test suite: test_heatmap.py (green)

### Success Criteria
- [ ] 5 HTML heatmaps exist at outputs/heatmaps/
- [ ] 10 forecast plots exist at outputs/plots/
- [ ] Heatmap renders in ≤5 seconds for 10 nodes
- [ ] test_heatmap.py passes
- [ ] Heatmap contains 10 CircleMarker elements (verified in test)

### Dependencies
- Phase 3 (trained models for predictions)

---

## Phase 5: Streamlit Demo & Documentation

**Goal:** Build and deploy complete 5-page Streamlit demo with full documentation.

**Week:** Week 5

### Requirements Covered
- DEMO-01 to DEMO-08 (Streamlit Demo)
- INFRA-02 to INFRA-06 (Infrastructure)

### Deliverables
1. Streamlit app with 5 pages:
   - Page 1: Project Overview
   - Page 2: Live AQI Forecaster
   - Page 3: Pollution Heatmap
   - Page 4: Model Comparison Dashboard
   - Page 5: Technical Details
2. Model caching with @st.cache_resource
3. Error handling (no raw tracebacks)
4. Deployment to Streamlit Cloud or HuggingFace Spaces
5. README with setup, data, train, demo instructions
6. All public functions have docstrings and type hints
7. Architecture diagram and report figures

### Success Criteria
- [ ] `streamlit run app/streamlit_app.py` launches without error
- [ ] All 5 pages render correctly
- [ ] Inference latency ≤2 seconds
- [ ] 20 consecutive predictions without crash
- [ ] README includes all setup instructions
- [ ] Live demo URL deployed and accessible
- [ ] All report figures saved to outputs/report_figs/

### Dependencies
- Phase 4 (heatmap and plot visualizations)

---

## Progress Tracking

| Phase | Status | Started | Completed |
|-------|--------|---------|-----------|
| Phase 1: Data Pipeline & IoT Simulation | 🔲 Pending | - | - |
| Phase 2: ARIMA & LSTM Models | 🔲 Pending | - | - |
| Phase 3: ST-GNN & Full Evaluation | 🔲 Pending | - | - |
| Phase 4: Geospatial Visualization | 🔲 Pending | - | - |
| Phase 5: Streamlit Demo & Documentation | 🔲 Pending | - | - |

**Legend:** 🔲 Pending | 🔄 In Progress | ✅ Complete | ⚠️ Blocked

---

## Risk Mitigation

| Risk | Mitigation | Phase |
|------|------------|-------|
| OpenAQ limited Pune data | Fall back to Kaggle dataset | 1 |
| ST-GNN OOM on 6GB VRAM | Use AMP, reduce batch to 16, fallback to GRU+GCNConv | 3 |
| torch_geometric_temporal install fails | Install from source or manual GRU+GCNConv | 3 |
| Streamlit Cloud memory limit | Load only ST-GNN, use caching, record demo video | 5 |

---
*Roadmap created: 2026-04-02*
*Last updated: 2026-04-02 after initialization*
