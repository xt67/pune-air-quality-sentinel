# Pune Air Quality Sentinel (PAQS)

## What This Is

An AI-based air quality forecasting system that predicts hyper-local AQI values 24-48 hours in advance for specific Pune neighborhoods. It combines time-series forecasting, geospatial analysis, and IoT simulation into a unified pipeline — progressing from ARIMA baseline through LSTM to Spatio-Temporal GNN — delivering neighborhood-level predictions with uncertainty estimates via a Streamlit web app with Folium heatmaps.

## Core Value

**Accurate, hyper-local AQI forecasting that answers: "Is it safe to take a morning run in my neighborhood tomorrow?"** — the ST-GNN model must achieve ≤12 AQI MAE with spatial diffusion modeling that outperforms location-independent approaches.

## Requirements

### Validated

(None yet — ship to validate)

### Active

**Pipeline A — Data Ingestion & Preprocessing**
- [ ] Multi-source data fetch (OpenAQ API, Kaggle Air Quality India, Open-Meteo weather)
- [ ] Data cleaning, interpolation, and normalization pipeline
- [ ] AQI computation from sub-indices using CPCB formula
- [ ] Feature engineering (lag features, rolling stats, wind vectors, calendar features)
- [ ] Train/val/test chronological split (70/15/15)

**Pipeline B — IoT Simulation**
- [ ] Synthetic sensor grid generation for 10 Pune neighborhoods (N01-N10)
- [ ] 2-year hourly time series with node-specific biases and patterns
- [ ] Temporal patterns (rush-hour, Diwali, monsoon effects)

**Pipeline C — Model Training & Evaluation**
- [ ] ARIMA baseline model with auto-order selection (pmdarima)
- [ ] Stacked LSTM with attention mechanism (PyTorch)
- [ ] ST-GNN using A3TGCN (PyTorch Geometric Temporal)
- [ ] Graph construction (adjacency matrix, edge weights by distance/wind)
- [ ] Evaluation metrics (MAE, RMSE, MAPE, category accuracy, skill score)
- [ ] Model comparison framework

**Pipeline D — Geospatial Visualization**
- [ ] Folium pollution heatmap with AQI color coding
- [ ] Interactive node markers with tooltips
- [ ] Time-series forecast plots (actual vs predicted)
- [ ] Loss curves and training diagnostics

**Pipeline E — Streamlit Demo**
- [ ] Page 1: Project Overview
- [ ] Page 2: Live AQI Forecaster (location + date → prediction)
- [ ] Page 3: Pollution Heatmap (interactive Folium map)
- [ ] Page 4: Model Comparison Dashboard
- [ ] Page 5: Technical Details

**Infrastructure & Quality**
- [ ] Comprehensive test suite (pytest)
- [ ] Error handling and robustness framework
- [ ] Full reproducibility (seed management, config files)
- [ ] Structured logging (JSON format)
- [ ] Documentation (README, docstrings, type hints)

### Out of Scope

- Real-time sensor hardware deployment — simulated IoT only
- CPCB API integration — requires registration; use public data sources
- Mobile app — web-based Streamlit demo only
- Multi-city generalization — Pune-focused MVP
- Commercial hosting — Streamlit Cloud/HuggingFace Spaces demo

## Context

**Technical Environment:**
- Windows 11 with ASUS laptop: Intel i5-12500H, 32GB RAM, RTX 4050 6GB VRAM
- Python 3.10.x with CUDA 12.1
- PyTorch 2.2.0 + PyTorch Geometric + torch-geometric-temporal

**Target Locations (10 Pune Nodes):**
| ID | Neighborhood | Characteristic |
|----|--------------|----------------|
| N01 | Shivajinagar | Reference CPCB station |
| N02 | Hinjewadi IT Park | Traffic + construction |
| N03 | Pimpri-Chinchwad MIDC | Heavy industry |
| N04 | Hadapsar | Mixed residential-commercial |
| N05 | Katraj | Dense residential |
| N06 | Deccan Gymkhana | High vehicular density |
| N07 | Viman Nagar | Airport proximity |
| N08 | Kothrud | Residential, cleaner |
| N09 | Talegaon MIDC | Outer industrial |
| N10 | Mundhwa | Downstream residential |

**Success Metrics:**
- MAE (ARIMA): ≤30 AQI units
- MAE (LSTM): ≤20 AQI units
- MAE (ST-GNN): ≤12 AQI units
- RMSE (ST-GNN): ≤18 AQI units
- Inference latency: ≤2 seconds
- Demo stability: No crashes during 30-min viva

**Portfolio Positioning:**
- Demonstrates ARIMA → LSTM → ST-GNN progression
- Sustainability angle for Germany MS applications
- Hyper-local Pune focus with real place names
- Quantitative evaluation with hard numbers

## Constraints

- **Timeline**: 5 weeks with week-by-week milestones
- **Hardware**: RTX 4050 6GB VRAM — requires mixed precision training, batch size tuning
- **Data**: Public sources only (OpenAQ, Kaggle, Open-Meteo) — no CPCB API
- **Deployment**: Streamlit Cloud (1GB memory limit) or HuggingFace Spaces
- **Dependencies**: PyTorch Geometric Temporal may have install issues — fallback to manual GRU+GCNConv

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| A3TGCN over Transformer | Purpose-built for spatio-temporal; proven for traffic/pollution; fits VRAM | — Pending |
| Simulated IoT over real sensors | No hardware available; controlled experiments | — Pending |
| Kaggle + OpenAQ over CPCB API | Public access without registration | — Pending |
| HuberLoss over MSE | Robust to AQI spike outliers | — Pending |
| Chronological split over random | Prevents data leakage in time series | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-02 after initialization*
