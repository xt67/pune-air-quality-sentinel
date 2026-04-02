# Pune Air Quality Sentinel (PAQS)

AI-Based Air Quality Forecaster for Urban Pollution Monitoring

## Overview

PAQS forecasts hyper-local Air Quality Index (AQI) values 24-48 hours in advance for Pune neighborhoods. It combines time-series forecasting, geospatial analysis, and IoT simulation into a coherent ML pipeline.

**Models:** ARIMA (baseline) → LSTM → Spatio-Temporal GNN

## Features

- 🌍 **10 Pune neighborhoods** — Shivajinagar, Hinjewadi, Pimpri-Chinchwad, and more
- 📊 **Multi-horizon forecasting** — 24h and 48h ahead predictions
- 🗺️ **Interactive heatmap** — Folium-based pollution visualization
- ⚡ **Health advisories** — AQI category labels with recommendations

## Project Structure

```
pune-air-quality-sentinel/
├── src/
│   ├── data/          # Data pipeline modules
│   ├── models/        # ARIMA, LSTM, ST-GNN architectures
│   ├── evaluate/      # Metrics and comparison
│   ├── viz/           # Visualization (Folium, plots)
│   └── utils/         # Config, logging, utilities
├── app/               # Streamlit demo
├── configs/           # YAML configuration files
├── tests/             # pytest test suite
└── notebooks/         # EDA and prototyping
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/xt67/pune-air-quality-sentinel.git
cd pune-air-quality-sentinel
pip install -r requirements.txt

# Generate simulated IoT data
python -m src.data.iot_sim

# Run demo
streamlit run app/streamlit_app.py
```

## Tech Stack

- **ML:** PyTorch, PyTorch Geometric, pmdarima
- **Data:** Pandas, NumPy, GeoPandas
- **Viz:** Folium, Matplotlib, Plotly
- **App:** Streamlit, FastAPI
- **Data Sources:** OpenAQ, Open-Meteo, Kaggle Air Quality India

## Success Metrics

| Model | Target MAE (AQI) |
|-------|------------------|
| ARIMA | ≤ 30 |
| LSTM | ≤ 20 |
| ST-GNN | ≤ 12 |

## License

MIT

## Author

Rayan — B.Tech Final Year Project (Data Science / AI-ML)
