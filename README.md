# Pune Air Quality Sentinel (PAQS)

AI-Based Air Quality Forecaster for Urban Pollution Monitoring

## 🎯 Overview

PAQS forecasts hyper-local Air Quality Index (AQI) values 24-48 hours in advance for Pune neighborhoods. It combines time-series forecasting, geospatial analysis, and IoT simulation into a coherent ML pipeline.

**Core Question:** "Is it safe to take a morning run in my neighborhood tomorrow?"

**Models:** ARIMA (baseline) → LSTM → Spatio-Temporal GNN

## ✨ Features

- 🌍 **10 Pune neighborhoods** — Shivajinagar, Hinjewadi, Pimpri-Chinchwad, and more
- 📊 **Multi-horizon forecasting** — 24h and 48h ahead predictions
- 🗺️ **Interactive heatmap** — Folium-based pollution visualization
- ⚡ **Health advisories** — AQI category labels with recommendations

## 🏗️ Project Structure

```
pune-air-quality-sentinel/
├── app/                    # Streamlit application
│   ├── components/         # Reusable UI components
│   └── streamlit_app.py    # Main app entry point
├── configs/                # YAML configuration files
│   ├── data.yaml          # Data pipeline config
│   ├── lstm.yaml          # LSTM model config
│   └── stgnn.yaml         # ST-GNN model config
├── data/
│   ├── raw/               # Downloaded datasets
│   ├── processed/         # Cleaned and feature-engineered data
│   ├── simulated/         # IoT simulation outputs
│   ├── splits/            # Train/val/test indices
│   └── graph/             # Node coordinates and adjacency
├── models/                # Trained model checkpoints
├── notebooks/             # Jupyter notebooks for exploration
├── outputs/
│   ├── logs/              # JSON structured logs
│   ├── plots/             # Visualization outputs
│   └── heatmaps/          # Folium heatmap HTMLs
├── src/
│   ├── data/              # Data ingestion and preprocessing
│   ├── models/            # ARIMA, LSTM, ST-GNN implementations
│   ├── evaluate/          # Metrics and comparison
│   ├── viz/               # Visualization modules
│   └── utils/             # Shared utilities
└── tests/                 # pytest test suite
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1 (for GPU acceleration)

### Installation

```bash
# Clone and setup
git clone https://github.com/xt67/pune-air-quality-sentinel.git
cd pune-air-quality-sentinel
pip install -r requirements.txt

# Set up Kaggle credentials (for data download)
# Create .env with KAGGLE_API_TOKEN=your_token
```

### Data Pipeline

```bash
# Run full data pipeline
python -m src.data.pipeline

# Or run individual components
python -m src.data.fetch      # Download datasets
python -m src.data.preprocess # Clean and engineer features
python -m src.data.iot_sim    # Generate IoT simulation
```

### Training

```bash
# Train ARIMA baseline
python -m src.models.train_arima

# Train LSTM
python -m src.models.train_lstm --config configs/lstm.yaml

# Train ST-GNN
python -m src.models.train_stgnn --config configs/stgnn.yaml
```

### Demo

```bash
# Run Streamlit app
streamlit run app/streamlit_app.py
```

## 📍 Target Locations (10 Pune Nodes)

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

## 📊 Success Metrics

| Model | Target MAE (AQI) | RMSE |
|-------|------------------|------|
| ARIMA | ≤ 30 | - |
| LSTM | ≤ 20 | - |
| ST-GNN | ≤ 12 | ≤ 18 |

## 🛠️ Tech Stack

- **ML:** PyTorch, PyTorch Geometric, pmdarima
- **Data:** Pandas, NumPy, GeoPandas
- **Viz:** Folium, Matplotlib, Plotly
- **App:** Streamlit, FastAPI
- **Data Sources:** OpenAQ, Open-Meteo, Kaggle Air Quality India

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## 📝 License

MIT License

## 👨‍💻 Author

Rayan — B.Tech Final Year Project (Data Science / AI-ML)
