# Pune Air Quality Sentinel (PAQS)

An AI-based air quality forecasting system that predicts hyper-local AQI values 24-48 hours in advance for specific Pune neighborhoods.

## 🎯 Core Value

**Accurate, hyper-local AQI forecasting that answers: "Is it safe to take a morning run in my neighborhood tomorrow?"**

## 🏗️ Project Structure

```
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
- Conda (recommended)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd pune-air-quality-sentinel

# Create conda environment
conda create -n paqs python=3.10
conda activate paqs

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Edit .env with your Kaggle credentials
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

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fetch.py -v
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

## 📊 Model Performance Targets

| Model | MAE (AQI units) | RMSE |
|-------|-----------------|------|
| ARIMA | ≤30 | - |
| LSTM | ≤20 | - |
| ST-GNN | ≤12 | ≤18 |

## 📁 Configuration

All hyperparameters are stored in `configs/`:

- `data.yaml`: Data paths, preprocessing parameters
- `lstm.yaml`: LSTM architecture and training settings
- `stgnn.yaml`: ST-GNN architecture and graph construction

## 🧪 Testing

```bash
# Run full test suite with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run only Phase 1 tests
pytest tests/test_fetch.py tests/test_preprocess.py tests/test_iot_sim.py -v
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- Data sources: OpenAQ, Open-Meteo, Kaggle Air Quality India
- Built with: PyTorch, PyTorch Geometric, Streamlit, Folium
