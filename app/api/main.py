"""FastAPI main application."""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router, set_models

logger = logging.getLogger(__name__)

# Application metadata
APP_TITLE = "Pune Air Quality Sentinel API"
APP_DESCRIPTION = """
## AI-based Air Quality Forecaster for Urban Pollution Monitoring

This API provides AQI (Air Quality Index) predictions for Pune city monitoring stations
using LSTM and ARIMA models trained on CPCB data.

### Features
- **24-72 hour forecasts** for multiple stations
- **LSTM model** with temporal attention (MAE ~60)
- **ARIMA baseline** for comparison (MAE ~87)
- **Real-time inference** with model caching

### Stations
Currently monitoring 3 Pune stations:
- MH020: Karve Road
- MH021: Shivajinagar  
- MH022: Hadapsar
"""
APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup: Load models
    logger.info("Loading models...")
    
    lstm_model = None
    arima_model = None
    
    # Try to load LSTM model
    lstm_path = Path("models/lstm_best.pth")
    if lstm_path.exists():
        try:
            import torch
            from src.models.lstm import LSTMForecaster
            
            checkpoint = torch.load(lstm_path, map_location="cpu")
            lstm_model = LSTMForecaster(input_size=9, horizon=24)
            lstm_model.load_state_dict(checkpoint["model_state_dict"])
            lstm_model.eval()
            logger.info(f"LSTM model loaded from {lstm_path}")
        except Exception as e:
            logger.warning(f"Failed to load LSTM model: {e}")
    
    # Try to load ARIMA model
    arima_path = Path("models/arima_MH020.pkl")
    if arima_path.exists():
        try:
            import joblib
            arima_model = joblib.load(arima_path)
            logger.info(f"ARIMA model loaded from {arima_path}")
        except Exception as e:
            logger.warning(f"Failed to load ARIMA model: {e}")
    
    # Set models in routes
    set_models(lstm_model, arima_model)
    
    if lstm_model is None and arima_model is None:
        logger.warning("No models loaded - predictions will use mock data")
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("API shutdown")


# Create FastAPI app
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": APP_TITLE,
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="0.0.0.0", port=8000, reload=True)
