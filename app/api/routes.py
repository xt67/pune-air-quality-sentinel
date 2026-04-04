"""API routes for AQI prediction service."""
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.api.models import (
    ErrorResponse,
    HealthResponse,
    HourlyPrediction,
    PredictionRequest,
    PredictionResponse,
    StationInfo,
    StationsResponse,
    get_aqi_category,
)

router = APIRouter()

# Pune stations data
STATIONS = {
    "MH020": StationInfo(
        station_id="MH020",
        name="Pune - Karve Road",
        latitude=18.5018,
        longitude=73.8167,
        city="Pune",
        state="Maharashtra",
    ),
    "MH021": StationInfo(
        station_id="MH021",
        name="Pune - Shivajinagar",
        latitude=18.5308,
        longitude=73.8475,
        city="Pune",
        state="Maharashtra",
    ),
    "MH022": StationInfo(
        station_id="MH022",
        name="Pune - Hadapsar",
        latitude=18.5089,
        longitude=73.9260,
        city="Pune",
        state="Maharashtra",
    ),
}

# Model cache (populated on startup)
_models = {"lstm": None, "arima": None}
_model_metrics = {"lstm": {"mae": 59.93}, "arima": {"mae": 87.36}}


def set_models(lstm_model=None, arima_model=None):
    """Set models from startup event."""
    global _models
    _models["lstm"] = lstm_model
    _models["arima"] = arima_model


def get_models():
    """Get loaded models."""
    return _models


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and model status."""
    models_loaded = _models["lstm"] is not None or _models["arima"] is not None
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=models_loaded,
        timestamp=datetime.utcnow(),
    )


@router.get("/stations", response_model=StationsResponse, tags=["Stations"])
async def list_stations():
    """Get list of available monitoring stations."""
    return StationsResponse(
        stations=list(STATIONS.values()),
        count=len(STATIONS),
    )


@router.get("/stations/{station_id}", response_model=StationInfo, tags=["Stations"])
async def get_station(station_id: str):
    """Get details for a specific station."""
    if station_id not in STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station {station_id} not found",
        )
    return STATIONS[station_id]


@router.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_aqi(request: PredictionRequest):
    """Generate AQI predictions for a station."""
    # Validate station
    if request.station_id not in STATIONS:
        raise HTTPException(
            status_code=404,
            detail=f"Station {request.station_id} not found",
        )
    
    # Validate model type
    if request.model_type not in ["lstm", "arima"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type: {request.model_type}. Use 'lstm' or 'arima'",
        )
    
    # Generate predictions
    predictions = []
    base_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    
    # Use model if loaded, otherwise generate mock predictions
    model = _models.get(request.model_type)
    
    if model is not None:
        # TODO: Implement actual model inference
        # For now, generate reasonable mock data
        import random
        base_aqi = random.uniform(80, 150)
    else:
        # Mock prediction when model not loaded
        import random
        base_aqi = random.uniform(80, 150)
    
    for hour in range(request.horizon):
        # Simulate daily pattern (higher AQI in morning/evening rush hours)
        hour_of_day = (base_time.hour + hour) % 24
        if 7 <= hour_of_day <= 10 or 17 <= hour_of_day <= 20:
            aqi = base_aqi * 1.2 + random.uniform(-10, 10)
        else:
            aqi = base_aqi + random.uniform(-15, 15)
        
        aqi = max(0, min(500, aqi))  # Clamp to valid range
        category, color = get_aqi_category(aqi)
        
        predictions.append(
            HourlyPrediction(
                hour=hour,
                timestamp=base_time + timedelta(hours=hour),
                aqi=round(aqi, 1),
                category=category,
                color=color,
            )
        )
    
    return PredictionResponse(
        station_id=request.station_id,
        model_type=request.model_type,
        horizon=request.horizon,
        generated_at=datetime.utcnow(),
        predictions=predictions,
        current_aqi=round(predictions[0].aqi, 1) if predictions else None,
        mae=_model_metrics.get(request.model_type, {}).get("mae"),
    )


@router.get("/predict/{station_id}", response_model=PredictionResponse, tags=["Predictions"])
async def predict_aqi_get(
    station_id: str,
    model_type: str = Query(default="lstm", description="Model type: 'lstm' or 'arima'"),
    horizon: int = Query(default=24, ge=1, le=72, description="Forecast horizon in hours"),
):
    """Generate AQI predictions (GET endpoint for convenience)."""
    request = PredictionRequest(
        station_id=station_id,
        model_type=model_type,
        horizon=horizon,
    )
    return await predict_aqi(request)
