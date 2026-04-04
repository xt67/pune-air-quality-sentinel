"""Pydantic models for API request/response schemas."""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    models_loaded: bool = Field(..., example=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StationInfo(BaseModel):
    """Air quality monitoring station information."""
    station_id: str = Field(..., example="MH020")
    name: str = Field(..., example="Pune - Karve Road")
    latitude: float = Field(..., example=18.5018)
    longitude: float = Field(..., example=73.8167)
    city: str = Field(..., example="Pune")
    state: str = Field(..., example="Maharashtra")


class StationsResponse(BaseModel):
    """List of available stations."""
    stations: List[StationInfo]
    count: int


class PredictionRequest(BaseModel):
    """Request for AQI prediction."""
    station_id: str = Field(..., example="MH020")
    model_type: str = Field(default="lstm", example="lstm", description="Model type: 'lstm' or 'arima'")
    horizon: int = Field(default=24, ge=1, le=72, example=24, description="Forecast horizon in hours")


class HourlyPrediction(BaseModel):
    """Single hour prediction."""
    hour: int = Field(..., ge=0, le=71)
    timestamp: datetime
    aqi: float = Field(..., ge=0, le=500)
    category: str = Field(..., example="Moderate")
    color: str = Field(..., example="#FFFF00")


class PredictionResponse(BaseModel):
    """AQI prediction response."""
    station_id: str
    model_type: str
    horizon: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    predictions: List[HourlyPrediction]
    current_aqi: Optional[float] = None
    mae: Optional[float] = Field(None, description="Model's Mean Absolute Error on test set")


class ErrorResponse(BaseModel):
    """Error response."""
    detail: str
    code: str = Field(default="ERROR")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# AQI category mapping
AQI_CATEGORIES = [
    (0, 50, "Good", "#00E400"),
    (51, 100, "Satisfactory", "#92D050"),
    (101, 200, "Moderate", "#FFFF00"),
    (201, 300, "Poor", "#FF7E00"),
    (301, 400, "Very Poor", "#FF0000"),
    (401, 500, "Severe", "#8F3F97"),
]


def get_aqi_category(aqi: float) -> tuple:
    """Get AQI category name and color."""
    for low, high, name, color in AQI_CATEGORIES:
        if low <= aqi <= high:
            return name, color
    return "Severe", "#8F3F97"
