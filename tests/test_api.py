"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_api_info(self):
        """Root returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestHealthEndpoint:
    """Tests for health check endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint returns 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
    
    def test_health_contains_required_fields(self):
        """Health response has required fields."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "timestamp" in data
    
    def test_health_status_is_healthy(self):
        """Health status is healthy."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestStationsEndpoint:
    """Tests for stations endpoints."""
    
    def test_list_stations_returns_200(self):
        """List stations returns 200."""
        response = client.get("/api/v1/stations")
        assert response.status_code == 200
    
    def test_list_stations_returns_array(self):
        """List stations returns array of stations."""
        response = client.get("/api/v1/stations")
        data = response.json()
        assert "stations" in data
        assert "count" in data
        assert isinstance(data["stations"], list)
        assert len(data["stations"]) == data["count"]
    
    def test_station_has_required_fields(self):
        """Each station has required fields."""
        response = client.get("/api/v1/stations")
        data = response.json()
        for station in data["stations"]:
            assert "station_id" in station
            assert "name" in station
            assert "latitude" in station
            assert "longitude" in station
            assert "city" in station
    
    def test_get_station_by_id(self):
        """Get specific station by ID."""
        response = client.get("/api/v1/stations/MH020")
        assert response.status_code == 200
        data = response.json()
        assert data["station_id"] == "MH020"
        assert data["city"] == "Pune"
    
    def test_get_station_not_found(self):
        """Unknown station returns 404."""
        response = client.get("/api/v1/stations/INVALID")
        assert response.status_code == 404


class TestPredictEndpoint:
    """Tests for prediction endpoints."""
    
    def test_predict_post_returns_200(self):
        """POST predict returns 200."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020", "model_type": "lstm", "horizon": 24}
        )
        assert response.status_code == 200
    
    def test_predict_get_returns_200(self):
        """GET predict returns 200."""
        response = client.get("/api/v1/predict/MH020?model_type=lstm&horizon=24")
        assert response.status_code == 200
    
    def test_predict_returns_predictions_array(self):
        """Predict returns array of predictions."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020", "horizon": 12}
        )
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 12
    
    def test_predict_contains_required_fields(self):
        """Prediction response has required fields."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020"}
        )
        data = response.json()
        assert "station_id" in data
        assert "model_type" in data
        assert "horizon" in data
        assert "predictions" in data
        assert "generated_at" in data
    
    def test_prediction_hour_has_required_fields(self):
        """Each hourly prediction has required fields."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020", "horizon": 6}
        )
        data = response.json()
        for pred in data["predictions"]:
            assert "hour" in pred
            assert "timestamp" in pred
            assert "aqi" in pred
            assert "category" in pred
            assert "color" in pred
    
    def test_predict_invalid_station(self):
        """Invalid station returns 404."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "INVALID"}
        )
        assert response.status_code == 404
    
    def test_predict_invalid_model_type(self):
        """Invalid model type returns 400."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020", "model_type": "invalid"}
        )
        assert response.status_code == 400
    
    def test_predict_arima_model(self):
        """ARIMA model prediction works."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020", "model_type": "arima"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["model_type"] == "arima"
    
    def test_predict_different_horizons(self):
        """Different horizons return correct number of predictions."""
        for horizon in [6, 12, 24, 48, 72]:
            response = client.post(
                "/api/v1/predict",
                json={"station_id": "MH020", "horizon": horizon}
            )
            data = response.json()
            assert len(data["predictions"]) == horizon
    
    def test_aqi_values_in_valid_range(self):
        """AQI values are within valid range."""
        response = client.post(
            "/api/v1/predict",
            json={"station_id": "MH020", "horizon": 24}
        )
        data = response.json()
        for pred in data["predictions"]:
            assert 0 <= pred["aqi"] <= 500
