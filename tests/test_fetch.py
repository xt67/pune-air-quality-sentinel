"""Unit tests for data fetch functions."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd


@pytest.fixture
def mock_open_meteo_response():
    """Mock Open-Meteo API response."""
    return {
        "hourly": {
            "time": ["2022-01-01T00:00", "2022-01-01T01:00", "2022-01-01T02:00"],
            "temperature_2m": [15.0, 14.5, 14.0],
            "relative_humidity_2m": [80, 82, 85],
            "precipitation": [0.0, 0.0, 0.1],
            "wind_speed_10m": [5.0, 4.5, 5.2],
            "wind_direction_10m": [180, 175, 185],
            "pressure_msl": [1013.0, 1013.5, 1014.0]
        }
    }


@pytest.fixture
def mock_openaq_locations_response():
    """Mock OpenAQ locations response."""
    return {
        "results": [
            {"id": 123, "name": "Pune Station 1"},
            {"id": 456, "name": "Pune Station 2"}
        ]
    }


@pytest.fixture
def mock_openaq_measurements_response():
    """Mock OpenAQ measurements response."""
    return {
        "results": [
            {
                "date": {"utc": "2022-01-01T00:00:00Z"},
                "parameter": "pm25",
                "value": 45.0,
                "unit": "µg/m³"
            },
            {
                "date": {"utc": "2022-01-01T01:00:00Z"},
                "parameter": "pm25",
                "value": 48.0,
                "unit": "µg/m³"
            }
        ]
    }


@pytest.fixture
def mock_node_coords():
    """Mock node coordinates."""
    return {
        "nodes": [
            {"id": "N01", "name": "Shivajinagar", "lat": 18.5308, "lon": 73.8475},
            {"id": "N02", "name": "Hinjewadi", "lat": 18.5912, "lon": 73.7389}
        ],
        "metadata": {"city": "Pune"}
    }


class TestCreateRetrySession:
    """Tests for create_retry_session function."""
    
    def test_creates_session_with_retry(self):
        """Test that session is created with retry adapter."""
        from src.data.fetch import create_retry_session
        
        session = create_retry_session(retries=3, backoff_factor=1.0)
        
        assert session is not None
        assert hasattr(session, 'get')
        # Check adapters are mounted
        assert 'http://' in session.adapters
        assert 'https://' in session.adapters


class TestFetchWeather:
    """Tests for Open-Meteo weather fetch."""
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_weather_success(self, mock_session_factory, mock_open_meteo_response):
        """Test successful weather data fetch."""
        from src.data.fetch import fetch_weather
        
        # Setup mock
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_open_meteo_response
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_factory.return_value = mock_session
        
        # Call function
        df = fetch_weather(
            lat=18.5308,
            lon=73.8475,
            start_date="2022-01-01",
            end_date="2022-01-01"
        )
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "timestamp" in df.columns
        assert "temperature" in df.columns
        assert "wind_speed" in df.columns
        assert "humidity" in df.columns
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_weather_calls_correct_url(self, mock_session_factory, mock_open_meteo_response):
        """Test that correct API URL is called."""
        from src.data.fetch import fetch_weather
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = mock_open_meteo_response
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_factory.return_value = mock_session
        
        fetch_weather(lat=18.5, lon=73.8, start_date="2022-01-01", end_date="2022-01-01")
        
        # Check URL contains open-meteo
        call_args = mock_session.get.call_args
        assert "archive-api.open-meteo.com" in call_args[0][0]


class TestFetchWeatherAllNodes:
    """Tests for fetching weather for all nodes."""
    
    @patch('src.data.fetch.fetch_weather')
    def test_fetch_weather_all_nodes(self, mock_fetch, mock_node_coords, tmp_path):
        """Test fetching weather for all nodes."""
        from src.data.fetch import fetch_weather_all_nodes
        
        # Setup mock - return DataFrame with all required columns
        mock_df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-01 00:00:00", "2022-01-01 01:00:00"]),
            "temperature": [15.0, 14.5],
            "humidity": [80, 82],
            "precipitation": [0.0, 0.0],
            "wind_speed": [5.0, 4.5],
            "wind_direction": [180, 175],
            "pressure": [1013.0, 1013.5]
        })
        mock_fetch.return_value = mock_df.copy()
        
        # Create temp nodes file
        nodes_file = tmp_path / "node_coords.json"
        with open(nodes_file, "w") as f:
            json.dump(mock_node_coords, f)
        
        # Call function
        result = fetch_weather_all_nodes(
            nodes_file=str(nodes_file),
            start_date="2022-01-01",
            end_date="2022-01-01",
            output_dir=str(tmp_path)
        )
        
        # Should have data for each node
        assert len(result) == 2
        assert "N01" in result
        assert "N02" in result


class TestFetchOpenAQ:
    """Tests for OpenAQ fetch."""
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_openaq_no_locations(self, mock_session_factory):
        """Test handling when no locations found."""
        from src.data.fetch import fetch_openaq
        
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_factory.return_value = mock_session
        
        df = fetch_openaq(city="Pune", country="IN")
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_openaq_api_error(self, mock_session_factory):
        """Test handling API errors gracefully."""
        from src.data.fetch import fetch_openaq
        import requests
        
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.RequestException("API Error")
        mock_session_factory.return_value = mock_session
        
        df = fetch_openaq(city="Pune", country="IN")
        
        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestFetchKaggleAQI:
    """Tests for Kaggle AQI fetch."""
    
    def test_fetch_kaggle_loads_cached(self, tmp_path):
        """Test that cached data is loaded if exists."""
        from src.data.fetch import fetch_kaggle_aqi
        
        # Create cached file with proper path
        cache_file = tmp_path / "pune_aqi_kaggle.parquet"
        test_df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-01"]),
            "pm25": [50.0],
            "aqi": [100]
        })
        test_df.to_parquet(str(cache_file), index=False)
        
        # Should load from cache
        result = fetch_kaggle_aqi(output_dir=str(tmp_path))
        
        assert len(result) == 1
        assert result["pm25"].iloc[0] == 50.0


class TestIntegration:
    """Integration tests (may require network)."""
    
    @pytest.mark.skip(reason="Requires network access")
    def test_real_open_meteo_fetch(self):
        """Test real Open-Meteo API call."""
        from src.data.fetch import fetch_weather
        
        df = fetch_weather(
            lat=18.5308,
            lon=73.8475,
            start_date="2024-01-01",
            end_date="2024-01-02"
        )
        
        assert not df.empty
        assert "temperature" in df.columns
