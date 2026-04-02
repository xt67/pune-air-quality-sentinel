---
phase: 1
plan: 2
title: "Data Fetch Modules"
wave: 2
depends_on: [1]
files_modified:
  - src/data/fetch.py
  - tests/test_fetch.py
requirements_addressed: [DATA-01, DATA-02, DATA-03]
autonomous: true
---

# Plan 2: Data Fetch Modules

<objective>
Implement data fetching functions for Kaggle AQI dataset, Open-Meteo weather API, and OpenAQ pollution API with retry logic and error handling.
</objective>

<must_haves>
- Kaggle AQI dataset download with Pune filtering
- Open-Meteo historical weather fetch for all 10 nodes
- OpenAQ pollution data fetch as backup
- Retry logic with exponential backoff
- Proper error handling and logging
</must_haves>

## Tasks

<task id="2.1">
<title>Implement Kaggle AQI data fetch</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-RESEARCH.md (Kaggle API patterns)
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (data source decisions)
</read_first>
<action>
Create src/data/fetch.py with Kaggle download function:

```python
"""Data fetching functions for AQI and weather data sources."""
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def create_retry_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: tuple = (429, 500, 502, 503, 504)
) -> requests.Session:
    """
    Create a requests session with retry logic.
    
    Args:
        retries: Number of retries
        backoff_factor: Multiplier for exponential backoff
        status_forcelist: HTTP status codes to retry
        
    Returns:
        Configured requests session
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def fetch_kaggle_aqi(
    output_dir: str = "data/raw",
    dataset_name: str = "rohanrao/air-quality-data-in-india"
) -> pd.DataFrame:
    """
    Download and filter Kaggle Air Quality India dataset for Pune.
    
    Args:
        output_dir: Directory to save raw data
        dataset_name: Kaggle dataset identifier
        
    Returns:
        DataFrame with Pune AQI data
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle API not installed. Run: pip install kaggle")
        raise
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for cached data
    cache_file = output_path / "pune_aqi_kaggle.parquet"
    if cache_file.exists():
        logger.info(f"Loading cached Kaggle data from {cache_file}")
        return pd.read_parquet(cache_file)
    
    # Download dataset
    logger.info(f"Downloading Kaggle dataset: {dataset_name}")
    api = KaggleApi()
    api.authenticate()
    
    api.dataset_download_files(
        dataset_name,
        path=str(output_path),
        unzip=True
    )
    
    # Find and load the city_day.csv file
    csv_files = list(output_path.glob("*.csv"))
    city_day_file = None
    for f in csv_files:
        if "city_day" in f.name.lower():
            city_day_file = f
            break
    
    if city_day_file is None:
        raise FileNotFoundError(f"city_day.csv not found in {output_path}")
    
    logger.info(f"Loading data from {city_day_file}")
    df = pd.read_csv(city_day_file)
    
    # Filter for Pune
    df_pune = df[df["City"].str.lower() == "pune"].copy()
    logger.info(f"Filtered {len(df_pune)} Pune records from {len(df)} total")
    
    # Parse dates
    df_pune["Date"] = pd.to_datetime(df_pune["Date"])
    
    # Rename columns for consistency
    df_pune = df_pune.rename(columns={
        "Date": "timestamp",
        "PM2.5": "pm25",
        "PM10": "pm10",
        "NO2": "no2",
        "SO2": "so2",
        "CO": "co",
        "O3": "o3",
        "AQI": "aqi"
    })
    
    # Select relevant columns
    cols = ["timestamp", "pm25", "pm10", "no2", "so2", "co", "o3", "aqi"]
    df_pune = df_pune[[c for c in cols if c in df_pune.columns]]
    
    # Cache for future use
    df_pune.to_parquet(cache_file, index=False)
    logger.info(f"Cached Pune AQI data to {cache_file}")
    
    return df_pune
```
</action>
<acceptance_criteria>
- `grep -l "def fetch_kaggle_aqi" src/data/fetch.py` returns file path
- `grep -l "KaggleApi" src/data/fetch.py` returns file path
- `grep -l "City.*pune" src/data/fetch.py` returns file path (case-insensitive filter)
</acceptance_criteria>
</task>

<task id="2.2">
<title>Implement Open-Meteo weather fetch</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-RESEARCH.md (Open-Meteo API structure)
- data/graph/node_coords.json (node coordinates)
</read_first>
<action>
Add Open-Meteo fetch function to src/data/fetch.py:

```python
def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    timezone: str = "Asia/Kolkata"
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.
    
    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timezone: Timezone for data
        
    Returns:
        DataFrame with hourly weather data
    """
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_direction_10m,pressure_msl",
        "timezone": timezone
    }
    
    session = create_retry_session()
    
    logger.info(f"Fetching weather for ({lat}, {lon}) from {start_date} to {end_date}")
    response = session.get(base_url, params=params, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    
    # Parse hourly data
    hourly = data.get("hourly", {})
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly.get("time", [])),
        "temperature": hourly.get("temperature_2m", []),
        "humidity": hourly.get("relative_humidity_2m", []),
        "precipitation": hourly.get("precipitation", []),
        "wind_speed": hourly.get("wind_speed_10m", []),
        "wind_direction": hourly.get("wind_direction_10m", []),
        "pressure": hourly.get("pressure_msl", [])
    })
    
    logger.info(f"Fetched {len(df)} hourly weather records")
    return df


def fetch_weather_all_nodes(
    nodes_file: str = "data/graph/node_coords.json",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    output_dir: str = "data/raw"
) -> Dict[str, pd.DataFrame]:
    """
    Fetch weather data for all Pune nodes.
    
    Args:
        nodes_file: Path to node coordinates JSON
        start_date: Start date
        end_date: End date
        output_dir: Output directory for cached data
        
    Returns:
        Dictionary mapping node_id to weather DataFrame
    """
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(nodes_file, "r") as f:
        nodes_data = json.load(f)
    
    weather_data = {}
    
    for node in nodes_data["nodes"]:
        node_id = node["id"]
        cache_file = output_path / f"weather_{node_id}.parquet"
        
        if cache_file.exists():
            logger.info(f"Loading cached weather for {node_id}")
            weather_data[node_id] = pd.read_parquet(cache_file)
        else:
            df = fetch_weather(
                lat=node["lat"],
                lon=node["lon"],
                start_date=start_date,
                end_date=end_date
            )
            df["node_id"] = node_id
            df.to_parquet(cache_file, index=False)
            weather_data[node_id] = df
            
            # Small delay to be nice to the API
            time.sleep(0.5)
    
    logger.info(f"Fetched weather for {len(weather_data)} nodes")
    return weather_data
```
</action>
<acceptance_criteria>
- `grep -l "def fetch_weather" src/data/fetch.py` returns file path
- `grep -l "archive-api.open-meteo.com" src/data/fetch.py` returns file path
- `grep -l "wind_speed_10m" src/data/fetch.py` returns file path
- `grep -l "def fetch_weather_all_nodes" src/data/fetch.py` returns file path
</acceptance_criteria>
</task>

<task id="2.3">
<title>Implement OpenAQ pollution fetch</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-RESEARCH.md (OpenAQ API structure)
</read_first>
<action>
Add OpenAQ fetch function to src/data/fetch.py:

```python
def fetch_openaq(
    city: str = "Pune",
    country: str = "IN",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    parameters: List[str] = None
) -> pd.DataFrame:
    """
    Fetch pollution data from OpenAQ API.
    
    Args:
        city: City name
        country: Country code (ISO)
        start_date: Start date
        end_date: End date
        parameters: List of pollutant parameters to fetch
        
    Returns:
        DataFrame with pollution measurements
    """
    if parameters is None:
        parameters = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    
    base_url = "https://api.openaq.org/v3"
    session = create_retry_session()
    
    # First, get locations in the city
    logger.info(f"Fetching OpenAQ locations for {city}, {country}")
    
    locations_url = f"{base_url}/locations"
    params = {
        "city": city,
        "country": country,
        "limit": 100
    }
    
    try:
        response = session.get(locations_url, params=params, timeout=30)
        response.raise_for_status()
        locations_data = response.json()
    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenAQ API error: {e}. Returning empty DataFrame.")
        return pd.DataFrame()
    
    results = locations_data.get("results", [])
    if not results:
        logger.warning(f"No OpenAQ locations found for {city}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(results)} OpenAQ locations in {city}")
    
    # Fetch measurements for each location
    all_measurements = []
    
    for location in results[:5]:  # Limit to first 5 locations
        location_id = location.get("id")
        location_name = location.get("name", "Unknown")
        
        logger.info(f"Fetching measurements for {location_name} (ID: {location_id})")
        
        measurements_url = f"{base_url}/measurements"
        params = {
            "location_id": location_id,
            "date_from": start_date,
            "date_to": end_date,
            "limit": 10000
        }
        
        try:
            response = session.get(measurements_url, params=params, timeout=60)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                logger.warning(f"Rate limited. Waiting {retry_after}s")
                time.sleep(retry_after)
                continue
            
            response.raise_for_status()
            measurements = response.json().get("results", [])
            
            for m in measurements:
                all_measurements.append({
                    "timestamp": m.get("date", {}).get("utc"),
                    "location": location_name,
                    "parameter": m.get("parameter"),
                    "value": m.get("value"),
                    "unit": m.get("unit")
                })
            
            time.sleep(1)  # Rate limiting
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching measurements for {location_name}: {e}")
            continue
    
    if not all_measurements:
        logger.warning("No measurements retrieved from OpenAQ")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_measurements)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Pivot to wide format
    df_wide = df.pivot_table(
        index=["timestamp", "location"],
        columns="parameter",
        values="value",
        aggfunc="mean"
    ).reset_index()
    
    logger.info(f"Fetched {len(df_wide)} OpenAQ records")
    return df_wide
```
</action>
<acceptance_criteria>
- `grep -l "def fetch_openaq" src/data/fetch.py` returns file path
- `grep -l "api.openaq.org" src/data/fetch.py` returns file path
- `grep -l "Retry-After" src/data/fetch.py` returns file path
</acceptance_criteria>
</task>

<task id="2.4">
<title>Create test_fetch.py unit tests</title>
<read_first>
- src/data/fetch.py (functions to test)
- .planning/phases/01-data-pipeline-iot-simulation/01-RESEARCH.md (testing patterns)
</read_first>
<action>
Create tests/test_fetch.py with mocked API tests:

```python
"""Unit tests for data fetch functions."""
import json
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd


@pytest.fixture
def mock_open_meteo_response():
    """Mock Open-Meteo API response."""
    return {
        "hourly": {
            "time": [
                "2023-01-01T00:00",
                "2023-01-01T01:00",
                "2023-01-01T02:00"
            ],
            "temperature_2m": [20.5, 21.0, 21.5],
            "relative_humidity_2m": [60, 62, 65],
            "precipitation": [0.0, 0.0, 0.1],
            "wind_speed_10m": [5.2, 5.5, 5.0],
            "wind_direction_10m": [180, 185, 190],
            "pressure_msl": [1013.0, 1012.8, 1012.5]
        }
    }


@pytest.fixture
def mock_openaq_locations_response():
    """Mock OpenAQ locations response."""
    return {
        "results": [
            {"id": 12345, "name": "Shivajinagar"},
            {"id": 12346, "name": "Pashan"}
        ]
    }


@pytest.fixture
def mock_openaq_measurements_response():
    """Mock OpenAQ measurements response."""
    return {
        "results": [
            {
                "date": {"utc": "2023-01-01T00:00:00Z"},
                "parameter": "pm25",
                "value": 45.5,
                "unit": "µg/m³"
            },
            {
                "date": {"utc": "2023-01-01T01:00:00Z"},
                "parameter": "pm25",
                "value": 48.2,
                "unit": "µg/m³"
            }
        ]
    }


class TestCreateRetrySession:
    """Tests for create_retry_session function."""
    
    def test_creates_session(self):
        from src.data.fetch import create_retry_session
        session = create_retry_session()
        assert session is not None
        
    def test_custom_retries(self):
        from src.data.fetch import create_retry_session
        session = create_retry_session(retries=5)
        assert session is not None


class TestFetchWeather:
    """Tests for fetch_weather function."""
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_weather_success(self, mock_session_func, mock_open_meteo_response):
        from src.data.fetch import fetch_weather
        
        mock_response = MagicMock()
        mock_response.json.return_value = mock_open_meteo_response
        mock_response.raise_for_status = MagicMock()
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_func.return_value = mock_session
        
        df = fetch_weather(18.52, 73.85, "2023-01-01", "2023-01-02")
        
        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "temperature" in df.columns
        assert "wind_speed" in df.columns
        assert len(df) == 3
        
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_weather_columns(self, mock_session_func, mock_open_meteo_response):
        from src.data.fetch import fetch_weather
        
        mock_response = MagicMock()
        mock_response.json.return_value = mock_open_meteo_response
        mock_response.raise_for_status = MagicMock()
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_func.return_value = mock_session
        
        df = fetch_weather(18.52, 73.85, "2023-01-01", "2023-01-02")
        
        expected_cols = ["timestamp", "temperature", "humidity", "precipitation", 
                        "wind_speed", "wind_direction", "pressure"]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


class TestFetchOpenAQ:
    """Tests for fetch_openaq function."""
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_openaq_no_locations(self, mock_session_func):
        from src.data.fetch import fetch_openaq
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = MagicMock()
        mock_response.status_code = 200
        
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_func.return_value = mock_session
        
        df = fetch_openaq()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @patch('src.data.fetch.create_retry_session')
    def test_fetch_openaq_handles_rate_limit(self, mock_session_func):
        from src.data.fetch import fetch_openaq
        
        # First call returns locations, second returns 429
        mock_response_locations = MagicMock()
        mock_response_locations.json.return_value = {"results": [{"id": 1, "name": "Test"}]}
        mock_response_locations.raise_for_status = MagicMock()
        mock_response_locations.status_code = 200
        
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "1"}
        
        mock_session = MagicMock()
        mock_session.get.side_effect = [mock_response_locations, mock_response_429]
        mock_session_func.return_value = mock_session
        
        df = fetch_openaq()
        
        # Should handle 429 gracefully
        assert isinstance(df, pd.DataFrame)


class TestFetchWeatherAllNodes:
    """Tests for fetch_weather_all_nodes function."""
    
    @patch('src.data.fetch.fetch_weather')
    def test_fetch_all_nodes(self, mock_fetch_weather, tmp_path):
        from src.data.fetch import fetch_weather_all_nodes
        
        # Create temp node coords file
        nodes = {
            "nodes": [
                {"id": "N01", "name": "Test1", "lat": 18.5, "lon": 73.8, "characteristic": "test"},
                {"id": "N02", "name": "Test2", "lat": 18.6, "lon": 73.9, "characteristic": "test"}
            ],
            "metadata": {}
        }
        nodes_file = tmp_path / "nodes.json"
        with open(nodes_file, "w") as f:
            json.dump(nodes, f)
        
        # Mock weather fetch
        mock_df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="H"),
            "temperature": [20, 21, 22],
            "humidity": [60, 62, 65],
            "precipitation": [0, 0, 0],
            "wind_speed": [5, 5, 5],
            "wind_direction": [180, 180, 180],
            "pressure": [1013, 1013, 1013]
        })
        mock_fetch_weather.return_value = mock_df
        
        result = fetch_weather_all_nodes(
            nodes_file=str(nodes_file),
            output_dir=str(tmp_path / "raw")
        )
        
        assert len(result) == 2
        assert "N01" in result
        assert "N02" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```
</action>
<acceptance_criteria>
- `Test-Path tests/test_fetch.py` returns True
- `grep -l "def test_fetch_weather_success" tests/test_fetch.py` returns file path
- `grep -l "def test_fetch_openaq_handles_rate_limit" tests/test_fetch.py` returns file path
- `python -m pytest tests/test_fetch.py -v --collect-only` shows at least 5 test items
</acceptance_criteria>
</task>

## Verification

```powershell
# Verify imports work
python -c "from src.data.fetch import fetch_kaggle_aqi, fetch_weather, fetch_openaq, fetch_weather_all_nodes; print('All fetch functions imported')"

# Run tests (with mocked APIs)
python -m pytest tests/test_fetch.py -v

# Verify test count
python -m pytest tests/test_fetch.py --collect-only -q | Select-String "test"
```
