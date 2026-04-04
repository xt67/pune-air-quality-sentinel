"""Data fetching functions for AQI and weather data sources."""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for cached data BEFORE importing kaggle
    cache_file = output_path / "pune_aqi_kaggle.parquet"
    if cache_file.exists():
        logger.info(f"Loading cached Kaggle data from {cache_file}")
        return pd.read_parquet(cache_file)
    
    # Only import kaggle if we need to download
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle API not installed. Run: pip install kaggle")
        raise
    
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
    
    # Filter for Pune (case-insensitive)
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


def fetch_openaq(
    city: str = "Pune",
    country: str = "IN",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    parameters: Optional[List[str]] = None
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


def fetch_open_meteo_weather(
    nodes_file: str = "data/graph/node_coords.json",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    output_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Fetch and combine weather data for all nodes into a single DataFrame.
    
    This is a convenience wrapper around fetch_weather_all_nodes.
    
    Args:
        nodes_file: Path to node coordinates JSON
        start_date: Start date
        end_date: End date
        output_dir: Output directory for cached data
        
    Returns:
        Combined DataFrame with all nodes' weather data
    """
    weather_dict = fetch_weather_all_nodes(
        nodes_file=nodes_file,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    # Combine all DataFrames
    dfs = list(weather_dict.values())
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined weather data: {len(combined)} rows for {len(weather_dict)} nodes")
    return combined


def fetch_openaq_pune(
    output_dir: str = "data/raw"
) -> pd.DataFrame:
    """
    Fetch OpenAQ data for Pune with caching.
    
    Args:
        output_dir: Output directory for cached data
        
    Returns:
        DataFrame with Pune OpenAQ data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cache_file = output_path / "openaq_pune.parquet"
    
    if cache_file.exists():
        logger.info(f"Loading cached OpenAQ data from {cache_file}")
        return pd.read_parquet(cache_file)
    
    df = fetch_openaq(city="Pune", country="IN")
    
    if not df.empty:
        df.to_parquet(cache_file, index=False)
        logger.info(f"Cached OpenAQ data to {cache_file}")
    
    return df
