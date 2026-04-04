"""IoT sensor simulation for Pune air quality grid."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Node-specific AQI biases (from PRD)
# Positive = higher pollution, negative = cleaner
NODE_BIASES = {
    "N01": 0,     # Shivajinagar - reference station
    "N02": 15,    # Hinjewadi IT Park - traffic + construction
    "N03": 25,    # Pimpri-Chinchwad MIDC - heavy industry
    "N04": 5,     # Hadapsar - mixed
    "N05": 8,     # Katraj - dense residential
    "N06": 10,    # Deccan Gymkhana - vehicular density
    "N07": 12,    # Viman Nagar - airport NOx
    "N08": -10,   # Kothrud - cleaner residential
    "N09": 20,    # Talegaon MIDC - outer industrial
    "N10": 3,     # Mundhwa - receives city drift
}

# Pune seasonal patterns
MONSOON_MONTHS = [7, 8, 9]  # July-September
WINTER_MONTHS = [11, 12, 1, 2]  # Nov-Feb (higher pollution)

# Diwali dates (approximate)
DIWALI_DATES = [
    "2022-10-24",
    "2023-11-12",
    "2024-11-01",
]


def load_node_coords(filepath: str = "data/graph/node_coords.json") -> Dict:
    """Load node coordinates from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def generate_base_signal(
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    base_aqi: float = 100.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate base AQI signal with temporal patterns.
    
    This creates a city-wide base signal that individual nodes modify.
    
    Args:
        start_date: Start of simulation period
        end_date: End of simulation period
        base_aqi: Mean AQI value for the city
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with timestamp and base_aqi columns
    """
    set_seed(seed)
    
    # Create hourly timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq="h")
    n_hours = len(timestamps)
    
    logger.info(f"Generating {n_hours} hours of base signal ({start_date} to {end_date})")
    
    # Initialize with base AQI + small random walk
    random_walk = np.cumsum(np.random.randn(n_hours) * 0.5)
    random_walk = random_walk - random_walk.mean()  # Center around 0
    random_walk = np.clip(random_walk, -20, 20)  # Limit drift
    
    aqi = base_aqi + random_walk
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "base_aqi": aqi
    })
    
    return df


def apply_hourly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rush-hour patterns to AQI.
    
    Rush hours (7-9 AM, 5-8 PM): +15 AQI
    Night hours (11 PM - 5 AM): -10 AQI (less traffic)
    """
    df = df.copy()
    hour = df["timestamp"].dt.hour
    
    # Morning rush: 7-9 AM
    morning_rush = (hour >= 7) & (hour < 9)
    df.loc[morning_rush, "base_aqi"] += 15
    
    # Evening rush: 5-8 PM
    evening_rush = (hour >= 17) & (hour < 20)
    df.loc[evening_rush, "base_aqi"] += 15
    
    # Night hours: 11 PM - 5 AM (cleaner)
    night = (hour >= 23) | (hour < 5)
    df.loc[night, "base_aqi"] -= 10
    
    logger.debug("Applied hourly rush-hour patterns")
    return df


def apply_seasonal_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply seasonal patterns to AQI.
    
    - Monsoon (Jul-Sep): -20 AQI (rain washes pollutants)
    - Winter (Nov-Feb): +15 AQI (temperature inversion)
    """
    df = df.copy()
    month = df["timestamp"].dt.month
    
    # Monsoon dip
    monsoon = month.isin(MONSOON_MONTHS)
    df.loc[monsoon, "base_aqi"] -= 20
    
    # Winter spike (temperature inversion traps pollution)
    winter = month.isin(WINTER_MONTHS)
    df.loc[winter, "base_aqi"] += 15
    
    logger.debug("Applied seasonal patterns (monsoon dip, winter spike)")
    return df


def apply_diwali_spike(df: pd.DataFrame, spike_aqi: float = 80) -> pd.DataFrame:
    """
    Apply Diwali week pollution spike.
    
    Diwali fireworks cause massive PM2.5/PM10 increase.
    Effect lasts ~5 days (2 days before, Diwali day, 2 days after).
    """
    df = df.copy()
    
    for diwali_str in DIWALI_DATES:
        diwali_date = pd.Timestamp(diwali_str)
        start = diwali_date - pd.Timedelta(days=2)
        end = diwali_date + pd.Timedelta(days=2)
        
        mask = (df["timestamp"].dt.date >= start.date()) & \
               (df["timestamp"].dt.date <= end.date())
        
        # Diwali day itself gets highest spike
        diwali_day = df["timestamp"].dt.date == diwali_date.date()
        df.loc[mask, "base_aqi"] += spike_aqi * 0.6
        df.loc[diwali_day, "base_aqi"] += spike_aqi * 0.4  # Additional spike
        
        logger.debug(f"Applied Diwali spike around {diwali_str}")
    
    return df


def generate_node_series(
    base_df: pd.DataFrame,
    node_id: str,
    noise_std: float = 5.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate time series for a specific node.
    
    Applies node-specific bias and Gaussian noise.
    
    Args:
        base_df: Base signal DataFrame
        node_id: Node identifier (N01-N10)
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed
        
    Returns:
        DataFrame with node-specific AQI time series
    """
    set_seed(seed + hash(node_id) % 1000)  # Node-specific seed
    
    df = base_df.copy()
    
    # Apply node-specific bias
    bias = NODE_BIASES.get(node_id, 0)
    df["aqi"] = df["base_aqi"] + bias
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, len(df))
    df["aqi"] += noise
    
    # Clip to valid AQI range
    df["aqi"] = df["aqi"].clip(0, 500)
    
    # Add node identifier
    df["node_id"] = node_id
    
    # Generate synthetic pollutant values from AQI
    # Using reverse CPCB breakpoints (approximate)
    df["pm25"] = df["aqi"] * 0.6 + np.random.normal(0, 3, len(df))
    df["pm10"] = df["aqi"] * 0.8 + np.random.normal(0, 5, len(df))
    df["no2"] = df["aqi"] * 0.3 + np.random.normal(0, 2, len(df))
    df["so2"] = df["aqi"] * 0.15 + np.random.normal(0, 1, len(df))
    df["co"] = df["aqi"] * 0.01 + np.random.normal(0, 0.1, len(df))
    df["o3"] = df["aqi"] * 0.4 + np.random.normal(0, 3, len(df))
    
    # Clip pollutants to non-negative
    for col in ["pm25", "pm10", "no2", "so2", "co", "o3"]:
        df[col] = df[col].clip(lower=0)
    
    # Drop base_aqi column
    df = df.drop(columns=["base_aqi"])
    
    # Reorder columns
    cols = ["timestamp", "node_id", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3"]
    df = df[cols]
    
    logger.info(f"Generated {len(df)} hours for node {node_id} (bias: {bias:+d})")
    
    return df


def generate_all_nodes(
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    base_aqi: float = 100.0,
    noise_std: float = 5.0,
    seed: int = 42,
    output_dir: str = "data/simulated"
) -> Dict[str, pd.DataFrame]:
    """
    Generate IoT simulation data for all 10 Pune nodes.
    
    Args:
        start_date: Start of simulation period
        end_date: End of simulation period
        base_aqi: Mean AQI for the city
        noise_std: Gaussian noise std
        seed: Random seed
        output_dir: Directory to save CSV files
        
    Returns:
        Dictionary mapping node_id to DataFrame
    """
    set_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate base signal
    base_df = generate_base_signal(start_date, end_date, base_aqi, seed)
    
    # Apply temporal patterns
    base_df = apply_hourly_pattern(base_df)
    base_df = apply_seasonal_pattern(base_df)
    base_df = apply_diwali_spike(base_df)
    
    # Generate each node's time series
    node_data = {}
    
    for node_id in NODE_BIASES.keys():
        df = generate_node_series(base_df, node_id, noise_std, seed)
        node_data[node_id] = df
        
        # Save to CSV
        csv_path = output_path / f"node_{node_id}_simulated.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")
    
    logger.info(f"Generated IoT simulation for {len(node_data)} nodes")
    
    return node_data


def validate_simulation(node_data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate generated simulation data.
    
    Checks:
    - All 10 nodes present
    - No NaN values
    - AQI in [0, 500]
    - Identical timestamps across nodes
    
    Args:
        node_data: Dictionary of node DataFrames
        
    Returns:
        True if validation passes
    """
    errors = []
    
    # Check all 10 nodes
    if len(node_data) != 10:
        errors.append(f"Expected 10 nodes, got {len(node_data)}")
    
    expected_nodes = set(NODE_BIASES.keys())
    actual_nodes = set(node_data.keys())
    missing = expected_nodes - actual_nodes
    if missing:
        errors.append(f"Missing nodes: {missing}")
    
    # Check each node's data
    reference_timestamps = None
    for node_id, df in node_data.items():
        # Check for NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            errors.append(f"Node {node_id} has {nan_count} NaN values")
        
        # Check AQI range
        if df["aqi"].min() < 0 or df["aqi"].max() > 500:
            errors.append(f"Node {node_id} AQI out of range: "
                         f"[{df['aqi'].min():.1f}, {df['aqi'].max():.1f}]")
        
        # Check timestamps match
        if reference_timestamps is None:
            reference_timestamps = df["timestamp"].tolist()
        elif df["timestamp"].tolist() != reference_timestamps:
            errors.append(f"Node {node_id} has mismatched timestamps")
    
    if errors:
        for error in errors:
            logger.error(f"Validation failed: {error}")
        return False
    
    logger.info("IoT simulation validation passed ✓")
    return True


def run_iot_simulation(
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42,
    output_dir: str = "data/simulated"
) -> Dict[str, pd.DataFrame]:
    """
    Run complete IoT simulation pipeline.
    
    Args:
        start_date: Start date
        end_date: End date
        seed: Random seed
        output_dir: Output directory
        
    Returns:
        Dictionary of node DataFrames
    """
    logger.info("Starting IoT simulation...")
    
    node_data = generate_all_nodes(
        start_date=start_date,
        end_date=end_date,
        seed=seed,
        output_dir=output_dir
    )
    
    # Validate
    valid = validate_simulation(node_data)
    if not valid:
        raise ValueError("IoT simulation validation failed")
    
    logger.info(f"IoT simulation complete: {len(node_data)} nodes, "
                f"{len(list(node_data.values())[0])} hours each")
    
    return node_data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_iot_simulation()
