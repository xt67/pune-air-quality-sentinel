"""Shared pytest fixtures for all test modules."""
import json
import tempfile
from pathlib import Path
from typing import Dict, Generator

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def seed() -> int:
    """Fixed random seed for reproducibility."""
    return 42


@pytest.fixture(scope="function")
def tmp_path_func(tmp_path) -> Path:
    """Function-scoped temp path for test isolation."""
    return tmp_path


@pytest.fixture
def sample_aqi_df() -> pd.DataFrame:
    """Sample AQI DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    
    timestamps = pd.date_range("2023-01-01", periods=n_rows, freq="h")
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "node_id": ["N01"] * n_rows,
        "pm25": np.random.uniform(20, 150, n_rows),
        "pm10": np.random.uniform(30, 200, n_rows),
        "no2": np.random.uniform(10, 80, n_rows),
        "so2": np.random.uniform(5, 50, n_rows),
        "co": np.random.uniform(0.5, 5, n_rows),
        "o3": np.random.uniform(20, 100, n_rows),
        "wind_speed": np.random.uniform(1, 15, n_rows),
        "wind_direction": np.random.uniform(0, 360, n_rows),
        "temperature": np.random.uniform(15, 35, n_rows),
        "humidity": np.random.uniform(40, 90, n_rows),
    })


@pytest.fixture
def sample_aqi_df_with_gaps() -> pd.DataFrame:
    """Sample DataFrame with intentional gaps for testing cleaning."""
    np.random.seed(42)
    timestamps = pd.date_range("2023-01-01", periods=50, freq="h")
    values = np.random.uniform(50, 150, 50)
    
    # Inject some NaN values
    values[5:8] = np.nan  # Small gap (3 hours)
    values[20:30] = np.nan  # Large gap (10 hours)
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "pm25": values,
    })


@pytest.fixture
def sample_multi_node_df() -> pd.DataFrame:
    """Sample DataFrame with multiple nodes for testing."""
    np.random.seed(42)
    n_rows_per_node = 48  # 2 days of hourly data
    nodes = ["N01", "N02", "N03"]
    
    dfs = []
    for node in nodes:
        timestamps = pd.date_range("2023-01-01", periods=n_rows_per_node, freq="h")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "node_id": node,
            "aqi": np.random.uniform(50, 200, n_rows_per_node),
            "pm25": np.random.uniform(20, 150, n_rows_per_node),
        })
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


@pytest.fixture
def node_coords_json(tmp_path) -> Path:
    """Create temporary node coordinates JSON file."""
    coords = {
        "N01": {"name": "Shivajinagar", "lat": 18.5308, "lon": 73.8475},
        "N02": {"name": "Hinjewadi", "lat": 18.5912, "lon": 73.7390},
    }
    
    filepath = tmp_path / "node_coords.json"
    with open(filepath, "w") as f:
        json.dump(coords, f)
    
    return filepath


@pytest.fixture
def mock_config_yaml(tmp_path) -> Path:
    """Create temporary config YAML file."""
    config_content = """
seed: 42
data:
  raw_path: data/raw
  processed_path: data/processed
preprocessing:
  max_gap_hours: 12
  train_ratio: 0.7
  val_ratio: 0.15
"""
    
    filepath = tmp_path / "config.yaml"
    filepath.write_text(config_content)
    
    return filepath


# Marker for slow tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


# Auto-skip network tests unless explicitly requested
def pytest_collection_modifyitems(config, items):
    """Skip network tests by default."""
    if not config.getoption("--run-network", default=False):
        skip_network = pytest.mark.skip(reason="network tests disabled by default")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require network access"
    )
