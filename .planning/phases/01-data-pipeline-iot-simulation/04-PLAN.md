---
phase: 1
plan: 4
title: "IoT Sensor Simulation"
wave: 2
depends_on: [3]
files_modified:
  - src/data/iot_sim.py
  - tests/test_iot_sim.py
requirements_addressed: [IOT-01, IOT-02, IOT-03, IOT-04, IOT-05, IOT-06]
autonomous: true
---

# Plan 4: IoT Sensor Simulation

<objective>
Generate synthetic IoT sensor data for 10 Pune neighborhoods with realistic temporal patterns, node-specific biases, and seasonal variations. This simulates a distributed sensor network without physical hardware.
</objective>

<must_haves>
- 2-year hourly time series per node (17,520 rows each)
- 10 nodes with unique biases (Pimpri +25, Kothrud -10, etc.)
- Rush-hour spikes (7-9 AM, 5-8 PM: +15 AQI)
- Diwali week spike (+80 AQI)
- Monsoon dip (July-Sept: -20 AQI)
- Gaussian noise (std=5 AQI)
- Output: data/simulated/node_{id}_simulated.csv
- Validation: identical timestamps, no NaN, AQI in [0, 500]
</must_haves>

## Tasks

<task id="4.1">
<title>Implement IoT sensor simulation module</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (node grid, simulation rules)
- data/graph/node_coords.json (node coordinates)
</read_first>
<action>
Create src/data/iot_sim.py:

```python
"""IoT sensor simulation for Pune air quality grid."""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    timestamps = pd.date_range(start=start_date, end=end_date, freq="H")
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
        df.loc[diwali_day, "base_aqi"] += spike_aqi * 0.4  # Extra on the day
        
        if mask.sum() > 0:
            logger.info(f"Applied Diwali spike for {diwali_str}: {mask.sum()} hours affected")
    
    return df


def apply_weekend_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply weekend pattern (slightly lower AQI due to reduced traffic).
    """
    df = df.copy()
    is_weekend = df["timestamp"].dt.dayofweek >= 5
    df.loc[is_weekend, "base_aqi"] -= 5
    
    logger.debug("Applied weekend pattern (-5 AQI)")
    return df


def generate_node_data(
    base_df: pd.DataFrame,
    node_id: str,
    bias: float,
    noise_std: float = 5.0,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate data for a single node with bias and noise.
    
    Args:
        base_df: DataFrame with base_aqi column
        node_id: Node identifier (e.g., "N01")
        bias: Node-specific AQI bias
        noise_std: Standard deviation of Gaussian noise
        seed: Random seed for noise (different per node)
        
    Returns:
        DataFrame with node-specific AQI values
    """
    if seed is not None:
        np.random.seed(seed)
    
    df = base_df.copy()
    df["node_id"] = node_id
    
    # Apply node-specific bias
    df["aqi_simulated"] = df["base_aqi"] + bias
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, len(df))
    df["aqi_simulated"] += noise
    
    # Clip to valid AQI range
    df["aqi_simulated"] = np.clip(df["aqi_simulated"], 0, 500)
    
    # Derive PM2.5 and PM10 from AQI (approximate inverse mapping)
    # These are rough estimates for simulation purposes
    df["pm25"] = estimate_pm25_from_aqi(df["aqi_simulated"].values)
    df["pm10"] = estimate_pm10_from_aqi(df["aqi_simulated"].values)
    df["no2"] = estimate_no2_from_aqi(df["aqi_simulated"].values)
    
    # Select output columns
    output_cols = ["timestamp", "node_id", "pm25", "pm10", "no2", "aqi_simulated"]
    df = df[output_cols]
    
    # Round to reasonable precision
    df["pm25"] = df["pm25"].round(2)
    df["pm10"] = df["pm10"].round(2)
    df["no2"] = df["no2"].round(2)
    df["aqi_simulated"] = df["aqi_simulated"].round(1)
    
    return df


def estimate_pm25_from_aqi(aqi_values: np.ndarray) -> np.ndarray:
    """Estimate PM2.5 from AQI using inverse CPCB mapping."""
    # Simplified inverse mapping (PM2.5 breakpoints)
    # AQI 0-50: PM2.5 0-30, AQI 51-100: PM2.5 31-60, etc.
    pm25 = np.zeros_like(aqi_values)
    
    for i, aqi in enumerate(aqi_values):
        if aqi <= 50:
            pm25[i] = aqi * 30 / 50
        elif aqi <= 100:
            pm25[i] = 30 + (aqi - 50) * 30 / 50
        elif aqi <= 200:
            pm25[i] = 60 + (aqi - 100) * 30 / 100
        elif aqi <= 300:
            pm25[i] = 90 + (aqi - 200) * 30 / 100
        elif aqi <= 400:
            pm25[i] = 120 + (aqi - 300) * 130 / 100
        else:
            pm25[i] = 250 + (aqi - 400) * 250 / 100
    
    return pm25


def estimate_pm10_from_aqi(aqi_values: np.ndarray) -> np.ndarray:
    """Estimate PM10 from AQI using inverse CPCB mapping."""
    # PM10 has similar shape but higher concentrations
    pm10 = estimate_pm25_from_aqi(aqi_values) * 1.5
    return np.clip(pm10, 0, 600)


def estimate_no2_from_aqi(aqi_values: np.ndarray) -> np.ndarray:
    """Estimate NO2 from AQI using inverse CPCB mapping."""
    # NO2 is typically 40-60% of AQI contribution
    no2 = aqi_values * 0.5
    return np.clip(no2, 0, 500)


def generate_all_nodes(
    output_dir: str = "data/simulated",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42
) -> List[str]:
    """
    Generate simulated data for all 10 Pune nodes.
    
    Args:
        output_dir: Directory to save CSV files
        start_date: Start of simulation period
        end_date: End of simulation period
        seed: Master random seed
        
    Returns:
        List of paths to generated CSV files
    """
    set_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate base signal
    base_df = generate_base_signal(start_date, end_date, base_aqi=100, seed=seed)
    
    # Apply temporal patterns
    base_df = apply_hourly_pattern(base_df)
    base_df = apply_seasonal_pattern(base_df)
    base_df = apply_diwali_spike(base_df)
    base_df = apply_weekend_pattern(base_df)
    
    logger.info(f"Base signal generated with {len(base_df)} rows")
    
    saved_files = []
    
    for node_id, bias in NODE_BIASES.items():
        # Use different seed per node for independent noise
        node_seed = seed + int(node_id[1:])  # N01 -> seed+1, N02 -> seed+2, etc.
        
        node_df = generate_node_data(base_df, node_id, bias, noise_std=5.0, seed=node_seed)
        
        # Validate
        assert not node_df["aqi_simulated"].isna().any(), f"NaN found in {node_id}"
        assert node_df["aqi_simulated"].min() >= 0, f"Negative AQI in {node_id}"
        assert node_df["aqi_simulated"].max() <= 500, f"AQI > 500 in {node_id}"
        
        # Save CSV
        filepath = output_path / f"node_{node_id}_simulated.csv"
        node_df.to_csv(filepath, index=False)
        saved_files.append(str(filepath))
        
        logger.info(
            f"Generated {node_id}: mean_aqi={node_df['aqi_simulated'].mean():.1f}, "
            f"bias={bias:+d}, saved to {filepath}"
        )
    
    # Validate all timestamps match
    all_timestamps = []
    for filepath in saved_files:
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        all_timestamps.append(set(df["timestamp"]))
    
    reference_ts = all_timestamps[0]
    for i, ts_set in enumerate(all_timestamps[1:], start=1):
        if ts_set != reference_ts:
            raise ValueError(f"Timestamp mismatch between N01 and node {i+1}")
    
    logger.info(f"All {len(saved_files)} nodes validated - identical timestamps confirmed")
    
    return saved_files


def load_simulated_data(
    data_dir: str = "data/simulated",
    nodes: List[str] = None
) -> pd.DataFrame:
    """
    Load simulated data from all nodes into a single DataFrame.
    
    Args:
        data_dir: Directory containing simulated CSVs
        nodes: List of node IDs to load (None = all)
        
    Returns:
        Combined DataFrame with all nodes
    """
    data_path = Path(data_dir)
    
    if nodes is None:
        nodes = list(NODE_BIASES.keys())
    
    dfs = []
    for node_id in nodes:
        filepath = data_path / f"node_{node_id}_simulated.csv"
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue
            
        df = pd.read_csv(filepath, parse_dates=["timestamp"])
        dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No simulated data found in {data_dir}")
    
    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined)} rows from {len(dfs)} nodes")
    
    return combined


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    files = generate_all_nodes()
    print(f"Generated {len(files)} files")
```
</action>
<acceptance_criteria>
- `Test-Path src/data/iot_sim.py` returns True
- `grep -l "NODE_BIASES = {" src/data/iot_sim.py` returns file path
- `grep -l "def generate_all_nodes" src/data/iot_sim.py` returns file path
- `grep -l "DIWALI_DATES" src/data/iot_sim.py` returns file path
- `python -c "from src.data.iot_sim import NODE_BIASES; assert len(NODE_BIASES) == 10; print('10 nodes configured')"` exits 0
</acceptance_criteria>
</task>

<task id="4.2">
<title>Create test_iot_sim.py unit tests</title>
<read_first>
- src/data/iot_sim.py (functions to test)
</read_first>
<action>
Create tests/test_iot_sim.py:

```python
"""Unit tests for IoT sensor simulation."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestBaseSignal:
    """Tests for base signal generation."""
    
    def test_generate_base_signal_length(self):
        from src.data.iot_sim import generate_base_signal
        
        df = generate_base_signal(
            start_date="2023-01-01",
            end_date="2023-01-31",
            seed=42
        )
        
        # 31 days * 24 hours = 744 hours
        # But date_range includes both endpoints, so it's actually 30 days
        assert len(df) >= 720
        
    def test_generate_base_signal_columns(self):
        from src.data.iot_sim import generate_base_signal
        
        df = generate_base_signal("2023-01-01", "2023-01-07", seed=42)
        
        assert "timestamp" in df.columns
        assert "base_aqi" in df.columns
        
    def test_generate_base_signal_reproducible(self):
        from src.data.iot_sim import generate_base_signal
        
        df1 = generate_base_signal("2023-01-01", "2023-01-07", seed=42)
        df2 = generate_base_signal("2023-01-01", "2023-01-07", seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)


class TestTemporalPatterns:
    """Tests for temporal pattern application."""
    
    @pytest.fixture
    def sample_base_df(self):
        timestamps = pd.date_range("2023-01-01", periods=48, freq="H")
        return pd.DataFrame({
            "timestamp": timestamps,
            "base_aqi": [100.0] * 48
        })
    
    def test_apply_hourly_pattern(self, sample_base_df):
        from src.data.iot_sim import apply_hourly_pattern
        
        df = apply_hourly_pattern(sample_base_df)
        
        # Check rush hour spike
        morning_rush = df[df["timestamp"].dt.hour == 8]["base_aqi"].values[0]
        assert morning_rush > 100  # Should have +15 spike
        
    def test_apply_seasonal_pattern(self):
        from src.data.iot_sim import apply_seasonal_pattern
        
        # Create data spanning monsoon months
        timestamps = pd.date_range("2023-07-01", periods=24, freq="H")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "base_aqi": [100.0] * 24
        })
        
        df = apply_seasonal_pattern(df)
        
        # July should have monsoon dip
        assert df["base_aqi"].mean() < 100
        
    def test_apply_diwali_spike(self):
        from src.data.iot_sim import apply_diwali_spike
        
        # Create data around Diwali 2023
        timestamps = pd.date_range("2023-11-10", periods=72, freq="H")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "base_aqi": [100.0] * 72
        })
        
        df = apply_diwali_spike(df)
        
        # Diwali day (Nov 12) should have spike
        diwali_day = df[df["timestamp"].dt.date == pd.Timestamp("2023-11-12").date()]
        assert diwali_day["base_aqi"].mean() > 100


class TestNodeGeneration:
    """Tests for individual node data generation."""
    
    def test_node_bias_applied(self):
        from src.data.iot_sim import generate_base_signal, generate_node_data
        
        base_df = generate_base_signal("2023-01-01", "2023-01-07", base_aqi=100, seed=42)
        
        # Node with +25 bias
        node_df = generate_node_data(base_df, "N03", bias=25, noise_std=0, seed=42)
        
        # Mean should be close to base + bias
        mean_aqi = node_df["aqi_simulated"].mean()
        assert mean_aqi > 120  # base ~100 + patterns + bias 25
        
    def test_node_data_columns(self):
        from src.data.iot_sim import generate_base_signal, generate_node_data
        
        base_df = generate_base_signal("2023-01-01", "2023-01-07", seed=42)
        node_df = generate_node_data(base_df, "N01", bias=0, seed=42)
        
        assert "timestamp" in node_df.columns
        assert "node_id" in node_df.columns
        assert "pm25" in node_df.columns
        assert "pm10" in node_df.columns
        assert "no2" in node_df.columns
        assert "aqi_simulated" in node_df.columns
        
    def test_aqi_range_valid(self):
        from src.data.iot_sim import generate_base_signal, generate_node_data
        
        base_df = generate_base_signal("2023-01-01", "2023-01-31", seed=42)
        node_df = generate_node_data(base_df, "N01", bias=0, seed=42)
        
        assert node_df["aqi_simulated"].min() >= 0
        assert node_df["aqi_simulated"].max() <= 500
        
    def test_no_nan_values(self):
        from src.data.iot_sim import generate_base_signal, generate_node_data
        
        base_df = generate_base_signal("2023-01-01", "2023-01-31", seed=42)
        node_df = generate_node_data(base_df, "N01", bias=0, seed=42)
        
        assert not node_df["aqi_simulated"].isna().any()
        assert not node_df["pm25"].isna().any()


class TestGenerateAllNodes:
    """Tests for full simulation pipeline."""
    
    def test_generates_10_files(self, tmp_path):
        from src.data.iot_sim import generate_all_nodes
        
        files = generate_all_nodes(
            output_dir=str(tmp_path),
            start_date="2023-01-01",
            end_date="2023-01-07",
            seed=42
        )
        
        assert len(files) == 10
        
    def test_files_exist(self, tmp_path):
        from src.data.iot_sim import generate_all_nodes
        
        generate_all_nodes(
            output_dir=str(tmp_path),
            start_date="2023-01-01",
            end_date="2023-01-07",
            seed=42
        )
        
        for node_id in ["N01", "N02", "N03", "N04", "N05", "N06", "N07", "N08", "N09", "N10"]:
            filepath = tmp_path / f"node_{node_id}_simulated.csv"
            assert filepath.exists()
            
    def test_identical_timestamps(self, tmp_path):
        from src.data.iot_sim import generate_all_nodes
        
        generate_all_nodes(
            output_dir=str(tmp_path),
            start_date="2023-01-01",
            end_date="2023-01-07",
            seed=42
        )
        
        # Load first two nodes and compare timestamps
        df1 = pd.read_csv(tmp_path / "node_N01_simulated.csv", parse_dates=["timestamp"])
        df2 = pd.read_csv(tmp_path / "node_N03_simulated.csv", parse_dates=["timestamp"])
        
        pd.testing.assert_series_equal(
            df1["timestamp"].reset_index(drop=True),
            df2["timestamp"].reset_index(drop=True)
        )


class TestPM25Estimation:
    """Tests for PM2.5 estimation from AQI."""
    
    def test_low_aqi_low_pm25(self):
        from src.data.iot_sim import estimate_pm25_from_aqi
        
        pm25 = estimate_pm25_from_aqi(np.array([50]))
        assert pm25[0] <= 30  # Good AQI = PM2.5 <= 30
        
    def test_high_aqi_high_pm25(self):
        from src.data.iot_sim import estimate_pm25_from_aqi
        
        pm25 = estimate_pm25_from_aqi(np.array([300]))
        assert pm25[0] > 100  # Very Poor AQI = high PM2.5


class TestLoadSimulatedData:
    """Tests for loading simulated data."""
    
    def test_load_combined(self, tmp_path):
        from src.data.iot_sim import generate_all_nodes, load_simulated_data
        
        generate_all_nodes(
            output_dir=str(tmp_path),
            start_date="2023-01-01",
            end_date="2023-01-03",
            seed=42
        )
        
        df = load_simulated_data(str(tmp_path))
        
        # Should have data from all 10 nodes
        assert df["node_id"].nunique() == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```
</action>
<acceptance_criteria>
- `Test-Path tests/test_iot_sim.py` returns True
- `grep -l "def test_generates_10_files" tests/test_iot_sim.py` returns file path
- `grep -l "def test_identical_timestamps" tests/test_iot_sim.py` returns file path
- `python -m pytest tests/test_iot_sim.py --collect-only -q | Select-String "test"` shows at least 10 tests
</acceptance_criteria>
</task>

<task id="4.3">
<title>Add CLI entry point for simulation</title>
<read_first>
- src/data/iot_sim.py
- configs/data.yaml
</read_first>
<action>
Add CLI entry point to src/data/iot_sim.py (at the bottom):

```python
def main():
    """CLI entry point for IoT simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate simulated IoT sensor data")
    parser.add_argument(
        "--output-dir", "-o",
        default="data/simulated",
        help="Output directory for CSV files"
    )
    parser.add_argument(
        "--start-date",
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        default="2023-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    files = generate_all_nodes(
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed
    )
    
    print(f"\nGenerated {len(files)} CSV files in {args.output_dir}/")
    print("Files:", [Path(f).name for f in files])


if __name__ == "__main__":
    main()
```
</action>
<acceptance_criteria>
- `grep -l "def main():" src/data/iot_sim.py` returns file path
- `grep -l "argparse" src/data/iot_sim.py` returns file path
- `python -m src.data.iot_sim --help` shows usage info (exit 0)
</acceptance_criteria>
</task>

## Verification

```powershell
# Verify module loads
python -c "from src.data.iot_sim import generate_all_nodes, NODE_BIASES; print(f'{len(NODE_BIASES)} nodes configured')"

# Run quick simulation (7 days)
python -m src.data.iot_sim --start-date 2023-01-01 --end-date 2023-01-07 --output-dir data/simulated/test

# Verify output files
Get-ChildItem data/simulated/test/*.csv | Measure-Object | Select-Object -ExpandProperty Count

# Validate data
python -c "import pandas as pd; df = pd.read_csv('data/simulated/test/node_N01_simulated.csv'); print(f'Rows: {len(df)}, Columns: {list(df.columns)}')"

# Run tests
python -m pytest tests/test_iot_sim.py -v
```
