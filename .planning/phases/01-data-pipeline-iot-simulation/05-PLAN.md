---
phase: 1
plan: 5
title: "Integration and Validation"
wave: 3
depends_on: [2, 3, 4]
files_modified:
  - src/data/pipeline.py
  - tests/conftest.py
requirements_addressed: [DATA-11, DATA-12, IOT-06]
autonomous: true
---

# Plan 5: Integration and Validation

<objective>
Create a unified data pipeline that orchestrates fetch, preprocess, and IoT simulation modules. Add pytest fixtures and end-to-end validation to ensure all components work together.
</objective>

<must_haves>
- Unified pipeline.py that runs full data flow
- pytest conftest.py with shared fixtures
- E2E test that validates complete pipeline
- Summary statistics validation
- .gitignore for data/ and models/ directories
</must_haves>

## Tasks

<task id="5.1">
<title>Create unified data pipeline</title>
<read_first>
- src/data/fetch.py
- src/data/preprocess.py
- src/data/iot_sim.py
</read_first>
<action>
Create src/data/pipeline.py:

```python
"""Unified data pipeline for Pune Air Quality Sentinel."""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data import fetch, preprocess, iot_sim

logger = logging.getLogger(__name__)


def run_fetch_pipeline(
    config_path: str = "configs/data.yaml",
    use_kaggle: bool = True,
    use_weather: bool = True,
    use_openaq: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Run data fetching pipeline.
    
    Args:
        config_path: Path to data configuration
        use_kaggle: Fetch from Kaggle
        use_weather: Fetch weather from Open-Meteo
        use_openaq: Fetch from OpenAQ (optional)
        
    Returns:
        Dict of DataFrames by source
    """
    config = load_config(config_path)
    data = {}
    
    if use_kaggle:
        logger.info("Fetching Kaggle Air Quality India data...")
        try:
            df = fetch.fetch_kaggle_aqi()
            data["kaggle"] = df
            logger.info(f"Kaggle data: {len(df)} rows")
        except Exception as e:
            logger.error(f"Kaggle fetch failed: {e}")
    
    if use_weather:
        logger.info("Fetching Open-Meteo weather data...")
        try:
            df = fetch.fetch_open_meteo_weather()
            data["weather"] = df
            logger.info(f"Weather data: {len(df)} rows")
        except Exception as e:
            logger.error(f"Weather fetch failed: {e}")
    
    if use_openaq:
        logger.info("Fetching OpenAQ data...")
        try:
            df = fetch.fetch_openaq_pune()
            data["openaq"] = df
            logger.info(f"OpenAQ data: {len(df)} rows")
        except Exception as e:
            logger.error(f"OpenAQ fetch failed: {e}")
    
    return data


def run_preprocess_pipeline(
    df: pd.DataFrame,
    output_path: str = "data/processed/pune_aqi_processed.parquet",
    scaler_path: str = "models/scaler.pkl",
    splits_dir: str = "data/splits",
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Run preprocessing pipeline.
    
    Args:
        df: Raw input DataFrame
        output_path: Path to save processed data
        scaler_path: Path to save fitted scaler
        splits_dir: Directory to save split indices
        seed: Random seed
        
    Returns:
        Dict with train, val, test DataFrames
    """
    set_seed(seed)
    
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Clean data
    logger.info("Step 1: Cleaning data...")
    df = preprocess.clean_aqi_data(df)
    
    # Step 2: Clip outliers
    logger.info("Step 2: Clipping outliers...")
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    pollutant_cols = [c for c in pollutant_cols if c in df.columns]
    df = preprocess.clip_outliers(df, columns=pollutant_cols)
    
    # Step 3: Compute AQI if not present
    if "aqi" not in df.columns and "pm25" in df.columns:
        logger.info("Step 3: Computing AQI...")
        df["aqi"] = df.apply(preprocess.compute_aqi_row, axis=1)
    
    # Step 4: Feature engineering
    logger.info("Step 4: Engineering features...")
    df = preprocess.create_lag_features(df, "aqi", lags=[1, 6, 24])
    df = preprocess.create_rolling_features(df, "pm25", windows=[24, 72])
    
    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        df = preprocess.compute_wind_vectors(df)
    
    df = preprocess.create_calendar_features(df)
    
    # Step 5: Drop rows with NaN from feature engineering
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_rows - len(df)} rows after feature engineering")
    
    # Step 6: Create splits BEFORE normalization (to fit scaler on train only)
    logger.info("Step 5: Creating train/val/test splits...")
    train_df, val_df, test_df = preprocess.create_splits(
        df, train_ratio=0.7, val_ratio=0.15, output_dir=splits_dir, seed=seed
    )
    
    # Step 7: Normalize (fit on train only)
    logger.info("Step 6: Normalizing features...")
    train_df, scaler = preprocess.normalize_features(
        train_df, scaler_path=scaler_path, fit=True
    )
    val_df, _ = preprocess.normalize_features(
        val_df, scaler_path=scaler_path, fit=False
    )
    test_df, _ = preprocess.normalize_features(
        test_df, scaler_path=scaler_path, fit=False
    )
    
    # Save full processed data
    preprocess.save_processed(df, output_path)
    
    logger.info("Preprocessing pipeline complete!")
    
    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "full": df
    }


def run_iot_simulation(
    output_dir: str = "data/simulated",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42
) -> List[str]:
    """
    Run IoT sensor simulation.
    
    Args:
        output_dir: Directory to save simulated CSVs
        start_date: Simulation start date
        end_date: Simulation end date
        seed: Random seed
        
    Returns:
        List of generated file paths
    """
    logger.info(f"Generating IoT simulation: {start_date} to {end_date}")
    
    files = iot_sim.generate_all_nodes(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        seed=seed
    )
    
    logger.info(f"Generated {len(files)} simulated sensor files")
    return files


def run_full_pipeline(
    use_simulated: bool = True,
    seed: int = 42
) -> Dict:
    """
    Run complete data pipeline end-to-end.
    
    Args:
        use_simulated: Use simulated IoT data instead of real data
        seed: Random seed
        
    Returns:
        Dict with all pipeline outputs
    """
    set_seed(seed)
    results = {}
    
    # Step 1: Generate or fetch data
    if use_simulated:
        logger.info("=== Running IoT Simulation ===")
        files = run_iot_simulation(seed=seed)
        results["simulated_files"] = files
        
        # Load simulated data
        df = iot_sim.load_simulated_data()
        results["raw_data"] = df
    else:
        logger.info("=== Fetching Real Data ===")
        fetched = run_fetch_pipeline()
        results["fetched_data"] = fetched
        
        # Use Kaggle data as primary
        if "kaggle" in fetched:
            df = fetched["kaggle"]
        else:
            raise ValueError("No data available for preprocessing")
    
    # Step 2: Preprocess
    logger.info("=== Running Preprocessing ===")
    processed = run_preprocess_pipeline(df, seed=seed)
    results["processed"] = processed
    
    # Step 3: Summary statistics
    logger.info("=== Pipeline Summary ===")
    train_df = processed["train"]
    val_df = processed["val"]
    test_df = processed["test"]
    
    summary = {
        "total_rows": len(processed["full"]),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "features": list(train_df.columns),
        "n_features": len(train_df.columns),
    }
    
    if "aqi" in train_df.columns:
        summary["aqi_stats"] = {
            "train_mean": float(train_df["aqi"].mean()),
            "train_std": float(train_df["aqi"].std()),
            "val_mean": float(val_df["aqi"].mean()),
            "test_mean": float(test_df["aqi"].mean()),
        }
    
    results["summary"] = summary
    
    logger.info(f"Pipeline complete: {summary['total_rows']} total rows")
    logger.info(f"Splits: train={summary['train_rows']}, val={summary['val_rows']}, test={summary['test_rows']}")
    logger.info(f"Features: {summary['n_features']}")
    
    return results


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument(
        "--simulated", "-s",
        action="store_true",
        help="Use simulated IoT data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    results = run_full_pipeline(use_simulated=args.simulated, seed=args.seed)
    
    print("\n=== Pipeline Results ===")
    for key, value in results["summary"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
```
</action>
<acceptance_criteria>
- `Test-Path src/data/pipeline.py` returns True
- `grep -l "def run_full_pipeline" src/data/pipeline.py` returns file path
- `python -c "from src.data.pipeline import run_full_pipeline; print('Pipeline imports OK')"` exits 0
</acceptance_criteria>
</task>

<task id="5.2">
<title>Create pytest fixtures in conftest.py</title>
<read_first>
- tests/ directory structure
</read_first>
<action>
Create tests/conftest.py:

```python
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


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_aqi_df() -> pd.DataFrame:
    """Sample AQI DataFrame for testing."""
    np.random.seed(42)
    n_rows = 100
    
    timestamps = pd.date_range("2023-01-01", periods=n_rows, freq="H")
    
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
    timestamps = pd.date_range("2023-01-01", periods=50, freq="H")
    values = np.random.uniform(50, 150, 50)
    
    # Introduce various gap patterns
    values[10] = np.nan  # Single gap
    values[20:23] = np.nan  # 3-hour gap
    values[35:42] = np.nan  # 7-hour gap
    
    return pd.DataFrame({
        "timestamp": timestamps,
        "pm25": values.copy(),
        "aqi": values.copy()
    })


@pytest.fixture
def node_coords() -> Dict:
    """Sample node coordinates for testing."""
    return {
        "N01": {"name": "Shivajinagar", "lat": 18.5308, "lon": 73.8475},
        "N02": {"name": "Hinjewadi", "lat": 18.5912, "lon": 73.7389},
        "N03": {"name": "Pimpri-Chinchwad", "lat": 18.6298, "lon": 73.8037},
        "N04": {"name": "Hadapsar", "lat": 18.5089, "lon": 73.9260},
        "N05": {"name": "Katraj", "lat": 18.4529, "lon": 73.8669},
        "N06": {"name": "Deccan", "lat": 18.5196, "lon": 73.8399},
        "N07": {"name": "Viman Nagar", "lat": 18.5679, "lon": 73.9143},
        "N08": {"name": "Kothrud", "lat": 18.5074, "lon": 73.8077},
        "N09": {"name": "Talegaon", "lat": 18.7330, "lon": 73.6554},
        "N10": {"name": "Mundhwa", "lat": 18.5326, "lon": 73.9368},
    }


@pytest.fixture
def node_coords_file(tmp_path, node_coords) -> Path:
    """Create temporary node coordinates JSON file."""
    filepath = tmp_path / "node_coords.json"
    with open(filepath, "w") as f:
        json.dump(node_coords, f)
    return filepath


@pytest.fixture
def mock_kaggle_response() -> pd.DataFrame:
    """Mock Kaggle API response."""
    np.random.seed(42)
    n_rows = 50
    
    return pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "City": ["Pune"] * n_rows,
        "AQI": np.random.randint(50, 300, n_rows),
        "PM2.5": np.random.uniform(20, 150, n_rows),
        "PM10": np.random.uniform(30, 200, n_rows),
        "NO2": np.random.uniform(10, 80, n_rows),
        "SO2": np.random.uniform(5, 50, n_rows),
        "CO": np.random.uniform(0.5, 5, n_rows),
        "O3": np.random.uniform(20, 100, n_rows),
    })


@pytest.fixture
def mock_weather_response() -> Dict:
    """Mock Open-Meteo API response."""
    return {
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00", "2023-01-01T02:00"],
            "temperature_2m": [20.5, 19.8, 19.2],
            "relative_humidity_2m": [65, 68, 70],
            "wind_speed_10m": [5.2, 4.8, 5.1],
            "wind_direction_10m": [180, 185, 190],
            "precipitation": [0, 0, 0.2]
        }
    }


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(42)
    yield


# Skip markers for slow/integration tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring GPU")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on markers and CLI options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Use --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="Use --run-integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
```
</action>
<acceptance_criteria>
- `Test-Path tests/conftest.py` returns True
- `grep -l "@pytest.fixture" tests/conftest.py` returns file path
- `grep -l "sample_aqi_df" tests/conftest.py` returns file path
</acceptance_criteria>
</task>

<task id="5.3">
<title>Create E2E integration test</title>
<read_first>
- src/data/pipeline.py
- tests/conftest.py
</read_first>
<action>
Create tests/test_pipeline_e2e.py:

```python
"""End-to-end integration tests for data pipeline."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for complete pipeline."""
    
    def test_iot_to_preprocessing(self, tmp_path):
        """Test IoT simulation feeds into preprocessing correctly."""
        from src.data.iot_sim import generate_all_nodes, load_simulated_data
        from src.data.preprocess import (
            clean_aqi_data,
            clip_outliers,
            create_lag_features,
            create_splits
        )
        
        # Step 1: Generate IoT data
        sim_dir = tmp_path / "simulated"
        files = generate_all_nodes(
            output_dir=str(sim_dir),
            start_date="2023-01-01",
            end_date="2023-01-31",
            seed=42
        )
        assert len(files) == 10
        
        # Step 2: Load simulated data
        df = load_simulated_data(str(sim_dir))
        assert len(df) > 0
        assert "aqi_simulated" in df.columns
        
        # Rename for preprocessing compatibility
        df = df.rename(columns={"aqi_simulated": "aqi"})
        
        # Step 3: Clean data
        df = clean_aqi_data(df)
        assert not df["aqi"].isna().all()
        
        # Step 4: Feature engineering
        df = create_lag_features(df, "aqi")
        assert "aqi_lag_1h" in df.columns
        
        # Step 5: Create splits
        train, val, test = create_splits(
            df.dropna(),
            output_dir=str(tmp_path / "splits")
        )
        
        assert len(train) > len(val)
        assert len(val) > 0
        assert len(test) > 0
        
        # Verify chronological order
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()
    
    def test_pipeline_output_shapes(self, tmp_path):
        """Verify pipeline produces expected output shapes."""
        from src.data.iot_sim import generate_all_nodes, load_simulated_data
        from src.data.preprocess import create_splits
        
        # Generate minimal data
        sim_dir = tmp_path / "simulated"
        generate_all_nodes(
            output_dir=str(sim_dir),
            start_date="2023-01-01",
            end_date="2023-01-14",
            seed=42
        )
        
        df = load_simulated_data(str(sim_dir))
        df = df.rename(columns={"aqi_simulated": "aqi"})
        
        train, val, test = create_splits(
            df,
            output_dir=str(tmp_path / "splits")
        )
        
        # Verify ratio approximately correct
        total = len(train) + len(val) + len(test)
        assert abs(len(train) / total - 0.7) < 0.02
        assert abs(len(val) / total - 0.15) < 0.02
        assert abs(len(test) / total - 0.15) < 0.02
    
    def test_no_data_leakage(self, tmp_path):
        """Verify no temporal data leakage between splits."""
        from src.data.iot_sim import generate_all_nodes, load_simulated_data
        from src.data.preprocess import create_splits, normalize_features
        
        sim_dir = tmp_path / "simulated"
        generate_all_nodes(
            output_dir=str(sim_dir),
            start_date="2023-01-01",
            end_date="2023-01-14",
            seed=42
        )
        
        df = load_simulated_data(str(sim_dir))
        df = df.rename(columns={"aqi_simulated": "aqi"})
        
        train, val, test = create_splits(
            df,
            output_dir=str(tmp_path / "splits")
        )
        
        # Normalize with scaler fit on train only
        scaler_path = str(tmp_path / "scaler.pkl")
        train_norm, scaler = normalize_features(
            train, scaler_path=scaler_path, fit=True
        )
        
        # Apply same scaler to val/test
        val_norm, _ = normalize_features(
            val, scaler_path=scaler_path, fit=False
        )
        test_norm, _ = normalize_features(
            test, scaler_path=scaler_path, fit=False
        )
        
        # Verify scaler was fit only on train
        # Train should have values exactly in [0,1]
        numeric_cols = train_norm.select_dtypes(include=[np.number]).columns
        # Exclude binary columns
        check_cols = [c for c in numeric_cols 
                      if not train_norm[c].isin([0, 1]).all()
                      and c != "aqi"]
        
        for col in check_cols[:3]:  # Check first 3 numeric columns
            assert train_norm[col].min() >= 0, f"Train {col} min < 0"
            assert train_norm[col].max() <= 1, f"Train {col} max > 1"


class TestDataValidation:
    """Tests for data validation and quality."""
    
    def test_aqi_range_valid(self, sample_aqi_df):
        """Verify AQI values are in valid range."""
        from src.data.preprocess import compute_aqi_row
        
        sample_aqi_df["computed_aqi"] = sample_aqi_df.apply(
            compute_aqi_row, axis=1
        )
        
        assert sample_aqi_df["computed_aqi"].min() >= 0
        assert sample_aqi_df["computed_aqi"].max() <= 500
    
    def test_no_future_leakage_in_lags(self, sample_aqi_df):
        """Verify lag features don't leak future information."""
        from src.data.preprocess import create_lag_features
        
        df = create_lag_features(sample_aqi_df, "pm25", lags=[1, 6])
        
        # Lag 1 should be previous value
        for i in range(1, len(df)):
            if not pd.isna(df.iloc[i]["pm25_lag_1h"]):
                assert df.iloc[i]["pm25_lag_1h"] == df.iloc[i-1]["pm25"]
    
    def test_node_count_correct(self, tmp_path):
        """Verify all 10 nodes are generated."""
        from src.data.iot_sim import generate_all_nodes, NODE_BIASES
        
        files = generate_all_nodes(
            output_dir=str(tmp_path),
            start_date="2023-01-01",
            end_date="2023-01-03",
            seed=42
        )
        
        assert len(files) == len(NODE_BIASES)
        assert len(files) == 10


class TestSplitIntegrity:
    """Tests for train/val/test split integrity."""
    
    def test_splits_json_created(self, tmp_path, sample_aqi_df):
        """Verify splits.json is created with correct structure."""
        from src.data.preprocess import create_splits
        
        splits_dir = tmp_path / "splits"
        create_splits(sample_aqi_df, output_dir=str(splits_dir))
        
        splits_file = splits_dir / "splits.json"
        assert splits_file.exists()
        
        with open(splits_file) as f:
            splits_info = json.load(f)
        
        assert "train" in splits_info
        assert "val" in splits_info
        assert "test" in splits_info
        assert "seed" in splits_info
    
    def test_splits_reproducible(self, sample_aqi_df, tmp_path):
        """Verify splits are reproducible with same seed."""
        from src.data.preprocess import create_splits
        
        train1, val1, test1 = create_splits(
            sample_aqi_df,
            output_dir=str(tmp_path / "splits1"),
            seed=42
        )
        
        train2, val2, test2 = create_splits(
            sample_aqi_df,
            output_dir=str(tmp_path / "splits2"),
            seed=42
        )
        
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(val1, val2)
        pd.testing.assert_frame_equal(test1, test2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--run-integration"])
```
</action>
<acceptance_criteria>
- `Test-Path tests/test_pipeline_e2e.py` returns True
- `grep -l "def test_iot_to_preprocessing" tests/test_pipeline_e2e.py` returns file path
- `grep -l "def test_no_data_leakage" tests/test_pipeline_e2e.py` returns file path
- `python -m pytest tests/test_pipeline_e2e.py --collect-only -q | Select-String "test"` shows at least 5 tests
</acceptance_criteria>
</task>

<task id="5.4">
<title>Add data directories to .gitignore</title>
<read_first>
- .gitignore (if exists)
</read_first>
<action>
Create or update .gitignore:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/

# Data directories (too large for git)
data/raw/
data/processed/
data/simulated/
data/splits/
data/graph/*.npy

# Model checkpoints
models/*.pkl
models/*.pth
models/*.joblib

# Output files
outputs/
*.html
*.png
*.jpg
*.jpeg

# Logs
logs/
*.log

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Kaggle credentials (NEVER commit!)
kaggle.json
.kaggle/

# Test artifacts
.pytest_cache/
.coverage
htmlcov/
.tox/

# Keep directory structure
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/simulated/.gitkeep
!data/splits/.gitkeep
!models/.gitkeep
!outputs/.gitkeep
```
</action>
<acceptance_criteria>
- `Test-Path .gitignore` returns True
- `grep -l "data/simulated/" .gitignore` returns file path
- `grep -l "models/*.pth" .gitignore` returns file path
</acceptance_criteria>
</task>

<task id="5.5">
<title>Create .gitkeep files for directory structure</title>
<read_first>
- Project structure from PRD
</read_first>
<action>
Create .gitkeep files to preserve empty directories:

```powershell
# Create directories and .gitkeep files
$dirs = @(
    "data/raw",
    "data/processed", 
    "data/simulated",
    "data/splits",
    "data/graph",
    "models",
    "outputs",
    "notebooks"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir
    "" | Out-File -FilePath "$dir/.gitkeep" -Encoding utf8
}
```

Also create data/graph/node_coords.json:

```json
{
    "N01": {"name": "Shivajinagar", "lat": 18.5308, "lon": 73.8475, "type": "reference"},
    "N02": {"name": "Hinjewadi IT Park", "lat": 18.5912, "lon": 73.7389, "type": "it_corridor"},
    "N03": {"name": "Pimpri-Chinchwad MIDC", "lat": 18.6298, "lon": 73.8037, "type": "industrial"},
    "N04": {"name": "Hadapsar", "lat": 18.5089, "lon": 73.9260, "type": "mixed"},
    "N05": {"name": "Katraj", "lat": 18.4529, "lon": 73.8669, "type": "residential"},
    "N06": {"name": "Deccan Gymkhana", "lat": 18.5196, "lon": 73.8399, "type": "central"},
    "N07": {"name": "Viman Nagar", "lat": 18.5679, "lon": 73.9143, "type": "airport"},
    "N08": {"name": "Kothrud", "lat": 18.5074, "lon": 73.8077, "type": "residential"},
    "N09": {"name": "Talegaon MIDC", "lat": 18.7330, "lon": 73.6554, "type": "industrial"},
    "N10": {"name": "Mundhwa", "lat": 18.5326, "lon": 73.9368, "type": "residential"}
}
```
</action>
<acceptance_criteria>
- `Test-Path data/graph/node_coords.json` returns True
- `Test-Path models/.gitkeep` returns True
- `python -c "import json; data = json.load(open('data/graph/node_coords.json')); assert len(data) == 10; print('10 nodes loaded')"` exits 0
</acceptance_criteria>
</task>

## Verification

```powershell
# Verify all Phase 1 modules load
python -c "from src.data import fetch, preprocess, iot_sim, pipeline; print('All data modules import OK')"

# Run all unit tests
python -m pytest tests/test_preprocess.py tests/test_iot_sim.py tests/test_fetch.py -v

# Run integration tests
python -m pytest tests/test_pipeline_e2e.py -v --run-integration

# Quick pipeline smoke test
python -c "
from src.data.iot_sim import generate_all_nodes, load_simulated_data
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    files = generate_all_nodes(tmpdir, '2023-01-01', '2023-01-07', 42)
    df = load_simulated_data(tmpdir)
    print(f'Generated {len(files)} files, {len(df)} total rows')
    print(f'Nodes: {df.node_id.unique().tolist()}')
    print(f'AQI range: {df.aqi_simulated.min():.1f} - {df.aqi_simulated.max():.1f}')
"

# Verify git ignores data files
git status --short
```
