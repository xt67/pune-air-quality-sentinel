---
phase: 1
plan: 3
title: "Data Preprocessing Pipeline"
wave: 2
depends_on: [1]
files_modified:
  - src/data/preprocess.py
  - tests/test_preprocess.py
requirements_addressed: [DATA-04, DATA-05, DATA-06, DATA-07, DATA-08, DATA-09, DATA-10, DATA-11, DATA-12]
autonomous: true
---

# Plan 3: Data Preprocessing Pipeline

<objective>
Implement the complete data preprocessing pipeline including cleaning, AQI computation, feature engineering, normalization, and train/val/test splitting.
</objective>

<must_haves>
- Gap filling (forward-fill ≤3h, interpolate 3-12h, drop >12h)
- Outlier clipping at 99.9th percentile
- CPCB AQI computation from sub-indices
- Lag features (1h, 6h, 24h)
- Rolling features (24h mean, 72h mean, 24h std)
- Wind vector computation
- Calendar features (hour encoding, day-of-week, Diwali flag)
- MinMaxScaler normalization (fit on train only)
- Chronological train/val/test split
- Parquet I/O
</must_haves>

## Tasks

<task id="3.1">
<title>Implement CPCB AQI breakpoint computation</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-RESEARCH.md (CPCB breakpoint tables)
</read_first>
<action>
Create src/data/preprocess.py with AQI computation:

```python
"""Data preprocessing pipeline for AQI data."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

logger = logging.getLogger(__name__)

# CPCB AQI Breakpoint Tables
# Format: (aqi_low, aqi_high, conc_low, conc_high)
BREAKPOINTS = {
    "pm25": [
        (0, 50, 0, 30),
        (51, 100, 31, 60),
        (101, 200, 61, 90),
        (201, 300, 91, 120),
        (301, 400, 121, 250),
        (401, 500, 251, 500),
    ],
    "pm10": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 200, 101, 250),
        (201, 300, 251, 350),
        (301, 400, 351, 430),
        (401, 500, 431, 600),
    ],
    "no2": [
        (0, 50, 0, 40),
        (51, 100, 41, 80),
        (101, 200, 81, 180),
        (201, 300, 181, 280),
        (301, 400, 281, 400),
        (401, 500, 401, 500),
    ],
    "so2": [
        (0, 50, 0, 40),
        (51, 100, 41, 80),
        (101, 200, 81, 380),
        (201, 300, 381, 800),
        (301, 400, 801, 1600),
        (401, 500, 1601, 2000),
    ],
    "co": [
        (0, 50, 0, 1.0),
        (51, 100, 1.1, 2.0),
        (101, 200, 2.1, 10.0),
        (201, 300, 10.1, 17.0),
        (301, 400, 17.1, 34.0),
        (401, 500, 34.1, 50.0),
    ],
    "o3": [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 200, 101, 168),
        (201, 300, 169, 208),
        (301, 400, 209, 748),
        (401, 500, 749, 1000),
    ],
}


def compute_sub_index(concentration: float, pollutant: str) -> float:
    """
    Compute AQI sub-index for a single pollutant using CPCB breakpoints.
    
    Args:
        concentration: Pollutant concentration value
        pollutant: Pollutant name (pm25, pm10, no2, so2, co, o3)
        
    Returns:
        AQI sub-index value
    """
    if pd.isna(concentration) or concentration < 0:
        return np.nan
    
    breakpoints = BREAKPOINTS.get(pollutant.lower())
    if breakpoints is None:
        logger.warning(f"Unknown pollutant: {pollutant}")
        return np.nan
    
    for aqi_low, aqi_high, conc_low, conc_high in breakpoints:
        if conc_low <= concentration <= conc_high:
            # Linear interpolation
            sub_index = (
                (aqi_high - aqi_low) / (conc_high - conc_low)
            ) * (concentration - conc_low) + aqi_low
            return sub_index
    
    # Above highest breakpoint
    if concentration > breakpoints[-1][3]:
        return 500.0
    
    return np.nan


def compute_aqi(
    pm25: float = None,
    pm10: float = None,
    no2: float = None,
    so2: float = None,
    co: float = None,
    o3: float = None
) -> float:
    """
    Compute overall AQI from pollutant concentrations using CPCB formula.
    
    AQI = max(sub-indices for all pollutants)
    
    Args:
        pm25: PM2.5 concentration (µg/m³)
        pm10: PM10 concentration (µg/m³)
        no2: NO2 concentration (µg/m³)
        so2: SO2 concentration (µg/m³)
        co: CO concentration (mg/m³)
        o3: O3 concentration (µg/m³)
        
    Returns:
        Overall AQI value (0-500)
    """
    sub_indices = []
    
    pollutants = [
        ("pm25", pm25),
        ("pm10", pm10),
        ("no2", no2),
        ("so2", so2),
        ("co", co),
        ("o3", o3),
    ]
    
    for name, value in pollutants:
        if value is not None and not pd.isna(value):
            sub_idx = compute_sub_index(value, name)
            if not pd.isna(sub_idx):
                sub_indices.append(sub_idx)
    
    if not sub_indices:
        return np.nan
    
    aqi = max(sub_indices)
    return np.clip(aqi, 0, 500)


def compute_aqi_row(row: pd.Series) -> float:
    """Compute AQI for a DataFrame row."""
    return compute_aqi(
        pm25=row.get("pm25"),
        pm10=row.get("pm10"),
        no2=row.get("no2"),
        so2=row.get("so2"),
        co=row.get("co"),
        o3=row.get("o3"),
    )
```
</action>
<acceptance_criteria>
- `grep -l "BREAKPOINTS = {" src/data/preprocess.py` returns file path
- `grep -l "def compute_sub_index" src/data/preprocess.py` returns file path
- `grep -l "def compute_aqi" src/data/preprocess.py` returns file path
- `python -c "from src.data.preprocess import compute_aqi; aqi = compute_aqi(pm25=45, pm10=80); assert 50 < aqi < 100; print(f'AQI={aqi}')"` exits 0
</acceptance_criteria>
</task>

<task id="3.2">
<title>Implement data cleaning functions</title>
<read_first>
- src/data/preprocess.py (existing code)
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (cleaning rules)
</read_first>
<action>
Add cleaning functions to src/data/preprocess.py:

```python
def clean_aqi_data(
    df: pd.DataFrame,
    gap_fill_limit: int = 3,
    interpolate_limit: int = 12
) -> pd.DataFrame:
    """
    Clean AQI data with gap filling strategy.
    
    Strategy:
    - Forward-fill gaps up to gap_fill_limit hours
    - Linear interpolation for gaps up to interpolate_limit hours
    - Drop remaining gaps (>interpolate_limit hours)
    
    Args:
        df: Input DataFrame with 'timestamp' column
        gap_fill_limit: Max hours for forward-fill (default: 3)
        interpolate_limit: Max hours for interpolation (default: 12)
        
    Returns:
        Cleaned DataFrame
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Ensure hourly frequency
    df = df.set_index("timestamp")
    df = df.resample("1H").asfreq()
    df = df.reset_index()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        original_nan = df[col].isna().sum()
        
        # Step 1: Forward-fill short gaps
        df[col] = df[col].fillna(method="ffill", limit=gap_fill_limit)
        
        # Step 2: Linear interpolation for medium gaps
        df[col] = df[col].interpolate(method="linear", limit=interpolate_limit - gap_fill_limit)
        
        remaining_nan = df[col].isna().sum()
        filled = original_nan - remaining_nan
        
        if filled > 0:
            logger.info(f"Filled {filled} gaps in {col}")
        if remaining_nan > 0:
            logger.warning(f"Dropping {remaining_nan} rows with >12h gaps in {col}")
    
    # Drop rows with any remaining NaN in critical columns
    critical_cols = ["pm25", "pm10", "aqi"]
    critical_cols = [c for c in critical_cols if c in df.columns]
    
    if critical_cols:
        initial_len = len(df)
        df = df.dropna(subset=critical_cols)
        dropped = initial_len - len(df)
        if dropped > 0:
            logger.warning(f"Dropped {dropped} rows with missing critical values")
    
    return df


def clip_outliers(
    df: pd.DataFrame,
    columns: List[str] = None,
    percentile: float = 0.999
) -> pd.DataFrame:
    """
    Clip outliers beyond the specified percentile.
    
    Args:
        df: Input DataFrame
        columns: Columns to clip (None = all numeric)
        percentile: Percentile threshold (default: 0.999)
        
    Returns:
        DataFrame with clipped values
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        threshold = df[col].quantile(percentile)
        outliers = df[col] > threshold
        outlier_count = outliers.sum()
        
        if outlier_count > 0:
            original_values = df.loc[outliers, col].tolist()
            df.loc[outliers, col] = threshold
            logger.warning(
                f"Clipped {outlier_count} outliers in {col} > {threshold:.2f}. "
                f"Original max: {max(original_values):.2f}"
            )
    
    return df
```
</action>
<acceptance_criteria>
- `grep -l "def clean_aqi_data" src/data/preprocess.py` returns file path
- `grep -l "def clip_outliers" src/data/preprocess.py` returns file path
- `grep -l "fillna.*ffill.*limit" src/data/preprocess.py` returns file path
</acceptance_criteria>
</task>

<task id="3.3">
<title>Implement feature engineering functions</title>
<read_first>
- src/data/preprocess.py (existing code)
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (feature specs)
</read_first>
<action>
Add feature engineering functions to src/data/preprocess.py:

```python
def create_lag_features(
    df: pd.DataFrame,
    column: str,
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lag features for a column.
    
    Args:
        df: Input DataFrame
        column: Column to create lags for
        lags: List of lag values in hours (default: [1, 6, 24])
        
    Returns:
        DataFrame with new lag columns
    """
    if lags is None:
        lags = [1, 6, 24]
    
    df = df.copy()
    
    for lag in lags:
        lag_col = f"{column}_lag_{lag}h"
        df[lag_col] = df[column].shift(lag)
        logger.debug(f"Created lag feature: {lag_col}")
    
    return df


def create_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Create rolling mean and std features.
    
    Args:
        df: Input DataFrame
        column: Column to compute rolling stats for
        windows: List of window sizes in hours (default: [24, 72])
        
    Returns:
        DataFrame with new rolling columns
    """
    if windows is None:
        windows = [24, 72]
    
    df = df.copy()
    
    for window in windows:
        mean_col = f"{column}_{window}h_mean"
        std_col = f"{column}_{window}h_std"
        
        # min_periods = half the window to allow partial calculations
        df[mean_col] = df[column].rolling(window=window, min_periods=window // 2).mean()
        df[std_col] = df[column].rolling(window=window, min_periods=window // 2).std()
        
        logger.debug(f"Created rolling features: {mean_col}, {std_col}")
    
    return df


def compute_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute wind vector components from speed and direction.
    
    wind_u = wind_speed * cos(wind_direction_rad)  [East-West component]
    wind_v = wind_speed * sin(wind_direction_rad)  [North-South component]
    
    Args:
        df: DataFrame with 'wind_speed' and 'wind_direction' columns
        
    Returns:
        DataFrame with 'wind_u' and 'wind_v' columns added
    """
    df = df.copy()
    
    if "wind_speed" not in df.columns or "wind_direction" not in df.columns:
        logger.warning("Wind columns not found, skipping wind vector computation")
        return df
    
    # Convert direction to radians (direction is in degrees, 0=North, 90=East)
    wind_dir_rad = np.radians(df["wind_direction"])
    
    df["wind_u"] = df["wind_speed"] * np.cos(wind_dir_rad)
    df["wind_v"] = df["wind_speed"] * np.sin(wind_dir_rad)
    
    logger.info("Computed wind vector components (wind_u, wind_v)")
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features for temporal patterns.
    
    Features:
    - Hour sine/cosine encoding (captures daily cyclicity)
    - Day-of-week one-hot encoding
    - is_holiday binary flag
    - is_diwali_week binary flag
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with calendar features added
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Hour sine/cosine encoding (captures rush-hour patterns)
    hour = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week encoding (0=Monday, 6=Sunday)
    dow = df["timestamp"].dt.dayofweek
    for i in range(7):
        df[f"dow_{i}"] = (dow == i).astype(int)
    
    # Is weekend flag
    df["is_weekend"] = (dow >= 5).astype(int)
    
    # Diwali dates (approximate, varies by year)
    # 2022: October 24, 2023: November 12
    diwali_dates = [
        pd.Timestamp("2022-10-24"),
        pd.Timestamp("2023-11-12"),
        pd.Timestamp("2024-11-01"),
    ]
    
    df["is_diwali_week"] = 0
    for diwali_date in diwali_dates:
        start = diwali_date - pd.Timedelta(days=2)
        end = diwali_date + pd.Timedelta(days=2)
        mask = (df["timestamp"].dt.date >= start.date()) & (df["timestamp"].dt.date <= end.date())
        df.loc[mask, "is_diwali_week"] = 1
    
    # Indian national holidays (simplified list)
    holidays = [
        "01-26",  # Republic Day
        "08-15",  # Independence Day
        "10-02",  # Gandhi Jayanti
        "12-25",  # Christmas
    ]
    
    df["is_holiday"] = 0
    for holiday in holidays:
        month, day = map(int, holiday.split("-"))
        mask = (df["timestamp"].dt.month == month) & (df["timestamp"].dt.day == day)
        df.loc[mask, "is_holiday"] = 1
    
    logger.info("Created calendar features: hour_sin/cos, dow_*, is_weekend, is_diwali_week, is_holiday")
    return df
```
</action>
<acceptance_criteria>
- `grep -l "def create_lag_features" src/data/preprocess.py` returns file path
- `grep -l "def create_rolling_features" src/data/preprocess.py` returns file path
- `grep -l "def compute_wind_vectors" src/data/preprocess.py` returns file path
- `grep -l "def create_calendar_features" src/data/preprocess.py` returns file path
- `grep -l "is_diwali_week" src/data/preprocess.py` returns file path
</acceptance_criteria>
</task>

<task id="3.4">
<title>Implement normalization and splitting</title>
<read_first>
- src/data/preprocess.py (existing code)
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (split requirements)
</read_first>
<action>
Add normalization and splitting functions to src/data/preprocess.py:

```python
def normalize_features(
    df: pd.DataFrame,
    scaler_path: str = "models/scaler.pkl",
    fit: bool = True,
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler.
    
    Args:
        df: Input DataFrame
        scaler_path: Path to save/load fitted scaler
        fit: If True, fit scaler on data; if False, load existing scaler
        exclude_cols: Columns to exclude from normalization
        
    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    df = df.copy()
    
    if exclude_cols is None:
        exclude_cols = ["timestamp", "node_id"]
    
    # Get numeric columns to normalize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]
    
    # Exclude binary columns (0/1 values only)
    binary_cols = [c for c in cols_to_normalize if df[c].isin([0, 1]).all()]
    cols_to_normalize = [c for c in cols_to_normalize if c not in binary_cols]
    
    scaler_dir = Path(scaler_path).parent
    scaler_dir.mkdir(parents=True, exist_ok=True)
    
    if fit:
        scaler = MinMaxScaler(feature_range=(0, 1))
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        logger.info(f"Fitted and saved scaler to {scaler_path}")
    else:
        if not Path(scaler_path).exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run with fit=True first.")
        
        scaler = joblib.load(scaler_path)
        df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
        logger.info(f"Loaded scaler from {scaler_path}")
    
    return df, scaler


def create_splits(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    output_dir: str = "data/splits",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create chronological train/val/test splits.
    
    NO SHUFFLING - preserves temporal order for time series.
    
    Args:
        df: Input DataFrame (must be sorted by timestamp)
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        output_dir: Directory to save split indices
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Verify no overlap
    assert train_df["timestamp"].max() < val_df["timestamp"].min(), "Train/Val overlap!"
    assert val_df["timestamp"].max() < test_df["timestamp"].min(), "Val/Test overlap!"
    
    # Save split info
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits_info = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(1 - train_ratio - val_ratio, 2),
        "train": {
            "start_idx": 0,
            "end_idx": train_end,
            "start_date": str(train_df["timestamp"].iloc[0]),
            "end_date": str(train_df["timestamp"].iloc[-1]),
            "count": len(train_df)
        },
        "val": {
            "start_idx": train_end,
            "end_idx": val_end,
            "start_date": str(val_df["timestamp"].iloc[0]),
            "end_date": str(val_df["timestamp"].iloc[-1]),
            "count": len(val_df)
        },
        "test": {
            "start_idx": val_end,
            "end_idx": n,
            "start_date": str(test_df["timestamp"].iloc[0]),
            "end_date": str(test_df["timestamp"].iloc[-1]),
            "count": len(test_df)
        }
    }
    
    splits_file = output_path / "splits.json"
    with open(splits_file, "w") as f:
        json.dump(splits_info, f, indent=2)
    
    logger.info(
        f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    logger.info(f"Saved split info to {splits_file}")
    
    return train_df, val_df, test_df


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save processed DataFrame to Parquet."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(path, compression="snappy", index=False)
    logger.info(f"Saved processed data to {path} ({len(df)} rows)")


def load_processed(path: str) -> pd.DataFrame:
    """Load processed DataFrame from Parquet."""
    df = pd.read_parquet(path)
    logger.info(f"Loaded processed data from {path} ({len(df)} rows)")
    return df
```
</action>
<acceptance_criteria>
- `grep -l "def normalize_features" src/data/preprocess.py` returns file path
- `grep -l "def create_splits" src/data/preprocess.py` returns file path
- `grep -l "def save_processed" src/data/preprocess.py` returns file path
- `grep -l "NO SHUFFLING" src/data/preprocess.py` returns file path
</acceptance_criteria>
</task>

<task id="3.5">
<title>Create test_preprocess.py unit tests</title>
<read_first>
- src/data/preprocess.py (functions to test)
</read_first>
<action>
Create tests/test_preprocess.py:

```python
"""Unit tests for data preprocessing functions."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestComputeAQI:
    """Tests for AQI computation."""
    
    def test_compute_sub_index_pm25(self):
        from src.data.preprocess import compute_sub_index
        
        # PM2.5 = 45 µg/m³ should be in 51-100 AQI range (31-60 conc)
        sub_idx = compute_sub_index(45, "pm25")
        assert 51 <= sub_idx <= 100
        
    def test_compute_sub_index_zero(self):
        from src.data.preprocess import compute_sub_index
        
        sub_idx = compute_sub_index(0, "pm25")
        assert sub_idx == 0
        
    def test_compute_sub_index_nan(self):
        from src.data.preprocess import compute_sub_index
        
        sub_idx = compute_sub_index(np.nan, "pm25")
        assert pd.isna(sub_idx)
        
    def test_compute_aqi_single_pollutant(self):
        from src.data.preprocess import compute_aqi
        
        aqi = compute_aqi(pm25=45)
        assert 0 <= aqi <= 500
        
    def test_compute_aqi_multiple_pollutants(self):
        from src.data.preprocess import compute_aqi
        
        aqi = compute_aqi(pm25=45, pm10=80, no2=30)
        assert 0 <= aqi <= 500
        
    def test_compute_aqi_max_selection(self):
        from src.data.preprocess import compute_aqi
        
        # High PM2.5 should dominate
        aqi = compute_aqi(pm25=200, pm10=50)
        assert aqi > 200  # PM2.5=200 is in "Very Poor" category
        
    def test_compute_aqi_range(self):
        from src.data.preprocess import compute_aqi
        
        # Test edge cases
        assert compute_aqi(pm25=0) == 0
        assert compute_aqi(pm25=500) == 500


class TestCleanAQIData:
    """Tests for data cleaning functions."""
    
    @pytest.fixture
    def sample_df_with_gaps(self):
        """Sample DataFrame with various gap patterns."""
        dates = pd.date_range("2023-01-01", periods=24, freq="H")
        values = [100.0] * 24
        
        # Introduce gaps
        values[5] = np.nan  # 1-hour gap
        values[10:13] = [np.nan, np.nan, np.nan]  # 3-hour gap
        values[18:23] = [np.nan] * 5  # 5-hour gap
        
        return pd.DataFrame({"timestamp": dates, "aqi": values, "pm25": values})
    
    def test_gap_filling_short(self, sample_df_with_gaps):
        from src.data.preprocess import clean_aqi_data
        
        df = clean_aqi_data(sample_df_with_gaps)
        
        # Short gaps should be filled
        assert df["aqi"].isna().sum() < sample_df_with_gaps["aqi"].isna().sum()
        
    def test_gap_filling_preserves_valid(self, sample_df_with_gaps):
        from src.data.preprocess import clean_aqi_data
        
        df = clean_aqi_data(sample_df_with_gaps)
        
        # Original valid values should be preserved
        valid_count = (~sample_df_with_gaps["aqi"].isna()).sum()
        assert len(df) >= valid_count - 5  # Some rows may be dropped


class TestClipOutliers:
    """Tests for outlier clipping."""
    
    def test_clips_high_values(self):
        from src.data.preprocess import clip_outliers
        
        df = pd.DataFrame({"aqi": [50, 100, 150, 200, 1000]})
        df_clipped = clip_outliers(df, columns=["aqi"], percentile=0.8)
        
        assert df_clipped["aqi"].max() < 1000
        
    def test_preserves_normal_values(self):
        from src.data.preprocess import clip_outliers
        
        df = pd.DataFrame({"aqi": [50, 100, 150, 200, 250]})
        df_clipped = clip_outliers(df, columns=["aqi"], percentile=0.999)
        
        # All values should be preserved (no extreme outliers)
        pd.testing.assert_frame_equal(df, df_clipped)


class TestFeatureEngineering:
    """Tests for feature engineering functions."""
    
    @pytest.fixture
    def sample_df(self):
        dates = pd.date_range("2023-01-01", periods=100, freq="H")
        return pd.DataFrame({
            "timestamp": dates,
            "aqi": np.random.uniform(50, 200, 100),
            "pm25": np.random.uniform(20, 100, 100),
            "wind_speed": np.random.uniform(1, 15, 100),
            "wind_direction": np.random.uniform(0, 360, 100)
        })
    
    def test_create_lag_features(self, sample_df):
        from src.data.preprocess import create_lag_features
        
        df = create_lag_features(sample_df, "aqi", lags=[1, 6, 24])
        
        assert "aqi_lag_1h" in df.columns
        assert "aqi_lag_6h" in df.columns
        assert "aqi_lag_24h" in df.columns
        
        # Check shift is correct
        assert df["aqi_lag_1h"].iloc[1] == df["aqi"].iloc[0]
        
    def test_create_rolling_features(self, sample_df):
        from src.data.preprocess import create_rolling_features
        
        df = create_rolling_features(sample_df, "pm25", windows=[24])
        
        assert "pm25_24h_mean" in df.columns
        assert "pm25_24h_std" in df.columns
        
    def test_compute_wind_vectors(self, sample_df):
        from src.data.preprocess import compute_wind_vectors
        
        df = compute_wind_vectors(sample_df)
        
        assert "wind_u" in df.columns
        assert "wind_v" in df.columns
        
        # Check vector magnitude roughly equals wind speed
        magnitude = np.sqrt(df["wind_u"]**2 + df["wind_v"]**2)
        np.testing.assert_array_almost_equal(magnitude, df["wind_speed"], decimal=5)
        
    def test_create_calendar_features(self, sample_df):
        from src.data.preprocess import create_calendar_features
        
        df = create_calendar_features(sample_df)
        
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "is_weekend" in df.columns
        assert "is_diwali_week" in df.columns
        
        # Check hour encoding range
        assert df["hour_sin"].min() >= -1 and df["hour_sin"].max() <= 1
        assert df["hour_cos"].min() >= -1 and df["hour_cos"].max() <= 1


class TestNormalization:
    """Tests for normalization functions."""
    
    def test_normalize_range(self, tmp_path):
        from src.data.preprocess import normalize_features
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="H"),
            "aqi": np.random.uniform(0, 500, 10),
            "pm25": np.random.uniform(0, 200, 10)
        })
        
        scaler_path = str(tmp_path / "scaler.pkl")
        df_norm, scaler = normalize_features(df, scaler_path=scaler_path, fit=True)
        
        # Check normalized range is [0, 1]
        assert df_norm["aqi"].min() >= 0
        assert df_norm["aqi"].max() <= 1
        assert df_norm["pm25"].min() >= 0
        assert df_norm["pm25"].max() <= 1


class TestSplits:
    """Tests for train/val/test splitting."""
    
    def test_chronological_split(self, tmp_path):
        from src.data.preprocess import create_splits
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
            "aqi": range(100)
        })
        
        train, val, test = create_splits(df, output_dir=str(tmp_path))
        
        # Check sizes
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        
        # Check no overlap
        assert train["timestamp"].max() < val["timestamp"].min()
        assert val["timestamp"].max() < test["timestamp"].min()
        
    def test_splits_file_created(self, tmp_path):
        from src.data.preprocess import create_splits
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="H"),
            "aqi": range(100)
        })
        
        create_splits(df, output_dir=str(tmp_path))
        
        assert (tmp_path / "splits.json").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```
</action>
<acceptance_criteria>
- `Test-Path tests/test_preprocess.py` returns True
- `grep -l "def test_compute_aqi_range" tests/test_preprocess.py` returns file path
- `grep -l "def test_create_lag_features" tests/test_preprocess.py` returns file path
- `grep -l "def test_chronological_split" tests/test_preprocess.py` returns file path
- `python -m pytest tests/test_preprocess.py --collect-only -q | Select-String "test"` shows at least 10 tests
</acceptance_criteria>
</task>

## Verification

```powershell
# Verify imports work
python -c "from src.data.preprocess import compute_aqi, clean_aqi_data, clip_outliers, create_lag_features, create_rolling_features, compute_wind_vectors, create_calendar_features, normalize_features, create_splits, save_processed, load_processed; print('All preprocess functions imported')"

# Run tests
python -m pytest tests/test_preprocess.py -v

# Test AQI computation
python -c "from src.data.preprocess import compute_aqi; print(f'AQI(pm25=45, pm10=80)={compute_aqi(pm25=45, pm10=80):.1f}')"
```
