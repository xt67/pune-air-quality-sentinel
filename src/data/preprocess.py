"""Data preprocessing pipeline for AQI data."""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Indian public holidays and Diwali dates
DIWALI_DATES = [
    "2022-10-24",
    "2023-11-12",
    "2024-11-01",
]


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


def _clean_single_node(
    df: pd.DataFrame,
    gap_fill_limit: int = 3,
    interpolate_limit: int = 12
) -> pd.DataFrame:
    """Clean a single node's data (no node_id grouping)."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Ensure hourly frequency
    df = df.set_index("timestamp")
    df = df.resample("1h").asfreq()
    df = df.reset_index()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        original_nan = df[col].isna().sum()
        
        # Step 1: Forward-fill short gaps
        df[col] = df[col].ffill(limit=gap_fill_limit)
        
        # Step 2: Linear interpolation for medium gaps
        df[col] = df[col].interpolate(method="linear", limit=interpolate_limit - gap_fill_limit)
        
        remaining_nan = df[col].isna().sum()
        filled = original_nan - remaining_nan
        
        if filled > 0:
            logger.debug(f"Filled {filled} gaps in {col}")
    
    return df


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
    # Handle multi-node data by processing each node separately
    if "node_id" in df.columns:
        node_ids = df["node_id"].unique()
        cleaned_dfs = []
        
        for node_id in node_ids:
            node_df = df[df["node_id"] == node_id].copy()
            cleaned_node = _clean_single_node(node_df, gap_fill_limit, interpolate_limit)
            cleaned_node["node_id"] = node_id
            cleaned_dfs.append(cleaned_node)
        
        df = pd.concat(cleaned_dfs, ignore_index=True)
        df = df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
    else:
        df = _clean_single_node(df, gap_fill_limit, interpolate_limit)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
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
    columns: Optional[List[str]] = None,
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
            
        upper_bound = df[col].quantile(percentile)
        lower_bound = df[col].quantile(1 - percentile)
        
        clipped_count = ((df[col] > upper_bound) | (df[col] < lower_bound)).sum()
        
        if clipped_count > 0:
            logger.info(f"Clipping {clipped_count} outliers in {col} "
                       f"(bounds: {lower_bound:.2f} - {upper_bound:.2f})")
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def engineer_lag_features(
    df: pd.DataFrame,
    target_col: str = "aqi",
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lag features for time series modeling.
    
    Args:
        df: Input DataFrame (must be sorted by timestamp)
        target_col: Column to create lags for
        lags: List of lag hours (default: [1, 6, 24])
        
    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = [1, 6, 24]
    
    df = df.copy()
    
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}h"
        df[col_name] = df[target_col].shift(lag)
        logger.debug(f"Created lag feature: {col_name}")
    
    return df


def engineer_rolling_features(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Create rolling window statistics.
    
    Args:
        df: Input DataFrame (must be sorted by timestamp)
        columns: Columns to compute rolling stats for
        windows: Window sizes in hours (default: [24, 72])
        
    Returns:
        DataFrame with rolling features added
    """
    if columns is None:
        columns = ["pm25"]
    if windows is None:
        windows = [24, 72]
    
    df = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            # Rolling mean
            mean_col = f"{col}_{window}h_mean"
            df[mean_col] = df[col].rolling(window=window, min_periods=1).mean()
            
            # Rolling std (only for longer windows)
            if window >= 24:
                std_col = f"{col}_{window}h_std"
                df[std_col] = df[col].rolling(window=window, min_periods=1).std()
            
            logger.debug(f"Created rolling features for {col} with window {window}h")
    
    return df


def create_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wind speed and direction to u/v vector components.
    
    Args:
        df: DataFrame with 'wind_speed' and 'wind_direction' columns
        
    Returns:
        DataFrame with wind_u and wind_v columns added
    """
    df = df.copy()
    
    if "wind_speed" not in df.columns or "wind_direction" not in df.columns:
        logger.warning("Missing wind columns, skipping wind vector creation")
        return df
    
    # Convert direction to radians
    direction_rad = np.deg2rad(df["wind_direction"])
    
    # U component (east-west, positive = eastward)
    df["wind_u"] = df["wind_speed"] * np.sin(direction_rad)
    
    # V component (north-south, positive = northward)
    df["wind_v"] = df["wind_speed"] * np.cos(direction_rad)
    
    logger.info("Created wind vector components (wind_u, wind_v)")
    
    return df


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create calendar-based features for temporal patterns.
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        DataFrame with calendar features added
    """
    df = df.copy()
    
    if "timestamp" not in df.columns:
        logger.warning("No timestamp column found")
        return df
    
    ts = pd.to_datetime(df["timestamp"])
    
    # Hour encoding (cyclical)
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    
    # Day of week (one-hot would be better, but cyclical for simplicity)
    df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)
    
    # Month encoding (for seasonal patterns)
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    
    # Weekend flag
    df["is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype(int)
    
    # Diwali week flag
    diwali_dates = pd.to_datetime(DIWALI_DATES)
    df["is_diwali_week"] = 0
    for diwali in diwali_dates:
        # 5-day window around Diwali
        start = diwali - pd.Timedelta(days=2)
        end = diwali + pd.Timedelta(days=2)
        mask = (ts >= start) & (ts <= end)
        df.loc[mask, "is_diwali_week"] = 1
    
    # Monsoon flag (July-September)
    df["is_monsoon"] = ts.dt.month.isin([7, 8, 9]).astype(int)
    
    logger.info("Created calendar features")
    
    return df


def normalize_features(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    scaler_path: Optional[str] = None,
    fit: bool = True
) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """
    Normalize features using MinMaxScaler.
    
    Args:
        df: Input DataFrame
        exclude_cols: Columns to exclude from normalization
        scaler_path: Path to save/load scaler
        fit: If True, fit scaler; if False, transform only
        
    Returns:
        Tuple of (normalized DataFrame, fitted scaler)
    """
    df = df.copy()
    
    if exclude_cols is None:
        exclude_cols = ["timestamp", "node_id"]
    
    # Identify columns to normalize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [c for c in numeric_cols if c not in exclude_cols]
    
    if fit:
        scaler = MinMaxScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        
        if scaler_path:
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")
    else:
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
            df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])
        else:
            raise ValueError(f"Scaler not found at {scaler_path}")
    
    logger.info(f"Normalized {len(cols_to_normalize)} features")
    
    return df, scaler


def create_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create chronological train/val/test split.
    
    Args:
        df: Input DataFrame (must be sorted by timestamp)
        train_ratio: Training set ratio (default: 0.7)
        val_ratio: Validation set ratio (default: 0.15)
        test_ratio: Test set ratio (default: 0.15)
        output_path: Path to save split indices
        
    Returns:
        Dictionary with split indices and DataFrames
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    splits = {
        "train_indices": list(range(0, train_end)),
        "val_indices": list(range(train_end, val_end)),
        "test_indices": list(range(val_end, n)),
        "train_df": df.iloc[:train_end].copy(),
        "val_df": df.iloc[train_end:val_end].copy(),
        "test_df": df.iloc[val_end:].copy(),
    }
    
    logger.info(f"Split sizes - Train: {len(splits['train_df'])}, "
                f"Val: {len(splits['val_df'])}, Test: {len(splits['test_df'])}")
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "train_indices": splits["train_indices"],
                "val_indices": splits["val_indices"],
                "test_indices": splits["test_indices"],
            }, f)
        logger.info(f"Saved split indices to {output_path}")
    
    return splits


def preprocess_pipeline(
    df: pd.DataFrame,
    output_parquet: str = "data/processed/pune_aqi_processed.parquet",
    scaler_path: str = "models/scaler.pkl",
    splits_path: str = "data/splits/splits.json"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run full preprocessing pipeline.
    
    Args:
        df: Raw input DataFrame
        output_parquet: Path to save processed data
        scaler_path: Path to save scaler
        splits_path: Path to save split indices
        
    Returns:
        Tuple of (processed DataFrame, splits dictionary)
    """
    logger.info("Starting preprocessing pipeline...")
    
    # Step 1: Clean data
    df = clean_aqi_data(df)
    logger.info(f"After cleaning: {len(df)} rows")
    
    # Step 2: Clip outliers
    df = clip_outliers(df)
    
    # Step 3: Compute AQI if not present
    if "aqi" not in df.columns:
        df["aqi"] = df.apply(compute_aqi_row, axis=1)
        logger.info("Computed AQI from pollutant values")
    
    # Step 4: Engineer lag features
    df = engineer_lag_features(df, target_col="aqi")
    
    # Step 5: Engineer rolling features
    df = engineer_rolling_features(df, columns=["pm25", "aqi"])
    
    # Step 6: Create wind vectors
    df = create_wind_vectors(df)
    
    # Step 7: Create calendar features
    df = create_calendar_features(df)
    
    # Drop rows with NaN from lag/rolling features
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_len - len(df)} rows with NaN after feature engineering")
    
    # Step 8: Create splits (before normalization to fit scaler on train only)
    splits = create_train_val_test_split(
        df,
        output_path=splits_path
    )
    
    # Step 9: Normalize (fit on train, transform all)
    train_df = splits["train_df"]
    _, scaler = normalize_features(train_df, scaler_path=scaler_path, fit=True)
    
    # Transform full dataset
    df, _ = normalize_features(df, scaler_path=scaler_path, fit=False)
    
    # Save processed data
    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    logger.info(f"Saved processed data to {output_parquet}")
    
    return df, splits
