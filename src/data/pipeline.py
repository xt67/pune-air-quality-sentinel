"""Unified data pipeline for Pune Air Quality Sentinel."""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            df = fetch.fetch_weather_all_nodes()
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
) -> Dict[str, Any]:
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
    if pollutant_cols:
        df = preprocess.clip_outliers(df, columns=pollutant_cols)
    
    # Step 3: Compute AQI if not present
    if "aqi" not in df.columns and "pm25" in df.columns:
        logger.info("Step 3: Computing AQI...")
        df["aqi"] = df.apply(preprocess.compute_aqi_row, axis=1)
    
    # Step 4: Feature engineering
    logger.info("Step 4: Engineering features...")
    if "aqi" in df.columns:
        df = preprocess.engineer_lag_features(df, "aqi", lags=[1, 6, 24])
    if "pm25" in df.columns:
        df = preprocess.engineer_rolling_features(df, "pm25", windows=[24, 72])
    
    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        df = preprocess.create_wind_vectors(df)
    
    df = preprocess.create_calendar_features(df)
    
    # Step 5: Drop rows with NaN from feature engineering
    initial_rows = len(df)
    df = df.dropna()
    logger.info(f"Dropped {initial_rows - len(df)} rows after feature engineering")
    
    if len(df) == 0:
        raise ValueError("No data remaining after preprocessing")
    
    # Save full processed data BEFORE splits
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")
    
    # Step 6: Create splits BEFORE normalization (to fit scaler on train only)
    logger.info("Step 5: Creating train/val/test splits...")
    Path(splits_dir).mkdir(parents=True, exist_ok=True)
    splits_path = str(Path(splits_dir) / "splits.json")
    splits = preprocess.create_train_val_test_split(
        df, train_ratio=0.7, val_ratio=0.15, output_path=splits_path
    )
    train_df = splits["train_df"]
    val_df = splits["val_df"]
    test_df = splits["test_df"]
    
    # Step 7: Normalize (fit on train only)
    logger.info("Step 6: Normalizing features...")
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Identify columns to exclude from normalization
    exclude_cols = ["timestamp", "node_id"]
    exclude_cols = [c for c in exclude_cols if c in train_df.columns]
    
    train_df_norm, _ = preprocess.normalize_features(
        train_df.copy(), scaler_path=scaler_path, fit=True, exclude_cols=exclude_cols
    )
    val_df_norm, _ = preprocess.normalize_features(
        val_df.copy(), scaler_path=scaler_path, fit=False, exclude_cols=exclude_cols
    )
    test_df_norm, _ = preprocess.normalize_features(
        test_df.copy(), scaler_path=scaler_path, fit=False, exclude_cols=exclude_cols
    )
    
    logger.info("Preprocessing pipeline complete!")
    
    return {
        "train": train_df_norm,
        "val": val_df_norm,
        "test": test_df_norm,
        "full": df
    }


def run_iot_simulation(
    output_dir: str = "data/simulated",
    start_date: str = "2022-01-01",
    end_date: str = "2023-12-31",
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Run IoT sensor simulation.
    
    Args:
        output_dir: Directory to save simulated CSVs
        start_date: Simulation start date
        end_date: Simulation end date
        seed: Random seed
        
    Returns:
        Dict mapping node_id to DataFrame
    """
    logger.info(f"Generating IoT simulation: {start_date} to {end_date}")
    
    node_data = iot_sim.generate_all_nodes(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        seed=seed
    )
    
    logger.info(f"Generated {len(node_data)} simulated sensor files")
    return node_data


def load_simulated_data(
    simulated_dir: str = "data/simulated"
) -> pd.DataFrame:
    """
    Load and combine all simulated node CSVs.
    
    Args:
        simulated_dir: Directory containing node_*.csv files
        
    Returns:
        Combined DataFrame with all nodes
    """
    path = Path(simulated_dir)
    csv_files = sorted(path.glob("node_*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No simulated CSVs found in {simulated_dir}")
    
    dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(["timestamp", "node_id"]).reset_index(drop=True)
    
    logger.info(f"Loaded {len(csv_files)} simulated files: {len(combined)} total rows")
    return combined


def run_full_pipeline(
    use_simulated: bool = True,
    seed: int = 42
) -> Dict[str, Any]:
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
        node_data = run_iot_simulation(seed=seed)
        results["simulated_nodes"] = node_data
        
        # Load simulated data
        df = load_simulated_data()
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


def validate_pipeline_outputs() -> Dict[str, bool]:
    """
    Validate that all expected pipeline outputs exist.
    
    Returns:
        Dict mapping check name to pass/fail status
    """
    checks = {}
    
    # Check processed parquet
    parquet_path = Path("data/processed/pune_aqi_processed.parquet")
    checks["parquet_exists"] = parquet_path.exists()
    
    # Check simulated CSVs
    simulated_path = Path("data/simulated")
    if simulated_path.exists():
        csv_files = list(simulated_path.glob("node_*.csv"))
        checks["simulated_csvs"] = len(csv_files) == 10
    else:
        checks["simulated_csvs"] = False
    
    # Check scaler
    scaler_path = Path("models/scaler.pkl")
    checks["scaler_exists"] = scaler_path.exists()
    
    # Check splits
    splits_path = Path("data/splits")
    if splits_path.exists():
        checks["splits_exist"] = (
            (splits_path / "train_idx.npy").exists() and
            (splits_path / "val_idx.npy").exists() and
            (splits_path / "test_idx.npy").exists()
        )
    else:
        checks["splits_exist"] = False
    
    # If parquet exists, validate content
    if checks["parquet_exists"]:
        df = pd.read_parquet(parquet_path)
        checks["no_nan_values"] = df.isna().sum().sum() == 0
        if "aqi" in df.columns:
            checks["aqi_in_range"] = (df["aqi"].min() >= 0) and (df["aqi"].max() <= 500)
        else:
            checks["aqi_in_range"] = True  # No AQI column to check
    
    return checks


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument(
        "--simulated", "-s",
        action="store_true",
        default=True,
        help="Use simulated IoT data (default: True)"
    )
    parser.add_argument(
        "--real", "-r",
        action="store_true",
        help="Use real data from APIs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing outputs"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    
    if args.validate:
        print("\n=== Validating Pipeline Outputs ===")
        checks = validate_pipeline_outputs()
        for check_name, passed in checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check_name}: {status}")
        
        all_passed = all(checks.values())
        print(f"\nOverall: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
        return
    
    use_simulated = not args.real
    results = run_full_pipeline(use_simulated=use_simulated, seed=args.seed)
    
    print("\n=== Pipeline Results ===")
    for key, value in results["summary"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
