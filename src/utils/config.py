"""Config loader utility with YAML support and sensible defaults."""
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "simulated_dir": "data/simulated",
        "splits_dir": "data/splits",
        "parquet_path": "data/processed/pune_aqi_processed.parquet",
    },
    "preprocessing": {
        "gap_fill_limit_hours": 3,
        "interpolate_limit_hours": 12,
        "outlier_percentile": 0.999,
    },
    "training": {
        "seed": 42,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
    },
}


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with fallback to defaults.
    
    Args:
        path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(path)
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        logger.warning(f"Config file not found at {path}, using defaults")
        return DEFAULT_CONFIG.copy()


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested config value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path (e.g., "data.raw_dir")
        default: Default value if key not found
        
    Returns:
        Config value or default
    """
    keys = key_path.split(".")
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
