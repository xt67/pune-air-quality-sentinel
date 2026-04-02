---
phase: 1
plan: 1
title: "Project Structure & Utilities Setup"
wave: 1
depends_on: []
files_modified:
  - src/utils/seed.py
  - src/utils/config.py
  - src/utils/logger.py
  - src/utils/error_guard.py
  - src/__init__.py
  - src/data/__init__.py
  - src/models/__init__.py
  - src/evaluate/__init__.py
  - src/viz/__init__.py
  - src/utils/__init__.py
  - configs/data.yaml
  - configs/lstm.yaml
  - configs/stgnn.yaml
  - .env.example
  - data/graph/node_coords.json
requirements_addressed: [INFRA-01]
autonomous: true
---

# Plan 1: Project Structure & Utilities Setup

<objective>
Create the full repository directory structure and core utility modules that all other plans depend on.
</objective>

<must_haves>
- All directory structure per PRD Section 4.2
- Seed utility with full reproducibility
- Config loader with YAML support and defaults
- JSON structured logger
- Error guard utilities
- Node coordinates JSON for 10 Pune locations
</must_haves>

## Tasks

<task id="1.1">
<title>Create repository directory structure</title>
<read_first>
- .planning/PROJECT.md (repository structure requirements)
</read_first>
<action>
Create the full directory structure:

```powershell
# Source directories
New-Item -ItemType Directory -Path src/data, src/models, src/evaluate, src/viz, src/utils -Force
New-Item -ItemType Directory -Path app/components -Force

# Data directories  
New-Item -ItemType Directory -Path data/raw, data/processed, data/simulated, data/splits, data/graph -Force

# Other directories
New-Item -ItemType Directory -Path models, configs, notebooks, tests, outputs/logs, outputs/plots, outputs/heatmaps -Force

# Create __init__.py files
$initPaths = @("src", "src/data", "src/models", "src/evaluate", "src/viz", "src/utils", "app", "app/components", "tests")
foreach ($path in $initPaths) {
    New-Item -Path "$path/__init__.py" -ItemType File -Force
}
```
</action>
<acceptance_criteria>
- `Test-Path src/data/__init__.py` returns True
- `Test-Path src/models/__init__.py` returns True
- `Test-Path src/utils/__init__.py` returns True
- `Test-Path data/processed` returns True
- `Test-Path data/simulated` returns True
- `Test-Path data/graph` returns True
- `Test-Path configs` returns True
- `Test-Path outputs/logs` returns True
</acceptance_criteria>
</task>

<task id="1.2">
<title>Implement seed utility for reproducibility</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (reproducibility requirements)
</read_first>
<action>
Create src/utils/seed.py with set_seed function:

```python
"""Seed utility for full reproducibility across all random generators."""
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```
</action>
<acceptance_criteria>
- `grep -l "def set_seed" src/utils/seed.py` returns file path
- `grep -l "torch.backends.cudnn.deterministic" src/utils/seed.py` returns file path
- `grep -l "np.random.seed" src/utils/seed.py` returns file path
- `python -c "from src.utils.seed import set_seed; set_seed(42)"` exits 0
</acceptance_criteria>
</task>

<task id="1.3">
<title>Implement config loader utility</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (config requirements)
</read_first>
<action>
Create src/utils/config.py with YAML loading and defaults:

```python
"""Config loader utility with YAML support and sensible defaults."""
import os
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
```
</action>
<acceptance_criteria>
- `grep -l "def load_config" src/utils/config.py` returns file path
- `grep -l "yaml.safe_load" src/utils/config.py` returns file path
- `grep -l "DEFAULT_CONFIG" src/utils/config.py` returns file path
- `python -c "from src.utils.config import load_config; c = load_config('nonexistent.yaml'); assert 'data' in c"` exits 0
</acceptance_criteria>
</task>

<task id="1.4">
<title>Implement JSON structured logger</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (logging requirements)
</read_first>
<action>
Create src/utils/logger.py with JSON line format:

```python
"""JSON structured logger with rotation support."""
import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict


class JsonFormatter(logging.Formatter):
    """Format log records as JSON lines."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, "extra"):
            log_obj["extra"] = record.extra
            
        return json.dumps(log_obj)


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    """
    Get a logger with JSON formatting and file rotation.
    
    Args:
        name: Logger name (usually module name)
        log_dir: Directory for log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation (50MB max, 5 backups)
    log_path = Path(log_dir) / f"{name}.log"
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JsonFormatter())
    
    # Console handler for warnings and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(JsonFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
```
</action>
<acceptance_criteria>
- `grep -l "class JsonFormatter" src/utils/logger.py` returns file path
- `grep -l "RotatingFileHandler" src/utils/logger.py` returns file path
- `grep -l "def get_logger" src/utils/logger.py` returns file path
- `python -c "from src.utils.logger import get_logger; log = get_logger('test'); log.info('test')"` exits 0
</acceptance_criteria>
</task>

<task id="1.5">
<title>Implement error guard utilities</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (error handling requirements)
</read_first>
<action>
Create src/utils/error_guard.py with validation helpers:

```python
"""Error guard utilities for input validation and error handling."""
import functools
import logging
from typing import Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class GraphConstructionError(Exception):
    """Raised when graph construction fails."""
    pass


def validate_no_nan(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Validate that tensor contains no NaN values.
    
    Args:
        tensor: PyTorch tensor to validate
        name: Name for error message
        
    Raises:
        DataValidationError: If tensor contains NaN
    """
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        nan_pct = (nan_count / tensor.numel()) * 100
        raise DataValidationError(
            f"{name} contains {nan_count} NaN values ({nan_pct:.2f}%), "
            f"shape: {tensor.shape}"
        )


def validate_aqi_range(
    values: Union[np.ndarray, torch.Tensor], 
    name: str = "AQI"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Validate and clip AQI values to [0, 500] range.
    
    Args:
        values: Array or tensor of AQI values
        name: Name for logging
        
    Returns:
        Clipped values
    """
    if isinstance(values, torch.Tensor):
        below_zero = (values < 0).sum().item()
        above_500 = (values > 500).sum().item()
        
        if below_zero > 0 or above_500 > 0:
            logger.warning(
                f"{name}: Clipping {below_zero} values below 0 and "
                f"{above_500} values above 500"
            )
        
        return torch.clamp(values, 0, 500)
    else:
        below_zero = (values < 0).sum()
        above_500 = (values > 500).sum()
        
        if below_zero > 0 or above_500 > 0:
            logger.warning(
                f"{name}: Clipping {below_zero} values below 0 and "
                f"{above_500} values above 500"
            )
        
        return np.clip(values, 0, 500)


def handle_oom(func):
    """
    Decorator to handle GPU out of memory errors.
    
    Catches CUDA OOM, clears cache, and re-raises with helpful message.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            logger.error(
                f"GPU OOM in {func.__name__}. "
                f"Try reducing batch_size or seq_len. "
                f"Original error: {e}"
            )
            raise
    return wrapper
```
</action>
<acceptance_criteria>
- `grep -l "class DataValidationError" src/utils/error_guard.py` returns file path
- `grep -l "def validate_no_nan" src/utils/error_guard.py` returns file path
- `grep -l "def validate_aqi_range" src/utils/error_guard.py` returns file path
- `grep -l "def handle_oom" src/utils/error_guard.py` returns file path
- `python -c "from src.utils.error_guard import validate_aqi_range; import numpy as np; r = validate_aqi_range(np.array([550, -10, 100])); assert r.max() == 500"` exits 0
</acceptance_criteria>
</task>

<task id="1.6">
<title>Create YAML config files</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (config specifications)
</read_first>
<action>
Create three config files:

configs/data.yaml:
```yaml
# Data pipeline configuration
data:
  raw_dir: data/raw
  processed_dir: data/processed
  simulated_dir: data/simulated
  splits_dir: data/splits
  graph_dir: data/graph
  parquet_path: data/processed/pune_aqi_processed.parquet
  scaler_path: models/scaler.pkl

preprocessing:
  gap_fill_limit_hours: 3
  interpolate_limit_hours: 12
  outlier_percentile: 0.999
  
training:
  seed: 42
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

api:
  open_meteo_base: https://archive-api.open-meteo.com/v1/archive
  openaq_base: https://api.openaq.org/v3
  retry_count: 3
  retry_backoff: [1, 2, 4]
```

configs/lstm.yaml:
```yaml
# LSTM model configuration
model:
  seq_len: 72
  forecast_horizon: 24
  hidden_size_1: 128
  hidden_size_2: 64
  num_attention_heads: 4
  dropout: 0.2
  fc_dropout: 0.3

training:
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 150
  early_stopping_patience: 10
  lr_scheduler_patience: 5
  lr_scheduler_factor: 0.5
  min_lr: 0.000001
  gradient_clip_norm: 1.0
  use_amp: true
  checkpoint_frequency: 5
```

configs/stgnn.yaml:
```yaml
# ST-GNN model configuration
model:
  seq_len: 72
  forecast_horizon: 24
  hidden_channels: 64
  num_nodes: 10

training:
  batch_size: 32
  learning_rate: 0.0005
  epochs: 100
  early_stopping_patience: 10
  use_amp: true

graph:
  distance_threshold_km: 15
  edge_weight_method: inverse_distance
  normalize_adjacency: true
```
</action>
<acceptance_criteria>
- `Test-Path configs/data.yaml` returns True
- `Test-Path configs/lstm.yaml` returns True
- `Test-Path configs/stgnn.yaml` returns True
- `grep -l "gap_fill_limit_hours: 3" configs/data.yaml` returns file path
- `grep -l "seq_len: 72" configs/lstm.yaml` returns file path
- `grep -l "num_nodes: 10" configs/stgnn.yaml` returns file path
</acceptance_criteria>
</task>

<task id="1.7">
<title>Create .env.example file</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (API key requirements)
</read_first>
<action>
Create .env.example with placeholder API keys:

```
# Pune Air Quality Sentinel - Environment Variables
# Copy this file to .env and fill in your values

# Kaggle API credentials (from kaggle.com/account)
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key

# Weights & Biases (optional, for experiment tracking)
WANDB_API_KEY=your_wandb_key

# Open-Meteo API (no key required - free)
OPEN_METEO_BASE_URL=https://archive-api.open-meteo.com/v1/archive

# OpenAQ API (no key required for public data)
OPENAQ_BASE_URL=https://api.openaq.org/v3
```
</action>
<acceptance_criteria>
- `Test-Path .env.example` returns True
- `grep -l "KAGGLE_USERNAME" .env.example` returns file path
- `grep -l "OPEN_METEO_BASE_URL" .env.example` returns file path
</acceptance_criteria>
</task>

<task id="1.8">
<title>Create node coordinates JSON</title>
<read_first>
- .planning/phases/01-data-pipeline-iot-simulation/01-CONTEXT.md (Pune node grid)
</read_first>
<action>
Create data/graph/node_coords.json with all 10 Pune locations:

```json
{
  "nodes": [
    {
      "id": "N01",
      "name": "Shivajinagar",
      "lat": 18.5308,
      "lon": 73.8475,
      "characteristic": "Reference CPCB station"
    },
    {
      "id": "N02",
      "name": "Hinjewadi IT Park",
      "lat": 18.5912,
      "lon": 73.7389,
      "characteristic": "Traffic + construction"
    },
    {
      "id": "N03",
      "name": "Pimpri-Chinchwad MIDC",
      "lat": 18.6298,
      "lon": 73.8037,
      "characteristic": "Heavy industry"
    },
    {
      "id": "N04",
      "name": "Hadapsar",
      "lat": 18.5089,
      "lon": 73.9260,
      "characteristic": "Mixed residential-commercial"
    },
    {
      "id": "N05",
      "name": "Katraj",
      "lat": 18.4529,
      "lon": 73.8669,
      "characteristic": "Dense residential"
    },
    {
      "id": "N06",
      "name": "Deccan Gymkhana",
      "lat": 18.5196,
      "lon": 73.8399,
      "characteristic": "High vehicular density"
    },
    {
      "id": "N07",
      "name": "Viman Nagar",
      "lat": 18.5679,
      "lon": 73.9143,
      "characteristic": "Airport proximity"
    },
    {
      "id": "N08",
      "name": "Kothrud",
      "lat": 18.5074,
      "lon": 73.8077,
      "characteristic": "Residential, cleaner"
    },
    {
      "id": "N09",
      "name": "Talegaon MIDC",
      "lat": 18.7330,
      "lon": 73.6554,
      "characteristic": "Outer industrial"
    },
    {
      "id": "N10",
      "name": "Mundhwa",
      "lat": 18.5326,
      "lon": 73.9368,
      "characteristic": "Downstream residential"
    }
  ],
  "metadata": {
    "city": "Pune",
    "country": "India",
    "timezone": "Asia/Kolkata",
    "coordinate_system": "WGS84"
  }
}
```
</action>
<acceptance_criteria>
- `Test-Path data/graph/node_coords.json` returns True
- `python -c "import json; d=json.load(open('data/graph/node_coords.json')); assert len(d['nodes']) == 10"` exits 0
- `grep -l "Shivajinagar" data/graph/node_coords.json` returns file path
- `grep -l "Pimpri-Chinchwad" data/graph/node_coords.json` returns file path
</acceptance_criteria>
</task>

## Verification

After all tasks complete:

```powershell
# Verify directory structure
$dirs = @("src/data", "src/models", "src/utils", "data/processed", "data/simulated", "data/graph", "configs", "tests")
foreach ($d in $dirs) { if (!(Test-Path $d)) { throw "Missing: $d" } }

# Verify Python imports work
python -c "from src.utils.seed import set_seed; from src.utils.config import load_config; from src.utils.logger import get_logger; from src.utils.error_guard import validate_no_nan; print('All imports OK')"

# Verify configs are valid YAML
python -c "import yaml; yaml.safe_load(open('configs/data.yaml')); yaml.safe_load(open('configs/lstm.yaml')); yaml.safe_load(open('configs/stgnn.yaml')); print('All configs valid')"

# Verify node coords
python -c "import json; nodes = json.load(open('data/graph/node_coords.json'))['nodes']; assert len(nodes) == 10; print('Node coords valid')"
```
