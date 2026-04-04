---
phase: 2
plan: 1
title: "Sliding Window Dataset"
wave: 1
depends_on: []
files_modified:
  - src/models/dataset.py
  - tests/test_dataset.py
requirements_addressed: [LSTM-01]
autonomous: true
---

# Plan 1: Sliding Window Dataset

<objective>
Create a PyTorch Dataset class that generates sliding windows from time series data for LSTM training. Support configurable sequence length and forecast horizon.
</objective>

<must_haves>
- Dataset class compatible with PyTorch DataLoader
- Configurable seq_len (default 72) and horizon (default 24)
- Support for multi-node data (node_id column)
- Proper train/val/test split handling
- Feature scaling using pre-fitted scaler
- Unit tests with >90% coverage
</must_haves>

## Tasks

<task id="1.1">
<title>Create AQIDataset class</title>
<action>
Create src/models/dataset.py:

```python
"""Sliding window dataset for AQI time series."""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List


class AQIDataset(Dataset):
    """
    Sliding window dataset for AQI forecasting.
    
    Args:
        df: DataFrame with columns [timestamp, node_id, aqi, ...]
        seq_len: Input sequence length (default 72 = 3 days)
        horizon: Forecast horizon (default 24 = 1 day)
        target_col: Target column name (default "aqi")
        feature_cols: Feature columns (None = all numeric except target)
        node_id: Filter to specific node (None = all nodes)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 72,
        horizon: int = 24,
        target_col: str = "aqi",
        feature_cols: Optional[List[str]] = None,
        node_id: Optional[str] = None
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.target_col = target_col
        
        # Filter by node if specified
        if node_id is not None:
            df = df[df["node_id"] == node_id].copy()
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Identify feature columns
        if feature_cols is None:
            exclude = ["timestamp", "node_id", target_col]
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                          if c not in exclude]
        
        self.feature_cols = feature_cols
        
        # Extract arrays
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(np.float32)
        
        # Calculate valid indices
        self.valid_indices = self._compute_valid_indices()
        
    def _compute_valid_indices(self) -> np.ndarray:
        """Compute indices where full window + horizon fits."""
        max_start = len(self.features) - self.seq_len - self.horizon
        if max_start < 0:
            return np.array([], dtype=np.int64)
        return np.arange(max_start + 1)
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.valid_indices[idx]
        end = start + self.seq_len
        
        # Input: (seq_len, n_features)
        x = torch.from_numpy(self.features[start:end])
        
        # Target: (horizon,)
        y = torch.from_numpy(self.targets[end:end + self.horizon])
        
        return x, y


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 72,
    horizon: int = 24,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple:
    """Create train/val/test DataLoaders."""
    from torch.utils.data import DataLoader
    
    train_ds = AQIDataset(train_df, seq_len, horizon)
    val_ds = AQIDataset(val_df, seq_len, horizon)
    test_ds = AQIDataset(test_df, seq_len, horizon)
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```
</action>
</task>

<task id="1.2">
<title>Create unit tests</title>
<action>
Create tests/test_dataset.py with comprehensive tests for:
- Dataset initialization
- __len__ and __getitem__
- Window shape verification
- Edge cases (short sequences)
- DataLoader integration
</action>
</task>

<acceptance_criteria>
- `python -c "from src.models.dataset import AQIDataset; print('OK')"` succeeds
- `pytest tests/test_dataset.py -v` all pass
- Dataset returns (seq_len, n_features) input and (horizon,) target shapes
</acceptance_criteria>
