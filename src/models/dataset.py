"""Sliding window dataset for time-series forecasting."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AQIDataset(Dataset):
    """
    Sliding window dataset for AQI time-series forecasting.
    
    Creates (input_window, forecast_horizon) pairs for LSTM training.
    Supports multi-node data with node_id column.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 72,
        horizon: int = 24,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "aqi",
        node_col: str = "node_id",
        timestamp_col: str = "timestamp"
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with features and target
            window_size: Number of historical timesteps (default 72h = 3 days)
            horizon: Forecast horizon (default 24h)
            feature_cols: List of feature columns. If None, auto-detect numeric cols
            target_col: Target column name
            node_col: Node identifier column
            timestamp_col: Timestamp column for sorting
        """
        self.window_size = window_size
        self.horizon = horizon
        self.target_col = target_col
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            exclude = {node_col, timestamp_col, "split"}
            feature_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        self.feature_cols = feature_cols
        
        # Build windows for each node
        self.windows: List[Tuple[np.ndarray, np.ndarray]] = []
        
        for node_id in df[node_col].unique():
            node_df = df[df[node_col] == node_id].sort_values(timestamp_col).reset_index(drop=True)
            self._build_node_windows(node_df)
            
    def _build_node_windows(self, node_df: pd.DataFrame) -> None:
        """Build sliding windows for a single node."""
        features = node_df[self.feature_cols].values
        target = node_df[self.target_col].values
        
        total_len = self.window_size + self.horizon
        n_samples = len(node_df) - total_len + 1
        
        for i in range(n_samples):
            x = features[i : i + self.window_size]
            y = target[i + self.window_size : i + total_len]
            self.windows.append((x.astype(np.float32), y.astype(np.float32)))
            
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.windows[idx]
        return torch.from_numpy(x), torch.from_numpy(y)
    
    @property
    def input_size(self) -> int:
        """Number of input features."""
        return len(self.feature_cols)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    window_size: int = 72,
    horizon: int = 24,
    batch_size: int = 64,
    feature_cols: Optional[List[str]] = None,
    num_workers: int = 0
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Optional test data
        window_size: Historical window size
        horizon: Forecast horizon
        batch_size: Batch size
        feature_cols: Feature columns
        num_workers: DataLoader workers
        
    Returns:
        Dict with 'train', 'val', and optionally 'test' DataLoaders
    """
    train_ds = AQIDataset(train_df, window_size, horizon, feature_cols)
    val_ds = AQIDataset(val_df, window_size, horizon, feature_cols)
    
    loaders = {
        "train": torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
    }
    
    if test_df is not None:
        test_ds = AQIDataset(test_df, window_size, horizon, feature_cols)
        loaders["test"] = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
    return loaders
