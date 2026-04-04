"""Tests for AQI sliding window dataset."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.dataset import AQIDataset, create_dataloaders


@pytest.fixture
def sample_df():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    n_hours = 200
    
    timestamps = pd.date_range("2023-01-01", periods=n_hours, freq="h")
    
    data = {
        "timestamp": timestamps,
        "node_id": "N01",
        "aqi": np.random.uniform(50, 200, n_hours),
        "pm25": np.random.uniform(10, 100, n_hours),
        "pm10": np.random.uniform(20, 150, n_hours),
        "temperature": np.random.uniform(15, 35, n_hours),
        "humidity": np.random.uniform(30, 90, n_hours),
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def multi_node_df(sample_df):
    """Create multi-node dataframe."""
    df1 = sample_df.copy()
    df1["node_id"] = "N01"
    
    df2 = sample_df.copy()
    df2["node_id"] = "N02"
    df2["aqi"] = df2["aqi"] + 20  # Different AQI for second node
    
    return pd.concat([df1, df2], ignore_index=True)


class TestAQIDataset:
    """Tests for AQIDataset class."""
    
    def test_init_basic(self, sample_df):
        """Test basic initialization."""
        ds = AQIDataset(sample_df, window_size=24, horizon=12)
        
        assert len(ds) > 0
        assert ds.input_size > 0
        
    def test_window_creation(self, sample_df):
        """Test correct number of windows created."""
        window_size = 24
        horizon = 12
        total_len = window_size + horizon
        expected_windows = len(sample_df) - total_len + 1
        
        ds = AQIDataset(sample_df, window_size=window_size, horizon=horizon)
        
        assert len(ds) == expected_windows
        
    def test_getitem_shapes(self, sample_df):
        """Test __getitem__ returns correct shapes."""
        window_size = 24
        horizon = 12
        
        ds = AQIDataset(sample_df, window_size=window_size, horizon=horizon)
        
        x, y = ds[0]
        
        assert x.shape == (window_size, ds.input_size)
        assert y.shape == (horizon,)
        
    def test_tensor_types(self, sample_df):
        """Test __getitem__ returns tensors."""
        ds = AQIDataset(sample_df)
        
        x, y = ds[0]
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.float32
        
    def test_multi_node(self, multi_node_df):
        """Test dataset handles multiple nodes."""
        window_size = 24
        horizon = 12
        
        ds = AQIDataset(multi_node_df, window_size=window_size, horizon=horizon)
        
        # Should have windows from both nodes
        single_node_windows = 200 - (window_size + horizon) + 1
        expected_total = single_node_windows * 2  # 2 nodes
        
        assert len(ds) == expected_total
        
    def test_custom_feature_cols(self, sample_df):
        """Test custom feature columns."""
        feature_cols = ["pm25", "temperature"]
        
        ds = AQIDataset(sample_df, feature_cols=feature_cols)
        
        assert ds.input_size == 2
        assert ds.feature_cols == feature_cols
        
    def test_input_size_property(self, sample_df):
        """Test input_size property matches features."""
        ds = AQIDataset(sample_df)
        
        x, _ = ds[0]
        
        assert ds.input_size == x.shape[1]


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""
    
    def test_creates_train_val(self, sample_df):
        """Test creates train and val loaders."""
        loaders = create_dataloaders(
            train_df=sample_df,
            val_df=sample_df,
            batch_size=16
        )
        
        assert "train" in loaders
        assert "val" in loaders
        
    def test_creates_test_when_provided(self, sample_df):
        """Test creates test loader when test_df provided."""
        loaders = create_dataloaders(
            train_df=sample_df,
            val_df=sample_df,
            test_df=sample_df,
            batch_size=16
        )
        
        assert "test" in loaders
        
    def test_batch_shapes(self, sample_df):
        """Test batch shapes from dataloader."""
        window_size = 24
        horizon = 12
        batch_size = 8
        
        loaders = create_dataloaders(
            train_df=sample_df,
            val_df=sample_df,
            window_size=window_size,
            horizon=horizon,
            batch_size=batch_size
        )
        
        for x, y in loaders["train"]:
            assert x.shape[1] == window_size  # seq_len
            assert y.shape[1] == horizon
            assert x.shape[0] <= batch_size
            break  # Just check first batch
            
    def test_train_shuffle(self, sample_df):
        """Test train loader shuffles data."""
        loaders = create_dataloaders(
            train_df=sample_df,
            val_df=sample_df,
            batch_size=16
        )
        
        # Get first batch twice
        batch1 = next(iter(loaders["train"]))[0]
        batch2 = next(iter(loaders["train"]))[0]
        
        # With shuffle, batches should differ (high probability)
        # This is a statistical test, might rarely fail
        assert not torch.allclose(batch1, batch2)
