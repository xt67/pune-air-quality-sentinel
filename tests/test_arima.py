"""Tests for ARIMA baseline model."""
import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.models.arima import ARIMAModel, train_all_nodes, evaluate_arima


@pytest.fixture
def sample_series():
    """Create sample AQI time series."""
    np.random.seed(42)
    n = 100  # Reduced from 500 for faster tests
    
    # Simple trend + seasonality + noise
    t = np.arange(n)
    trend = 100 + 0.01 * t
    seasonality = 20 * np.sin(2 * np.pi * t / 24)  # Daily cycle
    noise = np.random.normal(0, 5, n)
    
    return trend + seasonality + noise


@pytest.fixture
def sample_df():
    """Create sample dataframe with node data."""
    np.random.seed(42)
    n = 100  # Reduced from 500 for faster tests
    
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    
    # Generate series for two nodes
    t = np.arange(n)
    base = 100 + 20 * np.sin(2 * np.pi * t / 24)
    
    df = pd.DataFrame({
        "timestamp": list(timestamps) * 2,
        "node_id": ["N01"] * n + ["N02"] * n,
        "aqi": list(base + np.random.normal(0, 5, n)) + 
               list(base + 20 + np.random.normal(0, 5, n))  # N02 has higher AQI
    })
    
    return df


class TestARIMAModel:
    """Tests for ARIMAModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = ARIMAModel("N01")
        
        assert model.node_id == "N01"
        assert model.seasonal == True
        assert model.m == 24
        assert model.fitted == False
        
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        model = ARIMAModel(
            "N02",
            seasonal=False,
            m=12,
            max_p=3,
            max_q=3
        )
        
        assert model.seasonal == False
        assert model.m == 12
        assert model.max_p == 3
        
    def test_fit(self, sample_series):
        """Test model fitting."""
        model = ARIMAModel("N01", seasonal=False, max_p=2, max_q=2)  # Reduced for speed
        model.fit(sample_series[:80])
        
        assert model.fitted == True
        assert model.model is not None
        
    def test_predict_before_fit(self):
        """Test predict raises error before fit."""
        model = ARIMAModel("N01")
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(24)
            
    def test_predict_shape(self, sample_series):
        """Test predict returns correct shape."""
        model = ARIMAModel("N01", seasonal=False, max_p=2, max_q=2)
        model.fit(sample_series[:80])
        
        horizon = 24
        pred = model.predict(horizon)
        
        assert len(pred) == horizon
        
    def test_predict_reasonable_values(self, sample_series):
        """Test predictions are in reasonable range."""
        model = ARIMAModel("N01", seasonal=False, max_p=2, max_q=2)
        model.fit(sample_series[:80])
        
        pred = model.predict(24)
        
        # Predictions should be positive and not too far from training data
        assert all(pred > 0)
        assert all(pred < 300)  # AQI shouldn't exceed 300
        
    def test_save_load(self, sample_series, tmp_path):
        """Test save and load roundtrip."""
        model = ARIMAModel("N01", seasonal=False, max_p=2, max_q=2)
        model.fit(sample_series[:80])
        
        # Save
        save_path = str(tmp_path / "test_arima.pkl")
        model.save(save_path)
        
        assert Path(save_path).exists()
        
        # Load
        loaded = ARIMAModel.load(save_path)
        
        assert loaded.node_id == "N01"
        assert loaded.fitted == True
        
        # Same predictions
        original_pred = model.predict(12)
        loaded_pred = loaded.predict(12)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)


class TestTrainAllNodes:
    """Tests for train_all_nodes function."""
    
    def test_trains_all_nodes(self, sample_df, tmp_path):
        """Test training for all nodes."""
        # Use subset for speed
        subset = sample_df.groupby("node_id").head(80).reset_index(drop=True)
        
        models = train_all_nodes(
            subset,
            output_dir=str(tmp_path),
            target_col="aqi"
        )
        
        assert "N01" in models
        assert "N02" in models
        assert len(models) == 2
        
    def test_saves_models(self, sample_df, tmp_path):
        """Test models are saved to disk."""
        subset = sample_df.groupby("node_id").head(80).reset_index(drop=True)
        
        train_all_nodes(
            subset,
            output_dir=str(tmp_path),
            target_col="aqi"
        )
        
        assert (tmp_path / "arima_N01.pkl").exists()
        assert (tmp_path / "arima_N02.pkl").exists()


class TestEvaluateArima:
    """Tests for evaluate_arima function."""
    
    def test_returns_mae_per_node(self, sample_df, tmp_path):
        """Test evaluation returns MAE for each node."""
        # Use subset for speed
        subset = sample_df.groupby("node_id").head(80).reset_index(drop=True)
        
        # Train models
        models = train_all_nodes(
            subset,
            output_dir=str(tmp_path),
            target_col="aqi"
        )
        
        # Evaluate on same data (just for testing)
        results = evaluate_arima(models, subset, horizon=24)
        
        assert "N01" in results
        assert "N02" in results
        assert "average" in results
        
    def test_mae_reasonable(self, sample_df, tmp_path):
        """Test MAE is reasonable."""
        subset = sample_df.groupby("node_id").head(80).reset_index(drop=True)
        
        models = train_all_nodes(
            subset,
            output_dir=str(tmp_path),
            target_col="aqi"
        )
        
        results = evaluate_arima(models, subset, horizon=24)
        
        # MAE should be positive and not huge
        assert results["average"] > 0
        assert results["average"] < 100  # Reasonable for AQI
