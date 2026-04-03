"""Unit tests for data preprocessing functions."""
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_aqi_df():
    """Create sample AQI DataFrame for testing."""
    dates = pd.date_range("2022-01-01", periods=100, freq="1h")
    return pd.DataFrame({
        "timestamp": dates,
        "pm25": np.random.uniform(20, 100, 100),
        "pm10": np.random.uniform(30, 150, 100),
        "no2": np.random.uniform(10, 60, 100),
        "so2": np.random.uniform(5, 30, 100),
        "co": np.random.uniform(0.5, 2.0, 100),
        "o3": np.random.uniform(20, 80, 100),
        "aqi": np.random.uniform(50, 200, 100),
    })


@pytest.fixture
def sample_weather_df():
    """Create sample weather DataFrame for testing."""
    dates = pd.date_range("2022-01-01", periods=100, freq="1h")
    return pd.DataFrame({
        "timestamp": dates,
        "temperature": np.random.uniform(15, 35, 100),
        "humidity": np.random.uniform(40, 90, 100),
        "wind_speed": np.random.uniform(0, 15, 100),
        "wind_direction": np.random.uniform(0, 360, 100),
        "pressure": np.random.uniform(1000, 1020, 100),
    })


class TestComputeSubIndex:
    """Tests for CPCB sub-index computation."""
    
    def test_compute_sub_index_pm25_good(self):
        """Test PM2.5 sub-index in 'Good' category."""
        from src.data.preprocess import compute_sub_index
        
        # PM2.5 = 15 µg/m³ should give AQI around 25
        result = compute_sub_index(15, "pm25")
        assert 20 <= result <= 30
    
    def test_compute_sub_index_pm25_moderate(self):
        """Test PM2.5 sub-index in 'Moderate' category."""
        from src.data.preprocess import compute_sub_index
        
        # PM2.5 = 45 µg/m³ should give AQI around 75
        result = compute_sub_index(45, "pm25")
        assert 60 <= result <= 90
    
    def test_compute_sub_index_nan_input(self):
        """Test that NaN input returns NaN."""
        from src.data.preprocess import compute_sub_index
        
        result = compute_sub_index(np.nan, "pm25")
        assert pd.isna(result)
    
    def test_compute_sub_index_negative_input(self):
        """Test that negative input returns NaN."""
        from src.data.preprocess import compute_sub_index
        
        result = compute_sub_index(-10, "pm25")
        assert pd.isna(result)
    
    def test_compute_sub_index_above_max(self):
        """Test that very high values return 500."""
        from src.data.preprocess import compute_sub_index
        
        result = compute_sub_index(600, "pm25")
        assert result == 500


class TestComputeAQI:
    """Tests for overall AQI computation."""
    
    def test_compute_aqi_single_pollutant(self):
        """Test AQI with single pollutant."""
        from src.data.preprocess import compute_aqi
        
        aqi = compute_aqi(pm25=45)
        assert 50 < aqi < 100
    
    def test_compute_aqi_multiple_pollutants(self):
        """Test AQI with multiple pollutants (takes max)."""
        from src.data.preprocess import compute_aqi
        
        aqi = compute_aqi(pm25=45, pm10=80)
        assert 50 < aqi < 100
    
    def test_compute_aqi_no_pollutants(self):
        """Test AQI with no valid pollutants."""
        from src.data.preprocess import compute_aqi
        
        aqi = compute_aqi()
        assert pd.isna(aqi)
    
    def test_compute_aqi_clipped_to_500(self):
        """Test that AQI is clipped to max 500."""
        from src.data.preprocess import compute_aqi
        
        aqi = compute_aqi(pm25=600)  # Very high PM2.5
        assert aqi == 500


class TestCleanAQIData:
    """Tests for data cleaning functions."""
    
    def test_clean_aqi_data_fills_gaps(self, sample_aqi_df):
        """Test that small gaps are filled."""
        from src.data.preprocess import clean_aqi_data
        
        # Introduce a gap
        df = sample_aqi_df.copy()
        df.loc[5, "pm25"] = np.nan
        df.loc[6, "pm25"] = np.nan
        
        cleaned = clean_aqi_data(df)
        
        # Should have filled the gap
        assert cleaned["pm25"].isna().sum() == 0
    
    def test_clean_aqi_data_preserves_shape(self, sample_aqi_df):
        """Test that cleaning preserves reasonable data shape."""
        from src.data.preprocess import clean_aqi_data
        
        cleaned = clean_aqi_data(sample_aqi_df)
        
        # Should have similar number of rows (may add some for resampling)
        assert len(cleaned) >= len(sample_aqi_df) * 0.9


class TestClipOutliers:
    """Tests for outlier clipping."""
    
    def test_clip_outliers_reduces_extreme_values(self, sample_aqi_df):
        """Test that extreme values are clipped."""
        from src.data.preprocess import clip_outliers
        
        # Add an extreme value
        df = sample_aqi_df.copy()
        df.loc[0, "pm25"] = 10000  # Extreme outlier
        
        clipped = clip_outliers(df, columns=["pm25"])
        
        assert clipped["pm25"].max() < 10000
    
    def test_clip_outliers_preserves_normal_values(self, sample_aqi_df):
        """Test that normal values are preserved."""
        from src.data.preprocess import clip_outliers
        
        original_mean = sample_aqi_df["pm25"].mean()
        clipped = clip_outliers(sample_aqi_df)
        
        # Mean should be similar (within 10%)
        assert abs(clipped["pm25"].mean() - original_mean) / original_mean < 0.1


class TestEngineerLagFeatures:
    """Tests for lag feature creation."""
    
    def test_lag_features_created(self, sample_aqi_df):
        """Test that lag features are created."""
        from src.data.preprocess import engineer_lag_features
        
        result = engineer_lag_features(sample_aqi_df, lags=[1, 6, 24])
        
        assert "aqi_lag_1h" in result.columns
        assert "aqi_lag_6h" in result.columns
        assert "aqi_lag_24h" in result.columns
    
    def test_lag_features_values_correct(self, sample_aqi_df):
        """Test that lag values are correct."""
        from src.data.preprocess import engineer_lag_features
        
        result = engineer_lag_features(sample_aqi_df, lags=[1])
        
        # Value at index 1 should be original value at index 0
        assert result["aqi_lag_1h"].iloc[1] == sample_aqi_df["aqi"].iloc[0]


class TestEngineerRollingFeatures:
    """Tests for rolling feature creation."""
    
    def test_rolling_features_created(self, sample_aqi_df):
        """Test that rolling features are created."""
        from src.data.preprocess import engineer_rolling_features
        
        result = engineer_rolling_features(sample_aqi_df, columns=["pm25"], windows=[24])
        
        assert "pm25_24h_mean" in result.columns
        assert "pm25_24h_std" in result.columns


class TestCreateWindVectors:
    """Tests for wind vector computation."""
    
    def test_wind_vectors_created(self, sample_weather_df):
        """Test that wind vectors are created."""
        from src.data.preprocess import create_wind_vectors
        
        result = create_wind_vectors(sample_weather_df)
        
        assert "wind_u" in result.columns
        assert "wind_v" in result.columns
    
    def test_wind_vectors_north_wind(self):
        """Test wind vector for north wind (0°)."""
        from src.data.preprocess import create_wind_vectors
        
        df = pd.DataFrame({
            "wind_speed": [10.0],
            "wind_direction": [0.0]  # North
        })
        
        result = create_wind_vectors(df)
        
        # North wind: u≈0, v≈speed
        assert abs(result["wind_u"].iloc[0]) < 0.01
        assert abs(result["wind_v"].iloc[0] - 10.0) < 0.01
    
    def test_wind_vectors_east_wind(self):
        """Test wind vector for east wind (90°)."""
        from src.data.preprocess import create_wind_vectors
        
        df = pd.DataFrame({
            "wind_speed": [10.0],
            "wind_direction": [90.0]  # East
        })
        
        result = create_wind_vectors(df)
        
        # East wind: u≈speed, v≈0
        assert abs(result["wind_u"].iloc[0] - 10.0) < 0.01
        assert abs(result["wind_v"].iloc[0]) < 0.01


class TestCreateCalendarFeatures:
    """Tests for calendar feature creation."""
    
    def test_calendar_features_created(self, sample_aqi_df):
        """Test that calendar features are created."""
        from src.data.preprocess import create_calendar_features
        
        result = create_calendar_features(sample_aqi_df)
        
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns
        assert "dow_sin" in result.columns
        assert "is_weekend" in result.columns
        assert "is_diwali_week" in result.columns
        assert "is_monsoon" in result.columns
    
    def test_weekend_flag_correct(self):
        """Test that weekend flag is correct."""
        from src.data.preprocess import create_calendar_features
        
        # Create a known Saturday
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-01"])  # Saturday
        })
        
        result = create_calendar_features(df)
        assert result["is_weekend"].iloc[0] == 1


class TestNormalizeFeatures:
    """Tests for feature normalization."""
    
    def test_normalize_features_range(self, sample_aqi_df, tmp_path):
        """Test that normalized values are in [0, 1]."""
        from src.data.preprocess import normalize_features
        
        scaler_path = str(tmp_path / "scaler.pkl")
        result, scaler = normalize_features(
            sample_aqi_df, 
            scaler_path=scaler_path,
            exclude_cols=["timestamp"]
        )
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert result[col].min() >= 0
            assert result[col].max() <= 1
    
    def test_normalize_saves_scaler(self, sample_aqi_df, tmp_path):
        """Test that scaler is saved."""
        from src.data.preprocess import normalize_features
        
        scaler_path = tmp_path / "scaler.pkl"
        normalize_features(
            sample_aqi_df, 
            scaler_path=str(scaler_path),
            exclude_cols=["timestamp"]
        )
        
        assert scaler_path.exists()


class TestCreateSplit:
    """Tests for train/val/test split."""
    
    def test_split_ratios(self, sample_aqi_df, tmp_path):
        """Test that split ratios are respected."""
        from src.data.preprocess import create_train_val_test_split
        
        splits = create_train_val_test_split(
            sample_aqi_df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            output_path=str(tmp_path / "splits.json")
        )
        
        total = len(sample_aqi_df)
        assert len(splits["train_df"]) == int(total * 0.7)
        assert len(splits["val_df"]) == int(total * 0.85) - int(total * 0.7)
    
    def test_split_saves_indices(self, sample_aqi_df, tmp_path):
        """Test that split indices are saved."""
        from src.data.preprocess import create_train_val_test_split
        
        output_path = tmp_path / "splits.json"
        create_train_val_test_split(
            sample_aqi_df,
            output_path=str(output_path)
        )
        
        assert output_path.exists()
        
        with open(output_path) as f:
            saved = json.load(f)
        
        assert "train_indices" in saved
        assert "val_indices" in saved
        assert "test_indices" in saved
