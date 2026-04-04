"""Unit tests for IoT sensor simulation."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def base_signal():
    """Create a small base signal for testing."""
    from src.data.iot_sim import generate_base_signal
    return generate_base_signal(
        start_date="2022-01-01",
        end_date="2022-01-03",
        seed=42
    )


class TestGenerateBaseSignal:
    """Tests for base signal generation."""
    
    def test_base_signal_shape(self):
        """Test that base signal has correct shape."""
        from src.data.iot_sim import generate_base_signal
        
        df = generate_base_signal(
            start_date="2022-01-01",
            end_date="2022-01-02",
            seed=42
        )
        
        # Should have 25 hours (including both endpoints)
        assert len(df) == 25
        assert "timestamp" in df.columns
        assert "base_aqi" in df.columns
    
    def test_base_signal_reproducibility(self):
        """Test that same seed produces same results."""
        from src.data.iot_sim import generate_base_signal
        
        df1 = generate_base_signal(start_date="2022-01-01", end_date="2022-01-02", seed=42)
        df2 = generate_base_signal(start_date="2022-01-01", end_date="2022-01-02", seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_base_signal_different_seeds(self):
        """Test that different seeds produce different results."""
        from src.data.iot_sim import generate_base_signal
        
        df1 = generate_base_signal(start_date="2022-01-01", end_date="2022-01-02", seed=42)
        df2 = generate_base_signal(start_date="2022-01-01", end_date="2022-01-02", seed=123)
        
        # AQI values should differ
        assert not df1["base_aqi"].equals(df2["base_aqi"])


class TestApplyHourlyPattern:
    """Tests for rush-hour pattern application."""
    
    def test_morning_rush_increases_aqi(self, base_signal):
        """Test that morning rush hour increases AQI."""
        from src.data.iot_sim import apply_hourly_pattern
        
        original = base_signal.copy()
        result = apply_hourly_pattern(base_signal)
        
        # Check 8 AM hours have higher AQI
        morning_mask = result["timestamp"].dt.hour == 8
        if morning_mask.any():
            original_morning = original.loc[morning_mask, "base_aqi"].mean()
            result_morning = result.loc[morning_mask, "base_aqi"].mean()
            assert result_morning > original_morning
    
    def test_evening_rush_increases_aqi(self, base_signal):
        """Test that evening rush hour increases AQI."""
        from src.data.iot_sim import apply_hourly_pattern
        
        original = base_signal.copy()
        result = apply_hourly_pattern(base_signal)
        
        # Check 6 PM hours have higher AQI
        evening_mask = result["timestamp"].dt.hour == 18
        if evening_mask.any():
            original_evening = original.loc[evening_mask, "base_aqi"].mean()
            result_evening = result.loc[evening_mask, "base_aqi"].mean()
            assert result_evening > original_evening


class TestApplySeasonalPattern:
    """Tests for seasonal pattern application."""
    
    def test_monsoon_decreases_aqi(self):
        """Test that monsoon months have lower AQI."""
        from src.data.iot_sim import generate_base_signal, apply_seasonal_pattern
        
        # Generate data spanning monsoon
        df = generate_base_signal(
            start_date="2022-06-15",
            end_date="2022-08-15",
            seed=42
        )
        original = df.copy()
        result = apply_seasonal_pattern(df)
        
        # July-August should have lower AQI
        monsoon_mask = result["timestamp"].dt.month.isin([7, 8])
        if monsoon_mask.any():
            original_monsoon = original.loc[monsoon_mask, "base_aqi"].mean()
            result_monsoon = result.loc[monsoon_mask, "base_aqi"].mean()
            assert result_monsoon < original_monsoon
    
    def test_winter_increases_aqi(self):
        """Test that winter months have higher AQI."""
        from src.data.iot_sim import generate_base_signal, apply_seasonal_pattern
        
        # Generate data spanning winter
        df = generate_base_signal(
            start_date="2022-11-15",
            end_date="2023-01-15",
            seed=42
        )
        original = df.copy()
        result = apply_seasonal_pattern(df)
        
        # November-January should have higher AQI
        winter_mask = result["timestamp"].dt.month.isin([11, 12, 1])
        if winter_mask.any():
            original_winter = original.loc[winter_mask, "base_aqi"].mean()
            result_winter = result.loc[winter_mask, "base_aqi"].mean()
            assert result_winter > original_winter


class TestApplyDiwaliSpike:
    """Tests for Diwali spike application."""
    
    def test_diwali_spike_applied(self):
        """Test that Diwali period has higher AQI."""
        from src.data.iot_sim import generate_base_signal, apply_diwali_spike
        
        # Generate data around Diwali 2022 (Oct 24)
        df = generate_base_signal(
            start_date="2022-10-20",
            end_date="2022-10-28",
            seed=42
        )
        original = df.copy()
        result = apply_diwali_spike(df)
        
        # Diwali day should have much higher AQI
        diwali_mask = result["timestamp"].dt.date == pd.Timestamp("2022-10-24").date()
        if diwali_mask.any():
            original_diwali = original.loc[diwali_mask, "base_aqi"].mean()
            result_diwali = result.loc[diwali_mask, "base_aqi"].mean()
            # Spike should be at least 50 AQI higher
            assert result_diwali - original_diwali >= 50


class TestGenerateNodeSeries:
    """Tests for individual node series generation."""
    
    def test_node_series_shape(self, base_signal):
        """Test node series has correct columns."""
        from src.data.iot_sim import generate_node_series
        
        df = generate_node_series(base_signal, "N01", seed=42)
        
        expected_cols = ["timestamp", "node_id", "aqi", "pm25", "pm10", "no2", "so2", "co", "o3"]
        assert list(df.columns) == expected_cols
    
    def test_node_bias_applied(self, base_signal):
        """Test that node-specific bias is applied."""
        from src.data.iot_sim import generate_node_series, NODE_BIASES
        
        df_n01 = generate_node_series(base_signal, "N01", noise_std=0, seed=42)  # bias = 0
        df_n03 = generate_node_series(base_signal, "N03", noise_std=0, seed=42)  # bias = +25
        
        # N03 should have higher AQI than N01
        assert df_n03["aqi"].mean() > df_n01["aqi"].mean()
        # Difference should be approximately the bias
        diff = df_n03["aqi"].mean() - df_n01["aqi"].mean()
        assert 20 < diff < 30  # Allow some tolerance
    
    def test_node_series_aqi_range(self, base_signal):
        """Test that AQI is clipped to valid range."""
        from src.data.iot_sim import generate_node_series
        
        df = generate_node_series(base_signal, "N03", seed=42)
        
        assert df["aqi"].min() >= 0
        assert df["aqi"].max() <= 500
    
    def test_node_series_no_nan(self, base_signal):
        """Test that no NaN values in output."""
        from src.data.iot_sim import generate_node_series
        
        df = generate_node_series(base_signal, "N01", seed=42)
        
        assert df.isna().sum().sum() == 0


class TestGenerateAllNodes:
    """Tests for full simulation generation."""
    
    def test_generates_10_nodes(self, tmp_path):
        """Test that all 10 nodes are generated."""
        from src.data.iot_sim import generate_all_nodes
        
        node_data = generate_all_nodes(
            start_date="2022-01-01",
            end_date="2022-01-02",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        assert len(node_data) == 10
        assert all(f"N{i:02d}" in node_data for i in range(1, 11))
    
    def test_saves_csv_files(self, tmp_path):
        """Test that CSV files are saved."""
        from src.data.iot_sim import generate_all_nodes
        
        generate_all_nodes(
            start_date="2022-01-01",
            end_date="2022-01-02",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        # Check all 10 CSV files exist
        for i in range(1, 11):
            csv_path = tmp_path / f"node_N{i:02d}_simulated.csv"
            assert csv_path.exists(), f"Missing {csv_path}"
    
    def test_identical_timestamps(self, tmp_path):
        """Test that all nodes have identical timestamps."""
        from src.data.iot_sim import generate_all_nodes
        
        node_data = generate_all_nodes(
            start_date="2022-01-01",
            end_date="2022-01-02",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        reference_ts = node_data["N01"]["timestamp"].tolist()
        for node_id, df in node_data.items():
            assert df["timestamp"].tolist() == reference_ts


class TestValidateSimulation:
    """Tests for simulation validation."""
    
    def test_validates_correct_data(self, tmp_path):
        """Test that valid data passes validation."""
        from src.data.iot_sim import generate_all_nodes, validate_simulation
        
        node_data = generate_all_nodes(
            start_date="2022-01-01",
            end_date="2022-01-02",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        assert validate_simulation(node_data) is True
    
    def test_fails_missing_nodes(self):
        """Test that missing nodes fail validation."""
        from src.data.iot_sim import validate_simulation
        
        # Create incomplete data
        node_data = {
            "N01": pd.DataFrame({"timestamp": [], "aqi": []})
        }
        
        assert validate_simulation(node_data) is False


class TestRunIoTSimulation:
    """Tests for full simulation pipeline."""
    
    def test_run_iot_simulation(self, tmp_path):
        """Test complete simulation pipeline."""
        from src.data.iot_sim import run_iot_simulation
        
        node_data = run_iot_simulation(
            start_date="2022-01-01",
            end_date="2022-01-03",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        assert len(node_data) == 10
        assert all(df["aqi"].between(0, 500).all() for df in node_data.values())
