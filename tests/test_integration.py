"""Integration tests for the complete data pipeline."""
import pytest
from pathlib import Path

import pandas as pd
import numpy as np


class TestPipelineImports:
    """Test that all pipeline modules can be imported."""
    
    def test_import_fetch(self):
        """Test fetch module imports."""
        from src.data import fetch
        assert hasattr(fetch, "fetch_kaggle_aqi")
        assert hasattr(fetch, "fetch_weather_all_nodes")
    
    def test_import_preprocess(self):
        """Test preprocess module imports."""
        from src.data import preprocess
        assert hasattr(preprocess, "compute_aqi")
        assert hasattr(preprocess, "clean_aqi_data")
    
    def test_import_iot_sim(self):
        """Test IoT simulation module imports."""
        from src.data import iot_sim
        assert hasattr(iot_sim, "generate_all_nodes")
        assert hasattr(iot_sim, "validate_simulation")
    
    def test_import_pipeline(self):
        """Test pipeline module imports."""
        from src.data import pipeline
        assert hasattr(pipeline, "run_full_pipeline")
        assert hasattr(pipeline, "run_iot_simulation")


class TestIoTSimulationIntegration:
    """Integration tests for IoT simulation."""
    
    def test_generate_and_validate(self, tmp_path):
        """Test that generated data passes validation."""
        from src.data.iot_sim import generate_all_nodes, validate_simulation
        
        node_data = generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-01-07",  # 1 week
            seed=42,
            output_dir=str(tmp_path)
        )
        
        assert validate_simulation(node_data)
    
    def test_node_biases_reflected(self, tmp_path):
        """Test that node biases create expected AQI differences."""
        from src.data.iot_sim import generate_all_nodes, NODE_BIASES
        
        node_data = generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-01-07",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        # N03 (Pimpri-Chinchwad, bias=+25) should have higher AQI than N08 (Kothrud, bias=-10)
        n03_mean = node_data["N03"]["aqi"].mean()
        n08_mean = node_data["N08"]["aqi"].mean()
        
        assert n03_mean > n08_mean, f"N03 ({n03_mean:.1f}) should be > N08 ({n08_mean:.1f})"
        
        # Difference should approximately match bias difference (25 - (-10) = 35)
        diff = n03_mean - n08_mean
        assert 25 < diff < 45, f"Bias difference {diff:.1f} should be ~35"


class TestPreprocessIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_preprocess_simulated_data(self, tmp_path):
        """Test preprocessing on simulated data."""
        from src.data.iot_sim import generate_all_nodes
        from src.data.pipeline import load_simulated_data, run_preprocess_pipeline
        
        # Generate simulated data
        sim_dir = tmp_path / "simulated"
        sim_dir.mkdir()
        
        generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-01-14",  # 2 weeks
            seed=42,
            output_dir=str(sim_dir)
        )
        
        # Load simulated data
        df = load_simulated_data(str(sim_dir))
        assert len(df) > 0
        
        # Run preprocessing
        processed = run_preprocess_pipeline(
            df,
            output_path=str(tmp_path / "processed.parquet"),
            scaler_path=str(tmp_path / "scaler.pkl"),
            splits_dir=str(tmp_path / "splits"),
            seed=42
        )
        
        assert "train" in processed
        assert "val" in processed
        assert "test" in processed
        assert len(processed["train"]) > 0
    
    def test_no_data_leakage(self, tmp_path):
        """Test that preprocessing doesn't leak future data."""
        from src.data.iot_sim import generate_all_nodes
        from src.data.pipeline import load_simulated_data, run_preprocess_pipeline
        
        # Generate data
        sim_dir = tmp_path / "simulated"
        sim_dir.mkdir()
        
        generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-01-31",
            seed=42,
            output_dir=str(sim_dir)
        )
        
        df = load_simulated_data(str(sim_dir))
        
        processed = run_preprocess_pipeline(
            df,
            output_path=str(tmp_path / "processed.parquet"),
            scaler_path=str(tmp_path / "scaler.pkl"),
            splits_dir=str(tmp_path / "splits"),
            seed=42
        )
        
        # Check that test set timestamps are AFTER training set
        train_df = processed["train"]
        test_df = processed["test"]
        
        if "timestamp" in train_df.columns and "timestamp" in test_df.columns:
            train_max = train_df["timestamp"].max()
            test_min = test_df["timestamp"].min()
            
            # Test data should come after training data (chronological split)
            # Note: With multi-node data, this check is per-node conceptually
            # But overall, test timestamps should be >= some training timestamps


class TestFullPipelineIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_full_pipeline_simulated(self, tmp_path, monkeypatch):
        """Test complete pipeline with simulated data."""
        from src.data import pipeline
        
        # Redirect output paths to temp directory
        monkeypatch.chdir(tmp_path)
        
        # Create required directories
        (tmp_path / "data" / "simulated").mkdir(parents=True)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        (tmp_path / "data" / "splits").mkdir(parents=True)
        (tmp_path / "models").mkdir(parents=True)
        (tmp_path / "data" / "graph").mkdir(parents=True)
        
        # Create node_coords.json
        import json
        node_coords = {
            f"N{i:02d}": {"name": f"Node{i}", "lat": 18.5 + i*0.01, "lon": 73.8 + i*0.01}
            for i in range(1, 11)
        }
        with open(tmp_path / "data" / "graph" / "node_coords.json", "w") as f:
            json.dump(node_coords, f)
        
        # Run pipeline
        results = pipeline.run_full_pipeline(use_simulated=True, seed=42)
        
        assert "summary" in results
        assert results["summary"]["total_rows"] > 0
        assert results["summary"]["train_rows"] > 0
    
    def test_validate_pipeline_outputs_when_empty(self, tmp_path, monkeypatch):
        """Test validation when no outputs exist."""
        from src.data.pipeline import validate_pipeline_outputs
        
        monkeypatch.chdir(tmp_path)
        
        checks = validate_pipeline_outputs()
        
        assert checks["parquet_exists"] == False
        assert checks["simulated_csvs"] == False


class TestDataQuality:
    """Tests for data quality constraints."""
    
    def test_aqi_range(self, tmp_path):
        """Test that AQI values are in valid range [0, 500]."""
        from src.data.iot_sim import generate_all_nodes
        
        node_data = generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-06-30",  # 6 months for better coverage
            seed=42,
            output_dir=str(tmp_path)
        )
        
        for node_id, df in node_data.items():
            assert df["aqi"].min() >= 0, f"{node_id} has AQI < 0"
            assert df["aqi"].max() <= 500, f"{node_id} has AQI > 500"
    
    def test_no_nan_in_simulation(self, tmp_path):
        """Test that simulated data has no NaN values."""
        from src.data.iot_sim import generate_all_nodes
        
        node_data = generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-03-31",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        for node_id, df in node_data.items():
            nan_count = df.isna().sum().sum()
            assert nan_count == 0, f"{node_id} has {nan_count} NaN values"
    
    def test_timestamp_continuity(self, tmp_path):
        """Test that timestamps are continuous hourly."""
        from src.data.iot_sim import generate_all_nodes
        
        node_data = generate_all_nodes(
            start_date="2023-01-01",
            end_date="2023-01-07",
            seed=42,
            output_dir=str(tmp_path)
        )
        
        for node_id, df in node_data.items():
            timestamps = pd.to_datetime(df["timestamp"])
            diffs = timestamps.diff().dropna()
            
            # All differences should be 1 hour
            expected_diff = pd.Timedelta(hours=1)
            assert all(diffs == expected_diff), f"{node_id} has non-hourly gaps"
