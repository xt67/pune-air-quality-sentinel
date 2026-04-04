"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.models.metrics import (
    mae,
    rmse,
    mape,
    r2_score,
    aqi_to_category,
    category_accuracy,
    adjacent_category_accuracy,
    skill_score,
    persistence_baseline,
    compute_all_metrics,
    compute_horizon_metrics,
    generate_comparison_table,
)


class TestMAE:
    """Tests for MAE metric."""
    
    def test_perfect_prediction(self):
        """MAE of perfect predictions should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)
    
    def test_known_value(self):
        """MAE with known values."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])  # Each off by 1
        assert mae(y_true, y_pred) == pytest.approx(1.0)
    
    def test_symmetric(self):
        """MAE should be symmetric."""
        y1 = np.array([1.0, 2.0, 3.0])
        y2 = np.array([2.0, 3.0, 4.0])
        assert mae(y1, y2) == mae(y2, y1)
    
    def test_non_negative(self):
        """MAE should always be non-negative."""
        y_true = np.random.randn(100)
        y_pred = np.random.randn(100)
        assert mae(y_true, y_pred) >= 0


class TestRMSE:
    """Tests for RMSE metric."""
    
    def test_perfect_prediction(self):
        """RMSE of perfect predictions should be 0."""
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)
    
    def test_known_value(self):
        """RMSE with known values."""
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([2.0, 2.0, 2.0])  # Each off by 1
        assert rmse(y_true, y_pred) == pytest.approx(1.0)
    
    def test_greater_than_mae(self):
        """RMSE should be >= MAE."""
        y_true = np.random.randn(100) * 50 + 100
        y_pred = y_true + np.random.randn(100) * 10
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestMAPE:
    """Tests for MAPE metric."""
    
    def test_perfect_prediction(self):
        """MAPE of perfect predictions should be ~0."""
        y = np.array([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0, abs=1e-4)
    
    def test_known_value(self):
        """MAPE with known values."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 180.0])  # 10% error each
        assert mape(y_true, y_pred) == pytest.approx(10.0, rel=0.01)
    
    def test_handles_near_zero(self):
        """MAPE should handle near-zero values."""
        y_true = np.array([0.001, 100.0])
        y_pred = np.array([0.002, 100.0])
        result = mape(y_true, y_pred)
        assert not np.isnan(result)
        assert not np.isinf(result)


class TestR2Score:
    """Tests for R² score."""
    
    def test_perfect_prediction(self):
        """R² of perfect predictions should be 1.0."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert r2_score(y, y) == pytest.approx(1.0)
    
    def test_mean_baseline(self):
        """Predicting mean should give R² = 0."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)
    
    def test_can_be_negative(self):
        """R² can be negative for bad predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # Very wrong
        assert r2_score(y_true, y_pred) < 0


class TestAQIToCategory:
    """Tests for AQI category conversion."""
    
    def test_good_category(self):
        """AQI 0-50 should be category 0 (Good)."""
        assert aqi_to_category(25) == 0
        assert aqi_to_category(50) == 0
    
    def test_satisfactory_category(self):
        """AQI 51-100 should be category 1 (Satisfactory)."""
        assert aqi_to_category(51) == 1
        assert aqi_to_category(75) == 1
        assert aqi_to_category(100) == 1
    
    def test_moderate_category(self):
        """AQI 101-200 should be category 2 (Moderate)."""
        assert aqi_to_category(150) == 2
    
    def test_poor_category(self):
        """AQI 201-300 should be category 3 (Poor)."""
        assert aqi_to_category(250) == 3
    
    def test_very_poor_category(self):
        """AQI 301-400 should be category 4 (Very Poor)."""
        assert aqi_to_category(350) == 4
    
    def test_severe_category(self):
        """AQI 401+ should be category 5 (Severe)."""
        assert aqi_to_category(450) == 5
        assert aqi_to_category(500) == 5
    
    def test_array_input(self):
        """Should handle numpy arrays."""
        aqi = np.array([25, 75, 150, 250, 350, 450])
        expected = np.array([0, 1, 2, 3, 4, 5])
        result = aqi_to_category(aqi)
        np.testing.assert_array_equal(result, expected)


class TestCategoryAccuracy:
    """Tests for category accuracy metrics."""
    
    def test_perfect_accuracy(self):
        """Same values should give 100% accuracy."""
        y = np.array([50, 100, 150, 250])
        assert category_accuracy(y, y) == pytest.approx(1.0)
    
    def test_all_wrong(self):
        """Completely wrong categories should give 0%."""
        y_true = np.array([25, 25, 25])   # All Good
        y_pred = np.array([450, 450, 450])  # All Severe
        assert category_accuracy(y_true, y_pred) == pytest.approx(0.0)
    
    def test_partial_accuracy(self):
        """Mixed results should give partial accuracy."""
        y_true = np.array([25, 75, 150, 250])  # 0, 1, 2, 3
        y_pred = np.array([25, 75, 250, 350])  # 0, 1, 3, 4
        # 2 out of 4 correct
        assert category_accuracy(y_true, y_pred) == pytest.approx(0.5)
    
    def test_adjacent_accuracy(self):
        """Adjacent category accuracy should be more lenient."""
        y_true = np.array([25, 75, 150, 250])  # 0, 1, 2, 3
        y_pred = np.array([51, 101, 201, 301])  # 1, 2, 3, 4 (all off by 1)
        
        # Exact: 0%, Adjacent: 100%
        assert category_accuracy(y_true, y_pred) == pytest.approx(0.0)
        assert adjacent_category_accuracy(y_true, y_pred) == pytest.approx(1.0)


class TestSkillScore:
    """Tests for skill score."""
    
    def test_perfect_model(self):
        """Perfect model should have skill score = 1."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = y_true.copy()
        y_baseline = np.array([0.0, 0.0, 0.0])
        assert skill_score(y_true, y_pred, y_baseline) == pytest.approx(1.0)
    
    def test_same_as_baseline(self):
        """Same as baseline should have skill score = 0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        y_baseline = y_pred.copy()
        assert skill_score(y_true, y_pred, y_baseline) == pytest.approx(0.0)
    
    def test_worse_than_baseline(self):
        """Worse than baseline should have negative skill score."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 10.0, 10.0])  # Very wrong
        y_baseline = np.array([1.5, 2.5, 3.5])  # Close
        assert skill_score(y_true, y_pred, y_baseline) < 0


class TestPersistenceBaseline:
    """Tests for persistence baseline."""
    
    def test_horizon_shift(self):
        """Persistence should shift by horizon."""
        y = np.array([1, 2, 3, 4, 5, 6])
        baseline = persistence_baseline(y, horizon=2)
        # Values shifted by 2: [5, 6, 1, 2, 3, 4]
        expected = np.array([5, 6, 1, 2, 3, 4])
        np.testing.assert_array_equal(baseline, expected)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""
    
    def test_returns_all_keys(self):
        """Should return all expected metrics."""
        y_true = np.array([100.0, 150.0, 200.0, 250.0])
        y_pred = np.array([110.0, 140.0, 210.0, 240.0])
        
        metrics = compute_all_metrics(y_true, y_pred)
        
        expected_keys = {
            "mae", "rmse", "mape", "r2",
            "category_accuracy", "adjacent_category_accuracy"
        }
        assert set(metrics.keys()) == expected_keys
    
    def test_with_baseline(self):
        """Should include skill score when baseline provided."""
        y_true = np.array([100.0, 150.0, 200.0])
        y_pred = np.array([110.0, 140.0, 210.0])
        y_baseline = np.array([100.0, 100.0, 100.0])
        
        metrics = compute_all_metrics(y_true, y_pred, y_baseline)
        
        assert "skill_score" in metrics


class TestComputeHorizonMetrics:
    """Tests for compute_horizon_metrics function."""
    
    def test_multiple_horizons(self):
        """Should compute metrics for multiple horizons."""
        y_true = np.random.randn(100, 24) * 50 + 150
        y_pred = y_true + np.random.randn(100, 24) * 10
        
        metrics = compute_horizon_metrics(y_true, y_pred, horizons=[1, 6, 12, 24])
        
        assert "h1" in metrics
        assert "h6" in metrics
        assert "h12" in metrics
        assert "h24" in metrics
    
    def test_each_horizon_has_metrics(self):
        """Each horizon should have MAE, RMSE, category accuracy."""
        y_true = np.random.randn(50, 24) * 50 + 150
        y_pred = y_true + np.random.randn(50, 24) * 5
        
        metrics = compute_horizon_metrics(y_true, y_pred)
        
        for h_key in metrics:
            assert "mae" in metrics[h_key]
            assert "rmse" in metrics[h_key]
            assert "category_accuracy" in metrics[h_key]


class TestGenerateComparisonTable:
    """Tests for comparison table generation."""
    
    def test_markdown_format(self):
        """Should generate valid markdown table."""
        results = {
            "ARIMA": {"mae": 30.5, "rmse": 45.2},
            "LSTM": {"mae": 25.3, "rmse": 38.1},
        }
        
        table = generate_comparison_table(results)
        
        assert "| Model |" in table
        assert "| ARIMA |" in table
        assert "| LSTM |" in table
        assert "|---|" in table
    
    def test_custom_model_order(self):
        """Should respect custom model ordering."""
        results = {
            "C": {"mae": 30.0},
            "A": {"mae": 10.0},
            "B": {"mae": 20.0},
        }
        
        table = generate_comparison_table(results, model_names=["A", "B", "C"])
        
        # A should appear before B, B before C
        pos_a = table.find("| A |")
        pos_b = table.find("| B |")
        pos_c = table.find("| C |")
        
        assert pos_a < pos_b < pos_c
    
    def test_formats_floats(self):
        """Should format floats to 4 decimal places."""
        results = {"Model": {"mae": 25.123456789}}
        table = generate_comparison_table(results)
        assert "25.1235" in table  # Rounded to 4 decimals
