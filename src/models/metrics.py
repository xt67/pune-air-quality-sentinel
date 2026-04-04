"""
Evaluation metrics for AQI forecasting models.

Includes standard regression metrics and AQI-specific metrics.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        MAE score
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        RMSE score
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small value to avoid division by zero
        
    Returns:
        MAPE score (as percentage, 0-100)
    """
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score (1.0 is perfect, can be negative for bad predictions)
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return float(1 - (ss_res / ss_tot))


def aqi_to_category(aqi: Union[float, np.ndarray]) -> Union[int, np.ndarray]:
    """
    Convert AQI value(s) to category index.
    
    Categories (CPCB India):
        0: Good (0-50)
        1: Satisfactory (51-100)
        2: Moderate (101-200)
        3: Poor (201-300)
        4: Very Poor (301-400)
        5: Severe (401+)
    
    Args:
        aqi: AQI value(s)
        
    Returns:
        Category index/indices
    """
    breakpoints = [0, 51, 101, 201, 301, 401]
    
    if isinstance(aqi, np.ndarray):
        categories = np.zeros_like(aqi, dtype=int)
        for i, bp in enumerate(breakpoints[1:]):
            categories[aqi >= bp] = i + 1
        categories[aqi > 500] = 5
        return categories
    else:
        for i, bp in enumerate(breakpoints[1:]):
            if aqi < bp:
                return i
        return 5


def category_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AQI category prediction accuracy.
    
    Measures how often the predicted AQI falls in the correct category.
    
    Args:
        y_true: Ground truth AQI values
        y_pred: Predicted AQI values
        
    Returns:
        Accuracy (0-1)
    """
    true_cat = aqi_to_category(y_true)
    pred_cat = aqi_to_category(y_pred)
    return float(np.mean(true_cat == pred_cat))


def adjacent_category_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    AQI category prediction accuracy allowing adjacent categories.
    
    A prediction is correct if it's in the same or adjacent category.
    
    Args:
        y_true: Ground truth AQI values
        y_pred: Predicted AQI values
        
    Returns:
        Accuracy (0-1)
    """
    true_cat = aqi_to_category(y_true)
    pred_cat = aqi_to_category(y_pred)
    diff = np.abs(true_cat - pred_cat)
    return float(np.mean(diff <= 1))


def skill_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: np.ndarray,
) -> float:
    """
    Skill score comparing model to a baseline.
    
    SS = 1 - (MSE_model / MSE_baseline)
    
    Positive: better than baseline
    Zero: same as baseline
    Negative: worse than baseline
    
    Args:
        y_true: Ground truth values
        y_pred: Model predictions
        y_baseline: Baseline predictions (e.g., persistence, climatology)
        
    Returns:
        Skill score
    """
    mse_model = np.mean((y_true - y_pred) ** 2)
    mse_baseline = np.mean((y_true - y_baseline) ** 2)
    
    if mse_baseline == 0:
        return 0.0
    
    return float(1 - (mse_model / mse_baseline))


def persistence_baseline(y: np.ndarray, horizon: int = 24) -> np.ndarray:
    """
    Create persistence baseline (last known value).
    
    Args:
        y: Time series of actual values
        horizon: Forecast horizon
        
    Returns:
        Persistence predictions
    """
    return np.roll(y, horizon)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_baseline: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        y_baseline: Optional baseline predictions for skill score
        
    Returns:
        Dict with all metric values
    """
    metrics = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "category_accuracy": category_accuracy(y_true, y_pred),
        "adjacent_category_accuracy": adjacent_category_accuracy(y_true, y_pred),
    }
    
    if y_baseline is not None:
        metrics["skill_score"] = skill_score(y_true, y_pred, y_baseline)
    
    return metrics


def compute_horizon_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizons: List[int] = [1, 6, 12, 24],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for different forecast horizons.
    
    Args:
        y_true: Ground truth (samples, horizon)
        y_pred: Predictions (samples, horizon)
        horizons: List of horizon indices to evaluate
        
    Returns:
        Dict mapping horizon to metrics dict
    """
    results = {}
    
    for h in horizons:
        if h > y_true.shape[1]:
            continue
        
        y_t = y_true[:, h - 1]  # h is 1-indexed
        y_p = y_pred[:, h - 1]
        
        results[f"h{h}"] = {
            "mae": mae(y_t, y_p),
            "rmse": rmse(y_t, y_p),
            "category_accuracy": category_accuracy(y_t, y_p),
        }
    
    return results


def generate_comparison_table(
    results: Dict[str, Dict[str, float]],
    model_names: Optional[List[str]] = None,
) -> str:
    """
    Generate a markdown comparison table.
    
    Args:
        results: Dict mapping model name to metrics dict
        model_names: Optional list to order models
        
    Returns:
        Markdown table string
    """
    if model_names is None:
        model_names = list(results.keys())
    
    # Get all metrics
    all_metrics = set()
    for metrics in results.values():
        all_metrics.update(metrics.keys())
    metrics_list = sorted(all_metrics)
    
    # Build table
    lines = []
    header = "| Model | " + " | ".join(metrics_list) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics_list) + 1)) + "|"
    lines.append(header)
    lines.append(separator)
    
    for name in model_names:
        if name not in results:
            continue
        
        values = []
        for metric in metrics_list:
            val = results[name].get(metric, "N/A")
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        
        row = f"| {name} | " + " | ".join(values) + " |"
        lines.append(row)
    
    return "\n".join(lines)
