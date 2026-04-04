---
phase: 2
plan: 2
title: "ARIMA Baseline Model"
wave: 1
depends_on: []
files_modified:
  - src/models/arima.py
  - tests/test_arima.py
requirements_addressed: [ARIMA-01, ARIMA-02, ARIMA-03, ARIMA-04]
autonomous: true
---

# Plan 2: ARIMA Baseline Model

<objective>
Implement auto-ARIMA baseline model with seasonal component for each node. Provides benchmark MAE for LSTM/ST-GNN comparison.
</objective>

<must_haves>
- Auto-ARIMA with seasonal=True, m=24 (hourly data, daily cycle)
- Fallback to ARIMA(2,1,2) on convergence failure
- Train one model per node (10 total)
- 24h and 48h ahead forecasts
- Serialize with joblib
- Unit tests
</must_haves>

## Tasks

<task id="2.1">
<title>Install pmdarima</title>
<action>
```bash
pip install pmdarima
```
</action>
</task>

<task id="2.2">
<title>Create ARIMA wrapper</title>
<action>
Create src/models/arima.py:

```python
"""ARIMA baseline model for AQI forecasting."""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ARIMAModel:
    """
    Auto-ARIMA wrapper for single-node AQI forecasting.
    
    Uses pmdarima for automatic order selection with seasonal component.
    Falls back to ARIMA(2,1,2) on convergence failure.
    """
    
    def __init__(
        self,
        node_id: str,
        seasonal: bool = True,
        m: int = 24,  # Daily seasonality for hourly data
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2
    ):
        self.node_id = node_id
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.model = None
        self.fitted = False
        
    def fit(self, series: np.ndarray) -> "ARIMAModel":
        """
        Fit auto-ARIMA model to time series.
        
        Args:
            series: 1D array of AQI values (chronologically ordered)
            
        Returns:
            self for chaining
        """
        import pmdarima as pm
        
        logger.info(f"Fitting ARIMA for node {self.node_id} ({len(series)} points)")
        
        try:
            self.model = pm.auto_arima(
                series,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore"
            )
            self.fitted = True
            logger.info(f"Node {self.node_id}: {self.model.order}, seasonal={self.model.seasonal_order}")
            
        except Exception as e:
            logger.warning(f"Auto-ARIMA failed for {self.node_id}: {e}. Using fallback (2,1,2)")
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(series, order=(2, 1, 2)).fit()
            self.fitted = True
            
        return self
    
    def predict(self, horizon: int = 24) -> np.ndarray:
        """
        Generate forecast for given horizon.
        
        Args:
            horizon: Number of steps to forecast (default 24h)
            
        Returns:
            Array of predicted values
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        return self.model.predict(n_periods=horizon)
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved ARIMA model to {path}")
        
    @classmethod
    def load(cls, path: str) -> "ARIMAModel":
        """Load model from disk."""
        return joblib.load(path)


def train_all_nodes(
    df: pd.DataFrame,
    output_dir: str = "models",
    target_col: str = "aqi"
) -> Dict[str, ARIMAModel]:
    """
    Train ARIMA model for each node.
    
    Args:
        df: DataFrame with node_id and target columns
        output_dir: Directory to save models
        target_col: Target column name
        
    Returns:
        Dictionary mapping node_id to trained model
    """
    models = {}
    
    for node_id in df["node_id"].unique():
        node_df = df[df["node_id"] == node_id].sort_values("timestamp")
        series = node_df[target_col].values
        
        model = ARIMAModel(node_id)
        model.fit(series)
        
        # Save model
        model_path = f"{output_dir}/arima_{node_id}.pkl"
        model.save(model_path)
        
        models[node_id] = model
        
    logger.info(f"Trained {len(models)} ARIMA models")
    return models


def evaluate_arima(
    models: Dict[str, ARIMAModel],
    test_df: pd.DataFrame,
    horizon: int = 24,
    target_col: str = "aqi"
) -> Dict[str, float]:
    """
    Evaluate ARIMA models on test set.
    
    Returns MAE per node.
    """
    results = {}
    
    for node_id, model in models.items():
        node_test = test_df[test_df["node_id"] == node_id].sort_values("timestamp")
        
        if len(node_test) < horizon:
            logger.warning(f"Insufficient test data for {node_id}")
            continue
            
        # Get actual values
        actual = node_test[target_col].values[:horizon]
        
        # Get predictions
        pred = model.predict(horizon)
        
        # Calculate MAE
        mae = np.mean(np.abs(actual - pred))
        results[node_id] = mae
        
    avg_mae = np.mean(list(results.values()))
    results["average"] = avg_mae
    
    logger.info(f"ARIMA Average MAE: {avg_mae:.2f}")
    return results
```
</action>
</task>

<task id="2.3">
<title>Create unit tests</title>
<action>
Create tests/test_arima.py with tests for:
- Model initialization
- Fit on sample data
- Predict shapes
- Save/load roundtrip
</action>
</task>

<acceptance_criteria>
- `pip show pmdarima` shows installed
- `pytest tests/test_arima.py -v` all pass
- Model saves to models/arima_N01.pkl format
</acceptance_criteria>
