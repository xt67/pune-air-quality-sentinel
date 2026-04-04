"""Train and evaluate ARIMA and LSTM models on Pune data."""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import compute_aqi_row
from src.models.arima import ARIMAModel
from src.models.dataset import AQIDataset, create_dataloaders
from src.models.lstm import LSTMForecaster
from src.models.train import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_pune_data(path: str = "data/raw/MH020.csv") -> pd.DataFrame:
    """Load and preprocess Pune Karve Road station data."""
    logger.info(f"Loading data from {path}")
    
    df = pd.read_csv(path)
    
    # Rename columns for consistency
    df = df.rename(columns={
        "From Date": "timestamp",
        "PM2.5 (ug/m3)": "pm25",
        "PM10 (ug/m3)": "pm10",
        "NO2 (ug/m3)": "no2",
        "SO2 (ug/m3)": "so2",
        "CO (mg/m3)": "co",
        "Ozone (ug/m3)": "o3",
        "RH (%)": "humidity",
        "AT (degree C)": "temperature",
        "WS (m/s)": "wind_speed",
        "WD (degree)": "wind_dir",
    })
    
    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Select relevant columns
    cols = ["timestamp", "pm25", "pm10", "no2", "so2", "co", "o3", "temperature", "humidity", "wind_speed", "wind_dir"]
    df = df[[c for c in cols if c in df.columns]]
    
    # Add node_id for compatibility
    df["node_id"] = "MH020"
    
    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and calculate AQI."""
    logger.info("Preprocessing data...")
    
    # CO is already in mg/m3 in the dataset (e.g., 0.23 mg/m3) - DON'T convert
    # The CPCB breakpoints expect CO in mg/m3
    
    # Fill missing values with forward fill then backward fill
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].ffill().bfill()
    
    # Calculate AQI using compute_aqi_row
    df["aqi"] = df.apply(compute_aqi_row, axis=1)
    
    # Drop rows with NaN AQI
    initial_len = len(df)
    df = df.dropna(subset=["aqi"])
    logger.info(f"Dropped {initial_len - len(df)} rows with NaN AQI")
    
    # Log AQI stats
    logger.info(f"AQI stats: min={df['aqi'].min():.1f}, max={df['aqi'].max():.1f}, mean={df['aqi'].mean():.1f}")
    
    # Add time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    
    return df


def create_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple:
    """Create chronological train/val/test splits."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df


def train_arima(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train and evaluate ARIMA model."""
    logger.info("=" * 50)
    logger.info("Training ARIMA model...")
    
    # Get AQI series
    train_series = train_df["aqi"].values
    
    # Train model (non-seasonal for speed, use last 2000 points)
    model = ARIMAModel("MH020", seasonal=False, max_p=3, max_q=3)
    model.fit(train_series[-2000:])  # Use recent data for faster training
    
    # Predict on test set
    horizon = 24
    test_series = test_df["aqi"].values
    
    # Make predictions at intervals
    predictions = []
    actuals = []
    
    for i in range(0, len(test_series) - horizon, horizon):
        # Refit on recent data (rolling window)
        recent = np.concatenate([train_series[-1000:], test_series[:i]])[-2000:]
        model.fit(recent)
        
        pred = model.predict(horizon)
        actual = test_series[i:i + horizon]
        
        predictions.extend(pred)
        actuals.extend(actual)
        
        if len(predictions) >= 500:  # Limit for speed
            break
    
    # Calculate MAE
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    logger.info(f"ARIMA Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return {"mae": mae, "rmse": rmse, "predictions": predictions, "actuals": actuals}


def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train and evaluate LSTM model."""
    logger.info("=" * 50)
    logger.info("Training LSTM model...")
    
    # Feature columns - only use columns that exist
    possible_cols = ["pm25", "pm10", "temperature", "humidity", "wind_speed", "hour", "day_of_week", "month", "aqi"]
    feature_cols = [c for c in possible_cols if c in train_df.columns and train_df[c].notna().any()]
    
    logger.info(f"Using features: {feature_cols}")
    
    # Subset to feature columns plus identifiers
    train_subset = train_df[feature_cols + ["node_id", "timestamp"]].copy()
    val_subset = val_df[feature_cols + ["node_id", "timestamp"]].copy()
    test_subset = test_df[feature_cols + ["node_id", "timestamp"]].copy()
    
    # Normalize using sklearn directly
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    train_normalized = train_subset.copy()
    val_normalized = val_subset.copy()
    test_normalized = test_subset.copy()
    
    train_normalized[feature_cols] = scaler.fit_transform(train_subset[feature_cols])
    val_normalized[feature_cols] = scaler.transform(val_subset[feature_cols])
    test_normalized[feature_cols] = scaler.transform(test_subset[feature_cols])
    
    # Create dataloaders
    window_size = 72
    horizon = 24
    batch_size = 64
    
    loaders = create_dataloaders(
        train_normalized, val_normalized, test_normalized,
        window_size=window_size, horizon=horizon,
        batch_size=batch_size, feature_cols=feature_cols
    )
    
    logger.info(f"Train batches: {len(loaders['train'])}, Val batches: {len(loaders['val'])}, Test batches: {len(loaders['test'])}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create model with tuned hyperparameters
    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_sizes=(256, 128),  # Larger hidden sizes
        horizon=horizon,
        dropout=0.25
    ).to(device)
    
    # Initialize weights
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            torch.nn.init.xavier_uniform_(param)
        elif "bias" in name:
            torch.nn.init.zeros_(param)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Device: {device}")
    
    # Train with tuned hyperparameters
    trainer = Trainer(
        model, loaders["train"], loaders["val"],
        lr=5e-4,  # Lower learning rate
        device=device, 
        checkpoint_dir="models"
    )
    
    history = trainer.train(epochs=100, patience=20)  # More epochs and patience
    
    # Evaluate on test set
    model.eval()
    all_preds, all_actuals = [], []
    
    with torch.no_grad():
        for x, y in loaders["test"]:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_preds.extend(pred.flatten())
            all_actuals.extend(y.numpy().flatten())
    
    # Denormalize predictions (AQI is last feature)
    aqi_idx = feature_cols.index("aqi")
    aqi_min = scaler.data_min_[aqi_idx]
    aqi_max = scaler.data_max_[aqi_idx]
    
    predictions = np.array(all_preds) * (aqi_max - aqi_min) + aqi_min
    actuals = np.array(all_actuals) * (aqi_max - aqi_min) + aqi_min
    
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    logger.info(f"LSTM Results - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return {"mae": mae, "rmse": rmse, "history": history, "predictions": predictions, "actuals": actuals}


def main():
    """Main training pipeline."""
    logger.info("=" * 60)
    logger.info("Pune Air Quality Model Training")
    logger.info("=" * 60)
    
    # Load and preprocess data
    df = load_pune_data()
    df = preprocess_data(df)
    
    # Create splits
    train_df, val_df, test_df = create_splits(df)
    
    # Train models
    arima_results = train_arima(train_df, test_df)
    lstm_results = train_lstm(train_df, val_df, test_df)
    
    # Compare results
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<10} {'MAE':<10} {'RMSE':<10} {'Target MAE':<12}")
    logger.info("-" * 42)
    logger.info(f"{'ARIMA':<10} {arima_results['mae']:<10.2f} {arima_results['rmse']:<10.2f} {'≤30':<12}")
    logger.info(f"{'LSTM':<10} {lstm_results['mae']:<10.2f} {lstm_results['rmse']:<10.2f} {'≤20':<12}")
    logger.info("=" * 60)
    
    # Check if targets met
    if arima_results["mae"] <= 30:
        logger.info("✅ ARIMA meets target MAE ≤ 30")
    else:
        logger.info("❌ ARIMA does not meet target MAE ≤ 30")
        
    if lstm_results["mae"] <= 20:
        logger.info("✅ LSTM meets target MAE ≤ 20")
    else:
        logger.info("❌ LSTM does not meet target MAE ≤ 20")
    
    return arima_results, lstm_results


if __name__ == "__main__":
    main()
