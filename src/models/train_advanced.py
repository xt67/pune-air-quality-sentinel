"""
Advanced training with feature engineering and ensemble methods.

Improvements:
1. Rich feature engineering (rolling stats, lag features, temporal encoding)
2. Larger model with residual connections
3. Ensemble predictions from multiple models
4. Better loss function (combined MAE + MAPE)
5. Longer training with patience
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import compute_aqi
from src.models.metrics import mae, rmse, category_accuracy

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AdvancedLSTM(nn.Module):
    """Enhanced LSTM with attention and residual connections."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        horizon: int = 24,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Output layers with residual
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, horizon)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        
        # Output with residual-like structure
        out = self.dropout(self.relu(self.bn1(self.fc1(context))))
        out = self.dropout(self.relu(self.bn2(self.fc2(out))))
        out = self.fc3(out)
        
        return out


def load_all_pune_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load ALL available Pune station data."""
    logger.info("Loading ALL Pune station data...")
    
    all_data = []
    
    # Try all Maharashtra stations
    for file_path in Path(data_dir).glob("MH*.csv"):
        station = file_path.stem
        df = pd.read_csv(file_path)
        df["station_id"] = station
        
        # Rename columns
        col_mapping = {
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
        }
        df = df.rename(columns=col_mapping)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        all_data.append(df)
        logger.info(f"  {station}: {len(df)} rows")
    
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Total: {len(combined)} rows from {len(all_data)} stations")
    
    return combined


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extensive feature engineering for better predictions."""
    logger.info("Engineering features...")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # === Basic cleaning ===
    pollutants = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    weather = ["temperature", "humidity", "wind_speed"]
    
    for col in pollutants + weather:
        if col in df.columns:
            # Remove negatives
            df.loc[df[col] < 0, col] = np.nan
            # Cap extreme outliers
            p999 = df[col].quantile(0.999)
            if pd.notna(p999) and p999 > 0:
                df.loc[df[col] > p999 * 1.5, col] = p999 * 1.5
            # Interpolate
            df[col] = df[col].interpolate(method="linear", limit=12)
            df[col] = df[col].ffill().bfill()
            # Fill remaining with median
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    
    # === Compute AQI ===
    logger.info("Computing AQI values...")
    aqi_values = []
    for _, row in df.iterrows():
        aqi = compute_aqi(
            pm25=row.get("pm25"),
            pm10=row.get("pm10"),
            no2=row.get("no2"),
            so2=row.get("so2"),
            co=row.get("co"),
            o3=row.get("o3"),
        )
        aqi_values.append(aqi)
    df["aqi"] = aqi_values
    df = df.dropna(subset=["aqi"])
    
    # === Rolling statistics ===
    logger.info("Creating rolling features...")
    for col in ["pm25", "pm10", "aqi"]:
        if col in df.columns:
            # Rolling means
            df[f"{col}_roll_6h"] = df[col].rolling(6, min_periods=1).mean()
            df[f"{col}_roll_12h"] = df[col].rolling(12, min_periods=1).mean()
            df[f"{col}_roll_24h"] = df[col].rolling(24, min_periods=1).mean()
            
            # Rolling std (volatility)
            df[f"{col}_std_6h"] = df[col].rolling(6, min_periods=1).std().fillna(0)
            df[f"{col}_std_24h"] = df[col].rolling(24, min_periods=1).std().fillna(0)
            
            # Rate of change
            df[f"{col}_diff"] = df[col].diff().fillna(0)
    
    # === Lag features ===
    logger.info("Creating lag features...")
    for lag in [1, 3, 6, 12, 24]:
        df[f"aqi_lag_{lag}h"] = df["aqi"].shift(lag).fillna(df["aqi"].median())
        df[f"pm25_lag_{lag}h"] = df["pm25"].shift(lag).fillna(df["pm25"].median())
    
    # === Temporal encoding ===
    logger.info("Creating temporal features...")
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Cyclical encoding for hour and month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Rush hour indicator (7-9 AM, 5-8 PM)
    df["is_rush_hour"] = ((df["hour"] >= 7) & (df["hour"] <= 9) | 
                          (df["hour"] >= 17) & (df["hour"] <= 20)).astype(int)
    
    # === Weather interactions ===
    if "temperature" in df.columns and "humidity" in df.columns:
        df["temp_humidity"] = df["temperature"] * df["humidity"] / 100
    
    if "wind_speed" in df.columns:
        df["wind_speed_inv"] = 1 / (df["wind_speed"] + 0.1)  # Low wind = stagnation
    
    # === Seasonal indicators ===
    # Diwali approximation (Oct-Nov)
    df["is_diwali_season"] = ((df["month"] >= 10) & (df["month"] <= 11)).astype(int)
    # Monsoon (June-Sept)
    df["is_monsoon"] = ((df["month"] >= 6) & (df["month"] <= 9)).astype(int)
    # Winter (Dec-Feb)
    df["is_winter"] = ((df["month"] == 12) | (df["month"] <= 2)).astype(int)
    
    logger.info(f"Final features: {len(df.columns)} columns")
    logger.info(f"Final rows: {len(df)}")
    
    return df


def create_sequences(data: np.ndarray, seq_len: int, horizon: int, target_idx: int):
    """Create sequences with proper shapes."""
    X, Y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        # Predict only the value at horizon (not sequence)
        Y.append(data[i + seq_len + horizon - 1, target_idx])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32).reshape(-1, 1)


class CombinedLoss(nn.Module):
    """Combined loss: MAE + weighted MAPE for better optimization."""
    
    def __init__(self, alpha: float = 0.7):
        super().__init__()
        self.alpha = alpha
        self.mae = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mae_loss = self.mae(pred, target)
        # MAPE with epsilon for stability
        mape_loss = torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1))
        return self.alpha * mae_loss + (1 - self.alpha) * mape_loss * 100


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: nn.Module,
    device: str,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 5e-4,
) -> Dict:
    """Train a single model with all improvements."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    criterion = CombinedLoss(alpha=0.8)
    
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            
            # Handle horizon mismatch
            if pred.shape[1] > 1 and batch_y.shape[1] == 1:
                pred = pred[:, -1:] 
            
            loss = criterion(pred, batch_y)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                if pred.shape[1] > 1 and batch_y.shape[1] == 1:
                    pred = pred[:, -1:]
                val_loss += criterion(pred, batch_y).item()
        
        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            logger.info(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return {"best_val_loss": best_val_loss, "history": history}


def train_ensemble(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    n_features: int,
    n_models: int = 3,
    epochs: int = 150,
) -> List[nn.Module]:
    """Train an ensemble of models with different seeds."""
    logger.info(f"Training ensemble of {n_models} models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    
    for i in range(n_models):
        logger.info(f"\nTraining model {i + 1}/{n_models}...")
        
        # Set different seed for each model
        torch.manual_seed(42 + i * 100)
        np.random.seed(42 + i * 100)
        
        # Shuffle data differently for each model
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        Y_shuffled = Y_train[idx]
        
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_shuffled), torch.from_numpy(Y_shuffled)),
            batch_size=64, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
            batch_size=64
        )
        
        model = AdvancedLSTM(
            input_size=n_features,
            hidden_size=256 + i * 32,  # Vary architecture slightly
            num_layers=3,
            horizon=1,  # Single-step prediction
            dropout=0.2 + i * 0.05,
        ).to(device)
        
        train_model(train_loader, val_loader, model, device, epochs=epochs, patience=25)
        models.append(model)
    
    return models


def evaluate_ensemble(
    models: List[nn.Module],
    X_test: np.ndarray,
    Y_test: np.ndarray,
    scaler: StandardScaler,
    target_idx: int,
    device: str = "cuda",
) -> Dict:
    """Evaluate ensemble predictions."""
    
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
        batch_size=64
    )
    
    all_preds = []
    all_actuals = []
    
    for model in models:
        model.eval()
        model_preds = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                pred = model(batch_x)
                if pred.shape[1] > 1:
                    pred = pred[:, -1:]
                model_preds.extend(pred.cpu().numpy().flatten())
        all_preds.append(model_preds)
    
    # Get actuals
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            all_actuals.extend(batch_y.numpy().flatten())
    
    # Ensemble mean
    ensemble_preds = np.mean(all_preds, axis=0)
    actuals = np.array(all_actuals)
    
    # Denormalize
    dummy = np.zeros((len(ensemble_preds), scaler.n_features_in_))
    dummy[:, target_idx] = ensemble_preds
    preds_denorm = scaler.inverse_transform(dummy)[:, target_idx]
    
    dummy[:, target_idx] = actuals
    actuals_denorm = scaler.inverse_transform(dummy)[:, target_idx]
    
    # Clip to valid AQI range
    preds_denorm = np.clip(preds_denorm, 0, 500)
    actuals_denorm = np.clip(actuals_denorm, 0, 500)
    
    # Metrics
    results = {
        "mae": float(mae(preds_denorm, actuals_denorm)),
        "rmse": float(rmse(preds_denorm, actuals_denorm)),
        "cat_acc": float(category_accuracy(preds_denorm, actuals_denorm)),
        "n_samples": len(preds_denorm),
    }
    
    return results


def main():
    """Main training pipeline."""
    logger.info("=" * 70)
    logger.info("ADVANCED MODEL TRAINING WITH FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    # Load and engineer features
    df = load_all_pune_data("data/raw")
    df = engineer_features(df)
    
    # Select features
    feature_cols = [
        # Core pollutants
        "pm25", "pm10", "no2", "so2", "o3",
        # Target
        "aqi",
        # Weather
        "temperature", "humidity",
        # Rolling features
        "pm25_roll_6h", "pm25_roll_12h", "pm25_roll_24h",
        "aqi_roll_6h", "aqi_roll_12h", "aqi_roll_24h",
        "pm25_std_6h", "aqi_std_6h",
        "pm25_diff", "aqi_diff",
        # Lag features
        "aqi_lag_1h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
        "pm25_lag_1h", "pm25_lag_6h",
        # Temporal
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "is_rush_hour", "is_weekend",
        # Seasonal
        "is_diwali_season", "is_monsoon", "is_winter",
    ]
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
    
    # Ensure no NaN
    df = df[feature_cols].dropna()
    logger.info(f"Final clean dataset: {len(df)} rows")
    
    # Split
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Scale with RobustScaler (handles outliers better)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # Create sequences
    seq_len = 72  # 3 days of history
    horizon = 24  # Predict 24h ahead
    target_idx = feature_cols.index("aqi")
    
    X_train, Y_train = create_sequences(train_scaled, seq_len, horizon, target_idx)
    X_val, Y_val = create_sequences(val_scaled, seq_len, horizon, target_idx)
    X_test, Y_test = create_sequences(test_scaled, seq_len, horizon, target_idx)
    
    logger.info(f"Sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Train ensemble
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    models = train_ensemble(
        X_train, Y_train, X_val, Y_val,
        n_features=len(feature_cols),
        n_models=3,
        epochs=150,
    )
    
    # Evaluate
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION")
    logger.info("=" * 70)
    
    results = evaluate_ensemble(models, X_test, Y_test, scaler, target_idx, device)
    
    logger.info(f"Ensemble MAE: {results['mae']:.2f}")
    logger.info(f"Ensemble RMSE: {results['rmse']:.2f}")
    logger.info(f"Category Accuracy: {results['cat_acc'] * 100:.1f}%")
    
    # Save results
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    results_df = pd.DataFrame([{
        "Model": "Ensemble LSTM (3 models)",
        "MAE": results["mae"],
        "RMSE": results["rmse"],
        "Category_Accuracy": results["cat_acc"],
    }])
    results_df.to_csv(output_dir / "advanced_results.csv", index=False)
    logger.info(f"Results saved to {output_dir / 'advanced_results.csv'}")
    
    # Save best model
    torch.save({
        "model_states": [m.state_dict() for m in models],
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feature_cols": feature_cols,
        "target_idx": target_idx,
        "seq_len": seq_len,
    }, "models/ensemble_lstm.pth")
    logger.info("Models saved to models/ensemble_lstm.pth")
    
    return results


if __name__ == "__main__":
    main()
