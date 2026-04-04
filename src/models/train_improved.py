"""
Improved training with clean data and optimized hyperparameters.

Cleans data, removes NaN, uses interpolation, and trains with better settings.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import compute_aqi
from src.models.lstm import LSTMForecaster
from src.models.stgnn import STGNN
from src.models.graph import build_adjacency_matrix, normalize_adjacency, PUNE_STATIONS
from src.models.metrics import mae, rmse, category_accuracy, compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_and_clean_pune_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load and combine all Pune station data with thorough cleaning."""
    logger.info("Loading and cleaning Pune data...")
    
    # Find all Maharashtra files (Pune is MH020-MH041 roughly)
    pune_stations = ["MH020", "MH021", "MH022", "MH023", "MH024", "MH025"]
    
    all_data = []
    
    for station in pune_stations:
        file_path = Path(data_dir) / f"{station}.csv"
        if not file_path.exists():
            continue
            
        df = pd.read_csv(file_path)
        df["station_id"] = station
        
        # Rename columns
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
        })
        
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        
        # Select columns
        cols = ["timestamp", "station_id", "pm25", "pm10", "no2", "so2", "co", "o3", 
                "temperature", "humidity", "wind_speed"]
        df = df[[c for c in cols if c in df.columns]]
        
        all_data.append(df)
        logger.info(f"  {station}: {len(df)} rows")
    
    # Combine all stations
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined)} rows")
    
    return combined


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Thoroughly clean the data - remove NaN, interpolate, validate."""
    logger.info("Cleaning data...")
    
    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Remove rows with invalid timestamps
    df = df.dropna(subset=["timestamp"])
    
    # Pollutant columns
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    weather_cols = ["temperature", "humidity", "wind_speed"]
    
    # Step 1: Remove obvious outliers
    for col in pollutant_cols:
        if col in df.columns:
            # Remove negative values
            df.loc[df[col] < 0, col] = np.nan
            # Remove extreme outliers (> 99.9th percentile * 2)
            p999 = df[col].quantile(0.999)
            df.loc[df[col] > p999 * 2, col] = np.nan
    
    # Step 2: Interpolate missing values (linear for short gaps)
    for col in pollutant_cols + weather_cols:
        if col in df.columns:
            # Forward fill for gaps up to 6 hours
            df[col] = df[col].interpolate(method="linear", limit=6)
            # Then fill remaining with forward/backward fill
            df[col] = df[col].ffill().bfill()
    
    # Step 3: Ensure we have at least PM2.5 or PM10
    df = df.dropna(subset=["pm25", "pm10"], how="all")
    
    # Step 4: Fill any remaining NaN in required pollutants with median
    for col in pollutant_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled {col} NaN with median: {median_val:.2f}")
    
    logger.info(f"After cleaning: {len(df)} rows")
    logger.info(f"NaN remaining: {df[pollutant_cols].isnull().sum().sum()}")
    
    return df


def compute_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    """Compute AQI for each row."""
    logger.info("Computing AQI...")
    
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
    
    # Remove rows where AQI couldn't be computed
    df = df.dropna(subset=["aqi"])
    
    logger.info(f"AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    logger.info(f"Final dataset: {len(df)} rows")
    
    return df


def create_sequences(data: np.ndarray, seq_len: int, horizon: int, target_idx: int):
    """Create input/output sequences for training."""
    X, Y = [], []
    for i in range(len(data) - seq_len - horizon + 1):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + horizon, target_idx])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_improved_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    epochs: int = 100,
    patience: int = 15,
) -> Dict:
    """Train LSTM with improved hyperparameters."""
    logger.info("Training improved LSTM...")
    
    # Normalize with StandardScaler (often better than MinMax)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # Parameters
    seq_len = 48  # Shorter sequence for less noise
    horizon = 24
    target_idx = feature_cols.index("aqi")
    batch_size = 64
    
    # Create sequences
    X_train, Y_train = create_sequences(train_scaled, seq_len, horizon, target_idx)
    X_val, Y_val = create_sequences(val_scaled, seq_len, horizon, target_idx)
    X_test, Y_test = create_sequences(test_scaled, seq_len, horizon, target_idx)
    
    logger.info(f"Sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
        batch_size=batch_size
    )
    
    # Model with better architecture
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_sizes=(256, 128),  # Larger hidden layers
        horizon=horizon,
        dropout=0.3  # More dropout for regularization
    ).to(device)
    
    # Initialize weights properly
    for name, param in model.named_parameters():
        if "weight" in name and param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.SmoothL1Loss()  # Robust to outliers
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            all_preds.extend(pred.flatten())
            all_actuals.extend(batch_y.numpy().flatten())
    
    # Denormalize
    aqi_mean = scaler.mean_[target_idx]
    aqi_std = scaler.scale_[target_idx]
    
    predictions = np.array(all_preds) * aqi_std + aqi_mean
    actuals = np.array(all_actuals) * aqi_std + aqi_mean
    
    # Clip predictions to valid range
    predictions = np.clip(predictions, 0, 500)
    
    # Metrics
    results = compute_all_metrics(actuals, predictions)
    logger.info(f"LSTM Results - MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    logger.info(f"Category Accuracy: {results['category_accuracy']*100:.1f}%")
    
    return {**results, "predictions": predictions, "actuals": actuals, "model": model, "scaler": scaler}


def train_improved_stgnn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    epochs: int = 100,
    patience: int = 15,
) -> Dict:
    """Train ST-GNN with improved setup."""
    logger.info("Training improved ST-GNN...")
    
    # Normalize
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # Parameters
    seq_len = 48
    horizon = 24
    num_nodes = len(feature_cols)  # Treat features as nodes
    features_per_node = 1
    batch_size = 64
    
    def create_stgnn_sequences(data, seq_len, horizon):
        X, Y = [], []
        target_idx = feature_cols.index("aqi")
        for i in range(len(data) - seq_len - horizon + 1):
            x = data[i:i + seq_len].reshape(seq_len, num_nodes, features_per_node)
            y = data[i + seq_len:i + seq_len + horizon, target_idx]
            X.append(x)
            Y.append(y)
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    
    X_train, Y_train = create_stgnn_sequences(train_scaled, seq_len, horizon)
    X_val, Y_val = create_stgnn_sequences(val_scaled, seq_len, horizon)
    X_test, Y_test = create_stgnn_sequences(test_scaled, seq_len, horizon)
    
    logger.info(f"ST-GNN sequences: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    
    # Build adjacency (correlation-based for features)
    adj = np.corrcoef(train_df[feature_cols].T.values)
    adj = np.abs(adj)  # Use absolute correlation
    adj = normalize_adjacency(adj)
    adj_tensor = torch.from_numpy(adj.astype(np.float32))
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
        batch_size=batch_size
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
        batch_size=batch_size
    )
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STGNN(
        num_nodes=num_nodes,
        input_dim=features_per_node,
        hidden_dim=128,  # Larger
        output_dim=horizon,
        gru_layers=2,
        dropout=0.3
    ).to(device)
    
    logger.info(f"ST-GNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.SmoothL1Loss()
    
    adj_tensor = adj_tensor.to(device)
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    
    target_node_idx = feature_cols.index("aqi")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x, adj_tensor)
            pred = output[:, target_node_idx, :]
            loss = criterion(pred, batch_y)
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x, adj_tensor)
                pred = output[:, target_node_idx, :]
                val_loss += criterion(pred, batch_y).item()
        
        val_loss /= len(val_loader)
        train_loss /= len(train_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Evaluate
    model.eval()
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x, adj_tensor)
            pred = output[:, target_node_idx, :].cpu().numpy()
            all_preds.extend(pred.flatten())
            all_actuals.extend(batch_y.numpy().flatten())
    
    # Denormalize
    target_idx = feature_cols.index("aqi")
    aqi_mean = scaler.mean_[target_idx]
    aqi_std = scaler.scale_[target_idx]
    
    predictions = np.array(all_preds) * aqi_std + aqi_mean
    actuals = np.array(all_actuals) * aqi_std + aqi_mean
    predictions = np.clip(predictions, 0, 500)
    
    # Metrics
    results = compute_all_metrics(actuals, predictions)
    logger.info(f"ST-GNN Results - MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    logger.info(f"Category Accuracy: {results['category_accuracy']*100:.1f}%")
    
    return {**results, "predictions": predictions, "actuals": actuals, "model": model}


def main():
    """Main training pipeline with clean data."""
    logger.info("=" * 70)
    logger.info("IMPROVED MODEL TRAINING WITH CLEAN DATA")
    logger.info("=" * 70)
    
    # Load and clean data
    df = load_and_clean_pune_data()
    df = clean_data(df)
    df = compute_aqi_column(df)
    
    # Feature columns
    feature_cols = ["pm25", "pm10", "no2", "so2", "o3", "temperature", "humidity", "aqi"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    logger.info(f"Features: {feature_cols}")
    
    # Ensure no NaN in features
    df = df.dropna(subset=feature_cols)
    logger.info(f"Final clean dataset: {len(df)} rows")
    
    # Create splits (70/15/15)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    # Train models
    lstm_results = train_improved_lstm(train_df, val_df, test_df, feature_cols)
    stgnn_results = train_improved_stgnn(train_df, val_df, test_df, feature_cols)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<10} {'MAE':<10} {'RMSE':<10} {'Cat. Acc.':<12}")
    logger.info("-" * 42)
    logger.info(f"{'LSTM':<10} {lstm_results['mae']:<10.2f} {lstm_results['rmse']:<10.2f} {lstm_results['category_accuracy']*100:<10.1f}%")
    logger.info(f"{'ST-GNN':<10} {stgnn_results['mae']:<10.2f} {stgnn_results['rmse']:<10.2f} {stgnn_results['category_accuracy']*100:<10.1f}%")
    logger.info("=" * 70)
    
    # Save results
    results_df = pd.DataFrame([
        {"Model": "LSTM", "MAE": lstm_results["mae"], "RMSE": lstm_results["rmse"], 
         "Category_Accuracy": lstm_results["category_accuracy"]},
        {"Model": "ST-GNN", "MAE": stgnn_results["mae"], "RMSE": stgnn_results["rmse"],
         "Category_Accuracy": stgnn_results["category_accuracy"]},
    ])
    
    Path("outputs").mkdir(exist_ok=True)
    results_df.to_csv("outputs/improved_results.csv", index=False)
    logger.info("Results saved to outputs/improved_results.csv")
    
    return lstm_results, stgnn_results


if __name__ == "__main__":
    main()
