"""
Generate model comparison table: ARIMA vs LSTM vs ST-GNN.

Trains all models on Pune data and outputs comparison_table.csv.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocess import compute_aqi
from src.models.arima import ARIMAModel
from src.models.lstm import LSTMForecaster
from src.models.train import Trainer
from src.models.train import Trainer
from src.models.graph import PUNE_STATIONS, build_adjacency_matrix, normalize_adjacency
from src.models.stgnn import STGNN, STGNNTrainer
from src.models.metrics import (
    mae, rmse, mape, r2_score, 
    category_accuracy, adjacent_category_accuracy,
    skill_score, persistence_baseline,
    compute_all_metrics, generate_comparison_table
)

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
    
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data: fill NaN, compute AQI."""
    logger.info("Preprocessing data...")
    
    # Fill missing values with forward fill then backward fill
    pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3"]
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()
    
    # Compute AQI for each row
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
    
    # Drop rows with NaN AQI
    df = df.dropna(subset=["aqi"])
    
    logger.info(f"Preprocessed data shape: {df.shape}")
    logger.info(f"AQI range: {df['aqi'].min():.1f} - {df['aqi'].max():.1f}")
    
    return df


def create_splits(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple:
    """Create train/val/test splits."""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    logger.info(f"Splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def train_arima(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train ARIMA model and evaluate."""
    logger.info("Training ARIMA model...")
    
    # Prepare data
    train_aqi = train_df["aqi"].values
    test_aqi = test_df["aqi"].values
    
    # Fit ARIMA
    model = ARIMAModel(node_id="MH020", seasonal=False, m=24)
    model.fit(train_aqi)
    
    # Predict for test period
    horizon = min(len(test_aqi), 24)  # Predict up to 24 hours
    predictions = model.predict(horizon)
    actuals = test_aqi[:horizon]
    
    # Calculate metrics
    results = compute_all_metrics(actuals, predictions)
    
    # Add persistence baseline skill score
    if len(train_aqi) >= horizon:
        persistence = np.full(horizon, train_aqi[-1])  # Last known value
        results["skill_score"] = skill_score(actuals, predictions, persistence)
    
    logger.info(f"ARIMA Results - MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    
    return {**results, "predictions": predictions, "actuals": actuals, "model": model}


def train_lstm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train LSTM model and evaluate."""
    logger.info("Training LSTM model...")
    
    # Prepare features
    feature_cols = ["pm25", "pm10", "no2", "so2", "o3", "temperature", "humidity", "aqi"]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    # Normalize
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # Create sequences manually
    seq_len = 72
    horizon = 24
    target_idx = feature_cols.index("aqi")
    
    def create_sequences(data, seq_len, horizon, target_idx):
        X, Y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            X.append(data[i:i + seq_len])
            Y.append(data[i + seq_len:i + seq_len + horizon, target_idx])
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    
    X_train, Y_train = create_sequences(train_scaled, seq_len, horizon, target_idx)
    X_val, Y_val = create_sequences(val_scaled, seq_len, horizon, target_idx)
    X_test, Y_test = create_sequences(test_scaled, seq_len, horizon, target_idx)
    
    logger.info(f"LSTM data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(Y_test)
    )
    
    loaders = {
        "train": torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True),
        "val": torch.utils.data.DataLoader(val_dataset, batch_size=32),
        "test": torch.utils.data.DataLoader(test_dataset, batch_size=32),
    }
    
    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMForecaster(
        input_size=len(feature_cols),
        hidden_sizes=(256, 128),
        horizon=horizon,
        dropout=0.25
    ).to(device)
    
    logger.info(f"LSTM parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Custom training loop (avoiding NaN issues from Trainer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.HuberLoss(delta=1.0)
    
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in loaders["train"]:
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
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in loaders["val"]:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                pred = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
        
        val_loss /= len(loaders["val"])
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "models/lstm_best.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            logger.info(f"LSTM early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"LSTM Epoch {epoch + 1}: val_loss={val_loss:.4f}")
    
    # Load best and evaluate
    if Path("models/lstm_best.pt").exists():
        model.load_state_dict(torch.load("models/lstm_best.pt", weights_only=True))
    
    # Evaluate
    model.eval()
    all_preds, all_actuals = [], []
    
    with torch.no_grad():
        for x, y in loaders["test"]:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            all_preds.extend(pred.flatten())
            all_actuals.extend(y.numpy().flatten())
    
    # Denormalize
    aqi_idx = feature_cols.index("aqi")
    aqi_min = scaler.data_min_[aqi_idx]
    aqi_max = scaler.data_max_[aqi_idx]
    
    predictions = np.array(all_preds) * (aqi_max - aqi_min) + aqi_min
    actuals = np.array(all_actuals) * (aqi_max - aqi_min) + aqi_min
    
    # Calculate metrics
    results = compute_all_metrics(actuals, predictions)
    
    logger.info(f"LSTM Results - MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    
    return {**results, "predictions": predictions, "actuals": actuals, "model": model}


def train_stgnn(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """Train ST-GNN model and evaluate."""
    logger.info("Training ST-GNN model...")
    
    # For single-station data, we simulate multi-node by using different features
    # In production, this would use data from multiple stations
    feature_cols = ["pm25", "pm10", "no2", "so2", "o3", "temperature", "humidity", "aqi"]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    
    # Normalize
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[feature_cols])
    val_scaled = scaler.transform(val_df[feature_cols])
    test_scaled = scaler.transform(test_df[feature_cols])
    
    # For ST-GNN: treat each feature as a "node" for demonstration
    # Shape: (samples, seq_len, num_nodes, features_per_node)
    num_nodes = len(feature_cols)  # Each feature is a node
    features_per_node = 1
    seq_len = 72
    horizon = 24
    
    def create_stgnn_sequences(data, seq_len, horizon):
        """Create sequences for ST-GNN: (batch, seq_len, nodes, features)."""
        X, Y = [], []
        for i in range(len(data) - seq_len - horizon + 1):
            # X: (seq_len, num_features) -> (seq_len, num_nodes, 1)
            x = data[i:i + seq_len]
            x = x.reshape(seq_len, num_nodes, features_per_node)
            
            # Y: predict AQI (last column) for each "node" - use the AQI prediction
            # For simplicity, predict all features at horizon
            y = data[i + seq_len:i + seq_len + horizon, -1]  # Just AQI
            
            X.append(x)
            Y.append(y)
        
        return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)
    
    X_train, Y_train = create_stgnn_sequences(train_scaled, seq_len, horizon)
    X_val, Y_val = create_stgnn_sequences(val_scaled, seq_len, horizon)
    X_test, Y_test = create_stgnn_sequences(test_scaled, seq_len, horizon)
    
    logger.info(f"ST-GNN data shapes: X_train={X_train.shape}, Y_train={Y_train.shape}")
    
    # Build adjacency for nodes (features as nodes - simple correlation-based)
    # For real multi-station: use PUNE_STATIONS coordinates
    adj = np.eye(num_nodes, dtype=np.float32)  # Simple identity for single-station demo
    # Add some connections based on feature correlations
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                adj[i, j] = 0.3  # Weak connections between all features
    adj = normalize_adjacency(adj)
    adj_tensor = torch.from_numpy(adj)
    
    # Create dataloaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(Y_train)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(Y_val)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(Y_test)
    )
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # Model - simplified for single station
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = STGNN(
        num_nodes=num_nodes,
        input_dim=features_per_node,
        hidden_dim=64,
        output_dim=horizon,
        gru_layers=2,
        dropout=0.2
    ).to(device)
    
    logger.info(f"ST-GNN parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Custom training loop (using STGNNTrainer concepts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = torch.nn.HuberLoss(delta=1.0)
    
    best_val_loss = float("inf")
    patience_counter = 0
    max_patience = 10
    
    adj_tensor = adj_tensor.to(device)
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # ST-GNN outputs (batch, num_nodes, horizon) - we want just AQI node
            output = model(batch_x, adj_tensor)
            # Take last node (AQI) output
            pred = output[:, -1, :]  # (batch, horizon)
            
            loss = criterion(pred, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                output = model(batch_x, adj_tensor)
                pred = output[:, -1, :]
                val_loss += criterion(pred, batch_y).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path("models").mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), "models/stgnn_best.pt")
        else:
            patience_counter += 1
        
        if patience_counter >= max_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch + 1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}")
    
    # Load best model and evaluate
    model.load_state_dict(torch.load("models/stgnn_best.pt", weights_only=True))
    model.eval()
    
    all_preds, all_actuals = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x, adj_tensor)
            pred = output[:, -1, :].cpu().numpy()
            all_preds.extend(pred.flatten())
            all_actuals.extend(batch_y.numpy().flatten())
    
    # Denormalize
    aqi_idx = feature_cols.index("aqi")
    aqi_min = scaler.data_min_[aqi_idx]
    aqi_max = scaler.data_max_[aqi_idx]
    
    predictions = np.array(all_preds) * (aqi_max - aqi_min) + aqi_min
    actuals = np.array(all_actuals) * (aqi_max - aqi_min) + aqi_min
    
    # Calculate metrics
    results = compute_all_metrics(actuals, predictions)
    
    logger.info(f"ST-GNN Results - MAE: {results['mae']:.2f}, RMSE: {results['rmse']:.2f}")
    
    return {**results, "predictions": predictions, "actuals": actuals, "model": model}


def generate_comparison_csv(results: Dict[str, Dict], output_path: str = "outputs/comparison_table.csv"):
    """Generate CSV comparison table."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    for model_name, metrics in results.items():
        row = {
            "Model": model_name,
            "MAE": f"{metrics['mae']:.2f}",
            "RMSE": f"{metrics['rmse']:.2f}",
            "MAPE (%)": f"{metrics.get('mape', 0):.2f}",
            "R²": f"{metrics.get('r2', 0):.4f}",
            "Category Accuracy": f"{metrics.get('category_accuracy', 0)*100:.1f}%",
            "Adjacent Cat. Acc.": f"{metrics.get('adjacent_category_accuracy', 0)*100:.1f}%",
            "Target MAE": metrics.get("target_mae", "N/A"),
            "Target Met": "✓" if metrics["mae"] <= float(metrics.get("target_mae_val", 999)) else "✗"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Comparison table saved to {output_path}")
    
    # Also generate markdown
    md_path = output_path.replace(".csv", ".md")
    with open(md_path, "w") as f:
        f.write("# Model Comparison: Pune Air Quality Forecasting\n\n")
        f.write("## Overview\n")
        f.write("Comparison of ARIMA, LSTM, and ST-GNN models trained on Pune Karve Road (MH020) data.\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Notes\n")
        f.write("- **Horizon**: 24-hour forecast\n")
        f.write("- **Sequence Length**: 72 hours (3 days)\n")
        f.write("- **Dataset**: MH020.csv (Karve Road, Pune)\n")
        f.write("- **Category Accuracy**: Matches CPCB AQI category (Good/Satisfactory/Moderate/Poor/Very Poor/Severe)\n")
    
    logger.info(f"Markdown report saved to {md_path}")
    
    return df


def main():
    """Main training and comparison pipeline."""
    logger.info("=" * 70)
    logger.info("MODEL COMPARISON: ARIMA vs LSTM vs ST-GNN")
    logger.info("=" * 70)
    
    # Load and preprocess data
    df = load_pune_data()
    df = preprocess_data(df)
    
    # Create splits
    train_df, val_df, test_df = create_splits(df)
    
    # Train all models
    results = {}
    
    # ARIMA
    arima_results = train_arima(train_df, test_df)
    results["ARIMA"] = {**arima_results, "target_mae": "≤30", "target_mae_val": 30}
    
    # LSTM
    lstm_results = train_lstm(train_df, val_df, test_df)
    results["LSTM"] = {**lstm_results, "target_mae": "≤20", "target_mae_val": 20}
    
    # ST-GNN
    stgnn_results = train_stgnn(train_df, val_df, test_df)
    results["ST-GNN"] = {**stgnn_results, "target_mae": "≤15", "target_mae_val": 15}
    
    # Generate comparison table
    comparison_df = generate_comparison_csv(results)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 70)
    print(comparison_df.to_string(index=False))
    
    # Check targets
    logger.info("\n" + "-" * 40)
    for model_name, r in results.items():
        target = r.get("target_mae_val", 999)
        if r["mae"] <= target:
            logger.info(f"✅ {model_name} meets target MAE ≤ {target} (actual: {r['mae']:.2f})")
        else:
            logger.info(f"❌ {model_name} does not meet target MAE ≤ {target} (actual: {r['mae']:.2f})")
    
    return results


if __name__ == "__main__":
    main()
