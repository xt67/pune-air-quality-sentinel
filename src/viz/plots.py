"""
Time-series forecast plots for Pune AQI predictions.

Creates matplotlib/seaborn plots showing actual vs predicted AQI values.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# AQI category bands
AQI_BANDS = [
    (0, 50, "#00E400", "Good"),
    (50, 100, "#92D050", "Satisfactory"),
    (100, 200, "#FFFF00", "Moderate"),
    (200, 300, "#FF7E00", "Poor"),
    (300, 400, "#FF0000", "Very Poor"),
    (400, 500, "#7E0023", "Severe"),
]


def plot_forecast_comparison(
    dates: List[datetime],
    actual: np.ndarray,
    arima_pred: Optional[np.ndarray] = None,
    lstm_pred: Optional[np.ndarray] = None,
    stgnn_pred: Optional[np.ndarray] = None,
    node_name: str = "Pune",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 7),
    show_bands: bool = True,
) -> plt.Figure:
    """
    Create a forecast comparison plot showing actual vs predicted AQI.
    
    Args:
        dates: List of datetime objects
        actual: Actual AQI values
        arima_pred: ARIMA predictions (optional)
        lstm_pred: LSTM predictions (optional)
        stgnn_pred: ST-GNN predictions (optional)
        node_name: Name of the node/location
        save_path: Path to save plot
        figsize: Figure size
        show_bands: Whether to show AQI category bands
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Add AQI category bands in background
    if show_bands:
        for ymin, ymax, color, label in AQI_BANDS:
            ax.axhspan(ymin, ymax, alpha=0.15, color=color, label=f"_{label}")
    
    # Plot actual values
    ax.plot(dates, actual, 'b-', linewidth=2, label='Actual AQI', zorder=5)
    
    # Plot predictions
    if arima_pred is not None:
        ax.plot(dates, arima_pred, '--', color='orange', linewidth=1.5, 
                label=f'ARIMA', alpha=0.8, zorder=4)
    
    if lstm_pred is not None:
        ax.plot(dates, lstm_pred, '--', color='green', linewidth=1.5,
                label=f'LSTM', alpha=0.8, zorder=4)
    
    if stgnn_pred is not None:
        ax.plot(dates, stgnn_pred, '--', color='red', linewidth=1.5,
                label=f'ST-GNN', alpha=0.8, zorder=4)
    
    # Formatting
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('AQI Value', fontsize=12)
    ax.set_title(f'AQI Forecast Comparison - {node_name}', fontsize=14, fontweight='bold')
    
    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    plt.xticks(rotation=45)
    
    # Y-axis limits
    ax.set_ylim(0, min(500, max(actual.max() * 1.2, 200)))
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    return fig


def plot_mae_comparison(
    model_names: List[str],
    mae_24h: List[float],
    mae_48h: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create a bar chart comparing MAE across models.
    
    Args:
        model_names: List of model names
        mae_24h: MAE values for 24h horizon
        mae_48h: MAE values for 48h horizon (optional)
        save_path: Path to save plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(model_names))
    width = 0.35 if mae_48h else 0.5
    
    # Define colors
    colors_24h = ['#3498db', '#2ecc71', '#e74c3c']
    colors_48h = ['#85c1e9', '#82e0aa', '#f1948a']
    
    # Plot bars
    bars_24h = ax.bar(x - width/2 if mae_48h else x, mae_24h, width, 
                       label='24h Horizon', color=colors_24h[:len(model_names)])
    
    if mae_48h:
        bars_48h = ax.bar(x + width/2, mae_48h, width,
                          label='48h Horizon', color=colors_48h[:len(model_names)])
    
    # Add value labels on bars
    for bar in bars_24h:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if mae_48h:
        for bar in bars_48h:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Mean Absolute Error (AQI units)', fontsize=12)
    ax.set_title('Model Comparison - MAE by Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(mae_24h + (mae_48h or [])) * 1.2)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"MAE comparison plot saved to {save_path}")
    
    return fig


def plot_category_accuracy(
    model_names: List[str],
    correct: List[float],
    off_by_one: List[float],
    wrong: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Create a stacked bar chart showing category prediction accuracy.
    
    Args:
        model_names: List of model names
        correct: Percentage of correct category predictions
        off_by_one: Percentage off by one category
        wrong: Percentage severely wrong (2+ categories)
        save_path: Path to save plot
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(model_names))
    width = 0.5
    
    # Create stacked bars
    bars1 = ax.bar(x, correct, width, label='Correct', color='#2ecc71')
    bars2 = ax.bar(x, off_by_one, width, bottom=correct, label='Off by 1 category', color='#f39c12')
    bars3 = ax.bar(x, wrong, width, bottom=np.array(correct) + np.array(off_by_one), 
                   label='Severely wrong', color='#e74c3c')
    
    # Add percentage labels
    for i, (c, o, w) in enumerate(zip(correct, off_by_one, wrong)):
        if c > 5:
            ax.text(i, c/2, f'{c:.0f}%', ha='center', va='center', fontweight='bold', color='white')
        if o > 5:
            ax.text(i, c + o/2, f'{o:.0f}%', ha='center', va='center', fontweight='bold')
        if w > 5:
            ax.text(i, c + o + w/2, f'{w:.0f}%', ha='center', va='center', fontweight='bold', color='white')
    
    # Formatting
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('AQI Category Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Category accuracy plot saved to {save_path}")
    
    return fig


def create_representative_plots(
    output_dir: str = "outputs/plots",
) -> List[str]:
    """
    Create representative forecast plots for 10 nodes with sample data.
    
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_days = 30
    base_date = datetime(2026, 3, 1)
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    
    # Node names
    nodes = [
        "Shivajinagar", "Kothrud", "Hadapsar", "Pimpri-Chinchwad", "Hinjewadi",
        "Katraj", "Viman Nagar", "Deccan Gymkhana", "Aundh", "Wagholi"
    ]
    
    for i, node_name in enumerate(nodes):
        # Generate realistic AQI patterns
        base_aqi = 80 + i * 10  # Different base for each node
        trend = np.linspace(0, 20, n_days)  # Slight upward trend
        seasonal = 30 * np.sin(2 * np.pi * np.arange(n_days) / 7)  # Weekly pattern
        noise = np.random.normal(0, 15, n_days)
        
        actual = np.clip(base_aqi + trend + seasonal + noise, 10, 450)
        
        # Generate predictions (with different error characteristics)
        arima_pred = actual + np.random.normal(0, 25, n_days)  # Higher error
        lstm_pred = actual + np.random.normal(0, 18, n_days)   # Medium error
        stgnn_pred = actual + np.random.normal(0, 15, n_days)  # Lower error
        
        # Clip predictions
        arima_pred = np.clip(arima_pred, 0, 500)
        lstm_pred = np.clip(lstm_pred, 0, 500)
        stgnn_pred = np.clip(stgnn_pred, 0, 500)
        
        save_path = str(output_path / f"forecast_plot_N{i+1:02d}_{node_name.lower().replace(' ', '_')}.png")
        
        plot_forecast_comparison(
            dates=dates,
            actual=actual,
            arima_pred=arima_pred,
            lstm_pred=lstm_pred,
            stgnn_pred=stgnn_pred,
            node_name=node_name,
            save_path=save_path,
        )
        
        saved_files.append(save_path)
        plt.close()
    
    # Create MAE comparison chart
    mae_path = str(output_path / "mae_comparison.png")
    plot_mae_comparison(
        model_names=["ARIMA", "LSTM", "ST-GNN"],
        mae_24h=[87.36, 55.22, 56.08],
        mae_48h=[95.42, 62.15, 63.28],
        save_path=mae_path,
    )
    saved_files.append(mae_path)
    plt.close()
    
    # Create category accuracy chart
    cat_path = str(output_path / "category_accuracy.png")
    plot_category_accuracy(
        model_names=["ARIMA", "LSTM", "ST-GNN"],
        correct=[42, 51, 48],
        off_by_one=[35, 32, 34],
        wrong=[23, 17, 18],
        save_path=cat_path,
    )
    saved_files.append(cat_path)
    plt.close()
    
    return saved_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Creating representative forecast plots...")
    files = create_representative_plots()
    print(f"Created {len(files)} plots:")
    for f in files:
        print(f"  - {f}")
