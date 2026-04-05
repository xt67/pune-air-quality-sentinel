"""
Visualization modules for Pune Air Quality Sentinel.

Provides:
- Folium heatmaps with AQI color coding
- Matplotlib/Seaborn forecast plots
- Model comparison charts
"""

from src.viz.heatmap import (
    create_heatmap,
    create_representative_heatmaps,
    get_heatmap_html,
    get_aqi_category,
    get_aqi_color,
    get_health_advisory,
    PUNE_NODES,
    AQI_COLORS,
)

from src.viz.plots import (
    plot_forecast_comparison,
    plot_mae_comparison,
    plot_category_accuracy,
    create_representative_plots,
)

__all__ = [
    # Heatmap functions
    "create_heatmap",
    "create_representative_heatmaps",
    "get_heatmap_html",
    "get_aqi_category",
    "get_aqi_color",
    "get_health_advisory",
    "PUNE_NODES",
    "AQI_COLORS",
    # Plot functions
    "plot_forecast_comparison",
    "plot_mae_comparison",
    "plot_category_accuracy",
    "create_representative_plots",
]