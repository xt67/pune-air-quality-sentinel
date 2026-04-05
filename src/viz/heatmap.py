"""
Folium-based pollution heatmap visualization for Pune.

Creates interactive heatmaps showing predicted AQI values for 10 Pune nodes.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import folium
from folium.plugins import HeatMap
import branca.colormap as cm

logger = logging.getLogger(__name__)

# Pune node coordinates (10 locations as per PRD)
PUNE_NODES = {
    "N01": {"name": "Shivajinagar", "lat": 18.5308, "lon": 73.8475},
    "N02": {"name": "Kothrud", "lat": 18.5074, "lon": 73.8077},
    "N03": {"name": "Hadapsar", "lat": 18.5089, "lon": 73.9260},
    "N04": {"name": "Pimpri-Chinchwad", "lat": 18.6298, "lon": 73.7997},
    "N05": {"name": "Hinjewadi", "lat": 18.5912, "lon": 73.7380},
    "N06": {"name": "Katraj", "lat": 18.4575, "lon": 73.8680},
    "N07": {"name": "Viman Nagar", "lat": 18.5679, "lon": 73.9143},
    "N08": {"name": "Deccan Gymkhana", "lat": 18.5167, "lon": 73.8411},
    "N09": {"name": "Aundh", "lat": 18.5590, "lon": 73.8077},
    "N10": {"name": "Wagholi", "lat": 18.5817, "lon": 73.9771},
}

# AQI color scale per CPCB standards
AQI_COLORS = {
    "Good": {"range": (0, 50), "color": "#00E400", "hex": "#00E400"},
    "Satisfactory": {"range": (51, 100), "color": "#92D050", "hex": "#92D050"},
    "Moderate": {"range": (101, 200), "color": "#FFFF00", "hex": "#FFFF00"},
    "Poor": {"range": (201, 300), "color": "#FF7E00", "hex": "#FF7E00"},
    "Very Poor": {"range": (301, 400), "color": "#FF0000", "hex": "#FF0000"},
    "Severe": {"range": (401, 500), "color": "#7E0023", "hex": "#7E0023"},
}


def get_aqi_category(aqi: float) -> str:
    """Get AQI category from value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_aqi_color(aqi: float) -> str:
    """Get color for AQI value."""
    category = get_aqi_category(aqi)
    return AQI_COLORS[category]["color"]


def get_health_advisory(category: str) -> str:
    """Get health advisory for AQI category."""
    advisories = {
        "Good": "Air quality is satisfactory. Enjoy outdoor activities!",
        "Satisfactory": "Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion.",
        "Moderate": "May cause breathing discomfort to sensitive people. Reduce prolonged outdoor activities.",
        "Poor": "May cause breathing discomfort to people on prolonged exposure. Avoid outdoor activities.",
        "Very Poor": "May cause respiratory illness on prolonged exposure. Avoid outdoor activities, stay indoors.",
        "Severe": "Health alert! May affect healthy people. Avoid all outdoor activities, use air purifiers.",
    }
    return advisories.get(category, "Check air quality before outdoor activities.")


def create_aqi_colormap() -> cm.StepColormap:
    """Create a step colormap for AQI values."""
    colors = [
        AQI_COLORS["Good"]["color"],
        AQI_COLORS["Satisfactory"]["color"],
        AQI_COLORS["Moderate"]["color"],
        AQI_COLORS["Poor"]["color"],
        AQI_COLORS["Very Poor"]["color"],
        AQI_COLORS["Severe"]["color"],
    ]
    
    colormap = cm.StepColormap(
        colors=colors,
        index=[0, 50, 100, 200, 300, 400, 500],
        vmin=0,
        vmax=500,
        caption="Air Quality Index (AQI)"
    )
    
    return colormap


def create_heatmap(
    predictions: Dict[str, float],
    title: str = "Pune Air Quality Heatmap",
    save_path: Optional[str] = None,
    zoom_start: int = 12,
    show_heatmap: bool = True,
) -> folium.Map:
    """
    Create an interactive Folium heatmap for Pune AQI predictions.
    
    Args:
        predictions: Dict mapping node_id (N01-N10) to predicted AQI values
        title: Title for the heatmap
        save_path: Path to save HTML file (optional)
        zoom_start: Initial zoom level
        show_heatmap: Whether to show the heat layer (default True)
        
    Returns:
        folium.Map object
    """
    # Center on Pune
    pune_center = [18.52, 73.85]
    
    # Create base map with CartoDB Positron tiles (clean, free)
    m = folium.Map(
        location=pune_center,
        zoom_start=zoom_start,
        tiles="CartoDB Positron",
    )
    
    # Add title
    title_html = f'''
        <h3 style="position: fixed; 
                   top: 10px; left: 60px; 
                   z-index: 1000;
                   background-color: white;
                   padding: 10px;
                   border-radius: 5px;
                   box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            {title}
        </h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create colormap and add to map
    colormap = create_aqi_colormap()
    colormap.add_to(m)
    
    # Prepare data for heatmap
    heat_data = []
    
    # Add CircleMarkers for each node
    for node_id, node_info in PUNE_NODES.items():
        aqi = predictions.get(node_id, 100)  # Default to 100 if not provided
        category = get_aqi_category(aqi)
        color = get_aqi_color(aqi)
        advisory = get_health_advisory(category)
        
        # Radius proportional to AQI (min 8, max 25)
        radius = min(25, max(8, 8 + (aqi / 500) * 17))
        
        # Create tooltip content
        tooltip_html = f"""
            <div style="font-family: Arial, sans-serif; min-width: 200px;">
                <h4 style="margin: 0 0 5px 0; color: {color};">{node_info['name']}</h4>
                <p style="margin: 2px 0;"><b>Node:</b> {node_id}</p>
                <p style="margin: 2px 0;"><b>Predicted AQI:</b> {aqi:.0f}</p>
                <p style="margin: 2px 0;"><b>Category:</b> 
                    <span style="color: {color}; font-weight: bold;">{category}</span>
                </p>
                <hr style="margin: 5px 0;">
                <p style="margin: 2px 0; font-size: 11px;"><b>Advisory:</b> {advisory}</p>
            </div>
        """
        
        # Add CircleMarker
        folium.CircleMarker(
            location=[node_info["lat"], node_info["lon"]],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            weight=2,
            popup=folium.Popup(tooltip_html, max_width=300),
            tooltip=f"{node_info['name']}: AQI {aqi:.0f} ({category})",
        ).add_to(m)
        
        # Add to heatmap data (lat, lon, intensity)
        heat_data.append([node_info["lat"], node_info["lon"], aqi / 500])
    
    # Add HeatMap layer (conditional)
    if show_heatmap:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_opacity=0.8,
            radius=30,
            blur=20,
            gradient={
                0.0: "#00E400",
                0.2: "#92D050",
                0.4: "#FFFF00",
                0.6: "#FF7E00",
                0.8: "#FF0000",
                1.0: "#7E0023",
            },
        ).add_to(m)
    
    # Save if path provided
    if save_path:
        m.save(save_path)
        logger.info(f"Heatmap saved to {save_path}")
    
    return m


def create_representative_heatmaps(
    output_dir: str = "outputs/heatmaps",
) -> List[str]:
    """
    Create 5 representative heatmaps as required by PRD:
    1. Good day (low AQI)
    2. Moderate day (typical AQI)
    3. Diwali spike (high AQI)
    4. Post-monsoon (moderate-high)
    5. Peak summer (high)
    
    Returns:
        List of saved file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    # Scenario definitions
    scenarios = {
        "good_day": {
            "title": "Pune AQI - Good Air Quality Day",
            "predictions": {
                "N01": 35, "N02": 28, "N03": 42, "N04": 48, "N05": 32,
                "N06": 38, "N07": 45, "N08": 30, "N09": 25, "N10": 40,
            },
        },
        "moderate_day": {
            "title": "Pune AQI - Typical Day",
            "predictions": {
                "N01": 95, "N02": 78, "N03": 110, "N04": 145, "N05": 88,
                "N06": 102, "N07": 92, "N08": 85, "N09": 72, "N10": 118,
            },
        },
        "diwali_spike": {
            "title": "Pune AQI - Diwali Week (High Pollution)",
            "predictions": {
                "N01": 285, "N02": 245, "N03": 320, "N04": 380, "N05": 265,
                "N06": 295, "N07": 310, "N08": 275, "N09": 235, "N10": 345,
            },
        },
        "post_monsoon": {
            "title": "Pune AQI - Post-Monsoon Season",
            "predictions": {
                "N01": 125, "N02": 98, "N03": 155, "N04": 185, "N05": 115,
                "N06": 138, "N07": 142, "N08": 108, "N09": 92, "N10": 168,
            },
        },
        "peak_summer": {
            "title": "Pune AQI - Peak Summer (Hot & Dusty)",
            "predictions": {
                "N01": 165, "N02": 142, "N03": 195, "N04": 225, "N05": 155,
                "N06": 178, "N07": 188, "N08": 152, "N09": 135, "N10": 210,
            },
        },
    }
    
    for scenario_name, config in scenarios.items():
        save_path = str(output_path / f"heatmap_{scenario_name}.html")
        create_heatmap(
            predictions=config["predictions"],
            title=config["title"],
            save_path=save_path,
        )
        saved_files.append(save_path)
        logger.info(f"Created heatmap: {scenario_name}")
    
    return saved_files


def get_heatmap_html(predictions: Dict[str, float], title: str = "Pune AQI") -> str:
    """
    Get heatmap as HTML string for embedding in Streamlit.
    
    Args:
        predictions: Dict mapping node_id to AQI values
        title: Title for the map
        
    Returns:
        HTML string of the map
    """
    m = create_heatmap(predictions, title)
    return m._repr_html_()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create representative heatmaps
    print("Creating representative heatmaps...")
    files = create_representative_heatmaps()
    print(f"Created {len(files)} heatmaps:")
    for f in files:
        print(f"  - {f}")
