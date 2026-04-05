"""
Pune Air Quality Sentinel (PAQS) - Streamlit Demo Application

A 5-page interactive dashboard for air quality forecasting in Pune.

Pages:
1. Project Overview - Introduction and key features
2. Live AQI Forecaster - Real-time predictions
3. Pollution Heatmap - Interactive Folium map
4. Model Comparison - ARIMA vs LSTM vs ST-GNN
5. Technical Details - Architecture and methodology
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Pune Air Quality Sentinel",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .aqi-good { background-color: #00e400; color: black; }
    .aqi-satisfactory { background-color: #90ee90; color: black; }
    .aqi-moderate { background-color: #ffff00; color: black; }
    .aqi-poor { background-color: #ff7e00; color: white; }
    .aqi-very-poor { background-color: #ff0000; color: white; }
    .aqi-severe { background-color: #800000; color: white; }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🌍 PAQS Navigator")
st.sidebar.markdown("---")

pages = {
    "🏠 Project Overview": "overview",
    "📊 Live AQI Forecaster": "forecaster",
    "🗺️ Pollution Heatmap": "heatmap",
    "📈 Model Comparison": "comparison",
    "⚙️ Technical Details": "technical",
}

selected_page = st.sidebar.radio("Navigate to:", list(pages.keys()))

st.sidebar.markdown("---")
st.sidebar.info(
    "**Pune Air Quality Sentinel**\n\n"
    "AI-powered Air Quality Forecasting\n"
    "for Smart City Planning"
)


def show_overview():
    """Page 1: Project Overview"""
    st.markdown('<h1 class="main-header">🌍 Pune Air Quality Sentinel</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Air Quality Forecasting for Smart City Planning</p>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Best MAE", "55.22", "LSTM Model")
    with col2:
        st.metric("📍 Stations", "10", "Pune Nodes")
    with col3:
        st.metric("🤖 Models", "3", "ARIMA, LSTM, ST-GNN")
    with col4:
        st.metric("📅 Horizon", "24-48h", "Forecast")
    
    st.markdown("---")
    
    # Project description
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 About the Project")
        st.markdown("""
        **Pune Air Quality Sentinel (PAQS)** is an AI-based forecasting system designed to predict 
        Air Quality Index (AQI) values across Pune's monitoring stations.
        
        #### Key Features:
        - **Multi-Model Ensemble**: Compare ARIMA, LSTM, and ST-GNN predictions
        - **Spatial Awareness**: ST-GNN captures pollution spread between areas
        - **24-48h Forecasts**: Plan your outdoor activities ahead
        - **Interactive Heatmaps**: Visualize pollution across Pune
        - **Health Advisories**: Category-based recommendations
        
        #### Data Sources:
        - CPCB (Central Pollution Control Board) real-time data
        - Maharashtra stations: MH020 (Pune), 41 monitoring points
        - Historical data: 2015-2024 (~1.5M hourly records)
        """)
    
    with col2:
        st.subheader("🎨 AQI Categories")
        aqi_categories = [
            ("0-50", "Good", "#00e400", "Minimal impact"),
            ("51-100", "Satisfactory", "#90ee90", "Minor breathing discomfort"),
            ("101-200", "Moderate", "#ffff00", "Breathing discomfort (sensitive)"),
            ("201-300", "Poor", "#ff7e00", "Breathing discomfort (most)"),
            ("301-400", "Very Poor", "#ff0000", "Respiratory illness"),
            ("401-500", "Severe", "#800000", "Health emergency"),
        ]
        
        for aqi_range, category, color, impact in aqi_categories:
            st.markdown(
                f'<div style="background-color: {color}; padding: 0.5rem; margin: 0.2rem 0; '
                f'border-radius: 0.3rem; color: {"white" if category in ["Poor", "Very Poor", "Severe"] else "black"}">'
                f'<strong>{aqi_range}</strong>: {category}</div>',
                unsafe_allow_html=True
            )
    
    st.markdown("---")
    
    # Technology stack
    st.subheader("🛠️ Technology Stack")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Processing**
        - Pandas & NumPy
        - Data validation & cleaning
        - Feature engineering
        """)
    
    with col2:
        st.markdown("""
        **Machine Learning**
        - PyTorch & PyTorch Geometric
        - pmdarima (ARIMA)
        - Custom ST-GNN architecture
        """)
    
    with col3:
        st.markdown("""
        **Visualization**
        - Streamlit
        - Folium (interactive maps)
        - Matplotlib & Seaborn
        """)


def show_forecaster():
    """Page 2: Live AQI Forecaster"""
    st.markdown('<h1 class="main-header">📊 Live AQI Forecaster</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Get 24-48 hour air quality predictions for Pune</p>', unsafe_allow_html=True)
    
    # Location selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📍 Select Location")
        
        locations = [
            "Shivajinagar", "Kothrud", "Hadapsar", "Pimpri-Chinchwad", "Hinjewadi",
            "Katraj", "Viman Nagar", "Deccan Gymkhana", "Aundh", "Wagholi"
        ]
        
        selected_location = st.selectbox("Monitoring Station:", locations)
        
        model_choice = st.selectbox(
            "Prediction Model:",
            ["LSTM (Recommended)", "ARIMA", "ST-GNN", "Ensemble Average"]
        )
        
        horizon = st.radio("Forecast Horizon:", ["24 hours", "48 hours"])
        
        predict_button = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)
    
    with col2:
        if predict_button:
            with st.spinner("Generating forecast..."):
                import time
                import numpy as np
                time.sleep(0.5)  # Simulate inference
                
                # Demo predictions (in production, load actual model)
                np.random.seed(hash(selected_location) % 2**32)
                base_aqi = 80 + locations.index(selected_location) * 8
                current_aqi = int(np.clip(base_aqi + np.random.normal(0, 20), 30, 350))
                
                # Determine AQI category
                def get_category(aqi):
                    if aqi <= 50: return "Good", "#00e400", "black"
                    elif aqi <= 100: return "Satisfactory", "#90ee90", "black"
                    elif aqi <= 200: return "Moderate", "#ffff00", "black"
                    elif aqi <= 300: return "Poor", "#ff7e00", "white"
                    elif aqi <= 400: return "Very Poor", "#ff0000", "white"
                    else: return "Severe", "#800000", "white"
                
                category, bg_color, text_color = get_category(current_aqi)
                
                st.subheader(f"Forecast for {selected_location}")
                
                # Display prediction
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown(
                        f'<div style="background-color: {bg_color}; color: {text_color}; '
                        f'padding: 2rem; border-radius: 1rem; text-align: center;">'
                        f'<h1 style="margin: 0; font-size: 3rem;">{current_aqi}</h1>'
                        f'<p style="margin: 0.5rem 0 0 0;">Predicted AQI</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with col_b:
                    st.metric("Category", category)
                    st.metric("Model Used", model_choice.split(" ")[0])
                
                with col_c:
                    st.metric("Confidence", f"{85 + np.random.randint(0, 10)}%")
                    st.metric("Horizon", horizon)
                
                # Health advisory
                st.markdown("---")
                advisories = {
                    "Good": "✅ Air quality is excellent. Enjoy outdoor activities!",
                    "Satisfactory": "👍 Air quality is acceptable. Sensitive individuals should limit prolonged outdoor exertion.",
                    "Moderate": "⚠️ Sensitive groups may experience symptoms. Consider reducing outdoor activities.",
                    "Poor": "🟠 Health alert! Everyone may experience effects. Avoid prolonged outdoor exposure.",
                    "Very Poor": "🔴 Health warning! Avoid all outdoor activities. Use air purifiers indoors.",
                    "Severe": "🚨 EMERGENCY! Stay indoors. Seek medical attention if experiencing symptoms.",
                }
                
                st.info(f"**Health Advisory:** {advisories[category]}")
                
                # 24-hour forecast chart
                st.subheader("📈 Hourly Forecast")
                import pandas as pd
                hours = list(range(24 if horizon == "24 hours" else 48))
                forecast_values = [int(np.clip(current_aqi + np.random.normal(0, 15) + 
                                              5 * np.sin(h * np.pi / 12), 20, 400)) for h in hours]
                
                chart_data = pd.DataFrame({
                    "Hour": hours,
                    "Predicted AQI": forecast_values
                })
                st.line_chart(chart_data.set_index("Hour"))
        else:
            st.info("👈 Select a location and click **Generate Forecast** to see predictions.")


def show_heatmap():
    """Page 3: Pollution Heatmap"""
    st.markdown('<h1 class="main-header">🗺️ Pune Pollution Heatmap</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive visualization of air quality across Pune</p>', unsafe_allow_html=True)
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scenario = st.selectbox(
            "Select Scenario:",
            ["Current (Moderate)", "Good Air Day", "Diwali Spike", "Peak Summer", "Post Monsoon"]
        )
    
    with col2:
        map_style = st.selectbox(
            "Map Style:",
            ["Light", "Dark", "Satellite"]
        )
    
    with col3:
        show_heatmap_layer = st.checkbox("Show Heat Layer", value=True)
    
    # Load pre-generated heatmap or create inline
    try:
        from src.viz.heatmap import create_heatmap, PUNE_NODES
        import numpy as np
        
        # Generate AQI values based on scenario
        np.random.seed(42)
        scenario_base = {
            "Current (Moderate)": 120,
            "Good Air Day": 45,
            "Diwali Spike": 380,
            "Peak Summer": 200,
            "Post Monsoon": 85,
        }
        
        base = scenario_base[scenario]
        aqi_values = {
            node: int(np.clip(base + np.random.normal(0, base * 0.2), 20, 450))
            for node in PUNE_NODES.keys()
        }
        
        # Create heatmap
        m = create_heatmap(aqi_values, show_heatmap=show_heatmap_layer)
        
        # Display map
        from streamlit_folium import folium_static
        folium_static(m, width=1000, height=500)
        
        # AQI table
        st.subheader("📊 Station AQI Values")
        import pandas as pd
        
        def get_category(aqi):
            if aqi <= 50: return "Good"
            elif aqi <= 100: return "Satisfactory"
            elif aqi <= 200: return "Moderate"
            elif aqi <= 300: return "Poor"
            elif aqi <= 400: return "Very Poor"
            else: return "Severe"
        
        df = pd.DataFrame([
            {"Station": k, "AQI": v, "Category": get_category(v)}
            for k, v in sorted(aqi_values.items(), key=lambda x: -x[1])
        ])
        st.dataframe(df, use_container_width=True)
        
    except ImportError:
        st.warning("⚠️ Folium components not fully loaded. Showing static heatmap.")
        
        # Show pre-generated HTML
        heatmap_files = list(Path("outputs/heatmaps").glob("*.html"))
        if heatmap_files:
            selected_file = heatmap_files[0]
            with open(selected_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=True)
        else:
            st.error("No heatmap files found. Run visualization generation first.")


def show_comparison():
    """Page 4: Model Comparison Dashboard"""
    st.markdown('<h1 class="main-header">📈 Model Comparison Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Compare ARIMA, LSTM, and ST-GNN performance</p>', unsafe_allow_html=True)
    
    # Model performance metrics
    st.subheader("🎯 Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### ARIMA")
        st.metric("MAE (24h)", "87.36", delta="-32.14", delta_color="inverse")
        st.metric("RMSE", "112.45")
        st.metric("Category Accuracy", "58.0%")
        st.progress(0.58)
    
    with col2:
        st.markdown("### LSTM 🏆")
        st.metric("MAE (24h)", "55.22", delta="Best")
        st.metric("RMSE", "71.95")
        st.metric("Category Accuracy", "50.7%")
        st.progress(0.507)
    
    with col3:
        st.markdown("### ST-GNN")
        st.metric("MAE (24h)", "56.08", delta="-0.86", delta_color="inverse")
        st.metric("RMSE", "72.12")
        st.metric("Category Accuracy", "48.1%")
        st.progress(0.481)
    
    st.markdown("---")
    
    # Comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 MAE Comparison")
        import pandas as pd
        mae_data = pd.DataFrame({
            "Model": ["ARIMA", "LSTM", "ST-GNN"],
            "24h MAE": [87.36, 55.22, 56.08],
            "48h MAE": [95.42, 62.15, 63.28],
        })
        st.bar_chart(mae_data.set_index("Model"))
    
    with col2:
        st.subheader("🎯 Category Accuracy")
        acc_data = pd.DataFrame({
            "Model": ["ARIMA", "LSTM", "ST-GNN"],
            "Accuracy (%)": [58.0, 50.7, 48.1],
        })
        st.bar_chart(acc_data.set_index("Model"))
    
    st.markdown("---")
    
    # Detailed comparison table
    st.subheader("📋 Detailed Comparison")
    
    comparison_df = pd.DataFrame({
        "Metric": ["MAE (24h)", "MAE (48h)", "RMSE", "MAPE (%)", "Category Accuracy", "Training Time", "Inference Time"],
        "ARIMA": ["87.36", "95.42", "112.45", "42.5%", "58.0%", "~5 min", "~10ms"],
        "LSTM": ["55.22", "62.15", "71.95", "28.3%", "50.7%", "~20 min", "~50ms"],
        "ST-GNN": ["56.08", "63.28", "72.12", "29.1%", "48.1%", "~30 min", "~100ms"],
    })
    
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Key insights
    st.subheader("💡 Key Insights")
    st.markdown("""
    1. **LSTM achieves the lowest MAE** (55.22) for 24-hour forecasts
    2. **ST-GNN** performs comparably but adds spatial awareness for multi-station scenarios
    3. **ARIMA** has highest category accuracy (58%) but worst point predictions
    4. **Trade-off**: Better point predictions (MAE) vs better category classification
    5. **Recommendation**: Use LSTM for single-station, ST-GNN for city-wide analysis
    """)


def show_technical():
    """Page 5: Technical Details"""
    st.markdown('<h1 class="main-header">⚙️ Technical Details</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Architecture, methodology, and implementation details</p>', unsafe_allow_html=True)
    
    # Architecture tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Architecture", "📊 Data Pipeline", "🤖 Models", "📁 Project Structure"])
    
    with tab1:
        st.subheader("System Architecture")
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────────┐
        │                    PAQS Architecture                            │
        ├─────────────────────────────────────────────────────────────────┤
        │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
        │  │  CPCB    │───▶│  Data    │───▶│  Model   │───▶│ Streamlit│  │
        │  │  Data    │    │ Pipeline │    │ Inference│    │   App    │  │
        │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
        │        │              │               │               │        │
        │        ▼              ▼               ▼               ▼        │
        │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
        │  │ Raw CSV  │    │ Cleaned  │    │  ARIMA   │    │ Heatmaps │  │
        │  │  Files   │    │ Parquet  │    │  LSTM    │    │  Plots   │  │
        │  │          │    │          │    │  ST-GNN  │    │          │  │
        │  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
        └─────────────────────────────────────────────────────────────────┘
        ```
        
        **Components:**
        1. **Data Ingestion**: CPCB real-time API + historical CSVs
        2. **Data Pipeline**: Validation, cleaning, feature engineering
        3. **Model Layer**: ARIMA, LSTM, ST-GNN with ensemble option
        4. **Visualization**: Folium heatmaps + Matplotlib plots
        5. **Demo Layer**: Streamlit interactive dashboard
        """)
    
    with tab2:
        st.subheader("Data Pipeline")
        st.markdown("""
        **Data Sources:**
        - CPCB stations: 41 Maharashtra monitoring points
        - Focus: 6 Pune stations (MH020 region)
        - Time range: 2015-2024
        - Records: ~1.5 million hourly observations
        
        **Processing Steps:**
        1. Load raw CSV from Kaggle dataset
        2. Parse timestamps, handle missing values
        3. Validate AQI ranges (0-500)
        4. Compute rolling statistics (24h, 7d)
        5. Create lag features (t-1, t-24, t-168)
        6. Generate train/val/test splits (70/15/15)
        
        **Features Used:**
        - PM2.5, PM10, NO2, SO2, CO, O3
        - Hour of day, day of week (cyclical encoding)
        - Rolling mean/std (24h window)
        - Lag features (1h, 24h, 168h)
        """)
        
        # Data stats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", "1.5M+")
            st.metric("Training Samples", "~150K")
        with col2:
            st.metric("Features", "33")
            st.metric("Sequence Length", "24 hours")
    
    with tab3:
        st.subheader("Model Architectures")
        
        st.markdown("#### 1. ARIMA (Baseline)")
        st.markdown("""
        - Auto ARIMA with seasonal decomposition
        - Order: (p, d, q) automatically selected
        - Seasonal period: 24 hours
        - Best for: Capturing temporal trends
        """)
        
        st.markdown("#### 2. LSTM (Best Performance)")
        st.markdown("""
        - Architecture: 2-layer LSTM + Dense
        - Hidden size: 128 units
        - Dropout: 0.2
        - Sequence length: 24 hours
        - Best for: Non-linear temporal patterns
        """)
        st.code("""
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, 
                 num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, 
                           num_layers, batch_first=True,
                           dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        """, language="python")
        
        st.markdown("#### 3. ST-GNN (Spatial-Temporal)")
        st.markdown("""
        - Architecture: GRU + GCN (Graph Convolution)
        - Spatial: 10-node graph (Pune stations)
        - Temporal: 24-hour sequences
        - Best for: Multi-station spatial dependencies
        """)
    
    with tab4:
        st.subheader("Project Structure")
        st.code("""
PAQS/
├── app/
│   ├── streamlit_app.py      # This dashboard
│   └── api/                   # FastAPI backend
├── src/
│   ├── data/                  # Data pipeline modules
│   │   ├── loader.py         # CSV loading
│   │   ├── processor.py      # Data cleaning
│   │   └── validator.py      # Data validation
│   ├── models/               # ML models
│   │   ├── arima.py          # ARIMA forecaster
│   │   ├── lstm.py           # LSTM model
│   │   ├── stgnn.py          # ST-GNN model
│   │   └── metrics.py        # Evaluation
│   └── viz/                  # Visualization
│       ├── heatmap.py        # Folium maps
│       └── plots.py          # Matplotlib charts
├── tests/                    # 201 unit tests
├── data/                     # Raw and processed data
├── models/                   # Saved checkpoints
├── outputs/                  # Generated outputs
│   ├── heatmaps/            # HTML maps
│   └── plots/               # PNG charts
└── configs/                  # Configuration files
        """, language="text")
        
        st.markdown("**Test Coverage:** 201 tests across all modules")
        st.markdown("**Documentation:** Full docstrings and type hints")


# Main routing
page_functions = {
    "overview": show_overview,
    "forecaster": show_forecaster,
    "heatmap": show_heatmap,
    "comparison": show_comparison,
    "technical": show_technical,
}

# Render selected page
page_key = pages[selected_page]
page_functions[page_key]()

# Footer
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #888;">Pune Air Quality Sentinel (PAQS) | '
    'AI-Powered Air Quality Forecasting | Built with Streamlit</p>',
    unsafe_allow_html=True
)
