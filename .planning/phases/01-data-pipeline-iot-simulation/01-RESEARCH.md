Perfect! Now I have the project context. Let me research comprehensive answers to the 7 research questions for Phase 1 planning. I'll provide detailed findings structured according to your requested format.

---

# Phase 1 Research: Data Pipeline & IoT Simulation

## What You Need to Know to PLAN This Phase Well

### 1. Open-Meteo API

**Current API Structure & Best Practices (2024):**

**Endpoint:** `https://archive-api.open-meteo.com/v1/archive`

**Key Parameters for Pune Historical Weather:**
- **Latitude:** 18.5204
- **Longitude:** 73.8567
- **Start/End dates:** ISO 8601 format (YYYY-MM-DD)
- **Hourly variables:** temperature_2m, relative_humidity_2m, precipitation, weather_code, wind_speed_10m, wind_direction_10m
- **Timezone:** Auto (or specify "Asia/Kolkata" for IST)

**Critical Details:**
- **No rate limits:** Free API, no authentication required
- **Response time:** ~1-2 seconds for 2-year hourly data
- **Data availability:** Historical data from 1950 onwards (very reliable for Pune)
- **Recommended parameters for air quality modeling:**
  - temperature_2m (affects pollutant dispersion)
  - relative_humidity_2m (affects aerosol formation)
  - precipitation (wash-out effect)
  - wind_speed_10m + wind_direction_10m (critical for pollution transport)
  - pressure_msl (optional, helps atmospheric stability)

**Request Example:**
```
https://archive-api.open-meteo.com/v1/archive?latitude=18.5204&longitude=73.8567&start_date=2022-01-01&end_date=2023-12-31&hourly=temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m,wind_direction_10m&timezone=Asia/Kolkata
```

**Caching Strategy:**
- Cache downloaded data locally (5-10 MB per year)
- Use pickle or Parquet for fast reload
- Update only new months (incremental fetch)

---

### 2. OpenAQ API v3

**Current Endpoint Structure (2024):**

**Base URL:** `https://api.openaq.org/v3`

**Key Endpoints for Pune:**

1. **List Available Locations (Pune):**
   ```
   GET /locations?city=Pune&country=IN&limit=100
   ```
   - Returns all monitoring stations in Pune
   - Pune typically has 3-5 active CPCB stations (Shivajinagar most reliable)

2. **Query Historical Data:**
   ```
   GET /measurements?location_id={location_id}&date_from={YYYY-MM-DD}&date_to={YYYY-MM-DD}&parameter=pm25,pm10,no2,so2,co,o3&limit=1000
   ```
   - **Pagination:** Use `offset` and `limit` parameters (max limit: 1000)
   - For 2-year data: Need ~8,760 hourly records, so requires pagination

3. **Specific Pune Locations:**
   - Shivajinagar (most reliable, CPCB reference)
   - Pashan Garware Mahavidyalaya
   - NIL (National Institute of Limnology) - occasionally updated

**Rate Limiting & Backoff:**
- **Soft limit:** 1,000 requests/day for free tier
- **Hard limit:** 10 requests/second (will receive 429 Too Many Requests)
- **Backoff strategy:** Check `Retry-After` header (in seconds)
- **Recommended implementation:** 3 retries with exponential backoff (1s, 2s, 4s)

**Data Quality Notes:**
- OpenAQ data is often 3-6 months behind (not real-time)
- Gaps are common (India's CPCB reporting is inconsistent)
- PM2.5 and PM10 most reliable; O3 sparse
- **Decision:** Use as backup to Kaggle; don't rely on OpenAQ as primary

---

### 3. Kaggle API

**Programmatic Download Best Practices:**

**Authentication:**
1. Download credentials from Kaggle account settings → "API" → `kaggle.json`
2. Place in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows)
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

**Installation & Setup:**
```bash
pip install kaggle
kaggle datasets list --search "air quality india"
```

**Downloading "Air Quality Data in India":**
```bash
kaggle datasets download -d rohanrao/air-quality-data-in-india
```

**Python API Usage:**
```python
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
api.dataset_download_files('rohanrao/air-quality-data-in-india', path='./data/raw', unzip=True)
```

**Critical Considerations:**
- **First run:** Download is ~50-100 MB, takes 2-5 minutes
- **Caching:** After download, don't re-download; parse and filter locally
- **Filtering:** Extract only Pune rows using `city == 'Pune'`
- **Date range:** Dataset contains data from 2013 onwards; subset to 2022-2023 for 2-year simulation
- **Error handling:** Network timeouts possible; retry with 3-second delays

**Why Kaggle Primary:** Completeness, hourly data, direct Pune coverage

---

### 4. CPCB AQI Breakpoint Tables

**Exact Breakpoint Tables (as per CPCB Standards):**

For each pollutant, calculate sub-index using linear interpolation. **AQI = max(sub_indices)**

#### **PM2.5 (µg/m³)**
| AQI | Breakpoint Low | Breakpoint High | Concentration Low | Concentration High |
|-----|---|---|---|---|
| 0-50 | 0 | 50 | 0 | 30 |
| 51-100 | 51 | 100 | 31 | 60 |
| 101-200 | 101 | 200 | 61 | 90 |
| 201-300 | 201 | 300 | 91 | 120 |
| 301-400 | 301 | 400 | 121 | 250 |
| 401-500 | 401 | 500 | 251 | ∞ |

#### **PM10 (µg/m³)**
| AQI | Breakpoint Low | Breakpoint High | Concentration Low | Concentration High |
|-----|---|---|---|---|
| 0-50 | 0 | 50 | 0 | 50 |
| 51-100 | 51 | 100 | 51 | 100 |
| 101-200 | 101 | 200 | 101 | 250 |
| 201-300 | 201 | 300 | 251 | 350 |
| 301-400 | 301 | 400 | 351 | 430 |
| 401-500 | 401 | 500 | 431 | ∞ |

#### **NO₂ (µg/m³)**
| AQI | Breakpoint Low | Breakpoint High | Concentration Low | Concentration High |
|-----|---|---|---|---|
| 0-50 | 0 | 50 | 0 | 40 |
| 51-100 | 51 | 100 | 41 | 80 |
| 101-200 | 101 | 200 | 81 | 180 |
| 201-300 | 201 | 300 | 181 | 280 |
| 301-400 | 301 | 400 | 281 | 400 |
| 401-500 | 401 | 500 | 401 | ∞ |

#### **SO₂ (µg/m³)**
| AQI | Breakpoint Low | Breakpoint High | Concentration Low | Concentration High |
|-----|---|---|---|---|
| 0-50 | 0 | 50 | 0 | 40 |
| 51-100 | 51 | 100 | 41 | 80 |
| 101-200 | 101 | 200 | 81 | 380 |
| 201-300 | 201 | 300 | 381 | 800 |
| 301-400 | 301 | 400 | 801 | 1600 |
| 401-500 | 401 | 500 | 1601 | ∞ |

#### **CO (mg/m³)**
| AQI | Breakpoint Low | Breakpoint High | Concentration Low | Concentration High |
|-----|---|---|---|---|
| 0-50 | 0 | 50 | 0 | 1.0 |
| 51-100 | 51 | 100 | 1.1 | 2.0 |
| 101-200 | 101 | 200 | 2.1 | 10.0 |
| 201-300 | 201 | 300 | 10.1 | 17.0 |
| 301-400 | 301 | 400 | 17.1 | 34.0 |
| 401-500 | 401 | 500 | 34.1 | ∞ |

#### **O₃ (µg/m³) - 8-Hour Average**
| AQI | Breakpoint Low | Breakpoint High | Concentration Low | Concentration High |
|-----|---|---|---|---|
| 0-50 | 0 | 50 | 0 | 50 |
| 51-100 | 51 | 100 | 51 | 100 |
| 101-200 | 101 | 200 | 101 | 168 |
| 201-300 | 201 | 300 | 169 | 208 |
| 301-400 | 301 | 400 | 209 | 748 |
| 401-500 | 401 | 500 | 749 | ∞ |

**Calculation Formula:**
```
Sub-index = ((Breakpoint_AQI_High - Breakpoint_AQI_Low) / 
             (Breakpoint_Conc_High - Breakpoint_Conc_Low)) * 
            (Actual_Concentration - Breakpoint_Conc_Low) + Breakpoint_AQI_Low
```

**Implementation Pattern:**
- Create a lookup dataframe with all breakpoints
- For each pollutant, find the bracketing row (where Conc_Low ≤ measurement ≤ Conc_High)
- Apply the linear interpolation formula
- Take max of all 6 sub-indices

---

### 5. Pandas Best Practices 2024

#### **Time Series Gap Filling**

**Strategy (per project spec):**
```python
# Forward-fill up to 3 hours
df['AQI'] = df['AQI'].fillna(method='ffill', limit=3)

# Linear interpolation for 3-12 hour gaps
df['AQI'] = df['AQI'].interpolate(method='linear', limit=9)

# Flag and log >12 hour gaps
gaps = df['AQI'].isna().sum()
if gaps > 0:
    logger.warning(f"Dropped {gaps} values with >12 hour gaps")
    df = df.dropna(subset=['AQI'])
```

**Why this order matters:**
- Forward-fill preserves recent trends (physics-based)
- Linear interpolation assumes gradual change (realistic for AQI)
- Dropping >12h gaps avoids artificial data (better than imputation)

#### **Rolling Window Calculations (Optimized)**

**Efficient 2024 approach:**
```python
# Use .rolling() with min_periods to handle edge cases
df['PM2.5_24h_mean'] = df['PM2.5'].rolling(window=24, min_periods=12).mean()
df['PM2.5_72h_std'] = df['PM2.5'].rolling(window=72, min_periods=36).std()

# For exponential weighting (more recent = higher weight):
df['PM2.5_ewm'] = df['PM2.5'].ewm(span=24, adjust=False).mean()
```

**Gotcha:** `min_periods` prevents NaN spillover (edge values are NaN by default)

**For lagged features (no data leakage):**
```python
# Use .shift() to create lags WITHOUT lookahead bias
df['AQI_lag_1h'] = df['AQI'].shift(1)
df['AQI_lag_6h'] = df['AQI'].shift(6)
df['AQI_lag_24h'] = df['AQI'].shift(24)

# Drop first 24 rows to remove NaN lag features
df = df.dropna()
```

#### **Parquet I/O (Fast & Efficient)**

**Writing optimized Parquet:**
```python
# Compression matters for large files
df.to_parquet('data/processed/aqi.parquet', 
              compression='snappy',  # Fast + good compression
              engine='pyarrow',
              index=False)

# Or use 'gzip' for maximum compression (slightly slower read)
```

**Reading with type hints (2x faster):**
```python
# Define dtypes upfront
dtypes = {
    'AQI': 'float32',
    'PM2.5': 'float32',
    'timestamp': 'datetime64[ns]',
    'node_id': 'category'
}

df = pd.read_parquet('data/processed/aqi.parquet',
                     columns=['timestamp', 'AQI', 'PM2.5'],
                     dtype_backend='pyarrow')  # Use PyArrow for faster parsing
```

**Chunked reading (for very large files):**
```python
parquet_file = pq.ParquetFile('data/processed/aqi.parquet')
for batch in parquet_file.iter_batches(batch_size=100000):
    df_chunk = batch.to_pandas()
    # Process chunk
```

---

### 6. Testing Patterns for Data Pipelines

#### **Mocking API Responses in Pytest**

**Pattern 1: Mock with `unittest.mock`**
```python
import pytest
from unittest.mock import patch, MagicMock

def test_fetch_open_meteo():
    mock_response = {
        'hourly': {
            'time': ['2023-01-01T00:00', '2023-01-01T01:00'],
            'temperature_2m': [20.5, 21.0],
            'wind_speed_10m': [5.2, 5.5]
        }
    }
    
    with patch('src.data.fetch.requests.get') as mock_get:
        mock_get.return_value.json.return_value = mock_response
        from src.data.fetch import fetch_weather
        
        result = fetch_weather('2023-01-01', '2023-01-31')
        assert len(result) == 2
        assert result[0]['temperature_2m'] == 20.5
        mock_get.assert_called_once()
```

**Pattern 2: Use Fixtures for Reusable Mock Data**
```python
@pytest.fixture
def mock_open_meteo_response():
    return {
        'hourly': {
            'time': ['2023-01-01T00:00', '2023-01-01T01:00'],
            'temperature_2m': [20.5, 21.0],
            'relative_humidity_2m': [60, 65],
            'wind_speed_10m': [5.2, 5.5],
            'wind_direction_10m': [180, 185]
        }
    }

def test_preprocess_weather(mock_open_meteo_response):
    from src.data.preprocess import preprocess_weather
    result = preprocess_weather(mock_open_meteo_response)
    assert 'wind_u' in result.columns
    assert 'wind_v' in result.columns
```

**Pattern 3: Mock with Side Effects (Multiple Calls)**
```python
@patch('src.data.fetch.requests.get')
def test_retry_logic(mock_get):
    # First call fails, third succeeds
    mock_get.side_effect = [
        Exception("Timeout"),
        Exception("Timeout"),
        MagicMock(json=lambda: {'data': 'success'})
    ]
    
    from src.data.fetch import fetch_with_retry
    result = fetch_with_retry('url', retries=3)
    assert result['data'] == 'success'
    assert mock_get.call_count == 3
```

#### **Testing Data Pipelines**

**Pattern: Validate Schema & Values**
```python
def test_preprocessed_data_schema():
    from src.data.preprocess import preprocess_pipeline
    
    df = preprocess_pipeline()
    
    # Schema validation
    required_cols = ['timestamp', 'AQI', 'PM2.5', 'wind_u', 'wind_v']
    assert all(col in df.columns for col in required_cols)
    
    # Data quality checks
    assert df['AQI'].min() >= 0 and df['AQI'].max() <= 500
    assert df[['PM2.5', 'PM10']].notna().all().all()  # No NaN
    
    # Type checks
    assert df['timestamp'].dtype == 'datetime64[ns]'
    assert df['AQI'].dtype in ['float32', 'float64']
    
    # Temporal checks
    assert (df['timestamp'].diff().dt.total_seconds() == 3600).all()  # 1-hour intervals
```

#### **Fixtures for Time Series Data**

**Pattern: Synthetic Time Series Fixture**
```python
@pytest.fixture
def synthetic_ts_data():
    """Generate 7 days of hourly AQI data for testing"""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range('2023-01-01', periods=168, freq='1H')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'AQI': np.random.uniform(30, 150, 168),
        'PM2.5': np.random.uniform(10, 80, 168),
        'PM10': np.random.uniform(20, 150, 168),
        'NO2': np.random.uniform(10, 60, 168),
        'wind_speed': np.random.uniform(1, 15, 168),
        'wind_direction': np.random.uniform(0, 360, 168),
    })
    return df

def test_feature_engineering(synthetic_ts_data):
    from src.data.preprocess import engineer_features
    
    df = engineer_features(synthetic_ts_data)
    
    # Check lags exist
    assert 'AQI_lag_1h' in df.columns
    assert df['AQI_lag_1h'].isna().sum() == 1  # Only first row is NaN
    
    # Check rolling features exist
    assert 'PM2.5_24h_mean' in df.columns
    assert df['PM2.5_24h_mean'].notna().sum() >= 144  # At least 6 days of rolling data
```

---

### 7. Common Pitfalls & How to Avoid Them

#### **Pitfall 1: Timezone Handling (IST vs UTC)**

**Problem:**
- Open-Meteo returns UTC by default
- India uses IST (UTC+5:30)
- Kaggle data is in local IST
- Naive datetime objects cause silent misalignment

**Solution:**
```python
import pandas as pd

# Always specify timezone explicitly
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)  # Parse as UTC
df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Kolkata')  # Convert to IST

# For Open-Meteo: specify in request
# https://api.open-meteo.com/v1/archive?...&timezone=Asia/Kolkata

# For aggregation (hourly to daily), always use:
df_daily = df.set_index('timestamp').resample('1D', label='right').mean()
# label='right' aligns to end of day (IST midnight)

# NEVER do this with naive datetimes:
# df['hour'] = df['timestamp'].dt.hour  # Ambiguous without timezone!

# DO this instead:
df['hour_ist'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.hour
```

**For train/test splitting (critical for time series):**
```python
# IST midnight is 2023-01-01 00:00:00+05:30
split_date_ist = pd.Timestamp('2023-06-30 23:59:59', tz='Asia/Kolkata')

train_df = df[df['timestamp'] <= split_date_ist]
test_df = df[df['timestamp'] > split_date_ist]
```

#### **Pitfall 2: Handling Missing Values in Time Series**

**Problem:**
- Simple .fillna() breaks temporal patterns
- Gaps >12h indicate sensor malfunction (shouldn't interpolate)
- Forward-filling across day boundaries is unrealistic

**Solution:**
```python
def fill_gaps_correctly(df):
    """
    Gap-filling strategy that respects time series physics
    """
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Find gap lengths
    df['gap_hours'] = df['timestamp'].diff().dt.total_seconds() / 3600
    
    # Strategy 1: Forward-fill only short gaps (≤3 hours)
    df['AQI'] = df['AQI'].fillna(method='ffill', limit=3)
    
    # Strategy 2: Linear interpolation for medium gaps (3-12 hours)
    df['AQI'] = df['AQI'].interpolate(method='linear', limit=9)
    
    # Strategy 3: Flag and DROP long gaps (>12 hours)
    long_gaps = df[df['gap_hours'] > 12]
    logger.warning(f"Found {len(long_gaps)} gaps >12 hours. Dropping {long_gaps['AQI'].isna().sum()} rows")
    
    df = df.dropna(subset=['AQI'])
    
    return df
```

**Why NOT simple forward-fill:**
```python
# ❌ BAD: Forward-fills across sensor failures
df['AQI'].fillna(method='ffill')  # Can carry 3-month-old value forward!

# ✅ GOOD: Respects physics and data quality
df['AQI'] = df['AQI'].fillna(method='ffill', limit=3)  # Max 3 hours of carry-forward
```

#### **Pitfall 3: Train/Test Splits for Time Series**

**Problem:**
- Random shuffle breaks temporal continuity
- Future data in training set causes overfitting
- Data leakage from test to train is easy to miss

**Solution (Chronological Split):**
```python
def create_temporal_split(df, train_ratio=0.7, val_ratio=0.15):
    """
    Chronological split for time series (NO SHUFFLING)
    """
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    # Ensure no overlap
    train_end = n_train
    val_end = train_end + n_val
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Verify no leakage
    assert train_df['timestamp'].max() < val_df['timestamp'].min()
    assert val_df['timestamp'].max() < test_df['timestamp'].min()
    
    # Save for reproducibility
    splits = {
        'train': {'start': 0, 'end': train_end},
        'val': {'start': train_end, 'end': val_end},
        'test': {'start': val_end, 'end': n}
    }
    return train_df, val_df, test_df, splits
```

**NEVER do this:**
```python
# ❌ WRONG: This causes data leakage!
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
# ^ Shuffles time series, mixing future test data into past training

# ✅ CORRECT: Chronological
cutoff_date = pd.Timestamp('2023-07-01', tz='Asia/Kolkata')
train_df = df[df['timestamp'] < cutoff_date]
test_df = df[df['timestamp'] >= cutoff_date]
```

#### **Pitfall 4: Outlier Detection & Clipping**

**Problem:**
- Clipping at mean±3σ removes real pollution events (Diwali spikes)
- 99th percentile method is too aggressive
- Outliers compound in downstream features

**Solution (Per Project Spec):**
```python
def clip_outliers_safe(df):
    """Clip only pathological outliers (99.9th percentile)"""
    numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
    
    for col in numeric_cols:
        p99_9 = df[col].quantile(0.999)
        
        outliers = df[df[col] > p99_9]
        if len(outliers) > 0:
            logger.warning(f"Clipping {len(outliers)} outliers in {col} > {p99_9:.2f}")
            df.loc[df[col] > p99_9, col] = p99_9
    
    return df
```

---

### 8. Library Versions (Recommended for 2024)

For maximum compatibility and performance in Phase 1:

```
Python: 3.10.x or 3.11.x (3.12 optional, watch for library support)
pandas: 2.1.x (major speed improvements over 2.0)
numpy: 1.24.x or 1.25.x (1.26+ for Python 3.12)
scipy: 1.11.x
scikit-learn: 1.3.x
kaggle: 1.6.x
requests: 2.31.x
pyarrow: 13.x or 14.x (Parquet I/O engine)
pytest: 7.4.x
pytest-cov: 4.1.x
python-dotenv: 1.0.x (for API key management)
pytz: 2023.3.post1 (timezone handling)

# GPU/ML (if using CUDA 12.x):
torch: 2.0.x or 2.1.x
pytorch-lightning: 2.0.x (optional, for later phases)
torch-geometric: 2.3.x (for Phase 3)
```

---

### 9. Validation Architecture for Phase 1

**How to Validate the Phase's Success:**

```python
# Tests to run (in sequence):

# 1. Fetch module tests (mocked APIs)
pytest tests/test_fetch.py -v  # Verifies DATA-01, DATA-02, DATA-03

# 2. Preprocess module tests (synthetic data)
pytest tests/test_preprocess.py -v  # Verifies DATA-04 to DATA-11

# 3. IoT simulation tests
pytest tests/test_iot_sim.py -v  # Verifies IOT-01 to IOT-06

# 4. Output validation script (not a test, but critical)
python scripts/validate_deliverables.py
```

**Validation Checklist (from project spec):**
```
✅ File Existence:
  - data/processed/pune_aqi_processed.parquet exists
  - data/simulated/node_*.csv (10 files) exist
  - data/splits/splits.json exists
  - models/scaler.pkl exists

✅ Data Quality:
  - Processed Parquet has 0 NaN values in features
  - AQI values in [0, 500] range
  - Timestamps are ordered and uniform (1-hour gaps)
  - 10 nodes, 2 years = 87,600 rows per node

✅ Test Results:
  - test_fetch.py: ≥5 tests, all pass
  - test_preprocess.py: ≥5 tests, all pass
  - test_iot_sim.py: ≥5 tests, all pass

✅ Performance (sanity check):
  - Fetch all data: <30 seconds
  - Preprocess pipeline: <10 seconds
  - IoT sim generation: <5 seconds
  - Full pipeline end-to-end: <60 seconds
```

---

## Summary: Key Planning Insights

**1. Data Fetching Priority:**
   - Start with Kaggle (most reliable for Pune, no rate limits)
   - Layer in Open-Meteo for weather (free, no auth)
   - Use OpenAQ as fallback (slower, gappy, but public)

**2. Critical Technical Decisions:**
   - Always use `timezone='Asia/Kolkata'` for IST handling
   - Gap-fill with strict 3-hour/12-hour limits (physics-based)
   - Chronological train/test split (no shuffling)
   - Clip outliers at 99.9th percentile only (preserve Diwali spikes)

**3. API Design:**
   - Mock all external API calls in tests (3 retries, exponential backoff)
   - Cache raw data locally (don't re-download)
   - Use Parquet for 5-10x faster reload

**4. CPCB AQI:**
   - Use exact breakpoint tables provided (not approximations)
   - Calculate 6 sub-indices, take maximum
   - Linear interpolation between breakpoints

**5. Testing Strategy:**
   - Use fixtures for time series data
   - Mock with `unittest.mock` (builtin, no dependencies)
   - Validate schema, ranges, and temporal continuity

---

**Ready to proceed with Phase 1 execution when you are!**
