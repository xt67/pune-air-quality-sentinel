# Phase 1: Data Pipeline & IoT Simulation - Context

**Gathered:** 2026-04-02
**Status:** Ready for planning
**Source:** PRD Express Path (PuneAirQuality_PRD.docx)

<domain>
## Phase Boundary

This phase delivers the complete data ingestion, preprocessing, and IoT simulation pipelines. It creates the foundation for all downstream model training by:

1. **Data Ingestion**: Fetching historical AQI and weather data from multiple sources (Kaggle, OpenAQ, Open-Meteo)
2. **Data Preprocessing**: Cleaning, merging, normalizing, and feature engineering pipeline
3. **IoT Simulation**: Synthetic sensor grid generation for 10 Pune neighborhoods
4. **Test Suite**: Unit tests for all pipeline components

**Deliverables:**
- Processed Parquet file at `data/processed/pune_aqi_processed.parquet`
- 10 IoT CSVs at `data/simulated/node_*.csv`
- Test files: `test_fetch.py`, `test_preprocess.py`, `test_iot_sim.py` (all passing)

</domain>

<decisions>
## Implementation Decisions

### Repository Structure
- Create full directory structure as specified in PRD Section 4.2
- All source code in `src/` with subdirectories: data, models, evaluate, viz, utils
- Data directories: raw, processed, simulated, splits, graph
- Empty `__init__.py` in all source subdirectories

### Data Sources (Locked)
- **Primary AQI data**: Kaggle "Air Quality Data in India" dataset (Pune rows)
- **Weather data**: Open-Meteo API (free, no key required)
- **Backup AQI data**: OpenAQ API v3 (free, public data)
- Priority: Start with Kaggle for fast MVP, layer in APIs for final model

### Data Cleaning Rules (Locked)
- Forward-fill gaps up to 3 hours
- Interpolate linearly for gaps 3-12 hours
- Flag and drop gaps >12 hours with log entry
- Clip outliers beyond 99.9th percentile (log replacements)

### AQI Computation (Locked)
- Use CPCB breakpoint table linear interpolation
- AQI = max(sub-index for PM2.5, PM10, NO2, SO2, CO, O3)
- Output range: [0, 500]

### Feature Engineering (Locked)
- Lag features: AQI_lag_1h, AQI_lag_6h, AQI_lag_24h
- Rolling features: PM2.5_24h_mean, PM2.5_72h_mean, PM2.5_24h_std
- Wind vectors: wind_u = speed * cos(dir_rad), wind_v = speed * sin(dir_rad)
- Calendar: hour sine/cosine, day-of-week one-hot, is_holiday, is_diwali_week

### Normalization (Locked)
- MinMaxScaler fitted on training split ONLY
- Transform all continuous features to [0, 1]
- Save scaler to `models/scaler.pkl`

### Train/Val/Test Split (Locked)
- Chronological split (NO shuffling)
- Train: 70%, Val: 15%, Test: 15%
- Save indices to `data/splits/splits.json`

### Pune Node Grid (Locked - 10 Locations)
| Node ID | Neighborhood | Lat | Lon | Characteristic |
|---------|--------------|-----|-----|----------------|
| N01 | Shivajinagar | 18.5308 | 73.8475 | Reference CPCB station |
| N02 | Hinjewadi IT Park | 18.5912 | 73.7389 | Traffic + construction |
| N03 | Pimpri-Chinchwad | 18.6298 | 73.8037 | Heavy industry |
| N04 | Hadapsar | 18.5089 | 73.9260 | Mixed residential |
| N05 | Katraj | 18.4529 | 73.8669 | Dense residential |
| N06 | Deccan Gymkhana | 18.5196 | 73.8399 | High vehicular density |
| N07 | Viman Nagar | 18.5679 | 73.9143 | Airport proximity |
| N08 | Kothrud | 18.5074 | 73.8077 | Residential, cleaner |
| N09 | Talegaon MIDC | 18.7330 | 73.6554 | Outer industrial |
| N10 | Mundhwa | 18.5326 | 73.9368 | Downstream residential |

### IoT Simulation Rules (Locked)
- 2-year hourly time series per node
- Base signal: Kaggle Shivajinagar data as city-wide base
- Node biases: Pimpri +25, Kothrud -10, others defined proportionally
- Gaussian noise: std=5 AQI units
- Rush-hour spikes (7-9 AM, 5-8 PM): +15 AQI
- Diwali week spike: +80 AQI
- Monsoon dip (July-Sept): -20 AQI

### Agent's Discretion
- Specific retry/backoff parameters for API calls (recommended: 3 retries, exponential 1s/2s/4s)
- Exact logging format details (use JSON lines as specified)
- Error message wording
- Test fixture data generation approach

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Documentation
- `.planning/PROJECT.md` — Project context, core value, constraints
- `.planning/REQUIREMENTS.md` — Formal requirements with IDs (DATA-01 to DATA-12, IOT-01 to IOT-06)
- `.planning/ROADMAP.md` — Phase structure and success criteria

### Configuration
- `configs/data.yaml` — Data paths, API endpoints, normalization settings (to be created)

</canonical_refs>

<specifics>
## Specific Ideas

### CPCB Breakpoint Table
The AQI sub-index calculation uses linear interpolation between breakpoints. Each pollutant has a standard breakpoint table from CPCB guidelines.

### Diwali Dates for Simulation
Use actual Diwali dates for the 2-year simulation period. The 5-day window around Diwali should have +80 AQI spike.

### API Error Handling
- OpenAQ rate limit (429): Wait Retry-After header seconds
- Open-Meteo: No rate limits, but cache responses
- Kaggle: Download once, cache locally

</specifics>

<deferred>
## Deferred Ideas

- Real-time CPCB API integration — requires registration (v2)
- Satellite AOD data as input feature (v2)
- Real traffic data integration — needs permissions (v2)

</deferred>

---

*Phase: 01-data-pipeline-iot-simulation*
*Context gathered: 2026-04-02 via PRD Express Path*
