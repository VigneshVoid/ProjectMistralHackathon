# Pharma Surveillance System

A Streamlit-based epidemiological surveillance dashboard that detects public-health signals from pharmacy sales patterns.

## Run locally (localhost)

### 1) Prerequisites
- Python 3.11+
- `pip`

### 2) Create and activate a virtual environment
```bash
cd pharma-surveillance
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install -r backend/requirements.txt
```

### 4) Configure environment variables
Create `backend/.env` (same folder level referenced by `app/config.py`) with:
```env
MISTRAL_API_KEY=your_key_here
DATABASE_URL=sqlite:///./pharma.db
```

> `MISTRAL_API_KEY` is optional if you only want offline anomaly detection.

### 5) Start the app
```bash
streamlit run streamlit_app.py
```

Open: `http://localhost:8501`

## Project sections

### `streamlit_app.py`
Main UI application.
- Sidebar navigation and global toggle for Mistral AI.
- Five pages: Dashboard, Upload & Analyze, Anomaly Explorer, Disease Map, Alerts & Insights.
- Uses session state to persist current dataset and pipeline results.

### `backend/app/core/detection.py`
Anomaly detection engine.
- Aggregates daily sales into weekly totals.
- Runs three methods: z-score, IQR outlier, and % spike detection.
- Deduplicates anomalies by district+drug+week and keeps the most severe signal.

### `backend/app/core/pipeline.py`
Pipeline orchestrator.
- Runs detection.
- Groups anomalies per district and checks rule-based multi-drug correlations.
- Optionally calls Mistral to generate interpretations and district alerts.
- Returns a normalized payload used by all dashboard pages.

### `backend/app/core/mappings.py`
Domain knowledge and geography.
- Drug-to-condition map (`DRUG_CONDITION_MAP`) with confidence/category.
- Multi-drug correlation rules (`CORRELATION_RULES`) for outbreak patterns.
- District metadata (`DISTRICTS`) with coordinates for mapping.

### `backend/app/core/mistral_agent.py`
LLM integration layer.
- Centralized Mistral client creation and call helper.
- Three AI entry points: anomaly interpretation, alert generation, and cross-signal correlation.

### `backend/app/seed/generate_synthetic.py`
Synthetic data generator for demos/testing.
- Produces ~6 months of pharmacy data across 20 districts.
- Injects 4 realistic scenarios (Delhi respiratory, Chennai waterborne, Pune flu, Vizag thyroid trend).

### `backend/app/config.py`
Settings loading via `pydantic-settings` from `backend/.env`.

### `backend/requirements.txt`
Python dependency list for Streamlit UI, plotting, mapping, and analytics/LLM stack.
