# Pharma Surveillance System

**AI-Powered Epidemiological Intelligence from Pharmacy Sales Data**

Built for the **Mistral AI Hackathon 2026**, this system detects public health signals by analyzing pharmacy drug sales patterns across 20 Indian districts. It combines statistical anomaly detection with Mistral AI to deliver real-time alerts, structured interpretations, and actionable insights for district health officers.

---

## What It Does

Pharmacies are the first point of contact in India's healthcare system. A sudden spike in ORS sachets in Chennai could signal a waterborne outbreak; a cluster of respiratory drug sales in Delhi could indicate an air pollution event.

This system:
- **Ingests** pharmacy sales data (CSV upload or synthetic generation)
- **Detects** anomalies using 4 statistical methods (Z-Score, IQR, Percentage Spike, EWMA)
- **Interprets** findings with Mistral AI structured output and streaming analysis
- **Alerts** health officers with severity-ranked, actionable alerts in multiple languages
- **Enables** natural language querying, what-if simulations, and differential diagnosis

---

## Mistral AI Features Showcased

This project showcases **6 distinct Mistral SDK capabilities** through 15 AI-powered functions:

| # | Mistral Feature | How It's Used | Functions |
|---|----------------|---------------|-----------|
| 1 | **`chat.complete()`** | Batch anomaly interpretation and alert generation during pipeline execution | `interpret_anomaly_text()`, `generate_alert_text()`, `correlate_signals()`, `localize_alert()` |
| 2 | **Structured Output** (`response_format: json_object`) | Returns validated Pydantic models for alerts, briefings, diagnosis, interventions, and clustering | `interpret_anomaly()`, `generate_alert()`, `generate_district_briefing()`, `differential_diagnosis()`, `plan_interventions()`, `cluster_district_patterns()`, `explain_simulation()` |
| 3 | **Streaming** (`chat.stream()`) | Token-by-token live rendering for on-demand analysis in the Alerts page | `interpret_anomaly_stream()`, `correlate_signals_stream()` |
| 4 | **Function Calling** (`tools` + `tool_choice`) | Natural language query assistant that calls data filtering tools automatically | `query_assistant()` with tools: `filter_anomalies`, `get_district_risk`, `get_correlations`, `get_summary` |
| 5 | **Multilingual Generation** | Translates alerts into Hindi, Tamil, Telugu, Kannada, Bengali, Gujarati, Malayalam | `localize_alert()` |
| 6 | **Model Upgrade** (`mistral-medium-latest`) | All calls use the upgraded model for better epidemiological reasoning | Global `DEFAULT_MODEL` setting |

### Pydantic Response Models

All structured outputs are validated against strict schemas:

- `AnomalyInterpretation` - likely condition, confidence score, causes, actions
- `AlertResponse` - severity, affected area, evidence, urgency score (1-10)
- `DistrictBriefing` - risk level, active signals, monitoring metrics, escalation criteria
- `DifferentialDiagnosis` - ranked causes with probability, supporting/missing evidence
- `InterventionPlan` - immediate (24h) + short-term (7d) actions, resources needed
- `ClusterAnalysis` - regional event detection with confidence and evidence
- `SimulationExplanation` - impact summary, key changes, new risk areas

### Rate Limit Resilience

Built-in retry logic with exponential backoff (2s, 4s, 8s, 16s) handles free-tier rate limits gracefully. The pipeline runs sequentially with 2-second gaps between API calls and shows real-time progress.

---

## Dashboard Pages (9 Total)

| Page | Purpose |
|------|---------|
| **Dashboard** | KPI summary, district risk ranking, morning briefing generation, anomaly timeline |
| **Upload & Analyze** | CSV upload or synthetic data generation with progress-tracked pipeline |
| **Anomaly Explorer** | Multi-filter table, "Why flagged?" detail, differential diagnosis, intervention planner, CSV export |
| **Year-over-Year** | 2023 vs 2024 comparison by district and drug, seasonal spike detection |
| **Disease Map** | Folium geographic map with severity-colored markers and regional clustering analysis |
| **Alerts & Insights** | Structured alerts with lifecycle tracking (New/Investigating/Confirmed/Closed), multilingual translation, streaming on-demand analysis |
| **AI Assistant** | Natural language chat powered by Mistral function calling |
| **Scenario Simulator** | What-if analysis with configurable demand spikes and AI-powered impact explanation |
| **Evaluation** | Precision/Recall/F1 metrics against 4 known ground truth scenarios |

---

## Anomaly Detection Engine

Four complementary statistical methods run on weekly-aggregated pharmacy sales:

| Method | Algorithm | What It Catches |
|--------|-----------|-----------------|
| **Z-Score** | Rolling 8-week mean, threshold = 2.0 SD | Sudden departures from recent trend |
| **IQR Outlier** | Q1/Q3 with 1.5x IQR fence | Values outside historical distribution |
| **Percentage Spike** | Week-over-week change > 200% or < -66% | Rapid demand surges or drops |
| **EWMA** | Exponentially Weighted Moving Average (alpha=0.3) | Trend-responsive detection, smoothed |

All methods detect both **spikes** (demand surges) and **drops** (supply chain issues). Results are deduplicated by district + drug + week, keeping the highest severity signal.

**Severity Classification:**
- **Critical**: z-score >= 4.0 or % change >= 400%
- **High**: z-score >= 3.0 or % change >= 300%
- **Medium**: z-score >= 2.0 or % change >= 200%
- **Low**: below thresholds but still anomalous

---

## Synthetic Data & Injected Scenarios

The built-in data generator produces 18 months (Jan 2023 - Jun 2024) of pharmacy sales across **20 Indian districts**, **13 drugs**, and **100 pharmacies**. 2023 serves as a clean seasonal baseline; 2024 contains 4 injected anomaly scenarios:

| Scenario | Location | Weeks | Drugs Affected | Cause |
|----------|----------|-------|----------------|-------|
| **Delhi Respiratory Spike** | South/North/East Delhi | Weeks 18-20 | salbutamol (4x), budesonide (3.5x), montelukast (3x) | Air pollution event |
| **Chennai Waterborne Outbreak** | Chennai, Kanchipuram | Week 12 | ORS sachets (6x), metronidazole (5x), loperamide (4x) | Water contamination |
| **Pune Flu Cluster** | Pune, Pimpri-Chinchwad | Weeks 8-10 | oseltamivir (3.5x), paracetamol (2.5x), cetirizine (2x) | Influenza wave |
| **Vizag Thyroid Anomaly** | Visakhapatnam | Months 3-6 | levothyroxine (gradual 50% increase) | Industrial pollution |

---

## Getting Started

### Prerequisites

- Python 3.11+
- A Mistral API key from [console.mistral.ai](https://console.mistral.ai) (optional - the app works without it for offline anomaly detection)

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd pharma-surveillance
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create `backend/.env`:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
DATABASE_URL=sqlite:///./pharma.db
```

> **Note:** `MISTRAL_API_KEY` is optional. Without it, the app runs all anomaly detection and visualization features offline. Enable Mistral AI via the sidebar toggle to unlock AI interpretations, alerts, the AI Assistant, and multilingual translation.

### 5. Start the app

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 6. Quick demo

1. Click **"Generate Synthetic Data & Analyze"** on the Dashboard
2. Watch the progress bar: Validating -> Detecting -> Interpreting 1/5 -> Generating alert 1/10 -> Complete
3. Explore the Dashboard KPIs and navigate through all 9 pages
4. Try the **AI Assistant**: ask "Show critical anomalies in Delhi"
5. Try **Scenario Simulator**: inject a 5x respiratory spike in Bengaluru
6. Check **Evaluation**: see Precision/Recall/F1 against the 4 known ground truth events

---

## Project Structure

```
pharma-surveillance/
├── streamlit_app.py                  # Main Streamlit app (9 pages, ~1300 lines)
├── requirements.txt                  # Forwards to backend/requirements.txt
├── README.md
├── backend/
│   ├── .env                          # API keys and config
│   ├── requirements.txt              # All Python dependencies
│   └── app/
│       ├── config.py                 # Pydantic Settings (env loading)
│       ├── core/
│       │   ├── mistral_agent.py      # Mistral AI wrapper (15 functions, 6 SDK features)
│       │   ├── detection.py          # 4 anomaly detection algorithms
│       │   ├── pipeline.py           # Orchestrator: detect -> correlate -> interpret -> alert
│       │   ├── mappings.py           # Drug-condition maps, correlation rules, district metadata
│       │   ├── validation.py         # CSV data validation and cleaning
│       │   └── evaluation.py         # Ground truth evaluation (precision/recall/F1)
│       └── seed/
│           └── generate_synthetic.py # 18-month synthetic data generator
```

---

## CSV Data Format

To upload your own data, prepare a CSV with these columns:

| Column | Type | Example |
|--------|------|---------|
| `date` | ISO date | `2024-03-15` |
| `district` | string | `South Delhi` |
| `state` | string | `Delhi` |
| `pharmacy_id` | string | `SOUTH-PH001` |
| `drug_generic_name` | string | `salbutamol` |
| `drug_category` | string | `respiratory` |
| `quantity_sold` | integer | `42` |
| `unit_price` | float | `8.50` |
| `latitude` | float | `28.53` |
| `longitude` | float | `77.22` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **AI / LLM** | Mistral AI SDK (`mistralai >= 1.0`) with `mistral-medium-latest` |
| **Dashboard** | Streamlit 1.40+ |
| **Charts** | Plotly 5.24+ |
| **Maps** | Folium 0.18+ with streamlit-folium |
| **Data** | pandas 2.2+, numpy 2.0+, scipy 1.14+ |
| **Validation** | Pydantic 2.0+ |
| **Backend** | FastAPI 0.115+ (API-ready), SQLAlchemy 2.0+ |
| **Language** | Python 3.11+ |

---

## Deploy on Streamlit Community Cloud

1. Point **Main file path** to `streamlit_app.py`
2. Point **Requirements** to `requirements.txt`
3. Add `MISTRAL_API_KEY` to Streamlit Secrets

---

## License

Built for the Mistral AI Hackathon 2026.
