# Pharma Surveillance System — Mistral Hackathon

## What This Is
Real-time epidemiological surveillance system that uses pharmacy drug sales data as a proxy to detect disease outbreaks, pollution health effects, and health patterns — WITHOUT depending on doctor/hospital reporting.

**Core insight**: Every pharmacy transaction (already digitized via GST in India) encodes implicit health information. A spike in salbutamol sales in a district means respiratory disease is rising there — no doctor report needed. This system reads that signal.

## Tech Stack
- **Backend**: FastAPI (Python 3.11+)
- **Frontend**: React + Vite + TypeScript
- **AI Engine**: Mistral API via `mistralai` Python SDK
- **Database**: SQLite (dev) / PostgreSQL (prod) via SQLAlchemy
- **Maps**: Leaflet (react-leaflet) for geographic disease cluster visualization
- **Charts**: Recharts for time-series and anomaly plots
- **Anomaly Detection**: scipy stats (z-score), pandas rolling averages
- **Deployment**: Railway (backend) + Vercel (frontend)

## Environment Variables
```
MISTRAL_API_KEY=<key>
DATABASE_URL=sqlite:///./pharma.db
CORS_ORIGINS=http://localhost:5173
```

## Project Structure
```
pharma-surveillance/
├── CLAUDE.md
├── backend/
│   ├── pyproject.toml            # Python dependencies
│   ├── requirements.txt
│   ├── .env
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app entry, CORS, router mounts
│   │   ├── config.py             # Settings via pydantic-settings
│   │   ├── database.py           # SQLAlchemy engine, session, Base
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── drug_sale.py      # DrugSale ORM model
│   │   │   ├── alert.py          # Alert ORM model
│   │   │   └── anomaly.py        # Anomaly ORM model
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── drug_sale.py      # Pydantic request/response schemas
│   │   │   ├── alert.py
│   │   │   ├── anomaly.py
│   │   │   └── analysis.py       # Analysis request/response schemas
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── router.py         # Main API router aggregating all routes
│   │   │   ├── upload.py         # POST /api/upload — CSV upload & parse
│   │   │   ├── analysis.py       # POST /api/analyze — trigger anomaly detection
│   │   │   ├── alerts.py         # GET /api/alerts — fetch generated alerts
│   │   │   ├── map_data.py       # GET /api/map — geographic cluster data
│   │   │   └── insights.py       # POST /api/insights — Mistral interpretation
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── detection.py      # Anomaly detection engine (z-score, rolling avg, IQR)
│   │   │   ├── mistral_agent.py  # Mistral API wrapper — all LLM calls
│   │   │   ├── mappings.py       # Drug-to-condition mapping tables
│   │   │   └── pipeline.py       # Orchestrator: upload → detect → interpret → alert
│   │   └── seed/
│   │       ├── __init__.py
│   │       └── generate_synthetic.py  # Synthetic dataset generator
│   ├── tests/
│   │   ├── test_detection.py
│   │   ├── test_mistral_agent.py
│   │   └── test_upload.py
│   ├── Dockerfile
│   └── railway.json
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── index.html
│   ├── vercel.json
│   ├── .env                      # VITE_API_URL=http://localhost:8000
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx               # Main app with routing
│   │   ├── api/
│   │   │   └── client.ts         # Axios instance + API call functions
│   │   ├── components/
│   │   │   ├── Layout.tsx        # App shell — sidebar + header + main
│   │   │   ├── FileUpload.tsx    # CSV drag-and-drop upload component
│   │   │   ├── AnomalyChart.tsx  # Recharts time-series with anomaly highlights
│   │   │   ├── AlertPanel.tsx    # Real-time alert cards with severity badges
│   │   │   ├── DrugDiseaseMap.tsx # Leaflet map showing district-level clusters
│   │   │   ├── InsightCard.tsx   # Mistral-generated epidemiological insight display
│   │   │   └── Dashboard.tsx     # Main dashboard composing all widgets
│   │   ├── hooks/
│   │   │   ├── useAnalysis.ts    # React Query hook for analysis API
│   │   │   └── useAlerts.ts      # Polling/SSE hook for live alerts
│   │   ├── types/
│   │   │   └── index.ts          # TypeScript interfaces matching backend schemas
│   │   └── utils/
│   │       └── formatters.ts     # Date, number, severity formatting helpers
│   └── public/
│       └── india-districts.geojson  # India district boundaries for choropleth
└── README.md
```

## Backend Details

### Dependencies (requirements.txt)
```
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
sqlalchemy>=2.0
pydantic>=2.0
pydantic-settings>=2.0
mistralai>=1.0
pandas>=2.2
scipy>=1.14
numpy>=2.0
python-multipart>=0.0.18
python-dotenv>=1.0
aiosqlite>=0.20
```

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload pharma sales CSV, validate, store in DB |
| POST | `/api/analyze` | Run anomaly detection on stored data, return anomalies |
| GET | `/api/alerts` | Get generated alerts with severity levels |
| GET | `/api/map` | Get geographic cluster data for map visualization |
| POST | `/api/insights` | Send anomaly to Mistral for epidemiological interpretation |
| GET | `/api/health` | Health check |
| POST | `/api/seed` | Generate and load synthetic dataset |

### Anomaly Detection Engine (core/detection.py)
Three detection methods, all operating on drug sales grouped by (district, drug_category, week):
1. **Z-Score** — flag any (district, drug) pair where weekly sales exceed 2 standard deviations from the 8-week rolling mean
2. **IQR Outlier** — flag values outside 1.5× IQR of the district's historical sales for that drug
3. **Percentage Spike** — flag any week-over-week increase > 200%

Return anomalies as structured objects: `{ district, state, drug, drug_category, anomaly_type, severity, baseline_value, actual_value, z_score, date_range, confidence }`

### Mistral Agent (core/mistral_agent.py)
Use `mistralai` SDK. All Mistral calls go through this module.

**Functions to implement:**
1. `interpret_anomaly(anomaly)` — Given a detected anomaly, return epidemiological interpretation
   - System prompt: "You are an epidemiological analyst. Given pharmacy drug sales anomaly data from India, provide a public health interpretation. Include: likely condition, possible causes, severity assessment, recommended public health response."
   - Include drug-to-condition mappings as context
   
2. `generate_alert(anomalies: list)` — Given multiple anomalies in a region, generate a consolidated alert
   - System prompt: "Generate a public health alert based on these pharmacy sales anomalies. Include severity level (Critical/High/Medium/Low), affected area, suspected condition, and recommended actions for district health officers."

3. `correlate_signals(anomalies: list)` — Given anomalies across drugs/regions, identify if they point to a common cause
   - Example: salbutamol + antihistamine + corticosteroid spikes in same district = likely air pollution event

**Model**: Use `mistral-small-latest` for speed during hackathon. Can upgrade to `mistral-large-latest` for demo.

### Drug-to-Condition Mappings (core/mappings.py)
```python
DRUG_CONDITION_MAP = {
    # Tier 1 — High confidence (single drug → condition)
    "salbutamol": {"condition": "Respiratory disease", "category": "respiratory", "confidence": "high"},
    "budesonide": {"condition": "Asthma/COPD", "category": "respiratory", "confidence": "high"},
    "montelukast": {"condition": "Asthma", "category": "respiratory", "confidence": "high"},
    "oseltamivir": {"condition": "Influenza", "category": "infectious", "confidence": "high"},
    "ors_sachets": {"condition": "Diarrheal disease", "category": "waterborne", "confidence": "high"},
    "metronidazole": {"condition": "GI infection", "category": "waterborne", "confidence": "high"},
    "paracetamol": {"condition": "Fever/pain (non-specific)", "category": "general", "confidence": "low"},
    "levothyroxine": {"condition": "Thyroid disorder", "category": "endocrine", "confidence": "high"},
    "cetirizine": {"condition": "Allergic response", "category": "allergy", "confidence": "medium"},
    "azithromycin": {"condition": "Bacterial infection", "category": "infectious", "confidence": "medium"},
    "insulin_glargine": {"condition": "Diabetes", "category": "metabolic", "confidence": "high"},
    "amlodipine": {"condition": "Hypertension", "category": "cardiovascular", "confidence": "high"},
    
    # Tier 2 — Multi-drug correlation needed
    # ORS + metronidazole + loperamide in same district = waterborne outbreak (high confidence)
    # Paracetamol + oseltamivir + cetirizine in same district = flu cluster (high confidence)
    # Salbutamol + montelukast + budesonide in same district = air pollution event (high confidence)
}

CORRELATION_RULES = [
    {
        "name": "waterborne_outbreak",
        "drugs": ["ors_sachets", "metronidazole", "loperamide"],
        "min_match": 2,
        "condition": "Waterborne disease outbreak",
        "severity": "critical",
        "response": "Alert district water supply authority, deploy ORS distribution"
    },
    {
        "name": "respiratory_pollution",
        "drugs": ["salbutamol", "montelukast", "budesonide", "cetirizine"],
        "min_match": 2,
        "condition": "Air pollution respiratory cluster",
        "severity": "high",
        "response": "Correlate with AQI data, issue health advisory"
    },
    {
        "name": "flu_cluster",
        "drugs": ["oseltamivir", "paracetamol", "cetirizine"],
        "min_match": 2,
        "condition": "Influenza cluster",
        "severity": "high",
        "response": "Activate flu surveillance protocol, check vaccine stock"
    },
]
```

### Synthetic Data Generator (seed/generate_synthetic.py)
Generate 6 months of daily pharmacy sales data across 20 Indian districts.

**Baseline**: Each district has a normal daily sales volume per drug category with slight random variation (±15%).

**Injected Anomalies** (these are what the system should detect):
1. **Delhi respiratory spike** — Weeks 18-20: salbutamol, budesonide, montelukast sales 3-5x normal in South Delhi, North Delhi, East Delhi. Cause: simulated air pollution event.
2. **Chennai waterborne outbreak** — Week 12: ORS, metronidazole, loperamide spike 4-8x in Chennai, Kanchipuram. Cause: simulated water contamination.
3. **Pune flu cluster** — Weeks 8-10: oseltamivir, paracetamol, cetirizine 2-4x in Pune, Pimpri-Chinchwad. Cause: simulated influenza wave.
4. **Vizag thyroid anomaly** — Gradual 50% increase in levothyroxine over months 3-6 in Visakhapatnam. Cause: simulated industrial pollution.

**CSV columns**: `date,district,state,pharmacy_id,drug_generic_name,drug_category,quantity_sold,unit_price,latitude,longitude`

## Frontend Details

### Dependencies (package.json)
```json
{
  "dependencies": {
    "react": "^19",
    "react-dom": "^19",
    "react-router-dom": "^7",
    "axios": "^1.7",
    "@tanstack/react-query": "^5",
    "recharts": "^2.15",
    "react-leaflet": "^5",
    "leaflet": "^1.9",
    "react-dropzone": "^14",
    "lucide-react": "^0.460",
    "tailwindcss": "^4",
    "clsx": "^2"
  }
}
```

### Pages / Views
1. **Dashboard** (`/`) — Overview with summary cards (total anomalies, active alerts, districts affected), recent alerts feed, mini map
2. **Upload** (`/upload`) — Drag-and-drop CSV upload, preview parsed data table, trigger analysis
3. **Analysis** (`/analysis`) — Full anomaly detection results, time-series charts with highlighted anomalies, filter by district/drug/severity
4. **Map** (`/map`) — Full-screen Leaflet choropleth of India showing district-level disease signal intensity, click district for details
5. **Alerts** (`/alerts`) — All generated alerts, severity badges (Critical=red, High=orange, Medium=yellow, Low=blue), Mistral-generated insights expandable per alert

### UI Design Direction
- Clean, professional health-tech dashboard aesthetic
- Color scheme: Dark navy sidebar (#0f172a), white content area, accent colors for severity (red/orange/yellow/green)
- Cards with subtle shadows, clean typography
- Map should be the visual centerpiece of the demo

### Vercel Config (vercel.json)
```json
{
  "rewrites": [{ "source": "/(.*)", "destination": "/index.html" }]
}
```

## Build Order (for Claude Code)
Follow this exact sequence:

### Phase 1: Backend Foundation
1. Set up FastAPI app with CORS, config, database connection
2. Create SQLAlchemy models (DrugSale, Alert, Anomaly)
3. Build drug-to-condition mappings module
4. Build synthetic data generator with injected anomalies
5. Build CSV upload endpoint + parser

### Phase 2: Intelligence Layer
6. Build anomaly detection engine (z-score, IQR, percentage spike)
7. Build Mistral agent (interpret_anomaly, generate_alert, correlate_signals)
8. Build pipeline orchestrator (upload → detect → interpret → alert)
9. Wire up all API endpoints

### Phase 3: Frontend
10. Scaffold React + Vite + TypeScript + Tailwind
11. Build Layout + routing
12. Build FileUpload component with react-dropzone
13. Build Dashboard with summary cards
14. Build AnomalyChart with Recharts
15. Build DrugDiseaseMap with react-leaflet + India GeoJSON
16. Build AlertPanel with Mistral insights
17. Connect all components to backend API via React Query

### Phase 4: Polish & Deploy
18. Add loading states, error handling, empty states
19. Configure Railway deployment for backend
20. Configure Vercel deployment for frontend
21. Test full flow end-to-end

## Commands
```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev

# Generate synthetic data
cd backend && python -m app.seed.generate_synthetic

# Run tests
cd backend && pytest
```

## Key Demo Flow (for hackathon presentation)
1. Open dashboard — empty state
2. Upload pharma_sales.csv (pre-generated synthetic data)
3. System parses 50,000+ transactions across 20 districts
4. Click "Analyze" — anomaly detection runs, highlights appear on chart
5. Map lights up showing Delhi (red — respiratory), Chennai (red — waterborne), Pune (orange — flu)
6. Click Delhi cluster → Mistral generates: "Salbutamol and budesonide sales increased 340% in South Delhi over weeks 18-20. Combined with montelukast spike, this strongly suggests an acute respiratory event, likely correlated with air pollution. Recommend: cross-reference with CPCB AQI data, issue public health advisory for vulnerable populations."
7. Alerts panel shows prioritized alerts for district health officers

## Important Notes
- All data is synthetic and anonymized — no real patient data
- Drug names are generic, not brand — avoids pharma marketing issues
- This is the EPDS framework (Enterprise Software Perception Dependency Spectrum) applied to healthcare — disease surveillance is a P3 workflow (human-dependent) being converted to P1 (machine-readable) using already-digital proxy signals
- The GST e-invoicing infrastructure means this data ALREADY EXISTS in India — the barrier is architectural, not technological