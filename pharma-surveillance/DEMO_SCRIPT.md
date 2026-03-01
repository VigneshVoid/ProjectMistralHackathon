# Pharma Surveillance — 2-Minute Demo Script

**Mistral AI Hackathon 2026**
Total runtime: ~2:00 | ~300 words at natural pace

---

## [0:00–0:20] THE HOOK — Problem Statement
> **SCREEN: Title slide or Dashboard landing page**

"In India, pharmacies are the first point of contact for healthcare. When a waterborne outbreak hits Chennai, ORS sachet sales spike days before hospitals report cases. When Delhi's air quality crashes, inhaler sales explode across three districts simultaneously.

The data is already there — sitting in pharmacy sales records. The question is: can we read it fast enough to save lives?

This is **Pharma Surveillance** — an AI-powered early warning system that turns pharmacy sales data into actionable public health intelligence, powered entirely by Mistral AI."

---

## [0:20–0:45] THE ENGINE — How It Works
> **SCREEN: Click "Generate Synthetic Data & Analyze" — show progress bar running through steps**

"The system ingests pharmacy sales across 20 Indian districts and 13 drug categories. Four statistical methods — Z-Score, IQR, Percentage Spike, and EWMA — scan every week for anomalies.

Then Mistral takes over. Each anomaly gets interpreted, correlated across districts, and turned into a severity-ranked alert — all through structured output validated against Pydantic models. The entire pipeline runs with built-in retry logic and rate-limit resilience."

---

## [0:45–1:05] MISTRAL FEATURES — The AI Core
> **SCREEN: Quick cuts — Dashboard KPIs, then Alerts page with expander open**

"We showcase **six Mistral SDK capabilities** through 17 AI functions:

**Structured Output** powers nine validated Pydantic models — from differential diagnosis to intervention plans.
**Streaming** delivers token-by-token live analysis.
**Function Calling** drives our natural-language assistant — ask a question, Mistral picks the right data tool automatically.
**Multilingual generation** translates alerts into seven Indian languages — Hindi, Tamil, Telugu, and more.

Every response is typed, validated, and production-ready."

---

## [1:05–1:35] THE DEMO — Key Pages
> **SCREEN: Navigate pages as you speak — spend ~5 seconds on each**

"The **Dashboard** ranks districts by risk and generates morning briefings for health officers.

**[click Anomaly Explorer]** The Explorer lets you drill into any anomaly — 'Why was this flagged?' — and get a differential diagnosis with ranked probabilities and an intervention plan with 24-hour and 7-day actions.

**[click Disease Map]** The geographic map clusters signals by region with severity-colored markers.

**[click Alerts & Insights]** Alerts have full lifecycle tracking. You can translate any alert into local languages, stream a deeper analysis, or draft audience-specific public communications — a technical memo for health officers, a citizen advisory, or a press summary — all in one click.

**[click Scenario Simulator]** And the Simulator lets you ask 'what if?' — inject a 4x respiratory spike in Bengaluru and instantly see how it changes the risk landscape."

---

## [1:35–2:00] VALIDATION + CLOSE
> **SCREEN: Evaluation page showing F1 scores, then back to Dashboard**

"To prove this works, we injected four known outbreak scenarios into synthetic data — a Delhi respiratory spike, a Chennai waterborne outbreak, a Pune flu cluster, and a Vizag thyroid anomaly. The system achieves **100% recall** — every single ground truth event was detected.

Pharma Surveillance turns pharmacy counters into public health sensors. The data already exists. Mistral AI makes it speak.

Thank you."

---

## Recording Tips

- **Resolution**: 1920x1080, browser fullscreen
- **Pre-generate data** before recording so pages load instantly
- **Keep mouse movements smooth** — audiences follow the cursor
- **Voiceover pace**: ~150 words/min (natural, not rushed)
- **Background music**: Optional — subtle lo-fi at 10% volume
- **Tools**: OBS Studio (free) or Loom for screen + voice recording
