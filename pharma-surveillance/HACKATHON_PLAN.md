# Hackathon Improvement Plan (Pharma Surveillance + Mistral)

This plan is optimized for a hackathon: maximize demo impact quickly while still building toward a production-ready public-health tool.

## 1) What to improve first (highest impact)

### A. Data quality guardrails before analysis
**Why:** Uploaded CSVs can silently fail or produce misleading anomalies if columns/types are off.

**Implement:**
- Add schema validation (required columns + type coercion + clear error messages).
- Add missing-value diagnostics for date, district, drug, quantity.
- Add duplicate and outlier checks (e.g., impossible negative sales, huge single-point spikes).

**Hackathon value:** Prevents live-demo failures and builds trust with judges.

---

### B. Prioritized triage workflow
**Why:** The app surfaces anomalies, but users still need to decide where to act first.

**Implement:**
- Create a **Risk Score** per district combining severity, anomaly count, multi-drug correlation, and trend acceleration.
- Add “Top 5 districts to investigate today” card.
- Add one-click download of triage briefing (CSV/PDF markdown) for field teams.

**Hackathon value:** Turns analytics into immediate decision support.

---

### C. Better explainability in UI
**Why:** Judges and non-technical stakeholders need transparent reasoning.

**Implement:**
- Add “Why flagged?” panel per anomaly:
  - baseline window,
  - method(s) triggered (z-score/IQR/spike),
  - threshold crossed,
  - confidence band.
- Add confidence labels for insights and alerts.

**Hackathon value:** Stronger credibility and easier storytelling.

---

### D. Scenario simulation mode
**Why:** A strong demo needs interactive “what-if” controls.

**Implement:**
- Slider-based synthetic event injection (e.g., +40% respiratory drug demand in selected districts).
- Real-time rerun and comparison: before vs after.

**Hackathon value:** Makes the app feel product-like and interactive.

---

### E. Alert lifecycle tracking
**Why:** Useful surveillance systems close the loop.

**Implement:**
- Add alert statuses: New → Investigating → Confirmed → Closed.
- Add notes/owner/date in a lightweight local store.

**Hackathon value:** Demonstrates operational readiness, not only detection.

## 2) Suggested execution timeline

### Phase 1 (0–24 hours)
- CSV validation + friendly error reports.
- Risk score + “Top districts today” widget.
- “Why flagged?” mini-panel in anomaly explorer.

### Phase 2 (24–48 hours)
- Scenario simulation controls.
- Alert lifecycle status board.
- Export button for daily briefing.

### Phase 3 (48–72 hours)
- Evaluation page with precision proxy metrics on synthetic events.
- Polish: loading states, empty states, and concise microcopy.

## 3) More high-value Mistral use cases for this app

### 1) District briefing generator (morning report)
Input: anomalies + correlations + trend deltas.
Output: 8–10 line briefing for district medical officers with action checklist.

### 2) Differential-cause reasoning
Use Mistral to rank plausible causes for a spike (seasonal wave vs pollution vs reporting artifact), with confidence and missing evidence.

### 3) Cross-district pattern clustering
Given multiple district signals, summarize if they represent one regional event or isolated incidents.

### 4) Data quality anomaly narrator
When bad data is detected (missing district/date, sudden 1000x jumps), auto-generate remediation guidance for data-entry teams.

### 5) Public communication drafts
Generate audience-specific messaging:
- technical memo for health officers,
- short advisory for citizens,
- press-ready neutral summary.

### 6) Recommended intervention planner
From anomaly profile + mapped condition, generate tiered interventions:
- immediate (24h),
- short-term (7 days),
- monitoring metrics.

### 7) Query assistant for analysts
Natural language interface (“show high-severity anti-diarrheal spikes in coastal districts over the last 4 weeks”).

### 8) Policy memory + retrieval (RAG-ready)
Ground Mistral responses with local policy docs/SOPs so actions align with official protocol.

### 9) Counterfactual simulation explainer
After simulation mode changes, Mistral explains impact in plain language: what changed, why risk score moved, and what to monitor next.

### 10) Multi-lingual alert localization
Generate alerts in English + Hindi + regional language with controlled tone and factual consistency.

## 4) Demo story for judges (recommended)
1. Upload noisy CSV (show validation catches issues).
2. Run pipeline and open risk-ranked district list.
3. Drill into one district with “Why flagged?”.
4. Trigger Mistral district briefing + intervention plan.
5. Run simulation (+respiratory surge) and show changed priorities.
6. Export a daily action brief.

## 5) Success metrics to show at hackathon
- Time-to-insight (upload to actionable alert) < 2 minutes.
- % alerts with explainability traces.
- Correlated multi-drug events detected.
- Analyst clicks to produce district briefing.
- Demo reliability (no crash on malformed CSV).

## 6) Post-hackathon next steps
- Add unit tests around detection thresholds and mapping rules.
- Add evaluation harness using synthetic ground truth.
- Add auth/roles + secure audit logs for public-health workflows.
- Move from local state to durable backend (Postgres + API).
