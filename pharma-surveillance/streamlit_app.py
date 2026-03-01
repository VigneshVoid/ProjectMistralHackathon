"""
Pharma Surveillance System — Streamlit Dashboard

Real-time epidemiological surveillance using pharmacy drug sales data
as a proxy to detect disease outbreaks, pollution health effects, and health patterns.

Showcases 6 Mistral SDK features: structured output, streaming, function calling,
multilingual generation, chat.complete, and model upgrade to mistral-medium-latest.
"""

import sys
import html
import json
import math
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from app.core.pipeline import run_pipeline
from app.core.mappings import DISTRICTS, SEASONAL_PROFILES, STATE_LANGUAGES, get_seasonal_multiplier
from app.core.detection import aggregate_weekly
from app.seed.generate_synthetic import generate
from app.config import settings

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pharma Surveillance System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        -webkit-font-smoothing: antialiased;
    }
    
    .stApp {
        background: radial-gradient(circle at top center, #0f172a 0%, #020617 100%) !important;
    }
    
    /* LHS / Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.45) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Global Text Color Overrides for Dark Mode */
    [data-testid="stMainBlockContainer"] h1,
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] label,
    [data-testid="stMainBlockContainer"] li,
    [data-testid="stMainBlockContainer"] div {
        color: #cbd5e1;
    }

    .main .block-container { 
        padding-top: 2rem; 
        max-width: 1400px;
    }
    
    /* Glowing, Premium Severity Badges */
    .severity-critical { color: #fff !important; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85em; box-shadow: 0 4px 14px rgba(220, 38, 38, 0.4); text-shadow: 0 1px 2px rgba(0,0,0,0.2); }
    .severity-high { color: #fff !important; background: linear-gradient(135deg, #f97316 0%, #ea580c 100%); padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85em; box-shadow: 0 4px 14px rgba(234, 88, 12, 0.4); text-shadow: 0 1px 2px rgba(0,0,0,0.2); }
    .severity-medium { color: #1e293b !important; background: linear-gradient(135deg, #fde047 0%, #facc15 100%); padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85em; box-shadow: 0 4px 14px rgba(250, 204, 21, 0.3); }
    .severity-low { color: #fff !important; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 4px 14px; border-radius: 20px; font-weight: 600; font-size: 0.85em; box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4); text-shadow: 0 1px 2px rgba(0,0,0,0.2); }
    
    /* Glassmorphic Metric Cards with Micro-interactions */
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08); 
        border-radius: 24px;
        padding: 1.7rem; 
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2), 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(255,255,255,0) 100%);
        pointer-events: none;
    }
    .metric-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.3), 0 4px 16px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.15); 
        background: rgba(30, 41, 59, 0.7);
    }
    .metric-card h2 { margin: 0; font-size: 2.8rem; font-weight: 700; background: none; letter-spacing: -0.03em; line-height: 1.1; }
    .metric-card p { margin: 0; color: #94a3b8 !important; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; padding-top: 0.7rem; }
    
    /* Modern Alert Cards */
    .alert-card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-left: 6px solid; 
        padding: 1.4rem; 
        margin-bottom: 1.2rem;
        border-radius: 16px 20px 20px 16px;
        color: #f1f5f9;
        line-height: 1.6;
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.15), 0 3px 7px -3px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        border-top: 1px solid rgba(255,255,255,0.05);
        border-right: 1px solid rgba(255,255,255,0.05);
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .alert-card:hover {
        transform: translateX(6px);
        box-shadow: 0 12px 24px -4px rgba(0, 0, 0, 0.2), 0 6px 14px -6px rgba(0, 0, 0, 0.15);
        background: rgba(45, 60, 83, 0.7);
    }
    .alert-card strong { color: #f8fafc; font-weight: 700; letter-spacing: -0.01em; }
    .alert-card .severity-critical,
    .alert-card .severity-high,
    .alert-card .severity-medium,
    .alert-card .severity-low {
        margin-left: 0.6rem;
        vertical-align: text-bottom;
    }
    .alert-critical { border-left-color: #ef4444; }
    .alert-high { border-left-color: #f97316; }
    .alert-medium { border-left-color: #facc15; }
    .alert-low { border-left-color: #3b82f6; }
    
    /* Animated Direction indicators */
    .direction-spike { color: #fca5a5 !important; font-weight: 700; background: rgba(239, 68, 68, 0.2); padding: 3px 8px; border-radius: 8px; }
    .direction-drop { color: #93c5fd !important; font-weight: 700; background: rgba(59, 130, 246, 0.2); padding: 3px 8px; border-radius: 8px; }
    
    /* Playful primary buttons */
    div[data-testid="stButton"] > button {
        border-radius: 14px;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        border: 1px solid rgba(255,255,255,0.1);
        background: rgba(255, 255, 255, 0.05);
        color: #f8fafc;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div[data-testid="stButton"] > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255,255,255,0.2);
    }
    
    /* Input/Select boxes dark mode */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #f8fafc !important;
    }

    /* ── Phase 1A: Plotly transparent background ── */
    [data-testid="stPlotlyChart"] iframe { background: transparent !important; }

    /* ── Phase 2: Glassmorphic Dataframes ── */
    [data-testid="stDataFrame"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stDataFrame"] .dvn-scroller { background: transparent !important; }

    /* ── Phase 3A: Glassmorphic Expanders ── */
    [data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.35) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        margin-bottom: 0.8rem;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary { color: #f8fafc !important; font-weight: 600; }
    [data-testid="stExpander"] summary:hover { background: rgba(255, 255, 255, 0.03); }
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ── Phase 3B: Dark Pill Tabs ── */
    [data-testid="stTabs"] [role="tablist"] {
        background: rgba(15, 23, 42, 0.5);
        border-radius: 12px; padding: 4px; gap: 4px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stTabs"] [role="tab"] {
        color: #94a3b8 !important; border-radius: 8px;
        border: none !important; background: transparent;
        transition: all 0.2s ease;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #f8fafc !important;
        background: rgba(59, 130, 246, 0.2) !important;
        font-weight: 600;
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none; }

    /* ── Phase 3C: Gradient Fade Dividers ── */
    [data-testid="stMainBlockContainer"] hr {
        border: none !important; height: 1px !important;
        background: linear-gradient(90deg, rgba(255,255,255,0) 0%,
            rgba(148,163,184,0.2) 20%, rgba(148,163,184,0.2) 80%,
            rgba(255,255,255,0) 100%) !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Phase 3D: Dark File Uploader ── */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.4);
        border: 2px dashed rgba(148, 163, 184, 0.2) !important;
        border-radius: 16px; padding: 1rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(59, 130, 246, 0.4) !important;
        background: rgba(30, 41, 59, 0.6);
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background: transparent !important; border: none !important;
    }

    /* ── Phase 3E: Dark Text Inputs ── */
    [data-testid="stTextInput"] input {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #f8fafc !important; border-radius: 10px;
    }
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(59, 130, 246, 0.5) !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.15) !important;
    }

    /* ── Phase 4A: Glassmorphic Chat Bubbles ── */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px; margin-bottom: 0.8rem;
    }
    [data-testid="stChatInput"] {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 14px;
    }
    [data-testid="stChatInput"] textarea {
        color: #f8fafc !important; background: transparent !important;
    }

    /* ── Phase 4B: Dark Status Messages ── */
    [data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.5) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    [data-testid="stAlert"] p { color: #e2e8f0 !important; }

    /* ── Phase 4D: Gradient Progress Bars ── */
    [data-testid="stProgress"] > div { background: rgba(30, 41, 59, 0.5) !important; border-radius: 8px; }
    [data-testid="stProgress"] [role="progressbar"] > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; border-radius: 8px;
    }

    /* ── Phase 4E: Sidebar Polish ── */
    [data-testid="stSidebar"] [role="radiogroup"] label { transition: all 0.2s ease; }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        color: #f8fafc !important;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    [data-testid="stSidebar"] hr {
        border: none !important; height: 1px !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "alert_states" not in st.session_state:
    st.session_state.alert_states = {}
if "translations" not in st.session_state:
    st.session_state.translations = {}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital.png", width=60)
    st.title("Pharma Surveillance")
    st.caption("Epidemiological intelligence from pharmacy sales data")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Dashboard", "Upload & Analyze", "Anomaly Explorer", "Year-over-Year",
         "Disease Map", "Alerts & Insights", "AI Assistant", "Scenario Simulator", "Evaluation"],
        index=0,
    )

    st.divider()
    st.markdown("**Settings**")
    use_mistral = st.toggle(
        "Enable Mistral AI",
        value=bool(settings.mistral_api_key and settings.mistral_api_key != "your-mistral-api-key-here"),
        help="Toggle Mistral-powered interpretations. Requires a valid API key in .env",
    )
    seasonal_adjust = st.toggle(
        "Seasonal Exclusion",
        value=False,
        help="Exclude expected seasonal drug spikes from anomaly detection.",
    )

    st.divider()
    st.caption("Built for Mistral Hackathon 2026")
    st.caption("Powered by **mistral-medium-latest**")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def severity_badge(severity: str) -> str:
    icon = {"critical": "!!", "high": "!", "medium": "~", "low": "*"}.get(severity, "")
    return f'<span class="severity-{severity}">{icon} {severity.upper()}</span>'


def direction_badge(direction: str) -> str:
    if direction == "drop":
        return '<span class="direction-drop">&#x25BC; DROP</span>'
    return '<span class="direction-spike">&#x25B2; SPIKE</span>'


def render_validation(report: dict) -> None:
    if not report:
        return
    for error in report.get("errors", []):
        st.error(error)
    for warning in report.get("warnings", []):
        st.warning(warning)
    st.caption(
        f"Rows in: {report.get('rows_in', 0):,} | "
        f"Rows out: {report.get('rows_out', 0):,} | "
        f"Rows dropped: {report.get('rows_dropped', 0):,}"
    )


def metric_card(label: str, value, color: str = "#0f172a", delta: str = "") -> str:
    # Use gradient text mapping for beautiful shiny metrics
    gradient = f"color: {color};"  # Fallback
    if color == "#dc2626":
        gradient = "background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    elif color == "#ea580c":
        gradient = "background: linear-gradient(135deg, #f97316 0%, #c2410c 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    elif color == "#7c3aed":
        gradient = "background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"
    elif color == "#059669":
        gradient = "background: linear-gradient(135deg, #10b981 0%, #047857 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;"

    delta_html = ""
    if delta:
        d = str(delta)
        delta_color = "#ef4444" if d.startswith("+") else "#10b981" if d.startswith("-") else "#94a3b8"
        delta_html = f'<p style="margin:0;color:{delta_color} !important;font-size:0.85rem;font-weight:600;padding-top:0.3rem;">{d}</p>'

    return f"""
    <div class="metric-card">
        <h2 style="{gradient}">{value}</h2>
        <p>{label}</p>
        {delta_html}
    </div>
    """


# ---------------------------------------------------------------------------
# Plotly dark theme layout (reused by all charts)
# ---------------------------------------------------------------------------
PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#cbd5e1"),
    xaxis=dict(gridcolor="rgba(148,163,184,0.1)", zerolinecolor="rgba(148,163,184,0.15)",
               tickfont=dict(color="#94a3b8"), title_font=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="rgba(148,163,184,0.1)", zerolinecolor="rgba(148,163,184,0.15)",
               tickfont=dict(color="#94a3b8"), title_font=dict(color="#94a3b8")),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
    hoverlabel=dict(bgcolor="rgba(30,41,59,0.95)", font_color="#f8fafc",
                    bordercolor="rgba(255,255,255,0.1)"),
)


def _get_alert_text(alert: dict) -> str:
    """Extract displayable text from an alert (structured or legacy)."""
    data = alert.get("alert_data", {})
    if isinstance(data, dict):
        if "error" in data:
            return data["error"]
        parts = []
        if data.get("severity"):
            parts.append(f"**SEVERITY**: {data['severity']}")
        if data.get("affected_area"):
            parts.append(f"**AFFECTED AREA**: {data['affected_area']}")
        if data.get("suspected_condition"):
            parts.append(f"**SUSPECTED CONDITION**: {data['suspected_condition']}")
        if data.get("evidence_summary"):
            parts.append(f"**EVIDENCE**: {data['evidence_summary']}")
        if data.get("recommended_actions"):
            actions = "\n".join(f"{i+1}. {a}" for i, a in enumerate(data["recommended_actions"]))
            parts.append(f"**RECOMMENDED ACTIONS**:\n{actions}")
        if data.get("urgency_score"):
            parts.append(f"**URGENCY SCORE**: {data['urgency_score']}/10")
        return "\n\n".join(parts) if parts else str(data)
    # Legacy plain text
    if "alert_text" in alert:
        return alert["alert_text"]
    return str(data)


def _get_interpretation_content(interp: dict) -> str:
    """Extract displayable text from an interpretation (structured or legacy)."""
    data = interp.get("interpretation", {})
    if isinstance(data, dict):
        if "error" in data:
            return data["error"]
        parts = []
        if data.get("likely_condition"):
            parts.append(f"**Likely Condition**: {data['likely_condition']}")
        if data.get("confidence") is not None:
            parts.append(f"**Confidence**: {data['confidence']:.0%}")
        if data.get("severity_assessment"):
            parts.append(f"**Severity**: {data['severity_assessment']}")
        if data.get("possible_causes"):
            causes = "\n".join(f"- {c}" for c in data["possible_causes"])
            parts.append(f"**Possible Causes**:\n{causes}")
        if data.get("recommended_actions"):
            actions = "\n".join(f"{i+1}. {a}" for i, a in enumerate(data["recommended_actions"]))
            parts.append(f"**Recommended Actions**:\n{actions}")
        if data.get("additional_context"):
            parts.append(f"**Context**: {data['additional_context']}")
        return "\n\n".join(parts) if parts else str(data)
    return str(data)


# ---------------------------------------------------------------------------
# PAGE: Dashboard
# ---------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Surveillance Dashboard")

    if st.session_state.pipeline_results is None:
        st.info(
            "No data loaded yet. Go to **Upload & Analyze** to upload a CSV or generate synthetic data.",
            icon="📊",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Synthetic Data & Analyze", type="primary", use_container_width=True):
                with st.spinner("Generating 18 months of synthetic pharmacy data (2023-2024)..."):
                    df = generate()
                    st.session_state.df = df
                progress_bar = st.progress(0, text="Starting pipeline...")
                status_text = st.empty()

                def _on_progress(step, total, msg):
                    progress_bar.progress(min(step / max(total, 1), 1.0), text=msg)
                    status_text.caption(f"Step {step}/{total}: {msg}")

                results = run_pipeline(
                    df, use_mistral=use_mistral, seasonal_adjust=seasonal_adjust,
                    progress_callback=_on_progress,
                )
                st.session_state.pipeline_results = results
                progress_bar.empty()
                status_text.empty()
                st.rerun()
    else:
        results = st.session_state.pipeline_results
        summary = results["summary"]
        render_validation(summary.get("validation", {}))

        # Summary metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(metric_card("Total Records", f"{summary['total_records']:,}"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Anomalies Detected", summary["total_anomalies"], "#dc2626"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Districts Affected", summary["districts_affected"], "#ea580c"), unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card("Correlations", summary["correlations_found"], "#7c3aed"), unsafe_allow_html=True)
        with c5:
            st.markdown(metric_card("Alerts Generated", summary["alerts_generated"], "#059669"), unsafe_allow_html=True)

        st.divider()
        st.subheader("Top Districts to Investigate")
        risk_df = pd.DataFrame(results.get("district_risk", []))
        if not risk_df.empty:
            st.dataframe(risk_df.head(5), use_container_width=True)
        else:
            st.info("No district risk ranking available yet.")

        # District Morning Briefing
        if use_mistral and not risk_df.empty:
            st.divider()
            st.subheader("District Morning Briefing")
            briefing_district = st.selectbox(
                "Select district for briefing",
                risk_df["district"].tolist()[:10],
                key="briefing_district",
            )
            if st.button("Generate Morning Briefing", type="primary"):
                with st.spinner("Generating briefing with Mistral AI..."):
                    from app.core.mistral_agent import generate_district_briefing
                    district_anomalies = [a for a in results["anomalies"] if a["district"] == briefing_district]
                    district_corrs = [c for c in results["correlations"] if c["district"] == briefing_district]
                    briefing = generate_district_briefing(briefing_district, district_anomalies, district_corrs)

                st.markdown(f"### Morning Briefing: {briefing.get('district', briefing_district)}")
                st.markdown(f"**Risk Level**: {briefing.get('risk_level', 'N/A')}")

                if briefing.get("active_signals"):
                    st.markdown("**Active Signals:**")
                    for sig in briefing["active_signals"]:
                        st.markdown(f"- {sig}")

                if briefing.get("recommended_actions"):
                    st.markdown("**Recommended Actions:**")
                    for i, action in enumerate(briefing["recommended_actions"], 1):
                        st.markdown(f"{i}. {action}")

                if briefing.get("monitoring_metrics"):
                    st.markdown("**Monitoring Metrics:**")
                    for m in briefing["monitoring_metrics"]:
                        st.markdown(f"- {m}")

                if briefing.get("escalation_criteria"):
                    st.markdown(f"**Escalation Criteria**: {briefing['escalation_criteria']}")

                # Export briefing
                briefing_md = f"# Morning Briefing: {briefing.get('district', briefing_district)}\n\n"
                briefing_md += f"**Risk Level**: {briefing.get('risk_level', 'N/A')}\n\n"
                if briefing.get("active_signals"):
                    briefing_md += "## Active Signals\n" + "\n".join(f"- {s}" for s in briefing["active_signals"]) + "\n\n"
                if briefing.get("recommended_actions"):
                    briefing_md += "## Recommended Actions\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(briefing["recommended_actions"])) + "\n\n"
                if briefing.get("monitoring_metrics"):
                    briefing_md += "## Monitoring Metrics\n" + "\n".join(f"- {m}" for m in briefing["monitoring_metrics"]) + "\n\n"
                if briefing.get("escalation_criteria"):
                    briefing_md += f"## Escalation Criteria\n{briefing['escalation_criteria']}\n"
                st.download_button("Download Briefing", briefing_md, f"briefing_{briefing_district}.md", "text/markdown")

        st.divider()

        # Export daily summary
        if st.button("Export Daily Summary"):
            summary_md = f"# Daily Surveillance Summary\n\n"
            summary_md += f"- Total Records: {summary['total_records']:,}\n"
            summary_md += f"- Anomalies Detected: {summary['total_anomalies']}\n"
            summary_md += f"- Districts Affected: {summary['districts_affected']}\n"
            summary_md += f"- Correlations Found: {summary['correlations_found']}\n"
            summary_md += f"- Alerts Generated: {summary['alerts_generated']}\n\n"
            if not risk_df.empty:
                summary_md += "## Top Districts\n"
                for _, row in risk_df.head(5).iterrows():
                    summary_md += f"- {row['district']} (Risk: {row['risk_score']})\n"
            st.download_button("Download Summary", summary_md, "daily_summary.md", "text/markdown")

        # Severity breakdown + recent alerts side by side
        col_left, col_right = st.columns([1, 2])

        with col_left:
            st.subheader("Severity Breakdown")
            sev = summary["severity_breakdown"]
            sev_df = pd.DataFrame([
                {"Severity": k.capitalize(), "Count": v}
                for k, v in sev.items()
            ])
            if not sev_df.empty:
                colors = {"Critical": "#dc2626", "High": "#ea580c", "Medium": "#facc15", "Low": "#2563eb"}
                fig = px.pie(
                    sev_df, values="Count", names="Severity",
                    color="Severity", color_discrete_map=colors,
                    hole=0.45,
                )
                fig.update_layout(**PLOTLY_DARK_LAYOUT, margin=dict(t=20, b=20, l=20, r=20), height=280)
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Correlations Found")
            for corr in results["correlations"]:
                st.markdown(
                    f"**{corr['district']}**: {corr['condition']}  \n"
                    f"Matched drugs: {', '.join(corr['matched_drugs'])}  \n"
                    f"Severity: {severity_badge(corr['severity'])}",
                    unsafe_allow_html=True,
                )

        with col_right:
            st.subheader("Recent Alerts")
            if results["alerts"]:
                for alert in results["alerts"][:5]:
                    sev = alert.get("max_severity", "medium")
                    alert_text = html.escape(_get_alert_text(alert)[:300])
                    st.markdown(
                        f'<div class="alert-card alert-{sev}">'
                        f'<strong>{html.escape(alert["district"])}, {html.escape(alert["state"])}</strong> '
                        f'{severity_badge(sev)}<br>'
                        f'{alert_text}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No alerts generated. Enable Mistral AI or run analysis first.")

        # Anomaly timeline
        st.divider()
        st.subheader("Anomaly Timeline")
        anomalies_df = pd.DataFrame(results["anomalies"])
        if not anomalies_df.empty:
            timeline = anomalies_df.groupby(["week", "severity"]).size().reset_index(name="count")
            fig = px.bar(
                timeline, x="week", y="count", color="severity",
                color_discrete_map={"critical": "#dc2626", "high": "#ea580c", "medium": "#facc15", "low": "#2563eb"},
                labels={"week": "Week", "count": "Anomalies"},
            )
            fig.update_layout(**PLOTLY_DARK_LAYOUT, margin=dict(t=20, b=40), height=300, bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Upload & Analyze
# ---------------------------------------------------------------------------
elif page == "Upload & Analyze":
    st.title("Upload & Analyze")
    st.markdown("Upload a pharmacy sales CSV or generate synthetic data to run the surveillance pipeline.")

    tab1, tab2 = st.tabs(["Upload CSV", "Generate Synthetic"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload pharmacy sales CSV",
            type=["csv"],
            help="Expected columns: date, district, state, pharmacy_id, drug_generic_name, drug_category, quantity_sold, unit_price, latitude, longitude",
        )
        if uploaded:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"Loaded {len(df):,} records from CSV")
            st.dataframe(df.head(20), use_container_width=True)

            if st.button("Run Analysis", type="primary"):
                progress_bar = st.progress(0, text="Starting pipeline...")
                status_text = st.empty()

                def _on_progress(step, total, msg):
                    progress_bar.progress(min(step / max(total, 1), 1.0), text=msg)
                    status_text.caption(f"Step {step}/{total}: {msg}")

                results = run_pipeline(
                    df, use_mistral=use_mistral, seasonal_adjust=seasonal_adjust,
                    progress_callback=_on_progress,
                )
                st.session_state.pipeline_results = results
                progress_bar.empty()
                status_text.empty()
                st.success(f"Found {results['summary']['total_anomalies']} anomalies across {results['summary']['districts_affected']} districts")
                render_validation(results["summary"].get("validation", {}))
                st.rerun()

    with tab2:
        st.markdown(
            "Generate 18 months (Jan 2023 - Jun 2024) of synthetic pharmacy sales data across 20 Indian districts "
            "with **seasonal variation** and **4 injected anomaly scenarios** in 2024:"
        )
        st.markdown("""
        1. **Delhi Respiratory Spike** (Weeks 18-20 of 2024) — Air pollution event
        2. **Chennai Waterborne Outbreak** (Week 12 of 2024) — Water contamination
        3. **Pune Flu Cluster** (Weeks 8-10 of 2024) — Influenza wave
        4. **Vizag Thyroid Anomaly** (Months 3-6 of 2024) — Industrial pollution

        2023 data serves as a clean seasonal baseline for Year-over-Year comparison.
        """)

        if st.button("Generate & Analyze", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic data (2023-2024)..."):
                df = generate()
                st.session_state.df = df
            st.success(f"Generated {len(df):,} pharmacy sales records")

            progress_bar = st.progress(0, text="Starting pipeline...")
            status_text = st.empty()

            def _on_progress(step, total, msg):
                progress_bar.progress(min(step / max(total, 1), 1.0), text=msg)
                status_text.caption(f"Step {step}/{total}: {msg}")

            results = run_pipeline(
                df, use_mistral=use_mistral, seasonal_adjust=seasonal_adjust,
                progress_callback=_on_progress,
            )
            st.session_state.pipeline_results = results
            progress_bar.empty()
            status_text.empty()
            st.success(
                f"Analysis complete: {results['summary']['total_anomalies']} anomalies, "
                f"{results['summary']['districts_affected']} districts affected"
            )
            render_validation(results["summary"].get("validation", {}))
            st.rerun()

    if st.session_state.df is not None:
        st.divider()
        st.subheader("Current Dataset")
        df = st.session_state.df
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(metric_card("Total Records", f"{len(df):,}"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card("Districts", df["district"].nunique(), "#7c3aed"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card("Drugs", df["drug_generic_name"].nunique(), "#059669"), unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)

        # Data Quality Narrator — AI-powered remediation guidance
        if st.session_state.pipeline_results and use_mistral:
            validation_report = st.session_state.pipeline_results.get("summary", {}).get("validation", {})
            if validation_report and (validation_report.get("warnings") or validation_report.get("rows_dropped", 0) > 0):
                st.divider()
                st.subheader("AI Data Quality Narrator")
                if st.button("Generate Remediation Guidance", type="secondary"):
                    with st.spinner("Analyzing data quality issues with Mistral AI..."):
                        from app.core.mistral_agent import narrate_data_quality
                        narration = narrate_data_quality(validation_report)
                    if "error" not in narration:
                        st.markdown(f"**Summary:** {html.escape(narration['summary'])}")
                        if narration.get("issues_found"):
                            st.markdown("**Issues Found:**")
                            for issue in narration["issues_found"]:
                                st.markdown(f"- {html.escape(issue)}")
                        if narration.get("remediation_steps"):
                            st.markdown("**Remediation Steps:**")
                            for i, step in enumerate(narration["remediation_steps"], 1):
                                st.markdown(f"{i}. {html.escape(step)}")
                        if narration.get("data_entry_tips"):
                            with st.expander("Preventive Tips for Pharmacy Staff"):
                                for tip in narration["data_entry_tips"]:
                                    st.markdown(f"- {html.escape(tip)}")
                        st.info(f"**Risk Assessment:** {html.escape(narration['risk_assessment'])}")
                    else:
                        st.error(narration["error"])


# ---------------------------------------------------------------------------
# PAGE: Anomaly Explorer
# ---------------------------------------------------------------------------
elif page == "Anomaly Explorer":
    st.title("Anomaly Explorer")

    if st.session_state.pipeline_results is None:
        st.warning("Run analysis first from the Upload & Analyze page.")
    else:
        results = st.session_state.pipeline_results
        anomalies_df = pd.DataFrame(results["anomalies"])
        weekly = results["weekly_data"]

        if anomalies_df.empty:
            st.info("No anomalies detected.")
        else:
            # Filters
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sel_districts = st.multiselect("District", options=sorted(anomalies_df["district"].unique()), default=[])
            with col2:
                sel_drugs = st.multiselect("Drug", options=sorted(anomalies_df["drug"].unique()), default=[])
            with col3:
                sel_severity = st.multiselect("Severity", options=["critical", "high", "medium", "low"], default=[])
            with col4:
                direction_opts = sorted(anomalies_df["direction"].unique()) if "direction" in anomalies_df.columns else []
                sel_direction = st.multiselect("Direction", options=direction_opts, default=[])

            filtered = anomalies_df.copy()
            if sel_districts:
                filtered = filtered[filtered["district"].isin(sel_districts)]
            if sel_drugs:
                filtered = filtered[filtered["drug"].isin(sel_drugs)]
            if sel_severity:
                filtered = filtered[filtered["severity"].isin(sel_severity)]
            if sel_direction and "direction" in filtered.columns:
                filtered = filtered[filtered["direction"].isin(sel_direction)]

            st.markdown(f"**{len(filtered)} anomalies** matching filters")

            # Export filtered anomalies
            if not filtered.empty:
                csv_data = filtered.to_csv(index=False)
                st.download_button("Download Filtered Anomalies (CSV)", csv_data, "anomalies.csv", "text/csv")

            if not filtered.empty:
                st.subheader("Why flagged?")
                selected_row = st.selectbox(
                    "Select anomaly",
                    range(len(filtered)),
                    format_func=lambda i: (
                        f"{filtered.iloc[i]['drug']} | {filtered.iloc[i]['district']} | "
                        f"week {filtered.iloc[i]['week']} | {filtered.iloc[i]['severity']} | "
                        f"{filtered.iloc[i].get('direction', 'spike')}"
                    ),
                )
                row = filtered.iloc[selected_row]
                dir_str = row.get("direction", "spike")
                st.markdown(
                    f"**Method triggered:** `{row['anomaly_type']}`  \n"
                    f"**Direction:** {direction_badge(dir_str)}  \n"
                    f"**Baseline:** {row['baseline_value']} units/week  \n"
                    f"**Actual:** {row['actual_value']} units/week  \n"
                    f"**Z-score:** {row.get('z_score', 'N/A')}  \n"
                    f"**% change:** {row.get('pct_change', 'N/A')}  \n"
                    f"**Confidence:** {row.get('confidence', 'medium')}",
                    unsafe_allow_html=True,
                )

                # Differential diagnosis and intervention planner
                if use_mistral:
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if st.button("Differential Diagnosis"):
                            with st.spinner("Analyzing differential causes..."):
                                from app.core.mistral_agent import differential_diagnosis
                                dd = differential_diagnosis(row.to_dict())
                            st.markdown(f"**Most Likely**: {dd.get('most_likely', 'N/A')} (Confidence: {dd.get('overall_confidence', 0):.0%})")
                            for cause in dd.get("causes", []):
                                with st.expander(f"{cause['cause']} ({cause['probability']:.0%})"):
                                    st.markdown("**Supporting Evidence:**")
                                    for e in cause.get("supporting_evidence", []):
                                        st.markdown(f"- {e}")
                                    st.markdown("**Missing Evidence:**")
                                    for e in cause.get("missing_evidence", []):
                                        st.markdown(f"- {e}")
                                    st.markdown(f"**How to Confirm:** {cause.get('how_to_confirm', 'N/A')}")
                    with dc2:
                        if st.button("Plan Interventions"):
                            with st.spinner("Generating intervention plan..."):
                                from app.core.mistral_agent import plan_interventions
                                plan = plan_interventions(row.to_dict())
                            st.markdown("**Immediate Actions (24h):**")
                            for a in plan.get("immediate_actions", []):
                                st.markdown(f"- {a}")
                            st.markdown("**Short-term Actions (7d):**")
                            for a in plan.get("short_term_actions", []):
                                st.markdown(f"- {a}")
                            st.markdown("**Monitoring Metrics:**")
                            for m in plan.get("monitoring_metrics", []):
                                st.markdown(f"- {m}")
                            st.markdown(f"**Escalation:** {plan.get('escalation_criteria', 'N/A')}")

            # Anomaly table
            display_cols = [c for c in ["severity", "direction", "district", "state", "drug", "drug_category",
                           "anomaly_type", "baseline_value", "actual_value", "z_score", "week", "date_range"]
                           if c in filtered.columns]
            st.dataframe(
                filtered[display_cols].sort_values(["severity", "week"]),
                use_container_width=True,
                height=400,
            )

            st.divider()

            # Time series chart
            st.subheader("Time Series Deep Dive")
            col_ts1, col_ts2 = st.columns(2)
            with col_ts1:
                chart_district = st.selectbox("Select district", sorted(weekly["district"].unique()))
            with col_ts2:
                district_drugs = weekly[weekly["district"] == chart_district]["drug_generic_name"].unique()
                chart_drug = st.selectbox("Select drug", sorted(district_drugs))

            ts_data = weekly[
                (weekly["district"] == chart_district) & (weekly["drug_generic_name"] == chart_drug)
            ].sort_values("week")

            anomaly_weeks = set(
                filtered[
                    (filtered["district"] == chart_district) & (filtered["drug"] == chart_drug)
                ]["week"].tolist()
            )

            if not ts_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ts_data["week"], y=ts_data["total_sold"],
                    mode="lines+markers", name="Weekly Sales",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=6),
                ))

                anom_data = ts_data[ts_data["week"].isin(anomaly_weeks)]
                if not anom_data.empty:
                    fig.add_trace(go.Scatter(
                        x=anom_data["week"], y=anom_data["total_sold"],
                        mode="markers", name="Anomaly",
                        marker=dict(color="#dc2626", size=14, symbol="diamond",
                                    line=dict(width=2, color="#fff")),
                    ))

                fig.update_layout(
                    **PLOTLY_DARK_LAYOUT,
                    title=f"{chart_drug} sales in {chart_district}",
                    xaxis_title="Week", yaxis_title="Units Sold",
                    height=400, margin=dict(t=40, b=40),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Heatmap
            st.subheader("Anomaly Heatmap: District x Drug")
            if not filtered.empty:
                heat_data = filtered.groupby(["district", "drug"]).size().reset_index(name="count")
                heat_pivot = heat_data.pivot(index="district", columns="drug", values="count").fillna(0)
                fig_heat = px.imshow(
                    heat_pivot, color_continuous_scale="YlOrRd",
                    labels=dict(color="Anomaly Count"),
                    aspect="auto",
                )
                fig_heat.update_layout(**PLOTLY_DARK_LAYOUT, height=500, margin=dict(t=20, b=20))
                st.plotly_chart(fig_heat, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Year-over-Year Comparison
# ---------------------------------------------------------------------------
elif page == "Year-over-Year":
    st.title("Year-over-Year Comparison")
    st.markdown(
        "Compare drug sales for a selected month against the **same month last year** "
        "to identify true anomalies vs. expected seasonal patterns."
    )

    if st.session_state.df is None:
        st.warning("No data loaded. Go to **Upload & Analyze** to load data first.")
    else:
        df_raw = st.session_state.df.copy()
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw["month"] = df_raw["date"].dt.month
        df_raw["year"] = df_raw["date"].dt.year

        available_years = sorted(df_raw["year"].unique())

        if len(available_years) < 2:
            st.info("Year-over-Year comparison requires data spanning at least 2 years.")
        else:
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                current_year = max(available_years)
                prev_year = current_year - 1
                months_current = set(df_raw[df_raw["year"] == current_year]["month"].unique())
                months_prev = set(df_raw[df_raw["year"] == prev_year]["month"].unique())
                common_months = sorted(months_current & months_prev)

                if not common_months:
                    st.warning("No overlapping months between years for comparison.")
                    st.stop()

                month_names = {
                    1: "January", 2: "February", 3: "March", 4: "April",
                    5: "May", 6: "June", 7: "July", 8: "August",
                    9: "September", 10: "October", 11: "November", 12: "December",
                }
                sel_month = st.selectbox("Select Month", common_months, format_func=lambda m: month_names[m])
            with ctrl2:
                all_districts = sorted(df_raw["district"].unique())
                sel_yoy_district = st.selectbox("Select District", ["All Districts"] + all_districts)
            with ctrl3:
                all_drugs = sorted(df_raw["drug_generic_name"].unique())
                sel_yoy_drug = st.selectbox("Select Drug", ["All Drugs"] + all_drugs)

            df_current = df_raw[(df_raw["year"] == current_year) & (df_raw["month"] == sel_month)]
            df_prev = df_raw[(df_raw["year"] == prev_year) & (df_raw["month"] == sel_month)]

            if sel_yoy_district != "All Districts":
                df_current = df_current[df_current["district"] == sel_yoy_district]
                df_prev = df_prev[df_prev["district"] == sel_yoy_district]
            if sel_yoy_drug != "All Drugs":
                df_current = df_current[df_current["drug_generic_name"] == sel_yoy_drug]
                df_prev = df_prev[df_prev["drug_generic_name"] == sel_yoy_drug]

            def _agg_monthly(df_slice, label):
                agg = (
                    df_slice.groupby(["drug_generic_name", "drug_category"])
                    .agg(total_sold=("quantity_sold", "sum"))
                    .reset_index()
                )
                agg["period"] = label
                return agg

            agg_current = _agg_monthly(df_current, f"{month_names[sel_month]} {current_year}")
            agg_prev = _agg_monthly(df_prev, f"{month_names[sel_month]} {prev_year}")

            st.divider()
            total_current = int(agg_current["total_sold"].sum())
            total_prev = int(agg_prev["total_sold"].sum())
            yoy_change = ((total_current - total_prev) / max(total_prev, 1)) * 100

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(metric_card(f"{month_names[sel_month]} {prev_year}", f"{total_prev:,}"), unsafe_allow_html=True)
            with mc2:
                st.markdown(metric_card(f"{month_names[sel_month]} {current_year}", f"{total_current:,}"), unsafe_allow_html=True)
            with mc3:
                color = "#dc2626" if yoy_change > 50 else "#059669" if yoy_change < 20 else "#ea580c"
                st.markdown(metric_card("YoY Change", f"{yoy_change:+.1f}%", color), unsafe_allow_html=True)
            with mc4:
                seasonal_mult = get_seasonal_multiplier(
                    sel_yoy_drug if sel_yoy_drug != "All Drugs" else "paracetamol", sel_month
                )
                label = "Expected Seasonal" if sel_yoy_drug != "All Drugs" else "Avg Seasonal"
                st.markdown(metric_card(label, f"{seasonal_mult:.1f}x"), unsafe_allow_html=True)

            st.divider()

            st.subheader(f"Drug Sales: {month_names[sel_month]} {prev_year} vs {current_year}")
            combined = pd.concat([agg_prev, agg_current], ignore_index=True)
            if not combined.empty:
                fig_yoy = px.bar(
                    combined, x="drug_generic_name", y="total_sold", color="period",
                    barmode="group",
                    labels={"drug_generic_name": "Drug", "total_sold": "Total Units Sold", "period": "Period"},
                    color_discrete_sequence=["#94a3b8", "#3b82f6"],
                )
                fig_yoy.update_layout(
                    **{k: v for k, v in PLOTLY_DARK_LAYOUT.items() if k != "legend"},
                    margin=dict(t=20, b=40), height=420, xaxis_tickangle=-45,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

            st.subheader("Per-Drug Year-over-Year Change")
            merged = agg_prev[["drug_generic_name", "drug_category", "total_sold"]].merge(
                agg_current[["drug_generic_name", "total_sold"]],
                on="drug_generic_name", how="outer",
                suffixes=(f"_{prev_year}", f"_{current_year}"),
            ).fillna(0)

            merged["yoy_change_%"] = (
                (merged[f"total_sold_{current_year}"] - merged[f"total_sold_{prev_year}"])
                / merged[f"total_sold_{prev_year}"].replace(0, 1) * 100
            ).round(1)

            merged["seasonal_multiplier"] = merged["drug_generic_name"].apply(
                lambda d: get_seasonal_multiplier(d, sel_month)
            )
            merged["above_seasonal"] = merged.apply(
                lambda row: "YES" if row["yoy_change_%"] > (row["seasonal_multiplier"] - 1) * 100 + 50 else "no",
                axis=1,
            )
            st.dataframe(merged.sort_values("yoy_change_%", ascending=False), use_container_width=True, height=400)

            if sel_yoy_district == "All Districts":
                st.divider()
                st.subheader("District-Level YoY Change Heatmap")
                dist_current = df_current.groupby(["district", "drug_generic_name"]).agg(total=("quantity_sold", "sum")).reset_index()
                dist_prev = df_prev.groupby(["district", "drug_generic_name"]).agg(total=("quantity_sold", "sum")).reset_index()
                dist_merged = dist_prev.merge(dist_current, on=["district", "drug_generic_name"], how="outer", suffixes=("_prev", "_curr")).fillna(0)
                dist_merged["yoy_pct"] = ((dist_merged["total_curr"] - dist_merged["total_prev"]) / dist_merged["total_prev"].replace(0, 1) * 100).round(1)

                if sel_yoy_drug != "All Drugs":
                    dist_merged = dist_merged[dist_merged["drug_generic_name"] == sel_yoy_drug]

                if not dist_merged.empty:
                    heat_pivot = dist_merged.pivot(index="district", columns="drug_generic_name", values="yoy_pct").fillna(0)
                    fig_heat = px.imshow(heat_pivot, color_continuous_scale="RdYlBu_r", color_continuous_midpoint=0, labels=dict(color="YoY Change %"), aspect="auto")
                    fig_heat.update_layout(**PLOTLY_DARK_LAYOUT, height=500, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_heat, use_container_width=True)

            st.divider()
            st.subheader("Seasonal Profile Reference")
            st.caption("Expected monthly sales multipliers based on historical patterns.")
            if sel_yoy_drug != "All Drugs":
                profile_drugs = [sel_yoy_drug]
            else:
                profile_drugs = sorted(SEASONAL_PROFILES.keys())[:6]

            profile_rows = []
            for drug in profile_drugs:
                for m in range(1, 13):
                    profile_rows.append({"Drug": drug, "Month": month_names[m][:3], "Month_num": m, "Multiplier": get_seasonal_multiplier(drug, m)})
            profile_df = pd.DataFrame(profile_rows)
            fig_profile = px.line(profile_df, x="Month", y="Multiplier", color="Drug", markers=True, labels={"Multiplier": "Seasonal Multiplier"})
            fig_profile.add_hline(y=1.0, line_dash="dash", line_color="rgba(148,163,184,0.4)", annotation_text="Baseline", annotation_font_color="#94a3b8")
            fig_profile.update_layout(**{k: v for k, v in PLOTLY_DARK_LAYOUT.items() if k != "legend"}, margin=dict(t=20, b=40), height=350,
                                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                                  bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")))
            st.plotly_chart(fig_profile, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Disease Map
# ---------------------------------------------------------------------------
elif page == "Disease Map":
    st.title("Disease Signal Map")
    st.markdown("Geographic visualization of district-level disease signals detected from pharmacy sales.")

    if st.session_state.pipeline_results is None:
        st.warning("Run analysis first from the Upload & Analyze page.")
    else:
        results = st.session_state.pipeline_results
        anomalies_df = pd.DataFrame(results["anomalies"])

        if anomalies_df.empty:
            st.info("No anomalies to map.")
        else:
            district_lookup = {d["district"]: d for d in DISTRICTS}

            district_stats = (
                anomalies_df.groupby("district")
                .agg(
                    anomaly_count=("district", "size"),
                    max_severity=("severity", "first"),
                    drugs=("drug", lambda x: ", ".join(sorted(set(x)))),
                    categories=("drug_category", lambda x: ", ".join(sorted(set(x)))),
                )
                .reset_index()
            )

            severity_colors = {"critical": "red", "high": "orange", "medium": "beige", "low": "blue"}

            m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")

            for _, row in district_stats.iterrows():
                dist_info = district_lookup.get(row["district"])
                if not dist_info:
                    continue

                color = severity_colors.get(row["max_severity"], "gray")
                # Scale radius with sqrt so markers stay readable (8-30px range)
                radius = max(8, min(30, 8 + math.sqrt(row["anomaly_count"]) * 1.5))

                country = dist_info.get("country", "")
                popup_html = (
                    f"<b>{html.escape(row['district'])}</b><br>"
                    f"Country: {html.escape(country)}<br>"
                    f"Anomalies: {row['anomaly_count']}<br>"
                    f"Severity: {row['max_severity'].upper()}<br>"
                    f"Drugs: {html.escape(row['drugs'])}<br>"
                    f"Categories: {html.escape(row['categories'])}"
                )

                folium.CircleMarker(
                    location=[dist_info["lat"], dist_info["lon"]],
                    radius=radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{row['district']}: {row['anomaly_count']} anomalies",
                ).add_to(m)

            legend_html = """
            <div style="position:fixed; bottom:50px; left:50px; z-index:1000;
                        background:rgba(15,23,42,0.85); backdrop-filter:blur(12px);
                        padding:12px 16px; border-radius:12px;
                        border:1px solid rgba(255,255,255,0.1); font-size:13px;
                        color:#e2e8f0; font-family:Inter,sans-serif;
                        box-shadow:0 8px 24px rgba(0,0,0,0.3);">
                <b style="color:#f8fafc;">Severity</b><br>
                <span style="color:#ef4444;">&#9679;</span> Critical<br>
                <span style="color:#f97316;">&#9679;</span> High<br>
                <span style="color:#facc15;">&#9679;</span> Medium<br>
                <span style="color:#3b82f6;">&#9679;</span> Low
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, width=None, height=600, use_container_width=True)

            # Cross-district pattern clustering
            if use_mistral and len(anomalies_df) > 1:
                st.divider()
                st.subheader("Regional Pattern Analysis")
                if st.button("Analyze Cross-District Patterns"):
                    with st.spinner("Analyzing regional patterns with Mistral AI..."):
                        from app.core.mistral_agent import cluster_district_patterns
                        top_anomalies = [a for a in results["anomalies"] if a["severity"] in ("critical", "high")][:15]
                        if top_anomalies:
                            cluster = cluster_district_patterns(top_anomalies)
                            if cluster.get("is_regional_event"):
                                st.error(f"Regional Event Detected: {cluster.get('cluster_description', '')}")
                            else:
                                st.info(f"Analysis: {cluster.get('cluster_description', 'Isolated incidents')}")
                            st.markdown(f"**Likely Cause**: {cluster.get('likely_cause', 'N/A')}")
                            st.markdown(f"**Confidence**: {cluster.get('confidence', 0):.0%}")
                            if cluster.get("affected_districts"):
                                st.markdown(f"**Affected Districts**: {', '.join(cluster['affected_districts'])}")
                            if cluster.get("supporting_evidence"):
                                st.markdown("**Evidence:**")
                                for e in cluster["supporting_evidence"]:
                                    st.markdown(f"- {e}")

            st.divider()
            st.subheader("District Details")
            st.dataframe(district_stats, use_container_width=True)

            if results["correlations"]:
                st.divider()
                st.subheader("Detected Multi-Drug Correlations")
                for corr in results["correlations"]:
                    st.markdown(
                        f'<div class="alert-card alert-{corr["severity"]}">'
                        f'<strong>{html.escape(corr["district"])}, {html.escape(corr["state"])}</strong> — '
                        f'{html.escape(corr["condition"])} {severity_badge(corr["severity"])}<br>'
                        f'Matched drugs: {html.escape(", ".join(corr["matched_drugs"]))}<br>'
                        f'Recommended: {html.escape(corr["response"])}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# PAGE: Alerts & Insights
# ---------------------------------------------------------------------------
elif page == "Alerts & Insights":
    st.title("Alerts & AI Insights")

    if st.session_state.pipeline_results is None:
        st.warning("Run analysis first from the Upload & Analyze page.")
    else:
        results = st.session_state.pipeline_results

        # Alert lifecycle header
        st.subheader("Public Health Alerts")

        if results["alerts"]:
            # Export all alerts
            all_alerts_md = ""
            for alert in results["alerts"]:
                all_alerts_md += f"## {alert['district']}, {alert['state']}\n\n"
                all_alerts_md += _get_alert_text(alert) + "\n\n---\n\n"
            st.download_button("Download All Alerts (MD)", all_alerts_md, "alerts.md", "text/markdown")

            for idx, alert in enumerate(results["alerts"]):
                sev = alert.get("max_severity", "medium")
                alert_key = f"{alert['district']}_{idx}"

                # Alert lifecycle state
                if alert_key not in st.session_state.alert_states:
                    st.session_state.alert_states[alert_key] = {"status": "New", "owner": "", "notes": ""}

                state = st.session_state.alert_states[alert_key]
                icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(sev, "⚪")

                with st.expander(
                    f"{icon} {alert['district']}, {alert['state']} — {sev.upper()} "
                    f"({alert.get('anomaly_count', 0)} anomalies) [{state['status']}]",
                    expanded=(sev in ("critical", "high")),
                ):
                    st.markdown(_get_alert_text(alert))

                    # Lifecycle controls
                    st.divider()
                    lc1, lc2, lc3 = st.columns(3)
                    with lc1:
                        new_status = st.selectbox(
                            "Status", ["New", "Investigating", "Confirmed", "Closed"],
                            index=["New", "Investigating", "Confirmed", "Closed"].index(state["status"]),
                            key=f"status_{alert_key}",
                        )
                        st.session_state.alert_states[alert_key]["status"] = new_status
                    with lc2:
                        owner = st.text_input("Owner", value=state["owner"], key=f"owner_{alert_key}")
                        st.session_state.alert_states[alert_key]["owner"] = owner
                    with lc3:
                        notes = st.text_input("Notes", value=state["notes"], key=f"notes_{alert_key}")
                        st.session_state.alert_states[alert_key]["notes"] = notes

                    # Multi-lingual translation
                    if use_mistral:
                        alert_state = alert.get("state", "")
                        languages = STATE_LANGUAGES.get(alert_state, ["Hindi"])
                        lang_choice = st.selectbox(
                            "Translate alert to",
                            ["English (Original)"] + languages,
                            key=f"lang_{alert_key}",
                        )
                        if lang_choice != "English (Original)":
                            cache_key = f"{alert_key}_{lang_choice}"
                            if cache_key not in st.session_state.translations:
                                with st.spinner(f"Translating to {lang_choice}..."):
                                    from app.core.mistral_agent import localize_alert
                                    original = _get_alert_text(alert)
                                    translated = localize_alert(original, lang_choice)
                                    st.session_state.translations[cache_key] = translated
                            st.markdown(f"**{lang_choice} Translation:**")
                            st.markdown(st.session_state.translations[cache_key])

                        # Public communication drafts
                        comms_key = f"comms_{alert_key}"
                        if st.button("Draft Public Communications", key=f"comms_btn_{alert_key}"):
                            with st.spinner("Generating audience-specific drafts..."):
                                from app.core.mistral_agent import draft_public_communications
                                alert_text = _get_alert_text(alert)
                                comms = draft_public_communications(
                                    alert_text,
                                    district=alert.get("district", "Unknown"),
                                    severity=alert.get("max_severity", "unknown"),
                                )
                                st.session_state[comms_key] = comms

                        if comms_key in st.session_state:
                            comms = st.session_state[comms_key]
                            tab_tech, tab_citizen, tab_press = st.tabs([
                                "Technical Memo", "Citizen Advisory", "Press Summary"
                            ])
                            with tab_tech:
                                st.markdown(comms.get("technical_memo", ""))
                            with tab_citizen:
                                st.markdown(comms.get("citizen_advisory", ""))
                            with tab_press:
                                st.markdown(comms.get("press_summary", ""))
                            # Download all drafts
                            combined = (
                                "# Technical Memo\n\n" + comms.get("technical_memo", "") +
                                "\n\n---\n\n# Citizen Advisory\n\n" + comms.get("citizen_advisory", "") +
                                "\n\n---\n\n# Press Summary\n\n" + comms.get("press_summary", "")
                            )
                            st.download_button(
                                "Download All Drafts (MD)",
                                combined,
                                file_name=f"communications_{alert.get('district', 'alert')}.md",
                                mime="text/markdown",
                                key=f"dl_comms_{alert_key}",
                            )

            # Alert lifecycle summary
            st.divider()
            st.subheader("Alert Status Board")
            status_counts = {}
            for s in st.session_state.alert_states.values():
                status_counts[s["status"]] = status_counts.get(s["status"], 0) + 1
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("New", status_counts.get("New", 0))
            sc2.metric("Investigating", status_counts.get("Investigating", 0))
            sc3.metric("Confirmed", status_counts.get("Confirmed", 0))
            sc4.metric("Closed", status_counts.get("Closed", 0))

        else:
            st.info("No alerts generated. Enable Mistral AI in the sidebar and re-run analysis.")

        st.divider()

        # Insights section
        st.subheader("Mistral AI Interpretations")
        if results["insights"]:
            for insight in results["insights"]:
                a = insight["anomaly"]
                with st.expander(f"{a['drug']} in {a['district']} — {a['severity'].upper()} (Week {a['week']})"):
                    st.markdown(f"**Baseline:** {a['baseline_value']} → **Actual:** {a['actual_value']} units/week")
                    if a.get("z_score"):
                        st.markdown(f"**Z-Score:** {a['z_score']}")

                    interp = insight.get("interpretation", {})
                    if isinstance(interp, dict) and "confidence" in interp:
                        st.progress(interp["confidence"], text=f"Confidence: {interp['confidence']:.0%}")
                        st.markdown(f"**Likely Condition**: {interp.get('likely_condition', 'N/A')}")
                        st.markdown(f"**Severity**: {interp.get('severity_assessment', 'N/A')}")
                        if interp.get("possible_causes"):
                            st.markdown("**Possible Causes:**")
                            for c in interp["possible_causes"]:
                                st.markdown(f"- {c}")
                        if interp.get("recommended_actions"):
                            st.markdown("**Recommended Actions:**")
                            for i, action in enumerate(interp["recommended_actions"], 1):
                                st.checkbox(action, key=f"action_{a['drug']}_{a['district']}_{i}")
                        if interp.get("additional_context"):
                            st.markdown(f"**Context**: {interp['additional_context']}")
                    else:
                        st.markdown(_get_interpretation_content(insight))
        else:
            st.info("No AI interpretations available. Enable Mistral AI and re-run analysis.")

        # On-demand interpretation with streaming
        st.divider()
        st.subheader("On-Demand Analysis")
        anomalies_list = results.get("anomalies", [])
        if anomalies_list and use_mistral:
            selected_idx = st.selectbox(
                "Select an anomaly to interpret",
                range(len(anomalies_list)),
                format_func=lambda i: (
                    f"{anomalies_list[i]['drug']} in {anomalies_list[i]['district']} "
                    f"(week {anomalies_list[i]['week']}, {anomalies_list[i]['severity']})"
                ),
            )
            if st.button("Get Mistral Interpretation (Streaming)"):
                from app.core.mistral_agent import interpret_anomaly_stream
                st.write_stream(interpret_anomaly_stream(anomalies_list[selected_idx]))

            if len(anomalies_list) > 1:
                st.divider()
                if st.button("Correlate All Signals (Streaming)"):
                    from app.core.mistral_agent import correlate_signals_stream
                    top_10 = [a for a in anomalies_list if a["severity"] in ("critical", "high")][:10]
                    st.write_stream(correlate_signals_stream(top_10))
        elif not use_mistral:
            st.info("Enable Mistral AI in the sidebar for on-demand analysis.")


# ---------------------------------------------------------------------------
# PAGE: AI Assistant (Function Calling)
# ---------------------------------------------------------------------------
elif page == "AI Assistant":
    st.title("AI Surveillance Assistant")
    st.markdown("Ask questions about the surveillance data in natural language. Powered by Mistral function calling.")

    if st.session_state.pipeline_results is None:
        st.warning("Run analysis first from the Upload & Analyze page.")
    elif not use_mistral:
        st.info("Enable Mistral AI in the sidebar to use the assistant.")
    else:
        results = st.session_state.pipeline_results

        # Display chat history
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            elif msg["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])

        # Chat input
        user_query = st.chat_input("Ask about the surveillance data...")

        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    from app.core.mistral_agent import query_assistant, query_assistant_followup

                    response_text, tool_calls = query_assistant(
                        user_query, st.session_state.chat_history
                    )

                    if tool_calls:
                        # Execute tool calls locally
                        tool_results = []
                        for tc in tool_calls:
                            args = tc["arguments"]
                            if tc["name"] == "filter_anomalies":
                                filtered = results["anomalies"]
                                if args.get("district"):
                                    filtered = [a for a in filtered if a["district"].lower() == args["district"].lower()]
                                if args.get("drug"):
                                    filtered = [a for a in filtered if a["drug"].lower() == args["drug"].lower()]
                                if args.get("severity"):
                                    filtered = [a for a in filtered if a["severity"] == args["severity"]]
                                if args.get("min_week"):
                                    filtered = [a for a in filtered if a["week"] >= args["min_week"]]
                                if args.get("max_week"):
                                    filtered = [a for a in filtered if a["week"] <= args["max_week"]]
                                result = {"count": len(filtered), "anomalies": filtered[:20]}
                            elif tc["name"] == "get_district_risk":
                                risk = results.get("district_risk", [])
                                if args.get("district"):
                                    risk = [r for r in risk if r["district"].lower() == args["district"].lower()]
                                result = {"districts": risk[:10]}
                            elif tc["name"] == "get_correlations":
                                corrs = results["correlations"]
                                if args.get("district"):
                                    corrs = [c for c in corrs if c["district"].lower() == args["district"].lower()]
                                result = {"correlations": corrs}
                            elif tc["name"] == "get_summary":
                                result = results["summary"]
                            else:
                                result = {"error": f"Unknown tool: {tc['name']}"}

                            tool_results.append({"id": tc["id"], "name": tc["name"], "result": result})

                        # Build the assistant message object for followup
                        from mistralai.models import AssistantMessage, ToolCall, FunctionCall
                        assistant_msg = AssistantMessage(
                            role="assistant",
                            content=response_text or "",
                            tool_calls=[
                                ToolCall(
                                    id=tc["id"],
                                    function=FunctionCall(name=tc["name"], arguments=json.dumps(tc["arguments"])),
                                )
                                for tc in tool_calls
                            ],
                        )

                        final_response = query_assistant_followup(
                            st.session_state.chat_history,
                            tool_results,
                            assistant_msg,
                        )
                        st.markdown(final_response)
                        st.session_state.chat_history.append({"role": "user", "content": user_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": final_response})
                    else:
                        st.markdown(response_text)
                        st.session_state.chat_history.append({"role": "user", "content": user_query})
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()


# ---------------------------------------------------------------------------
# PAGE: Scenario Simulator
# ---------------------------------------------------------------------------
elif page == "Scenario Simulator":
    st.title("Scenario Simulator")
    st.markdown("Simulate 'what-if' scenarios by injecting synthetic drug demand changes and comparing results.")

    if st.session_state.df is None:
        st.warning("No data loaded. Go to **Upload & Analyze** to load data first.")
    elif st.session_state.pipeline_results is None:
        st.warning("Run analysis first from the Upload & Analyze page.")
    else:
        results = st.session_state.pipeline_results
        df_base = st.session_state.df.copy()

        st.subheader("Configure Scenario")
        sc1, sc2 = st.columns(2)
        with sc1:
            sim_districts = st.multiselect(
                "Target Districts",
                sorted(df_base["district"].unique()),
                default=["Bengaluru Urban"],
            )
            sim_drugs = st.multiselect(
                "Target Drug Category",
                sorted(df_base["drug_category"].unique()),
                default=["respiratory"],
            )
        with sc2:
            sim_multiplier = st.slider("Demand Multiplier", 1.0, 10.0, 4.0, 0.5)
            sim_weeks = st.slider("Duration (weeks)", 1, 8, 3)
            max_week = int(df_base["date"].apply(lambda d: pd.Timestamp(d)).max().timestamp() / (7 * 86400))
            sim_start_week = st.slider("Start Week (2024)", 60, 78, 70)

        if st.button("Run Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                df_sim = df_base.copy()
                df_sim["date"] = pd.to_datetime(df_sim["date"])
                min_date = df_sim["date"].min()
                df_sim["week"] = ((df_sim["date"] - min_date).dt.days // 7).astype(int)

                # Apply simulation: multiply quantity_sold for target rows
                mask = (
                    df_sim["district"].isin(sim_districts) &
                    df_sim["drug_category"].isin(sim_drugs) &
                    (df_sim["week"] >= sim_start_week) &
                    (df_sim["week"] < sim_start_week + sim_weeks)
                )
                df_sim.loc[mask, "quantity_sold"] = (df_sim.loc[mask, "quantity_sold"] * sim_multiplier).astype(int)

                sim_results = run_pipeline(df_sim, use_mistral=False, seasonal_adjust=seasonal_adjust)

            # Side-by-side comparison
            st.divider()
            st.subheader("Comparison: Original vs Simulated")

            original = results["summary"]
            simulated = sim_results["summary"]

            cc1, cc2, cc3, cc4 = st.columns(4)
            with cc1:
                delta = simulated["total_anomalies"] - original["total_anomalies"]
                st.markdown(metric_card("Anomalies", simulated["total_anomalies"], "#dc2626", f"{delta:+d}"), unsafe_allow_html=True)
            with cc2:
                delta = simulated["districts_affected"] - original["districts_affected"]
                st.markdown(metric_card("Districts Affected", simulated["districts_affected"], "#ea580c", f"{delta:+d}"), unsafe_allow_html=True)
            with cc3:
                delta = simulated["correlations_found"] - original["correlations_found"]
                st.markdown(metric_card("Correlations", simulated["correlations_found"], "#7c3aed", f"{delta:+d}"), unsafe_allow_html=True)
            with cc4:
                orig_crit = original["severity_breakdown"].get("critical", 0)
                sim_crit = simulated["severity_breakdown"].get("critical", 0)
                st.markdown(metric_card("Critical Alerts", sim_crit, "#dc2626", f"{sim_crit - orig_crit:+d}"), unsafe_allow_html=True)

            # Severity comparison chart
            sev_data = []
            for sev in ["critical", "high", "medium", "low"]:
                sev_data.append({"Severity": sev.capitalize(), "Count": original["severity_breakdown"].get(sev, 0), "Scenario": "Original"})
                sev_data.append({"Severity": sev.capitalize(), "Count": simulated["severity_breakdown"].get(sev, 0), "Scenario": "Simulated"})
            fig_comp = px.bar(
                pd.DataFrame(sev_data), x="Severity", y="Count", color="Scenario",
                barmode="group", color_discrete_sequence=["#94a3b8", "#dc2626"],
            )
            fig_comp.update_layout(**PLOTLY_DARK_LAYOUT, margin=dict(t=20, b=40), height=350)
            st.plotly_chart(fig_comp, use_container_width=True)

            # New anomalies in simulated
            orig_keys = {(a["district"], a["drug"], a["week"]) for a in results["anomalies"]}
            new_anomalies = [a for a in sim_results["anomalies"] if (a["district"], a["drug"], a["week"]) not in orig_keys]
            if new_anomalies:
                st.subheader(f"New Anomalies from Simulation ({len(new_anomalies)})")
                new_df = pd.DataFrame(new_anomalies)
                display_cols = [c for c in ["severity", "direction", "district", "drug", "anomaly_type", "baseline_value", "actual_value", "week"] if c in new_df.columns]
                st.dataframe(new_df[display_cols], use_container_width=True, height=300)

            # Mistral simulation explanation
            if use_mistral:
                st.divider()
                if st.button("Explain Simulation Impact"):
                    with st.spinner("Generating AI explanation..."):
                        from app.core.mistral_agent import explain_simulation
                        scenario_desc = (
                            f"{sim_multiplier}x demand increase in {', '.join(sim_districts)} "
                            f"for {', '.join(sim_drugs)} drugs over {sim_weeks} weeks starting week {sim_start_week}"
                        )
                        explanation = explain_simulation(original, simulated, scenario_desc)

                    st.markdown(f"**Impact Summary**: {explanation.get('impact_summary', 'N/A')}")
                    if explanation.get("key_changes"):
                        st.markdown("**Key Changes:**")
                        for c in explanation["key_changes"]:
                            st.markdown(f"- {c}")
                    if explanation.get("new_risk_areas"):
                        st.markdown("**New Risk Areas:**")
                        for r in explanation["new_risk_areas"]:
                            st.markdown(f"- {r}")
                    if explanation.get("monitoring_recommendations"):
                        st.markdown("**Monitoring Recommendations:**")
                        for m in explanation["monitoring_recommendations"]:
                            st.markdown(f"- {m}")


# ---------------------------------------------------------------------------
# PAGE: Evaluation
# ---------------------------------------------------------------------------
elif page == "Evaluation":
    st.title("Detection Evaluation")
    st.markdown(
        "Evaluate anomaly detection accuracy against the 4 known ground truth scenarios "
        "injected into the synthetic data."
    )

    if st.session_state.pipeline_results is None:
        st.warning("Run analysis first from the Upload & Analyze page.")
    else:
        results = st.session_state.pipeline_results
        from app.core.evaluation import evaluate_detections, GROUND_TRUTH_EVENTS

        eval_results = evaluate_detections(results["anomalies"])

        # Overall metrics
        st.subheader("Overall Detection Performance")
        overall = eval_results["overall"]
        oc1, oc2, oc3, oc4 = st.columns(4)
        with oc1:
            st.markdown(metric_card("Precision", f"{overall['precision']:.1%}", "#059669"), unsafe_allow_html=True)
        with oc2:
            st.markdown(metric_card("Recall", f"{overall['recall']:.1%}", "#7c3aed"), unsafe_allow_html=True)
        with oc3:
            st.markdown(metric_card("F1 Score", f"{overall['f1']:.1%}", "#ea580c"), unsafe_allow_html=True)
        with oc4:
            st.markdown(metric_card("Total Detected", overall["total_detected"], "#dc2626"), unsafe_allow_html=True)

        st.divider()

        # Per-event breakdown
        st.subheader("Per-Scenario Breakdown")
        for event_result in eval_results["events"]:
            with st.expander(
                f"{event_result['event']} — F1: {event_result['f1']:.0%} "
                f"(P: {event_result['precision']:.0%}, R: {event_result['recall']:.0%})",
                expanded=True,
            ):
                st.caption(event_result["scenario"])

                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown(metric_card("Expected Signals", event_result["expected_signals"]), unsafe_allow_html=True)
                with ec2:
                    st.markdown(metric_card("Detected Signals", event_result["detected_signals"], "#7c3aed"), unsafe_allow_html=True)
                with ec3:
                    st.markdown(metric_card("True Positives", event_result["true_positives"], "#059669"), unsafe_allow_html=True)

                if event_result["precision"] >= 0.8 and event_result["recall"] >= 0.8:
                    st.success(f"Detection quality: Excellent (F1 = {event_result['f1']:.0%})")
                elif event_result["recall"] >= 0.5:
                    st.warning(f"Detection quality: Partial (F1 = {event_result['f1']:.0%})")
                else:
                    st.error(f"Detection quality: Missed (F1 = {event_result['f1']:.0%})")

                if event_result["missed_details"]:
                    st.markdown("**Missed signals:**")
                    for d, drug in event_result["missed_details"]:
                        st.markdown(f"- {d}: {drug}")

        # Confusion matrix visualization
        st.divider()
        st.subheader("Detection Summary")
        summary_data = pd.DataFrame([
            {
                "Scenario": e["event"],
                "Expected": e["expected_signals"],
                "Detected": e["detected_signals"],
                "TP": e["true_positives"],
                "FN": e["false_negatives"],
                "Precision": f"{e['precision']:.0%}",
                "Recall": f"{e['recall']:.0%}",
                "F1": f"{e['f1']:.0%}",
            }
            for e in eval_results["events"]
        ])
        st.dataframe(summary_data, use_container_width=True)

        # Visualization
        fig_eval = px.bar(
            summary_data, x="Scenario", y=["Expected", "Detected", "TP"],
            barmode="group",
            labels={"value": "Signal Count", "variable": "Type"},
            color_discrete_sequence=["#94a3b8", "#3b82f6", "#059669"],
        )
        fig_eval.update_layout(**PLOTLY_DARK_LAYOUT, margin=dict(t=20, b=40), height=350)
        st.plotly_chart(fig_eval, use_container_width=True)
