"""
Pharma Surveillance System — Streamlit Dashboard

Real-time epidemiological surveillance using pharmacy drug sales data
as a proxy to detect disease outbreaks, pollution health effects, and health patterns.
"""

import sys
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
from app.core.mappings import DISTRICTS, SEASONAL_PROFILES, get_seasonal_multiplier
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
    .main .block-container { padding-top: 1rem; }
    .severity-critical { color: #fff; background: #dc2626; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85em; }
    .severity-high { color: #fff; background: #ea580c; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85em; }
    .severity-medium { color: #000; background: #facc15; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85em; }
    .severity-low { color: #fff; background: #2563eb; padding: 2px 10px; border-radius: 12px; font-weight: 600; font-size: 0.85em; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.2rem; text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #0f172a; }
    .metric-card p { margin: 0; color: #64748b; font-size: 0.9rem; }
    .alert-card {
        border-left: 4px solid; padding: 1rem; margin-bottom: 0.8rem;
        border-radius: 0 8px 8px 0; background: #f8fafc;
    }
    .alert-critical { border-color: #dc2626; }
    .alert-high { border-color: #ea580c; }
    .alert-medium { border-color: #facc15; }
    .alert-low { border-color: #2563eb; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = None


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
        ["Dashboard", "Upload & Analyze", "Anomaly Explorer", "Year-over-Year", "Disease Map", "Alerts & Insights"],
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
        help="Exclude expected seasonal drug spikes from anomaly detection. "
             "When enabled, drugs that normally sell high in the current month "
             "(e.g. flu drugs in winter) won't be flagged as anomalies.",
    )

    st.divider()
    st.caption("Built for Mistral Hackathon 2025")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def severity_badge(severity: str) -> str:
    return f'<span class="severity-{severity}">{severity.upper()}</span>'


def render_validation(report: dict) -> None:
    """Render validation diagnostics from the pipeline."""
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


def metric_card(label: str, value, color: str = "#0f172a") -> str:
    return f"""
    <div class="metric-card">
        <h2 style="color: {color};">{value}</h2>
        <p>{label}</p>
    </div>
    """


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
                with st.spinner("Generating 18 months of global pharmacy data (India, USA, Europe)..."):
                    df = generate()
                    st.session_state.df = df
                with st.spinner("Running anomaly detection pipeline..."):
                    results = run_pipeline(df, use_mistral=use_mistral, seasonal_adjust=seasonal_adjust)
                    st.session_state.pipeline_results = results
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

        st.divider()

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
                fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280)
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
                    sev = alert["max_severity"]
                    st.markdown(
                        f'<div class="alert-card alert-{sev}">'
                        f'<strong>{alert["district"]}, {alert["state"]}</strong> '
                        f'{severity_badge(sev)}<br>'
                        f'{alert["alert_text"][:300]}...'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No alerts generated. Enable Mistral AI or run analysis first.")

        # Quick anomaly time series
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
            fig.update_layout(margin=dict(t=20, b=40), height=300, bargap=0.1)
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
                with st.spinner("Running anomaly detection pipeline..."):
                    results = run_pipeline(df, use_mistral=use_mistral, seasonal_adjust=seasonal_adjust)
                    st.session_state.pipeline_results = results
                st.success(f"Found {results['summary']['total_anomalies']} anomalies across {results['summary']['districts_affected']} districts")
                render_validation(results["summary"].get("validation", {}))
                st.rerun()

    with tab2:
        st.markdown(
            "Generate 18 months (Jan 2023 – Jun 2024) of synthetic pharmacy sales data across "
            "**44 districts** in **India, USA, and Europe** "
            "with **seasonal variation** and **10 injected anomaly scenarios** in 2024:"
        )
        st.markdown("""
        **India:**
        1. **Delhi Respiratory Spike** (Weeks 18-20) — Air pollution event
        2. **Chennai Waterborne Outbreak** (Week 12) — Water contamination
        3. **Pune Flu Cluster** (Weeks 8-10) — Influenza wave
        4. **Vizag Thyroid Anomaly** (Months 3-6) — Industrial pollution

        **USA:**
        5. **NYC Flu Outbreak** (Weeks 4-6) — Severe winter influenza
        6. **Houston GI Outbreak** (Weeks 14-15) — Post-flood water contamination
        7. **LA Respiratory Spike** (Weeks 20-22) — Wildfire smoke event

        **Europe:**
        8. **London-Paris Flu Wave** (Weeks 6-8) — Cross-border influenza
        9. **Berlin-Munich Respiratory Cluster** (Weeks 10-12) — Industrial pollution
        10. **Madrid-Rome Diabetes Trend** (Weeks 5-20) — Post-holiday metabolic rise

        2023 data serves as a clean seasonal baseline for Year-over-Year comparison.
        """)

        if st.button("Generate & Analyze", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic data (2023-2024)..."):
                df = generate()
                st.session_state.df = df
            st.success(f"Generated {len(df):,} pharmacy sales records")

            with st.spinner("Running anomaly detection pipeline..."):
                results = run_pipeline(df, use_mistral=use_mistral, seasonal_adjust=seasonal_adjust)
                st.session_state.pipeline_results = results
            st.success(
                f"Analysis complete: {results['summary']['total_anomalies']} anomalies, "
                f"{results['summary']['districts_affected']} districts affected"
            )
            render_validation(results["summary"].get("validation", {}))
            st.rerun()

    # Show current data stats if loaded
    if st.session_state.df is not None:
        st.divider()
        st.subheader("Current Dataset")
        df = st.session_state.df
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Districts", df["district"].nunique())
        c3.metric("Drugs", df["drug_generic_name"].nunique())
        st.dataframe(df.head(10), use_container_width=True)


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
            col1, col2, col3 = st.columns(3)
            with col1:
                sel_districts = st.multiselect(
                    "District", options=sorted(anomalies_df["district"].unique()), default=[]
                )
            with col2:
                sel_drugs = st.multiselect(
                    "Drug", options=sorted(anomalies_df["drug"].unique()), default=[]
                )
            with col3:
                sel_severity = st.multiselect(
                    "Severity", options=["critical", "high", "medium", "low"], default=[]
                )

            filtered = anomalies_df.copy()
            if sel_districts:
                filtered = filtered[filtered["district"].isin(sel_districts)]
            if sel_drugs:
                filtered = filtered[filtered["drug"].isin(sel_drugs)]
            if sel_severity:
                filtered = filtered[filtered["severity"].isin(sel_severity)]

            st.markdown(f"**{len(filtered)} anomalies** matching filters")

            if not filtered.empty:
                st.subheader("Why flagged?")
                selected_row = st.selectbox(
                    "Select anomaly",
                    range(len(filtered)),
                    format_func=lambda i: (
                        f"{filtered.iloc[i]['drug']} | {filtered.iloc[i]['district']} | "
                        f"week {filtered.iloc[i]['week']} | {filtered.iloc[i]['severity']}"
                    ),
                )
                row = filtered.iloc[selected_row]
                st.markdown(
                    f"**Method triggered:** `{row['anomaly_type']}`  \n"
                    f"**Baseline:** {row['baseline_value']} units/week  \n"
                    f"**Actual:** {row['actual_value']} units/week  \n"
                    f"**Z-score:** {row.get('z_score', 'N/A')}  \n"
                    f"**% change:** {row.get('pct_change', 'N/A')}  \n"
                    f"**Confidence:** {row.get('confidence', 'medium')}"
                )

            # Anomaly table
            display_cols = ["severity", "district", "state", "drug", "drug_category",
                           "anomaly_type", "baseline_value", "actual_value", "z_score", "week", "date_range"]
            st.dataframe(
                filtered[display_cols].sort_values(["severity", "week"]),
                use_container_width=True,
                height=400,
            )

            st.divider()

            # Time series chart for selected district+drug
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

            # Get anomaly weeks for this combo
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

                # Highlight anomaly points
                anom_data = ts_data[ts_data["week"].isin(anomaly_weeks)]
                if not anom_data.empty:
                    fig.add_trace(go.Scatter(
                        x=anom_data["week"], y=anom_data["total_sold"],
                        mode="markers", name="Anomaly",
                        marker=dict(color="#dc2626", size=14, symbol="diamond",
                                    line=dict(width=2, color="#fff")),
                    ))

                fig.update_layout(
                    title=f"{chart_drug} sales in {chart_district}",
                    xaxis_title="Week", yaxis_title="Units Sold",
                    height=400, margin=dict(t=40, b=40),
                    hovermode="x unified",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Anomaly heatmap
            st.subheader("Anomaly Heatmap: District × Drug")
            if not filtered.empty:
                heat_data = filtered.groupby(["district", "drug"]).size().reset_index(name="count")
                heat_pivot = heat_data.pivot(index="district", columns="drug", values="count").fillna(0)
                fig_heat = px.imshow(
                    heat_pivot, color_continuous_scale="YlOrRd",
                    labels=dict(color="Anomaly Count"),
                    aspect="auto",
                )
                fig_heat.update_layout(height=500, margin=dict(t=20, b=20))
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
        df_raw["month_name"] = df_raw["date"].dt.strftime("%B")

        available_years = sorted(df_raw["year"].unique())

        if len(available_years) < 2:
            st.info(
                "Year-over-Year comparison requires data spanning at least 2 years. "
                "Generate synthetic data (which covers 2023-2024) to use this feature."
            )
        else:
            # Controls
            ctrl1, ctrl2, ctrl3 = st.columns(3)
            with ctrl1:
                # Only show months that exist in both years
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
                sel_month = st.selectbox(
                    "Select Month",
                    common_months,
                    format_func=lambda m: month_names[m],
                )
            with ctrl2:
                all_districts = sorted(df_raw["district"].unique())
                sel_yoy_district = st.selectbox("Select District", ["All Districts"] + all_districts)
            with ctrl3:
                all_drugs = sorted(df_raw["drug_generic_name"].unique())
                sel_yoy_drug = st.selectbox("Select Drug", ["All Drugs"] + all_drugs)

            # Filter data
            df_current = df_raw[(df_raw["year"] == current_year) & (df_raw["month"] == sel_month)]
            df_prev = df_raw[(df_raw["year"] == prev_year) & (df_raw["month"] == sel_month)]

            if sel_yoy_district != "All Districts":
                df_current = df_current[df_current["district"] == sel_yoy_district]
                df_prev = df_prev[df_prev["district"] == sel_yoy_district]
            if sel_yoy_drug != "All Drugs":
                df_current = df_current[df_current["drug_generic_name"] == sel_yoy_drug]
                df_prev = df_prev[df_prev["drug_generic_name"] == sel_yoy_drug]

            # Aggregate monthly totals
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

            # Summary metrics
            st.divider()
            total_current = int(agg_current["total_sold"].sum())
            total_prev = int(agg_prev["total_sold"].sum())
            yoy_change = ((total_current - total_prev) / max(total_prev, 1)) * 100

            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.markdown(
                    metric_card(f"{month_names[sel_month]} {prev_year}", f"{total_prev:,}"),
                    unsafe_allow_html=True,
                )
            with mc2:
                st.markdown(
                    metric_card(f"{month_names[sel_month]} {current_year}", f"{total_current:,}"),
                    unsafe_allow_html=True,
                )
            with mc3:
                color = "#dc2626" if yoy_change > 50 else "#059669" if yoy_change < 20 else "#ea580c"
                st.markdown(
                    metric_card("YoY Change", f"{yoy_change:+.1f}%", color),
                    unsafe_allow_html=True,
                )
            with mc4:
                seasonal_mult = get_seasonal_multiplier(
                    sel_yoy_drug if sel_yoy_drug != "All Drugs" else "paracetamol", sel_month
                )
                label = "Expected Seasonal" if sel_yoy_drug != "All Drugs" else "Avg Seasonal"
                st.markdown(
                    metric_card(label, f"{seasonal_mult:.1f}x"),
                    unsafe_allow_html=True,
                )

            st.divider()

            # Side-by-side bar chart: current year vs previous year by drug
            st.subheader(f"Drug Sales: {month_names[sel_month]} {prev_year} vs {current_year}")
            combined = pd.concat([agg_prev, agg_current], ignore_index=True)
            if not combined.empty:
                fig_yoy = px.bar(
                    combined,
                    x="drug_generic_name",
                    y="total_sold",
                    color="period",
                    barmode="group",
                    labels={"drug_generic_name": "Drug", "total_sold": "Total Units Sold", "period": "Period"},
                    color_discrete_sequence=["#94a3b8", "#3b82f6"],
                )
                fig_yoy.update_layout(
                    margin=dict(t=20, b=40), height=420,
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

            # Per-drug YoY change table
            st.subheader("Per-Drug Year-over-Year Change")
            merged = agg_prev[["drug_generic_name", "drug_category", "total_sold"]].merge(
                agg_current[["drug_generic_name", "total_sold"]],
                on="drug_generic_name",
                how="outer",
                suffixes=(f"_{prev_year}", f"_{current_year}"),
            ).fillna(0)

            merged["yoy_change_%"] = (
                (merged[f"total_sold_{current_year}"] - merged[f"total_sold_{prev_year}"])
                / merged[f"total_sold_{prev_year}"].replace(0, 1)
                * 100
            ).round(1)

            # Add seasonal context
            merged["seasonal_multiplier"] = merged["drug_generic_name"].apply(
                lambda d: get_seasonal_multiplier(d, sel_month)
            )
            merged["above_seasonal"] = merged.apply(
                lambda row: "YES" if row["yoy_change_%"] > (row["seasonal_multiplier"] - 1) * 100 + 50 else "no",
                axis=1,
            )

            st.dataframe(
                merged.sort_values("yoy_change_%", ascending=False),
                use_container_width=True,
                height=400,
            )

            # District-level YoY heatmap (if not filtered to single district)
            if sel_yoy_district == "All Districts":
                st.divider()
                st.subheader("District-Level YoY Change Heatmap")

                dist_current = (
                    df_current.groupby(["district", "drug_generic_name"])
                    .agg(total=("quantity_sold", "sum"))
                    .reset_index()
                )
                dist_prev = (
                    df_prev.groupby(["district", "drug_generic_name"])
                    .agg(total=("quantity_sold", "sum"))
                    .reset_index()
                )
                dist_merged = dist_prev.merge(
                    dist_current,
                    on=["district", "drug_generic_name"],
                    how="outer",
                    suffixes=("_prev", "_curr"),
                ).fillna(0)

                dist_merged["yoy_pct"] = (
                    (dist_merged["total_curr"] - dist_merged["total_prev"])
                    / dist_merged["total_prev"].replace(0, 1)
                    * 100
                ).round(1)

                if sel_yoy_drug != "All Drugs":
                    dist_merged = dist_merged[dist_merged["drug_generic_name"] == sel_yoy_drug]

                if not dist_merged.empty:
                    heat_pivot = dist_merged.pivot(
                        index="district", columns="drug_generic_name", values="yoy_pct"
                    ).fillna(0)
                    fig_heat = px.imshow(
                        heat_pivot,
                        color_continuous_scale="RdYlBu_r",
                        color_continuous_midpoint=0,
                        labels=dict(color="YoY Change %"),
                        aspect="auto",
                    )
                    fig_heat.update_layout(height=500, margin=dict(t=20, b=20))
                    st.plotly_chart(fig_heat, use_container_width=True)

            # Seasonal profile visualization
            st.divider()
            st.subheader("Seasonal Profile Reference")
            st.caption(
                "Expected monthly sales multipliers based on historical patterns. "
                "A multiplier of 1.8x means sales are expected to be 80% higher than average in that month."
            )
            if sel_yoy_drug != "All Drugs":
                profile_drugs = [sel_yoy_drug]
            else:
                profile_drugs = sorted(SEASONAL_PROFILES.keys())[:6]

            profile_rows = []
            for drug in profile_drugs:
                for m in range(1, 13):
                    profile_rows.append({
                        "Drug": drug,
                        "Month": month_names[m][:3],
                        "Month_num": m,
                        "Multiplier": get_seasonal_multiplier(drug, m),
                    })
            profile_df = pd.DataFrame(profile_rows)
            fig_profile = px.line(
                profile_df,
                x="Month",
                y="Multiplier",
                color="Drug",
                markers=True,
                labels={"Multiplier": "Seasonal Multiplier"},
            )
            fig_profile.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Baseline")
            fig_profile.update_layout(
                margin=dict(t=20, b=40), height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
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
            # Build district-level aggregation
            district_lookup = {d["district"]: d for d in DISTRICTS}

            district_stats = (
                anomalies_df.groupby("district")
                .agg(
                    anomaly_count=("district", "size"),
                    max_severity=("severity", "first"),  # already sorted by severity
                    drugs=("drug", lambda x: ", ".join(sorted(set(x)))),
                    categories=("drug_category", lambda x: ", ".join(sorted(set(x)))),
                )
                .reset_index()
            )

            # Add country info to district_stats for filtering
            district_stats["country"] = district_stats["district"].apply(
                lambda d: district_lookup.get(d, {}).get("country", "Unknown")
            )

            # Region filter
            available_regions = sorted(district_stats["country"].unique())
            map_col1, map_col2 = st.columns([1, 3])
            with map_col1:
                sel_region = st.selectbox(
                    "Region",
                    ["All Regions"] + available_regions,
                )

            if sel_region != "All Regions":
                district_stats = district_stats[district_stats["country"] == sel_region]

            # Color by severity
            severity_colors = {
                "critical": "red",
                "high": "orange",
                "medium": "beige",
                "low": "blue",
            }

            # Compute map center and zoom from the districts being shown
            visible_coords = []
            for _, row in district_stats.iterrows():
                dist_info = district_lookup.get(row["district"])
                if dist_info:
                    visible_coords.append((dist_info["lat"], dist_info["lon"]))

            if visible_coords:
                lats = [c[0] for c in visible_coords]
                lons = [c[1] for c in visible_coords]
                center_lat = (min(lats) + max(lats)) / 2
                center_lon = (min(lons) + max(lons)) / 2
                # Estimate zoom from span
                lat_span = max(lats) - min(lats)
                lon_span = max(lons) - min(lons)
                span = max(lat_span, lon_span, 1)
                if span > 100:
                    zoom = 2
                elif span > 40:
                    zoom = 3
                elif span > 20:
                    zoom = 4
                elif span > 10:
                    zoom = 5
                else:
                    zoom = 6
            else:
                center_lat, center_lon, zoom = 20.0, 0.0, 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, tiles="CartoDB positron")

            for _, row in district_stats.iterrows():
                dist_info = district_lookup.get(row["district"])
                if not dist_info:
                    continue

                color = severity_colors.get(row["max_severity"], "gray")
                radius = max(8, row["anomaly_count"] * 2)
                country = dist_info.get("country", "")

                popup_html = (
                    f"<b>{row['district']}</b> ({country})<br>"
                    f"State/Region: {dist_info.get('state', '')}<br>"
                    f"Anomalies: {row['anomaly_count']}<br>"
                    f"Severity: {row['max_severity'].upper()}<br>"
                    f"Drugs: {row['drugs']}<br>"
                    f"Categories: {row['categories']}"
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

            # Legend
            legend_html = """
            <div style="position:fixed; bottom:50px; left:50px; z-index:1000;
                        background:white; padding:10px; border-radius:8px;
                        border:1px solid #ccc; font-size:13px;">
                <b>Severity</b><br>
                <span style="color:red;">●</span> Critical<br>
                <span style="color:orange;">●</span> High<br>
                <span style="color:#c8b900;">●</span> Medium<br>
                <span style="color:blue;">●</span> Low
            </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))

            st_folium(m, width=None, height=600, use_container_width=True)

            # District detail table
            st.divider()
            st.subheader("District Details")
            st.dataframe(district_stats, use_container_width=True)

            # Correlations on map
            if results["correlations"]:
                st.divider()
                st.subheader("Detected Multi-Drug Correlations")
                for corr in results["correlations"]:
                    st.markdown(
                        f'<div class="alert-card alert-{corr["severity"]}">'
                        f'<strong>{corr["district"]}, {corr["state"]}</strong> — '
                        f'{corr["condition"]} {severity_badge(corr["severity"])}<br>'
                        f'Matched drugs: {", ".join(corr["matched_drugs"])}<br>'
                        f'Recommended: {corr["response"]}'
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

        # Alerts section
        st.subheader("Public Health Alerts")
        if results["alerts"]:
            for alert in results["alerts"]:
                sev = alert["max_severity"]
                with st.expander(
                    f"{'🔴' if sev == 'critical' else '🟠' if sev == 'high' else '🟡' if sev == 'medium' else '🔵'} "
                    f"{alert['district']}, {alert['state']} — {sev.upper()} ({alert['anomaly_count']} anomalies)",
                    expanded=(sev in ("critical", "high")),
                ):
                    st.markdown(alert["alert_text"])
        else:
            st.info("No alerts generated. Enable Mistral AI in the sidebar and re-run analysis.")

        st.divider()

        # Insights section
        st.subheader("Mistral AI Interpretations")
        if results["insights"]:
            for insight in results["insights"]:
                a = insight["anomaly"]
                with st.expander(
                    f"{a['drug']} in {a['district']} — {a['severity'].upper()} (Week {a['week']})"
                ):
                    st.markdown(f"**Baseline:** {a['baseline_value']} → **Actual:** {a['actual_value']} units/week")
                    if a.get("z_score"):
                        st.markdown(f"**Z-Score:** {a['z_score']}")
                    st.divider()
                    st.markdown(insight["interpretation"])
        else:
            st.info("No AI interpretations available. Enable Mistral AI and re-run analysis.")

        # On-demand interpretation
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
            if st.button("Get Mistral Interpretation"):
                with st.spinner("Querying Mistral AI..."):
                    from app.core.mistral_agent import interpret_anomaly
                    try:
                        interpretation = interpret_anomaly(anomalies_list[selected_idx])
                        st.markdown(interpretation)
                    except Exception as e:
                        st.error(f"Mistral API error: {e}")

            # Correlate signals button
            if len(anomalies_list) > 1:
                st.divider()
                if st.button("Correlate All Signals"):
                    with st.spinner("Analyzing cross-drug correlations with Mistral..."):
                        from app.core.mistral_agent import correlate_signals
                        try:
                            top_10 = [a for a in anomalies_list if a["severity"] in ("critical", "high")][:10]
                            correlation_text = correlate_signals(top_10)
                            st.markdown(correlation_text)
                        except Exception as e:
                            st.error(f"Mistral API error: {e}")
        elif not use_mistral:
            st.info("Enable Mistral AI in the sidebar for on-demand analysis.")
