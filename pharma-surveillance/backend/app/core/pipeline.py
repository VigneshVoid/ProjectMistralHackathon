"""
Pipeline orchestrator: data → detect → correlate → interpret → alert.

This module ties together the detection engine, Mistral agent, and correlation logic.
"""

from dataclasses import asdict

import pandas as pd

from app.core.detection import run_all_detections, aggregate_weekly, Anomaly
from app.core.mappings import check_correlations
from app.core.mistral_agent import interpret_anomaly, generate_alert, correlate_signals


def run_pipeline(df: pd.DataFrame, use_mistral: bool = True) -> dict:
    """Run the full surveillance pipeline on pharmacy sales data.

    Args:
        df: Raw pharmacy sales DataFrame.
        use_mistral: Whether to call Mistral for interpretations (set False for offline testing).

    Returns:
        Dict with keys: anomalies, correlations, alerts, insights, weekly_data, summary.
    """
    # Step 1: Detect anomalies
    anomalies = run_all_detections(df)
    anomaly_dicts = [a.to_dict() for a in anomalies]

    # Step 2: Group anomalies by district and check correlations
    district_anomalies: dict[str, list[dict]] = {}
    for a in anomaly_dicts:
        district_anomalies.setdefault(a["district"], []).append(a)

    correlations = []
    for district, dist_anomalies in district_anomalies.items():
        drugs_in_district = list({a["drug"] for a in dist_anomalies})
        matched_rules = check_correlations(drugs_in_district)
        for rule in matched_rules:
            correlations.append({
                "district": district,
                "state": dist_anomalies[0]["state"],
                **rule,
            })

    # Step 3: Generate Mistral interpretations (if enabled)
    insights = []
    alerts = []

    if use_mistral and anomaly_dicts:
        # Interpret top anomalies (limit to avoid API rate limits)
        top_anomalies = [a for a in anomaly_dicts if a["severity"] in ("critical", "high")][:10]
        for a in top_anomalies:
            try:
                interpretation = interpret_anomaly(a)
                insights.append({
                    "anomaly": a,
                    "interpretation": interpretation,
                })
            except Exception as e:
                insights.append({
                    "anomaly": a,
                    "interpretation": f"[Mistral unavailable: {e}]",
                })

        # Generate alerts per district cluster
        for district, dist_anomalies in district_anomalies.items():
            high_severity = [a for a in dist_anomalies if a["severity"] in ("critical", "high")]
            if high_severity:
                try:
                    alert_text = generate_alert(high_severity)
                    alerts.append({
                        "district": district,
                        "state": high_severity[0]["state"],
                        "alert_text": alert_text,
                        "anomaly_count": len(high_severity),
                        "max_severity": high_severity[0]["severity"],
                    })
                except Exception as e:
                    alerts.append({
                        "district": district,
                        "state": high_severity[0]["state"],
                        "alert_text": f"[Alert generation failed: {e}]",
                        "anomaly_count": len(high_severity),
                        "max_severity": high_severity[0]["severity"],
                    })

    # Step 4: Aggregate weekly data for charts
    weekly = aggregate_weekly(df)

    # Step 5: Build summary
    severity_counts = {}
    for a in anomaly_dicts:
        severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1

    districts_affected = len(district_anomalies)

    summary = {
        "total_records": len(df),
        "total_anomalies": len(anomaly_dicts),
        "severity_breakdown": severity_counts,
        "districts_affected": districts_affected,
        "correlations_found": len(correlations),
        "alerts_generated": len(alerts),
    }

    return {
        "anomalies": anomaly_dicts,
        "correlations": correlations,
        "alerts": alerts,
        "insights": insights,
        "weekly_data": weekly,
        "summary": summary,
    }
