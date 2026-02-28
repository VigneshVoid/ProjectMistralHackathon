"""
Pipeline orchestrator: data → detect → correlate → interpret → alert.

This module ties together the detection engine, Mistral agent, and correlation logic.
"""

import pandas as pd

from app.core.detection import run_all_detections, aggregate_weekly
from app.core.mappings import check_correlations
from app.core.mistral_agent import interpret_anomaly, generate_alert
from app.core.validation import validate_and_clean_sales_data


def _compute_district_risk(anomalies: list[dict], correlations: list[dict]) -> list[dict]:
    """Build district-level risk ranking for triage."""
    severity_weights = {"critical": 5, "high": 3, "medium": 2, "low": 1}
    corr_by_district: dict[str, int] = {}
    for c in correlations:
        corr_by_district[c["district"]] = corr_by_district.get(c["district"], 0) + 1

    district_scores: dict[str, dict] = {}
    for a in anomalies:
        district = a["district"]
        if district not in district_scores:
            district_scores[district] = {
                "district": district,
                "state": a["state"],
                "anomaly_count": 0,
                "weighted_severity": 0,
                "max_severity": a["severity"],
                "avg_pct_change": 0.0,
                "risk_score": 0.0,
            }

        row = district_scores[district]
        row["anomaly_count"] += 1
        row["weighted_severity"] += severity_weights.get(a["severity"], 0)
        if severity_weights.get(a["severity"], 0) > severity_weights.get(row["max_severity"], 0):
            row["max_severity"] = a["severity"]

        if a.get("pct_change"):
            row["avg_pct_change"] += float(a["pct_change"])

    for district, row in district_scores.items():
        corr_bonus = corr_by_district.get(district, 0) * 2
        trend_bonus = min(row["avg_pct_change"] / max(row["anomaly_count"], 1) / 100.0, 3)
        row["risk_score"] = round(row["weighted_severity"] + corr_bonus + trend_bonus, 2)
        row["correlations"] = corr_by_district.get(district, 0)

    return sorted(district_scores.values(), key=lambda x: x["risk_score"], reverse=True)


def run_pipeline(df: pd.DataFrame, use_mistral: bool = True) -> dict:
    """Run the full surveillance pipeline on pharmacy sales data.

    Args:
        df: Raw pharmacy sales DataFrame.
        use_mistral: Whether to call Mistral for interpretations (set False for offline testing).

    Returns:
        Dict with keys: anomalies, correlations, alerts, insights, weekly_data, summary.
    """
    # Step 0: Validate and clean input
    clean_df, validation_report = validate_and_clean_sales_data(df)
    if clean_df.empty:
        return {
            "anomalies": [],
            "correlations": [],
            "alerts": [],
            "insights": [],
            "weekly_data": pd.DataFrame(),
            "district_risk": [],
            "summary": {
                "total_records": len(df),
                "total_anomalies": 0,
                "severity_breakdown": {},
                "districts_affected": 0,
                "correlations_found": 0,
                "alerts_generated": 0,
                "validation": validation_report,
            },
        }

    # Step 1: Detect anomalies
    anomalies = run_all_detections(clean_df)
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
    weekly = aggregate_weekly(clean_df)

    # Step 5: Build summary
    severity_counts = {}
    for a in anomaly_dicts:
        severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1

    districts_affected = len(district_anomalies)

    district_risk = _compute_district_risk(anomaly_dicts, correlations)

    summary = {
        "total_records": len(clean_df),
        "total_anomalies": len(anomaly_dicts),
        "severity_breakdown": severity_counts,
        "districts_affected": districts_affected,
        "correlations_found": len(correlations),
        "alerts_generated": len(alerts),
        "validation": validation_report,
    }

    return {
        "anomalies": anomaly_dicts,
        "correlations": correlations,
        "alerts": alerts,
        "insights": insights,
        "weekly_data": weekly,
        "district_risk": district_risk,
        "summary": summary,
    }
