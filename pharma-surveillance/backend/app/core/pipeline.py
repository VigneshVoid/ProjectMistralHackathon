"""
Pipeline orchestrator: data → detect → correlate → interpret → alert.

This module ties together the detection engine, Mistral agent, and correlation logic.
Uses sequential API calls with delays to respect free-tier rate limits.
"""

import time
from typing import Callable, Optional

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


def _safe_interpret(anomaly: dict, delay: float = 2.0) -> dict:
    """Interpret a single anomaly, catching errors.

    Delay is applied BEFORE the API call to space out requests.
    """
    if delay > 0:
        time.sleep(delay)
    try:
        interpretation = interpret_anomaly(anomaly)
        return {"anomaly": anomaly, "interpretation": interpretation}
    except Exception as e:
        return {
            "anomaly": anomaly,
            "interpretation": {"error": f"Mistral unavailable: {e}"},
        }


def _safe_alert(district: str, state: str, high_severity: list[dict], delay: float = 2.0) -> dict:
    """Generate an alert for a district, catching errors.

    Delay is applied BEFORE the API call to space out requests.
    """
    if delay > 0:
        time.sleep(delay)
    try:
        alert_data = generate_alert(high_severity)
        return {
            "district": district,
            "state": state,
            "alert_data": alert_data,
            "anomaly_count": len(high_severity),
            "max_severity": high_severity[0]["severity"],
        }
    except Exception as e:
        return {
            "district": district,
            "state": state,
            "alert_data": {"error": f"Alert generation failed: {e}"},
            "anomaly_count": len(high_severity),
            "max_severity": high_severity[0]["severity"],
        }


def run_pipeline(
    df: pd.DataFrame,
    use_mistral: bool = True,
    seasonal_adjust: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Run the full surveillance pipeline on pharmacy sales data.

    Args:
        df: Raw pharmacy sales DataFrame.
        use_mistral: Whether to call Mistral for interpretations.
        seasonal_adjust: If True, exclude expected seasonal spikes from anomaly detection.
        progress_callback: Optional callable(step, total_steps, message) for UI progress.

    Returns:
        Dict with keys: anomalies, correlations, alerts, insights, weekly_data, summary.
    """
    # Compute total steps for progress reporting
    # Base steps: validate(1) + detect(2) + correlate(3) + aggregate(4) + summary(5)
    # Mistral steps added dynamically below
    _MAX_INTERPRETATIONS = 5
    _MAX_ALERT_DISTRICTS = 10

    def _progress(step: int, total: int, msg: str):
        if progress_callback:
            progress_callback(step, total, msg)

    # Step 0: Validate and clean input
    _progress(1, 5, "Validating data...")
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
    _progress(2, 5, "Detecting anomalies...")
    anomalies = run_all_detections(clean_df, seasonal_adjust=seasonal_adjust)
    anomaly_dicts = [a.to_dict() for a in anomalies]

    # Step 2: Group anomalies by district and check correlations
    _progress(3, 5, "Checking correlations...")
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

    # Step 3: Mistral AI — sequential with rate-limit-safe delays
    insights = []
    alerts = []

    if use_mistral and anomaly_dicts:
        # Select top anomalies (reduced from 10 to 5 for rate limit safety)
        top_anomalies = [a for a in anomaly_dicts if a["severity"] in ("critical", "high")][:_MAX_INTERPRETATIONS]
        num_interpretations = len(top_anomalies)

        # Select top districts by high-severity count (reduced to max 10)
        district_severity_counts = []
        for district, dist_anomalies in district_anomalies.items():
            high_severity = [a for a in dist_anomalies if a["severity"] in ("critical", "high")]
            if high_severity:
                district_severity_counts.append((district, dist_anomalies, high_severity))
        district_severity_counts.sort(key=lambda x: len(x[2]), reverse=True)
        top_districts = district_severity_counts[:_MAX_ALERT_DISTRICTS]
        num_alerts = len(top_districts)

        total_steps = 3 + num_interpretations + num_alerts + 2  # +2 for aggregate + summary

        # Interpret anomalies — fully sequential with 2s delays
        for i, anomaly in enumerate(top_anomalies):
            _progress(4 + i, total_steps, f"Interpreting anomaly {i + 1}/{num_interpretations}...")
            insights.append(_safe_interpret(anomaly, delay=2.0 if i > 0 else 0.5))

        # Generate alerts — fully sequential with 2s delays
        for i, (district, dist_anomalies, high_severity) in enumerate(top_districts):
            _progress(
                4 + num_interpretations + i,
                total_steps,
                f"Generating alert {i + 1}/{num_alerts}...",
            )
            alerts.append(
                _safe_alert(district, high_severity[0]["state"], high_severity, delay=2.0 if i > 0 else 0.5)
            )

        _progress(total_steps - 1, total_steps, "Building summary...")
    else:
        total_steps = 5
        _progress(4, total_steps, "Aggregating weekly data...")

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

    _progress(total_steps, total_steps, "Pipeline complete!")

    return {
        "anomalies": anomaly_dicts,
        "correlations": correlations,
        "alerts": alerts,
        "insights": insights,
        "weekly_data": weekly,
        "district_risk": district_risk,
        "summary": summary,
    }
