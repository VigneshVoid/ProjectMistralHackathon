"""
Mistral API wrapper — all LLM calls go through this module.

Uses mistralai SDK with mistral-small-latest for speed.
"""

import json
from mistralai import Mistral

from app.config import settings
from app.core.mappings import DRUG_CONDITION_MAP, CORRELATION_RULES


def _get_client() -> Mistral:
    return Mistral(api_key=settings.mistral_api_key)


def _drug_context() -> str:
    """Build drug-to-condition context string for system prompts."""
    lines = []
    for drug, info in DRUG_CONDITION_MAP.items():
        lines.append(f"- {drug}: {info['condition']} ({info['category']}, confidence: {info['confidence']})")
    return "\n".join(lines)


def _call_mistral(system_prompt: str, user_content: str, model: str = "mistral-small-latest") -> str:
    """Make a Mistral API call and return the response text."""
    client = _get_client()
    response = client.chat.complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def interpret_anomaly(anomaly: dict) -> str:
    """Given a detected anomaly, return an epidemiological interpretation."""
    system_prompt = (
        "You are an epidemiological analyst working for India's public health surveillance system. "
        "Given pharmacy drug sales anomaly data from India, provide a concise public health interpretation. "
        "Include: (1) likely health condition, (2) possible causes, (3) severity assessment, "
        "(4) recommended public health response. Be specific and actionable.\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )

    user_content = (
        f"Anomaly detected:\n"
        f"- District: {anomaly['district']}, {anomaly['state']}\n"
        f"- Drug: {anomaly['drug']} ({anomaly['drug_category']})\n"
        f"- Type: {anomaly['anomaly_type']}\n"
        f"- Severity: {anomaly['severity']}\n"
        f"- Baseline value: {anomaly['baseline_value']} units/week\n"
        f"- Actual value: {anomaly['actual_value']} units/week\n"
        f"- Z-score: {anomaly.get('z_score', 'N/A')}\n"
        f"- Period: {anomaly['date_range']}\n"
        f"- Percentage change: {anomaly.get('pct_change', 'N/A')}%\n"
    )

    return _call_mistral(system_prompt, user_content)


def generate_alert(anomalies: list[dict]) -> str:
    """Given multiple anomalies in a region, generate a consolidated public health alert."""
    system_prompt = (
        "You are a public health alert system for India. Generate a concise, structured public health alert "
        "based on pharmacy sales anomalies. Format:\n"
        "**SEVERITY**: [Critical/High/Medium/Low]\n"
        "**AFFECTED AREA**: [districts]\n"
        "**SUSPECTED CONDITION**: [condition]\n"
        "**EVIDENCE**: [brief summary of drug sales signals]\n"
        "**RECOMMENDED ACTIONS**: [numbered list for district health officers]\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )

    anomaly_summaries = []
    for a in anomalies:
        anomaly_summaries.append(
            f"- {a['drug']} in {a['district']}: {a['actual_value']} units "
            f"(baseline {a['baseline_value']}), severity={a['severity']}"
        )

    user_content = (
        f"Generate alert for these anomalies in {anomalies[0].get('state', 'unknown')}:\n"
        + "\n".join(anomaly_summaries)
    )

    return _call_mistral(system_prompt, user_content)


def correlate_signals(anomalies: list[dict]) -> str:
    """Given anomalies across drugs/regions, identify if they point to a common cause."""
    system_prompt = (
        "You are an epidemiological analyst. Analyze these pharmacy drug sales anomalies "
        "to determine if they indicate a common underlying cause (e.g., air pollution event, "
        "waterborne outbreak, seasonal flu wave). Provide:\n"
        "(1) Whether these signals are correlated\n"
        "(2) The likely common cause\n"
        "(3) Confidence level\n"
        "(4) Supporting evidence from the drug patterns\n\n"
        "Known correlation patterns:\n"
    )

    for rule in CORRELATION_RULES:
        system_prompt += (
            f"- {rule['name']}: {', '.join(rule['drugs'])} "
            f"(≥{rule['min_match']} match) → {rule['condition']}\n"
        )

    system_prompt += "\nDrug-to-condition reference:\n" + _drug_context()

    anomaly_lines = []
    for a in anomalies:
        anomaly_lines.append(
            f"- {a['drug']} in {a['district']}, {a['state']}: "
            f"week {a['week']}, {a['actual_value']} units (baseline {a['baseline_value']})"
        )

    user_content = "Analyze these anomalies for correlation:\n" + "\n".join(anomaly_lines)

    return _call_mistral(system_prompt, user_content)
