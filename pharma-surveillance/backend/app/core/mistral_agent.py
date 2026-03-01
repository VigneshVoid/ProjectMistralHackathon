"""
Mistral API wrapper — all LLM calls go through this module.

Showcases 6 Mistral SDK features:
1. chat.complete()     — batch pipeline calls
2. chat.parse()        — structured output with Pydantic models
3. chat.stream()       — streaming for on-demand analysis
4. Function calling    — natural language query assistant
5. Multilingual        — alert localization (Hindi + regional)
6. Model upgrade       — mistral-medium-latest for better reasoning
"""

import json
import time
import logging
from typing import Generator

from pydantic import BaseModel
from mistralai import Mistral, SDKError

from app.config import settings
from app.core.mappings import DRUG_CONDITION_MAP, CORRELATION_RULES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry configuration for rate-limited (429) requests
# ---------------------------------------------------------------------------
_MAX_RETRIES = 4
_BASE_DELAY = 2  # seconds — doubles each retry: 2, 4, 8, 16

# ---------------------------------------------------------------------------
# Default model — upgraded from mistral-small-latest
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "mistral-medium-latest"

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------
_client: Mistral | None = None


def _get_client() -> Mistral:
    global _client
    if _client is None:
        _client = Mistral(api_key=settings.mistral_api_key)
    return _client


# ---------------------------------------------------------------------------
# Pydantic response models for structured output
# ---------------------------------------------------------------------------
class AnomalyInterpretation(BaseModel):
    likely_condition: str
    confidence: float  # 0-1
    possible_causes: list[str]
    severity_assessment: str
    recommended_actions: list[str]
    additional_context: str


class AlertResponse(BaseModel):
    severity: str
    affected_area: str
    suspected_condition: str
    evidence_summary: str
    recommended_actions: list[str]
    urgency_score: int  # 1-10


class DistrictBriefing(BaseModel):
    district: str
    risk_level: str
    active_signals: list[str]
    recommended_actions: list[str]
    monitoring_metrics: list[str]
    escalation_criteria: str


class DifferentialCause(BaseModel):
    cause: str
    probability: float  # 0-1
    supporting_evidence: list[str]
    missing_evidence: list[str]
    how_to_confirm: str


class DifferentialDiagnosis(BaseModel):
    causes: list[DifferentialCause]
    most_likely: str
    overall_confidence: float


class InterventionPlan(BaseModel):
    immediate_actions: list[str]
    short_term_actions: list[str]
    monitoring_metrics: list[str]
    escalation_criteria: str
    resource_requirements: list[str]


class ClusterAnalysis(BaseModel):
    is_regional_event: bool
    cluster_description: str
    affected_districts: list[str]
    likely_cause: str
    confidence: float
    supporting_evidence: list[str]


class SimulationExplanation(BaseModel):
    impact_summary: str
    key_changes: list[str]
    new_risk_areas: list[str]
    monitoring_recommendations: list[str]


class DataQualityNarration(BaseModel):
    summary: str
    issues_found: list[str]
    remediation_steps: list[str]
    data_entry_tips: list[str]
    risk_assessment: str


class PublicCommunications(BaseModel):
    technical_memo: str
    citizen_advisory: str
    press_summary: str


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------
def _drug_context() -> str:
    """Build drug-to-condition context string for system prompts."""
    lines = []
    for drug, info in DRUG_CONDITION_MAP.items():
        lines.append(f"- {drug}: {info['condition']} ({info['category']}, confidence: {info['confidence']})")
    return "\n".join(lines)


def _anomaly_user_content(anomaly: dict) -> str:
    """Format a single anomaly dict into a user prompt string."""
    return (
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


# ---------------------------------------------------------------------------
# Core call helpers (with automatic retry on 429 rate-limit errors)
# ---------------------------------------------------------------------------
def _call_mistral(system_prompt: str, user_content: str, model: str = DEFAULT_MODEL) -> str:
    """Make a Mistral API call and return the response text.

    Retries up to _MAX_RETRIES times on 429 rate-limit errors with
    exponential backoff (2s, 4s, 8s, 16s).
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except SDKError as e:
            if e.status_code == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (429). Retry %d/%d in %ds...", attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise


def _call_mistral_structured(system_prompt: str, user_content: str, response_model, model: str = DEFAULT_MODEL):
    """Make a Mistral API call with structured output (Pydantic model).

    Retries up to _MAX_RETRIES times on 429 rate-limit errors.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            return response_model.model_validate_json(raw)
        except SDKError as e:
            if e.status_code == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (429). Retry %d/%d in %ds...", attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise


def _stream_mistral(system_prompt: str, user_content: str, model: str = DEFAULT_MODEL) -> Generator[str, None, None]:
    """Stream a Mistral response token-by-token.

    Retries the initial stream creation on 429 rate-limit errors.
    """
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    stream = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            stream = client.chat.stream(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            break
        except SDKError as e:
            if e.status_code == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (429). Retry %d/%d in %ds...", attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise
    if stream is None:
        return
    for chunk in stream:
        content = chunk.data.choices[0].delta.content
        if content:
            yield content


# ---------------------------------------------------------------------------
# 1. Interpret anomaly (structured output)
# ---------------------------------------------------------------------------
_INTERPRET_SYSTEM = (
    "You are an epidemiological analyst working for India's public health surveillance system. "
    "Given pharmacy drug sales anomaly data from India, provide a concise public health interpretation. "
    "Return a JSON object with these exact fields:\n"
    "- likely_condition (string): the most probable health condition\n"
    "- confidence (float 0-1): your confidence level\n"
    "- possible_causes (list of strings): 2-4 possible causes\n"
    "- severity_assessment (string): brief severity explanation\n"
    "- recommended_actions (list of strings): 2-5 actionable steps\n"
    "- additional_context (string): any extra epidemiological context\n\n"
    "Drug-to-condition reference:\n"
)


def interpret_anomaly(anomaly: dict) -> dict:
    """Return a structured epidemiological interpretation."""
    system_prompt = _INTERPRET_SYSTEM + _drug_context()
    user_content = _anomaly_user_content(anomaly)
    result = _call_mistral_structured(system_prompt, user_content, AnomalyInterpretation)
    return result.model_dump()


def interpret_anomaly_text(anomaly: dict) -> str:
    """Return a plain-text interpretation (legacy compatibility)."""
    system_prompt = (
        "You are an epidemiological analyst working for India's public health surveillance system. "
        "Given pharmacy drug sales anomaly data from India, provide a concise public health interpretation. "
        "Include: (1) likely health condition, (2) possible causes, (3) severity assessment, "
        "(4) recommended public health response. Be specific and actionable.\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )
    return _call_mistral(system_prompt, _anomaly_user_content(anomaly))


def interpret_anomaly_stream(anomaly: dict) -> Generator[str, None, None]:
    """Stream an interpretation token-by-token for on-demand UI."""
    system_prompt = (
        "You are an epidemiological analyst working for India's public health surveillance system. "
        "Given pharmacy drug sales anomaly data from India, provide a concise public health interpretation. "
        "Include: (1) likely health condition, (2) possible causes, (3) severity assessment, "
        "(4) recommended public health response. Be specific and actionable.\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )
    yield from _stream_mistral(system_prompt, _anomaly_user_content(anomaly))


# ---------------------------------------------------------------------------
# 2. Generate alert (structured output)
# ---------------------------------------------------------------------------
_ALERT_SYSTEM = (
    "You are a public health alert system for India. Generate a concise, structured alert "
    "based on pharmacy sales anomalies. Return a JSON object with these exact fields:\n"
    "- severity (string): Critical/High/Medium/Low\n"
    "- affected_area (string): affected districts\n"
    "- suspected_condition (string): likely condition\n"
    "- evidence_summary (string): brief evidence summary\n"
    "- recommended_actions (list of strings): 3-5 actions for health officers\n"
    "- urgency_score (integer 1-10): urgency level\n\n"
    "Drug-to-condition reference:\n"
)


def generate_alert(anomalies: list[dict]) -> dict:
    """Return a structured public health alert."""
    system_prompt = _ALERT_SYSTEM + _drug_context()
    summaries = []
    for a in anomalies:
        summaries.append(
            f"- {a['drug']} in {a['district']}: {a['actual_value']} units "
            f"(baseline {a['baseline_value']}), severity={a['severity']}"
        )
    user_content = (
        f"Generate alert for these anomalies in {anomalies[0].get('state', 'unknown')}:\n"
        + "\n".join(summaries)
    )
    result = _call_mistral_structured(system_prompt, user_content, AlertResponse)
    return result.model_dump()


def generate_alert_text(anomalies: list[dict]) -> str:
    """Return plain-text alert (legacy compatibility)."""
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
    summaries = []
    for a in anomalies:
        summaries.append(
            f"- {a['drug']} in {a['district']}: {a['actual_value']} units "
            f"(baseline {a['baseline_value']}), severity={a['severity']}"
        )
    user_content = (
        f"Generate alert for these anomalies in {anomalies[0].get('state', 'unknown')}:\n"
        + "\n".join(summaries)
    )
    return _call_mistral(system_prompt, user_content)


# ---------------------------------------------------------------------------
# 3. Correlate signals
# ---------------------------------------------------------------------------
def correlate_signals(anomalies: list[dict]) -> str:
    """Identify if anomalies across drugs/regions point to a common cause."""
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
            f"(>={rule['min_match']} match) -> {rule['condition']}\n"
        )
    system_prompt += "\nDrug-to-condition reference:\n" + _drug_context()

    lines = []
    for a in anomalies:
        lines.append(
            f"- {a['drug']} in {a['district']}, {a['state']}: "
            f"week {a['week']}, {a['actual_value']} units (baseline {a['baseline_value']})"
        )
    user_content = "Analyze these anomalies for correlation:\n" + "\n".join(lines)
    return _call_mistral(system_prompt, user_content)


def correlate_signals_stream(anomalies: list[dict]) -> Generator[str, None, None]:
    """Stream correlation analysis token-by-token."""
    system_prompt = (
        "You are an epidemiological analyst. Analyze these pharmacy drug sales anomalies "
        "to determine if they indicate a common underlying cause. Provide:\n"
        "(1) Whether these signals are correlated\n"
        "(2) The likely common cause\n"
        "(3) Confidence level\n"
        "(4) Supporting evidence from the drug patterns\n\n"
        "Known correlation patterns:\n"
    )
    for rule in CORRELATION_RULES:
        system_prompt += (
            f"- {rule['name']}: {', '.join(rule['drugs'])} "
            f"(>={rule['min_match']} match) -> {rule['condition']}\n"
        )
    system_prompt += "\nDrug-to-condition reference:\n" + _drug_context()

    lines = []
    for a in anomalies:
        lines.append(
            f"- {a['drug']} in {a['district']}, {a['state']}: "
            f"week {a['week']}, {a['actual_value']} units (baseline {a['baseline_value']})"
        )
    user_content = "Analyze these anomalies for correlation:\n" + "\n".join(lines)
    yield from _stream_mistral(system_prompt, user_content)


# ---------------------------------------------------------------------------
# 4. District morning briefing (structured output)
# ---------------------------------------------------------------------------
def generate_district_briefing(district: str, anomalies: list[dict], correlations: list[dict]) -> dict:
    """Generate a structured morning briefing for a district health officer."""
    system_prompt = (
        "You are generating a morning briefing for a district health officer in India. "
        "Return a JSON object with these exact fields:\n"
        "- district (string): district name\n"
        "- risk_level (string): Critical/High/Medium/Low\n"
        "- active_signals (list of strings): current active health signals\n"
        "- recommended_actions (list of strings): prioritized action items\n"
        "- monitoring_metrics (list of strings): what to track today\n"
        "- escalation_criteria (string): when to escalate to state level\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )
    anomaly_lines = []
    for a in anomalies:
        anomaly_lines.append(
            f"- {a['drug']} ({a['drug_category']}): {a['actual_value']} units "
            f"(baseline {a['baseline_value']}), severity={a['severity']}, week {a['week']}"
        )
    corr_lines = []
    for c in correlations:
        corr_lines.append(f"- {c['condition']}: drugs={', '.join(c['matched_drugs'])}")

    user_content = (
        f"Generate morning briefing for {district}.\n\n"
        f"Active anomalies:\n" + "\n".join(anomaly_lines) + "\n\n"
        f"Detected correlations:\n" + ("\n".join(corr_lines) if corr_lines else "None") + "\n"
    )
    result = _call_mistral_structured(system_prompt, user_content, DistrictBriefing)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 5. Differential cause reasoning (structured output)
# ---------------------------------------------------------------------------
def differential_diagnosis(anomaly: dict) -> dict:
    """Rank plausible causes for an anomaly with confidence and evidence."""
    system_prompt = (
        "You are an epidemiological analyst. For a detected pharmacy drug sales anomaly, "
        "provide a differential diagnosis ranking the top 3-5 plausible causes. "
        "Return a JSON object with these exact fields:\n"
        "- causes (list of objects, each with: cause, probability (0-1), "
        "supporting_evidence (list), missing_evidence (list), how_to_confirm (string))\n"
        "- most_likely (string): the most likely cause\n"
        "- overall_confidence (float 0-1): overall diagnostic confidence\n\n"
        "Consider: seasonal wave, pollution event, outbreak, reporting artifact, "
        "supply chain issue, policy change.\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )
    user_content = _anomaly_user_content(anomaly)
    result = _call_mistral_structured(system_prompt, user_content, DifferentialDiagnosis)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 6. Multi-lingual alert localization
# ---------------------------------------------------------------------------
def localize_alert(alert_text: str, target_language: str) -> str:
    """Translate an alert into the target language using Mistral."""
    system_prompt = (
        f"Translate the following public health alert into {target_language}. "
        "Maintain factual accuracy, medical terminology, and urgency level. "
        "Use clear, accessible language suitable for public communication. "
        "Do not add or remove information."
    )
    return _call_mistral(system_prompt, alert_text)


# ---------------------------------------------------------------------------
# 7. Intervention planner (structured output)
# ---------------------------------------------------------------------------
def plan_interventions(anomaly: dict) -> dict:
    """Generate tiered intervention plan from anomaly profile."""
    system_prompt = (
        "You are a public health intervention planner for India. "
        "Based on a pharmacy drug sales anomaly, generate a tiered response plan. "
        "Return a JSON object with these exact fields:\n"
        "- immediate_actions (list of strings): within 24 hours\n"
        "- short_term_actions (list of strings): within 7 days\n"
        "- monitoring_metrics (list of strings): what to track\n"
        "- escalation_criteria (string): when to escalate\n"
        "- resource_requirements (list of strings): resources needed\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )
    user_content = _anomaly_user_content(anomaly)
    result = _call_mistral_structured(system_prompt, user_content, InterventionPlan)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 8. Cross-district pattern clustering (structured output)
# ---------------------------------------------------------------------------
def cluster_district_patterns(anomalies: list[dict]) -> dict:
    """Determine if multi-district anomalies are one regional event or isolated."""
    system_prompt = (
        "You are an epidemiological analyst. Analyze anomalies across multiple districts "
        "to determine if they represent one regional event or isolated incidents. "
        "Return a JSON object with these exact fields:\n"
        "- is_regional_event (boolean): true if connected regional event\n"
        "- cluster_description (string): description of the pattern\n"
        "- affected_districts (list of strings): districts involved\n"
        "- likely_cause (string): most likely common cause\n"
        "- confidence (float 0-1): confidence level\n"
        "- supporting_evidence (list of strings): evidence for conclusion\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )
    lines = []
    for a in anomalies:
        lines.append(
            f"- {a['drug']} in {a['district']}, {a['state']}: "
            f"week {a['week']}, {a['actual_value']} units (baseline {a['baseline_value']}), "
            f"severity={a['severity']}"
        )
    user_content = "Analyze these multi-district anomalies:\n" + "\n".join(lines)
    result = _call_mistral_structured(system_prompt, user_content, ClusterAnalysis)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 9. Simulation explainer (structured output)
# ---------------------------------------------------------------------------
def explain_simulation(original_summary: dict, simulated_summary: dict, scenario_desc: str) -> dict:
    """Explain simulation impact in plain language."""
    system_prompt = (
        "You are a public health analyst. Compare original vs simulated surveillance results "
        "and explain the impact. Return a JSON object with:\n"
        "- impact_summary (string): plain-language summary of what changed\n"
        "- key_changes (list of strings): specific changes observed\n"
        "- new_risk_areas (list of strings): new areas of concern\n"
        "- monitoring_recommendations (list of strings): what to watch\n"
    )
    user_content = (
        f"Scenario: {scenario_desc}\n\n"
        f"Original results:\n"
        f"- Total anomalies: {original_summary.get('total_anomalies', 0)}\n"
        f"- Districts affected: {original_summary.get('districts_affected', 0)}\n"
        f"- Severity breakdown: {json.dumps(original_summary.get('severity_breakdown', {}))}\n\n"
        f"Simulated results:\n"
        f"- Total anomalies: {simulated_summary.get('total_anomalies', 0)}\n"
        f"- Districts affected: {simulated_summary.get('districts_affected', 0)}\n"
        f"- Severity breakdown: {json.dumps(simulated_summary.get('severity_breakdown', {}))}\n"
    )
    result = _call_mistral_structured(system_prompt, user_content, SimulationExplanation)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 10. Data quality anomaly narrator (structured output)
# ---------------------------------------------------------------------------
def narrate_data_quality(validation_report: dict) -> dict:
    """Auto-generate remediation guidance from data quality issues.

    Takes a validation report dict and uses Mistral to produce
    actionable guidance for data-entry teams.
    """
    system_prompt = (
        "You are a data quality specialist for India's pharmacy surveillance system. "
        "Given a data validation report, generate clear remediation guidance for "
        "the data-entry teams at district pharmacies. "
        "Return a JSON object with these exact fields:\n"
        "- summary (string): one-sentence overview of data health\n"
        "- issues_found (list of strings): specific issues detected\n"
        "- remediation_steps (list of strings): actionable steps to fix each issue\n"
        "- data_entry_tips (list of strings): preventive tips for pharmacy staff\n"
        "- risk_assessment (string): impact on surveillance accuracy if unfixed\n"
    )
    user_content = (
        f"Data Validation Report:\n"
        f"- Rows submitted: {validation_report.get('rows_in', 0)}\n"
        f"- Rows accepted: {validation_report.get('rows_out', 0)}\n"
        f"- Rows dropped: {validation_report.get('rows_dropped', 0)}\n"
        f"- Errors: {json.dumps(validation_report.get('errors', []))}\n"
        f"- Warnings: {json.dumps(validation_report.get('warnings', []))}\n"
    )
    result = _call_mistral_structured(system_prompt, user_content, DataQualityNarration)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 11. Public communication drafts (structured output)
# ---------------------------------------------------------------------------
def draft_public_communications(alert_text: str, district: str, severity: str) -> dict:
    """Generate audience-specific communications from a public health alert.

    Produces three versions:
    - Technical memo for health officers
    - Short advisory for citizens
    - Press-ready neutral summary
    """
    system_prompt = (
        "You are a public health communications specialist for India. "
        "Given a public health alert, generate three audience-specific versions. "
        "Return a JSON object with these exact fields:\n"
        "- technical_memo (string): detailed memo for district health officers with "
        "medical terminology, data points, and protocol references (200-300 words)\n"
        "- citizen_advisory (string): clear, simple advisory for the general public "
        "with practical health tips and reassurance, no jargon (100-150 words)\n"
        "- press_summary (string): neutral, fact-based summary suitable for media "
        "with balanced tone and official framing (100-150 words)\n"
    )
    user_content = (
        f"District: {district}\n"
        f"Severity: {severity}\n\n"
        f"Alert Content:\n{alert_text}"
    )
    result = _call_mistral_structured(system_prompt, user_content, PublicCommunications)
    return result.model_dump()


# ---------------------------------------------------------------------------
# 12. Natural language query assistant (function calling)
# ---------------------------------------------------------------------------
QUERY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "filter_anomalies",
            "description": "Filter and search detected anomalies by district, drug, severity, or week range",
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "District name to filter by (e.g., 'South Delhi', 'Chennai'). Leave empty for all.",
                    },
                    "drug": {
                        "type": "string",
                        "description": "Drug name to filter by (e.g., 'salbutamol', 'ors_sachets'). Leave empty for all.",
                    },
                    "severity": {
                        "type": "string",
                        "description": "Severity level: 'critical', 'high', 'medium', 'low'. Leave empty for all.",
                    },
                    "min_week": {
                        "type": "integer",
                        "description": "Minimum week number to filter by. Leave empty for no lower bound.",
                    },
                    "max_week": {
                        "type": "integer",
                        "description": "Maximum week number to filter by. Leave empty for no upper bound.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_district_risk",
            "description": "Get the risk score and ranking for a specific district or all districts",
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "District name. Leave empty for all districts ranked by risk.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlations",
            "description": "Get detected multi-drug correlation patterns, optionally filtered by district",
            "parameters": {
                "type": "object",
                "properties": {
                    "district": {
                        "type": "string",
                        "description": "District name to filter correlations. Leave empty for all.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_summary",
            "description": "Get overall surveillance summary statistics",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def query_assistant(user_query: str, chat_history: list[dict]) -> tuple[str, list[dict] | None]:
    """Process a natural language query using Mistral function calling.

    Returns:
        Tuple of (response_text, tool_calls_or_none).
        tool_calls is a list of dicts with 'name' and 'arguments' if tools were called.
    """
    client = _get_client()

    system_prompt = (
        "You are an AI assistant for the Pharma Surveillance System, a real-time epidemiological "
        "surveillance dashboard that detects public health signals from pharmacy drug sales data in India. "
        "You help analysts query and understand the surveillance data.\n\n"
        "Available tools let you filter anomalies, check district risk scores, find correlations, "
        "and get summary statistics. Use them to answer the user's questions with data.\n\n"
        "After receiving tool results, provide a clear, concise narrative summary.\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_query})

    # Retry on rate limit
    response = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.complete(
                model=DEFAULT_MODEL,
                messages=messages,
                tools=QUERY_TOOLS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )
            break
        except SDKError as e:
            if e.status_code == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (429). Retry %d/%d in %ds...", attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise

    message = response.choices[0].message

    # If the model called tools, return the tool calls for the app to execute
    if message.tool_calls:
        tool_calls = []
        for tc in message.tool_calls:
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments),
            })
        return message.content or "", tool_calls

    # Otherwise return the direct text response
    return message.content or "", None


def query_assistant_followup(
    chat_history: list[dict],
    tool_results: list[dict],
    assistant_message,
) -> str:
    """Continue the conversation after tool execution with results."""
    client = _get_client()

    system_prompt = (
        "You are an AI assistant for the Pharma Surveillance System. "
        "You just received tool results from querying the surveillance database. "
        "Provide a clear, concise narrative summary of the findings. "
        "Highlight the most important insights and any urgent items.\n\n"
        "Drug-to-condition reference:\n" + _drug_context()
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)

    # Add the assistant message with tool calls
    messages.append(assistant_message)

    # Add tool results
    for tr in tool_results:
        messages.append({
            "role": "tool",
            "name": tr["name"],
            "content": json.dumps(tr["result"]),
            "tool_call_id": tr["id"],
        })

    # Retry on rate limit
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = client.chat.complete(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except SDKError as e:
            if e.status_code == 429 and attempt < _MAX_RETRIES:
                delay = _BASE_DELAY * (2 ** attempt)
                logger.warning("Rate limited (429). Retry %d/%d in %ds...", attempt + 1, _MAX_RETRIES, delay)
                time.sleep(delay)
            else:
                raise
