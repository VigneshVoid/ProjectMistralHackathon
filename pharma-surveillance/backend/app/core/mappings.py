"""Drug-to-condition mapping tables for epidemiological inference."""

DRUG_CONDITION_MAP = {
    # Tier 1 — High confidence (single drug → condition)
    "salbutamol": {"condition": "Respiratory disease", "category": "respiratory", "confidence": "high"},
    "budesonide": {"condition": "Asthma/COPD", "category": "respiratory", "confidence": "high"},
    "montelukast": {"condition": "Asthma", "category": "respiratory", "confidence": "high"},
    "oseltamivir": {"condition": "Influenza", "category": "infectious", "confidence": "high"},
    "ors_sachets": {"condition": "Diarrheal disease", "category": "waterborne", "confidence": "high"},
    "metronidazole": {"condition": "GI infection", "category": "waterborne", "confidence": "high"},
    "loperamide": {"condition": "Diarrhea", "category": "waterborne", "confidence": "medium"},
    "paracetamol": {"condition": "Fever/pain (non-specific)", "category": "general", "confidence": "low"},
    "levothyroxine": {"condition": "Thyroid disorder", "category": "endocrine", "confidence": "high"},
    "cetirizine": {"condition": "Allergic response", "category": "allergy", "confidence": "medium"},
    "azithromycin": {"condition": "Bacterial infection", "category": "infectious", "confidence": "medium"},
    "insulin_glargine": {"condition": "Diabetes", "category": "metabolic", "confidence": "high"},
    "amlodipine": {"condition": "Hypertension", "category": "cardiovascular", "confidence": "high"},
}

CORRELATION_RULES = [
    {
        "name": "waterborne_outbreak",
        "drugs": ["ors_sachets", "metronidazole", "loperamide"],
        "min_match": 2,
        "condition": "Waterborne disease outbreak",
        "severity": "critical",
        "response": "Alert district water supply authority, deploy ORS distribution",
    },
    {
        "name": "respiratory_pollution",
        "drugs": ["salbutamol", "montelukast", "budesonide", "cetirizine"],
        "min_match": 2,
        "condition": "Air pollution respiratory cluster",
        "severity": "high",
        "response": "Correlate with AQI data, issue health advisory",
    },
    {
        "name": "flu_cluster",
        "drugs": ["oseltamivir", "paracetamol", "cetirizine"],
        "min_match": 2,
        "condition": "Influenza cluster",
        "severity": "high",
        "response": "Activate flu surveillance protocol, check vaccine stock",
    },
]

# All drugs used in synthetic data generation
ALL_DRUGS = list(DRUG_CONDITION_MAP.keys())

# District data with coordinates for map visualization
DISTRICTS = [
    {"district": "South Delhi", "state": "Delhi", "lat": 28.53, "lon": 77.22},
    {"district": "North Delhi", "state": "Delhi", "lat": 28.70, "lon": 77.20},
    {"district": "East Delhi", "state": "Delhi", "lat": 28.63, "lon": 77.30},
    {"district": "West Delhi", "state": "Delhi", "lat": 28.65, "lon": 77.10},
    {"district": "Chennai", "state": "Tamil Nadu", "lat": 13.08, "lon": 80.27},
    {"district": "Kanchipuram", "state": "Tamil Nadu", "lat": 12.83, "lon": 79.70},
    {"district": "Coimbatore", "state": "Tamil Nadu", "lat": 11.01, "lon": 76.96},
    {"district": "Pune", "state": "Maharashtra", "lat": 18.52, "lon": 73.86},
    {"district": "Pimpri-Chinchwad", "state": "Maharashtra", "lat": 18.63, "lon": 73.80},
    {"district": "Mumbai", "state": "Maharashtra", "lat": 19.08, "lon": 72.88},
    {"district": "Visakhapatnam", "state": "Andhra Pradesh", "lat": 17.69, "lon": 83.22},
    {"district": "Hyderabad", "state": "Telangana", "lat": 17.39, "lon": 78.49},
    {"district": "Bengaluru Urban", "state": "Karnataka", "lat": 12.97, "lon": 77.59},
    {"district": "Kolkata", "state": "West Bengal", "lat": 22.57, "lon": 88.36},
    {"district": "Lucknow", "state": "Uttar Pradesh", "lat": 26.85, "lon": 80.95},
    {"district": "Jaipur", "state": "Rajasthan", "lat": 26.91, "lon": 75.79},
    {"district": "Ahmedabad", "state": "Gujarat", "lat": 23.02, "lon": 72.57},
    {"district": "Bhopal", "state": "Madhya Pradesh", "lat": 23.26, "lon": 77.41},
    {"district": "Patna", "state": "Bihar", "lat": 25.61, "lon": 85.14},
    {"district": "Thiruvananthapuram", "state": "Kerala", "lat": 8.52, "lon": 76.94},
]


def get_condition_for_drug(drug_name: str) -> dict | None:
    """Look up the condition mapping for a drug."""
    return DRUG_CONDITION_MAP.get(drug_name)


def check_correlations(anomalous_drugs: list[str]) -> list[dict]:
    """Check if a set of anomalous drugs matches any correlation rules."""
    matched = []
    drug_set = set(anomalous_drugs)
    for rule in CORRELATION_RULES:
        overlap = drug_set & set(rule["drugs"])
        if len(overlap) >= rule["min_match"]:
            matched.append({**rule, "matched_drugs": list(overlap)})
    return matched
