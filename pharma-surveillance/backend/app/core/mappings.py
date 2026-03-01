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

# Monthly seasonal multipliers per drug (1.0 = baseline, >1 = expected higher sales)
# Based on Indian climate and disease patterns:
#   - Winter (Nov-Feb): Flu/cold drugs spike
#   - Summer (Apr-Jun): Waterborne diseases rise
#   - Monsoon (Jul-Sep): Waterborne + vector-borne peak
#   - Post-monsoon/Winter (Oct-Dec): Respiratory/pollution spike
SEASONAL_PROFILES: dict[str, dict[int, float]] = {
    # Month numbers: 1=Jan ... 12=Dec
    "salbutamol":      {1: 1.1, 2: 1.0, 3: 0.9, 4: 0.9, 5: 0.8, 6: 0.8, 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.5, 11: 1.8, 12: 1.7},
    "budesonide":      {1: 1.1, 2: 1.0, 3: 0.9, 4: 0.9, 5: 0.8, 6: 0.8, 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.4, 11: 1.7, 12: 1.6},
    "montelukast":     {1: 1.1, 2: 1.0, 3: 0.9, 4: 0.9, 5: 0.8, 6: 0.8, 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.4, 11: 1.6, 12: 1.5},
    "oseltamivir":     {1: 1.8, 2: 1.6, 3: 1.2, 4: 0.8, 5: 0.6, 6: 0.5, 7: 0.6, 8: 0.7, 9: 0.9, 10: 1.0, 11: 1.3, 12: 1.7},
    "paracetamol":     {1: 1.4, 2: 1.3, 3: 1.1, 4: 1.0, 5: 0.9, 6: 0.9, 7: 1.1, 8: 1.1, 9: 1.0, 10: 1.0, 11: 1.2, 12: 1.4},
    "cetirizine":      {1: 1.0, 2: 1.1, 3: 1.4, 4: 1.5, 5: 1.3, 6: 1.0, 7: 0.9, 8: 0.9, 9: 1.0, 10: 1.2, 11: 1.3, 12: 1.1},
    "ors_sachets":     {1: 0.7, 2: 0.8, 3: 1.0, 4: 1.3, 5: 1.6, 6: 1.8, 7: 1.9, 8: 1.8, 9: 1.5, 10: 1.0, 11: 0.8, 12: 0.7},
    "metronidazole":   {1: 0.8, 2: 0.8, 3: 0.9, 4: 1.2, 5: 1.5, 6: 1.7, 7: 1.8, 8: 1.7, 9: 1.4, 10: 1.0, 11: 0.8, 12: 0.8},
    "loperamide":      {1: 0.8, 2: 0.8, 3: 0.9, 4: 1.2, 5: 1.4, 6: 1.6, 7: 1.7, 8: 1.6, 9: 1.3, 10: 1.0, 11: 0.8, 12: 0.8},
    "levothyroxine":   {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
    "insulin_glargine": {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0},
    "amlodipine":      {1: 1.1, 2: 1.1, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.9, 7: 1.0, 8: 1.0, 9: 1.0, 10: 1.0, 11: 1.1, 12: 1.1},
    "azithromycin":    {1: 1.3, 2: 1.2, 3: 1.0, 4: 0.9, 5: 0.8, 6: 0.9, 7: 1.2, 8: 1.3, 9: 1.2, 10: 1.0, 11: 1.1, 12: 1.2},
}


def get_seasonal_multiplier(drug: str, month: int) -> float:
    """Get the expected seasonal multiplier for a drug in a given month."""
    profile = SEASONAL_PROFILES.get(drug)
    if not profile:
        return 1.0
    return profile.get(month, 1.0)


# All drugs used in synthetic data generation
ALL_DRUGS = list(DRUG_CONDITION_MAP.keys())

# District/city data with coordinates for map visualization
# Each entry includes a "country" field to support global surveillance
DISTRICTS = [
    # --- India ---
    {"district": "South Delhi", "state": "Delhi", "country": "India", "lat": 28.53, "lon": 77.22},
    {"district": "North Delhi", "state": "Delhi", "country": "India", "lat": 28.70, "lon": 77.20},
    {"district": "East Delhi", "state": "Delhi", "country": "India", "lat": 28.63, "lon": 77.30},
    {"district": "West Delhi", "state": "Delhi", "country": "India", "lat": 28.65, "lon": 77.10},
    {"district": "Chennai", "state": "Tamil Nadu", "country": "India", "lat": 13.08, "lon": 80.27},
    {"district": "Kanchipuram", "state": "Tamil Nadu", "country": "India", "lat": 12.83, "lon": 79.70},
    {"district": "Coimbatore", "state": "Tamil Nadu", "country": "India", "lat": 11.01, "lon": 76.96},
    {"district": "Pune", "state": "Maharashtra", "country": "India", "lat": 18.52, "lon": 73.86},
    {"district": "Pimpri-Chinchwad", "state": "Maharashtra", "country": "India", "lat": 18.63, "lon": 73.80},
    {"district": "Mumbai", "state": "Maharashtra", "country": "India", "lat": 19.08, "lon": 72.88},
    {"district": "Visakhapatnam", "state": "Andhra Pradesh", "country": "India", "lat": 17.69, "lon": 83.22},
    {"district": "Hyderabad", "state": "Telangana", "country": "India", "lat": 17.39, "lon": 78.49},
    {"district": "Bengaluru Urban", "state": "Karnataka", "country": "India", "lat": 12.97, "lon": 77.59},
    {"district": "Kolkata", "state": "West Bengal", "country": "India", "lat": 22.57, "lon": 88.36},
    {"district": "Lucknow", "state": "Uttar Pradesh", "country": "India", "lat": 26.85, "lon": 80.95},
    {"district": "Jaipur", "state": "Rajasthan", "country": "India", "lat": 26.91, "lon": 75.79},
    {"district": "Ahmedabad", "state": "Gujarat", "country": "India", "lat": 23.02, "lon": 72.57},
    {"district": "Bhopal", "state": "Madhya Pradesh", "country": "India", "lat": 23.26, "lon": 77.41},
    {"district": "Patna", "state": "Bihar", "country": "India", "lat": 25.61, "lon": 85.14},
    {"district": "Thiruvananthapuram", "state": "Kerala", "country": "India", "lat": 8.52, "lon": 76.94},
    # --- United States ---
    {"district": "Manhattan", "state": "New York", "country": "USA", "lat": 40.78, "lon": -73.97},
    {"district": "Brooklyn", "state": "New York", "country": "USA", "lat": 40.65, "lon": -73.95},
    {"district": "Los Angeles", "state": "California", "country": "USA", "lat": 34.05, "lon": -118.24},
    {"district": "San Francisco", "state": "California", "country": "USA", "lat": 37.77, "lon": -122.42},
    {"district": "Chicago", "state": "Illinois", "country": "USA", "lat": 41.88, "lon": -87.63},
    {"district": "Houston", "state": "Texas", "country": "USA", "lat": 29.76, "lon": -95.37},
    {"district": "Miami", "state": "Florida", "country": "USA", "lat": 25.76, "lon": -80.19},
    {"district": "Boston", "state": "Massachusetts", "country": "USA", "lat": 42.36, "lon": -71.06},
    {"district": "Seattle", "state": "Washington", "country": "USA", "lat": 47.61, "lon": -122.33},
    {"district": "Denver", "state": "Colorado", "country": "USA", "lat": 39.74, "lon": -104.99},
    {"district": "Atlanta", "state": "Georgia", "country": "USA", "lat": 33.75, "lon": -84.39},
    {"district": "Philadelphia", "state": "Pennsylvania", "country": "USA", "lat": 39.95, "lon": -75.17},
    # --- Europe ---
    {"district": "London", "state": "United Kingdom", "country": "Europe", "lat": 51.51, "lon": -0.13},
    {"district": "Manchester", "state": "United Kingdom", "country": "Europe", "lat": 53.48, "lon": -2.24},
    {"district": "Paris", "state": "France", "country": "Europe", "lat": 48.86, "lon": 2.35},
    {"district": "Lyon", "state": "France", "country": "Europe", "lat": 45.76, "lon": 4.84},
    {"district": "Berlin", "state": "Germany", "country": "Europe", "lat": 52.52, "lon": 13.41},
    {"district": "Munich", "state": "Germany", "country": "Europe", "lat": 48.14, "lon": 11.58},
    {"district": "Madrid", "state": "Spain", "country": "Europe", "lat": 40.42, "lon": -3.70},
    {"district": "Rome", "state": "Italy", "country": "Europe", "lat": 41.90, "lon": 12.50},
    {"district": "Amsterdam", "state": "Netherlands", "country": "Europe", "lat": 52.37, "lon": 4.90},
    {"district": "Stockholm", "state": "Sweden", "country": "Europe", "lat": 59.33, "lon": 18.07},
    {"district": "Vienna", "state": "Austria", "country": "Europe", "lat": 48.21, "lon": 16.37},
    {"district": "Brussels", "state": "Belgium", "country": "Europe", "lat": 50.85, "lon": 4.35},
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
