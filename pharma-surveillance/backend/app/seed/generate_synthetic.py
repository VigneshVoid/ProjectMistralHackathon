"""
Synthetic pharmacy sales data generator.

Generates 18 months of daily pharmacy sales across 44 districts in India,
USA, and Europe with seasonal variation and injected anomaly scenarios.
"""

import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from app.core.mappings import ALL_DRUGS, DISTRICTS, get_seasonal_multiplier

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Generation parameters
START_DATE = date(2023, 1, 1)  # Start from 2023 for Year-over-Year comparison
NUM_DAYS = 365 + 180  # ~18 months: full 2023 + Jan-Jun 2024
PHARMACIES_PER_DISTRICT = 3

# Baseline daily sales per drug per pharmacy (units)
BASELINE_SALES = {
    "paracetamol": 80,
    "cetirizine": 30,
    "azithromycin": 15,
    "salbutamol": 20,
    "budesonide": 10,
    "montelukast": 12,
    "oseltamivir": 5,
    "ors_sachets": 25,
    "metronidazole": 18,
    "loperamide": 12,
    "levothyroxine": 8,
    "insulin_glargine": 6,
    "amlodipine": 22,
}

# Unit prices (INR)
UNIT_PRICES = {
    "paracetamol": 2.5,
    "cetirizine": 4.0,
    "azithromycin": 15.0,
    "salbutamol": 8.0,
    "budesonide": 25.0,
    "montelukast": 12.0,
    "oseltamivir": 45.0,
    "ors_sachets": 5.0,
    "metronidazole": 6.0,
    "loperamide": 7.0,
    "levothyroxine": 10.0,
    "insulin_glargine": 120.0,
    "amlodipine": 8.0,
}


def _date_to_week(d: date) -> int:
    """Convert date to week number from START_DATE."""
    return (d - START_DATE).days // 7


def _generate_pharmacy_ids(district: str, n: int) -> list[str]:
    """Generate pharmacy IDs for a district."""
    prefix = district.replace(" ", "")[:6].upper()
    return [f"{prefix}-PH{str(i).zfill(3)}" for i in range(1, n + 1)]


def _apply_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Inject realistic anomaly patterns into the dataset.

    Anomalies are injected in year 2024 (the "current year") so that 2023
    serves as a clean historical baseline for Year-over-Year comparison.
    Week offsets are relative to START_DATE (2023-01-01), so 2024 weeks
    start around week 52.
    """
    df = df.copy()

    # Offset: week 0 = 2023-01-01, so 2024-01-01 ≈ week 52
    YEAR2_OFFSET = 52

    # --- Anomaly 1: Delhi respiratory spike (Weeks 18-20 of 2024) ---
    # Cause: simulated air pollution event
    delhi_districts = ["South Delhi", "North Delhi", "East Delhi"]
    respiratory_drugs = ["salbutamol", "budesonide", "montelukast"]
    mask_delhi = (
        df["district"].isin(delhi_districts)
        & df["drug_generic_name"].isin(respiratory_drugs)
        & df["week"].between(YEAR2_OFFSET + 18, YEAR2_OFFSET + 20)
    )
    multipliers = df.loc[mask_delhi, "drug_generic_name"].map(
        {"salbutamol": 4.0, "budesonide": 3.5, "montelukast": 3.0}
    )
    df.loc[mask_delhi, "quantity_sold"] = (
        df.loc[mask_delhi, "quantity_sold"] * multipliers * np.random.uniform(0.8, 1.2, mask_delhi.sum())
    ).astype(int)

    # --- Anomaly 2: Chennai waterborne outbreak (Week 12 of 2024) ---
    # Cause: simulated water contamination
    chennai_districts = ["Chennai", "Kanchipuram"]
    waterborne_drugs = ["ors_sachets", "metronidazole", "loperamide"]
    mask_chennai = (
        df["district"].isin(chennai_districts)
        & df["drug_generic_name"].isin(waterborne_drugs)
        & df["week"].between(YEAR2_OFFSET + 12, YEAR2_OFFSET + 13)
    )
    multipliers_ch = df.loc[mask_chennai, "drug_generic_name"].map(
        {"ors_sachets": 6.0, "metronidazole": 5.0, "loperamide": 4.0}
    )
    df.loc[mask_chennai, "quantity_sold"] = (
        df.loc[mask_chennai, "quantity_sold"] * multipliers_ch * np.random.uniform(0.8, 1.2, mask_chennai.sum())
    ).astype(int)

    # --- Anomaly 3: Pune flu cluster (Weeks 8-10 of 2024) ---
    # Cause: simulated influenza wave
    pune_districts = ["Pune", "Pimpri-Chinchwad"]
    flu_drugs = ["oseltamivir", "paracetamol", "cetirizine"]
    mask_pune = (
        df["district"].isin(pune_districts)
        & df["drug_generic_name"].isin(flu_drugs)
        & df["week"].between(YEAR2_OFFSET + 8, YEAR2_OFFSET + 10)
    )
    multipliers_pune = df.loc[mask_pune, "drug_generic_name"].map(
        {"oseltamivir": 3.5, "paracetamol": 2.5, "cetirizine": 2.0}
    )
    df.loc[mask_pune, "quantity_sold"] = (
        df.loc[mask_pune, "quantity_sold"] * multipliers_pune * np.random.uniform(0.8, 1.2, mask_pune.sum())
    ).astype(int)

    # --- Anomaly 4: Vizag thyroid anomaly (gradual over months 3-6 of 2024) ---
    # Cause: simulated industrial pollution
    mask_vizag = (
        (df["district"] == "Visakhapatnam")
        & (df["drug_generic_name"] == "levothyroxine")
        & (df["week"] >= YEAR2_OFFSET + 9)
    )
    weeks_vizag = df.loc[mask_vizag, "week"]
    ramp = 1.0 + 0.5 * ((weeks_vizag - (YEAR2_OFFSET + 9)) / (25 - 9)).clip(0, 1)
    df.loc[mask_vizag, "quantity_sold"] = (
        df.loc[mask_vizag, "quantity_sold"] * ramp * np.random.uniform(0.9, 1.1, mask_vizag.sum())
    ).astype(int)

    # --- Anomaly 5: NYC flu outbreak (Weeks 4-6 of 2024, deep winter) ---
    # Cause: simulated severe influenza season
    nyc_districts = ["Manhattan", "Brooklyn"]
    mask_nyc = (
        df["district"].isin(nyc_districts)
        & df["drug_generic_name"].isin(["oseltamivir", "paracetamol", "cetirizine"])
        & df["week"].between(YEAR2_OFFSET + 4, YEAR2_OFFSET + 6)
    )
    multipliers_nyc = df.loc[mask_nyc, "drug_generic_name"].map(
        {"oseltamivir": 4.0, "paracetamol": 3.0, "cetirizine": 2.5}
    )
    df.loc[mask_nyc, "quantity_sold"] = (
        df.loc[mask_nyc, "quantity_sold"] * multipliers_nyc * np.random.uniform(0.8, 1.2, mask_nyc.sum())
    ).astype(int)

    # --- Anomaly 6: Houston GI outbreak (Weeks 14-15 of 2024) ---
    # Cause: simulated water contamination after flooding
    mask_houston = (
        (df["district"] == "Houston")
        & df["drug_generic_name"].isin(["ors_sachets", "metronidazole", "loperamide"])
        & df["week"].between(YEAR2_OFFSET + 14, YEAR2_OFFSET + 15)
    )
    multipliers_hou = df.loc[mask_houston, "drug_generic_name"].map(
        {"ors_sachets": 5.0, "metronidazole": 4.5, "loperamide": 4.0}
    )
    df.loc[mask_houston, "quantity_sold"] = (
        df.loc[mask_houston, "quantity_sold"] * multipliers_hou * np.random.uniform(0.8, 1.2, mask_houston.sum())
    ).astype(int)

    # --- Anomaly 7: LA respiratory spike (Weeks 20-22 of 2024) ---
    # Cause: simulated wildfire smoke event
    mask_la = (
        (df["district"] == "Los Angeles")
        & df["drug_generic_name"].isin(["salbutamol", "budesonide", "montelukast"])
        & df["week"].between(YEAR2_OFFSET + 20, YEAR2_OFFSET + 22)
    )
    multipliers_la = df.loc[mask_la, "drug_generic_name"].map(
        {"salbutamol": 5.0, "budesonide": 4.0, "montelukast": 3.5}
    )
    df.loc[mask_la, "quantity_sold"] = (
        df.loc[mask_la, "quantity_sold"] * multipliers_la * np.random.uniform(0.8, 1.2, mask_la.sum())
    ).astype(int)

    # --- Anomaly 8: London-Paris flu wave (Weeks 6-8 of 2024, winter) ---
    # Cause: simulated cross-border influenza wave
    eu_flu_districts = ["London", "Manchester", "Paris"]
    mask_eu_flu = (
        df["district"].isin(eu_flu_districts)
        & df["drug_generic_name"].isin(["oseltamivir", "paracetamol", "cetirizine"])
        & df["week"].between(YEAR2_OFFSET + 6, YEAR2_OFFSET + 8)
    )
    multipliers_eu_flu = df.loc[mask_eu_flu, "drug_generic_name"].map(
        {"oseltamivir": 3.5, "paracetamol": 2.8, "cetirizine": 2.0}
    )
    df.loc[mask_eu_flu, "quantity_sold"] = (
        df.loc[mask_eu_flu, "quantity_sold"] * multipliers_eu_flu * np.random.uniform(0.8, 1.2, mask_eu_flu.sum())
    ).astype(int)

    # --- Anomaly 9: Berlin-Munich respiratory cluster (Weeks 10-12 of 2024) ---
    # Cause: simulated industrial pollution event
    mask_de = (
        df["district"].isin(["Berlin", "Munich"])
        & df["drug_generic_name"].isin(["salbutamol", "budesonide", "montelukast"])
        & df["week"].between(YEAR2_OFFSET + 10, YEAR2_OFFSET + 12)
    )
    multipliers_de = df.loc[mask_de, "drug_generic_name"].map(
        {"salbutamol": 3.5, "budesonide": 3.0, "montelukast": 2.5}
    )
    df.loc[mask_de, "quantity_sold"] = (
        df.loc[mask_de, "quantity_sold"] * multipliers_de * np.random.uniform(0.8, 1.2, mask_de.sum())
    ).astype(int)

    # --- Anomaly 10: Madrid-Rome diabetes spike (gradual, Weeks 5-20 of 2024) ---
    # Cause: simulated post-holiday metabolic trend
    mask_south_eu = (
        df["district"].isin(["Madrid", "Rome"])
        & (df["drug_generic_name"] == "insulin_glargine")
        & (df["week"] >= YEAR2_OFFSET + 5)
        & (df["week"] <= YEAR2_OFFSET + 20)
    )
    weeks_south_eu = df.loc[mask_south_eu, "week"]
    ramp_eu = 1.0 + 0.6 * ((weeks_south_eu - (YEAR2_OFFSET + 5)) / 15).clip(0, 1)
    df.loc[mask_south_eu, "quantity_sold"] = (
        df.loc[mask_south_eu, "quantity_sold"] * ramp_eu * np.random.uniform(0.9, 1.1, mask_south_eu.sum())
    ).astype(int)

    return df


def generate(output_path: str | Path | None = None) -> pd.DataFrame:
    """Generate the full synthetic dataset.

    Returns the DataFrame and optionally writes to CSV.
    """
    rows = []

    for day_offset in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_offset)
        week = _date_to_week(current_date)

        for dist_info in DISTRICTS:
            pharmacy_ids = _generate_pharmacy_ids(dist_info["district"], PHARMACIES_PER_DISTRICT)

            for pharm_id in pharmacy_ids:
                # Each pharmacy sells a random subset of drugs each day
                drugs_today = random.sample(ALL_DRUGS, k=random.randint(6, len(ALL_DRUGS)))

                for drug in drugs_today:
                    base = BASELINE_SALES[drug]
                    # Apply seasonal multiplier for this drug/month
                    seasonal_mult = get_seasonal_multiplier(drug, current_date.month)
                    # ±15% random daily variation on top of seasonal baseline
                    quantity = max(1, int(base * seasonal_mult * np.random.uniform(0.85, 1.15)))
                    price = UNIT_PRICES[drug]

                    rows.append({
                        "date": current_date.isoformat(),
                        "week": week,
                        "district": dist_info["district"],
                        "state": dist_info["state"],
                        "country": dist_info.get("country", "India"),
                        "pharmacy_id": pharm_id,
                        "drug_generic_name": drug,
                        "drug_category": _get_category(drug),
                        "quantity_sold": quantity,
                        "unit_price": price,
                        "latitude": dist_info["lat"] + np.random.uniform(-0.02, 0.02),
                        "longitude": dist_info["lon"] + np.random.uniform(-0.02, 0.02),
                    })

    df = pd.DataFrame(rows)
    df = _apply_anomalies(df)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df):,} records → {output_path}")

    return df


def _get_category(drug: str) -> str:
    """Get drug category from mappings."""
    from app.core.mappings import DRUG_CONDITION_MAP
    info = DRUG_CONDITION_MAP.get(drug, {})
    return info.get("category", "general")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent.parent / "data" / "pharma_sales.csv"
    generate(output_path=out)
