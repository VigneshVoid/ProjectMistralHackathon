"""Input data validation and cleaning utilities for pharmacy sales datasets."""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "district",
    "state",
    "pharmacy_id",
    "drug_generic_name",
    "drug_category",
    "quantity_sold",
    "unit_price",
    "latitude",
    "longitude",
]



def validate_and_clean_sales_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate and clean uploaded sales data.

    Returns a cleaned dataframe and a validation report with warnings/errors.
    """
    report = {
        "errors": [],
        "warnings": [],
        "rows_in": len(df),
        "rows_out": 0,
        "rows_dropped": 0,
    }

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report["errors"].append(
            f"Missing required columns: {', '.join(missing_cols)}"
        )
        report["rows_out"] = 0
        report["rows_dropped"] = len(df)
        return pd.DataFrame(), report

    clean = df.copy()

    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    for col in ["quantity_sold", "unit_price", "latitude", "longitude"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    critical_cols = ["date", "district", "state", "drug_generic_name", "drug_category", "quantity_sold"]
    missing_critical = clean[critical_cols].isna().any(axis=1)
    if missing_critical.any():
        dropped = int(missing_critical.sum())
        report["warnings"].append(f"Dropped {dropped} rows with missing critical values.")
        clean = clean[~missing_critical]

    negative_qty = clean["quantity_sold"] < 0
    if negative_qty.any():
        dropped = int(negative_qty.sum())
        report["warnings"].append(f"Dropped {dropped} rows with negative quantity_sold.")
        clean = clean[~negative_qty]

    dupes = clean.duplicated(subset=["date", "district", "pharmacy_id", "drug_generic_name"], keep="first")
    if dupes.any():
        dropped = int(dupes.sum())
        report["warnings"].append(f"Removed {dropped} duplicate rows.")
        clean = clean[~dupes]

    report["rows_out"] = len(clean)
    report["rows_dropped"] = report["rows_in"] - report["rows_out"]

    if clean.empty:
        report["errors"].append("No usable rows remain after validation.")

    return clean.reset_index(drop=True), report
