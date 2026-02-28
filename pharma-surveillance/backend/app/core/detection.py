"""
Anomaly detection engine for pharmacy sales data.

Three detection methods operating on drug sales grouped by (district, drug, week):
1. Z-Score — flag where weekly sales exceed 2σ from 8-week rolling mean
2. IQR Outlier — flag values outside 1.5× IQR of district history
3. Percentage Spike — flag week-over-week increase > 200%
"""

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from app.core.mappings import DRUG_CONDITION_MAP


@dataclass
class Anomaly:
    district: str
    state: str
    drug: str
    drug_category: str
    anomaly_type: str  # "z_score" | "iqr_outlier" | "pct_spike"
    severity: str  # "critical" | "high" | "medium" | "low"
    baseline_value: float
    actual_value: float
    z_score: float | None
    week: int
    date_range: str
    confidence: str
    pct_change: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_severity(z: float | None, pct: float | None) -> str:
    """Classify anomaly severity based on deviation magnitude."""
    score = abs(z) if z else 0
    if pct:
        score = max(score, pct / 100)
    if score >= 4:
        return "critical"
    if score >= 3:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _week_to_date_range(week: int, start_date: str = "2024-01-01") -> str:
    """Convert week number to a human-readable date range."""
    start = pd.Timestamp(start_date) + pd.Timedelta(weeks=week)
    end = start + pd.Timedelta(days=6)
    return f"{start.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"


def aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales to weekly by (district, state, drug)."""
    if "week" not in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        min_date = df["date"].min()
        df["week"] = ((df["date"] - min_date).dt.days // 7).astype(int)

    weekly = (
        df.groupby(["district", "state", "drug_generic_name", "drug_category", "week"])
        .agg(total_sold=("quantity_sold", "sum"))
        .reset_index()
    )
    return weekly.sort_values(["district", "drug_generic_name", "week"])


def detect_zscore(weekly: pd.DataFrame, window: int = 8, threshold: float = 2.0) -> list[Anomaly]:
    """Z-Score detection: flag weeks where sales exceed threshold σ from rolling mean."""
    anomalies = []

    for (district, state, drug, category), group in weekly.groupby(
        ["district", "state", "drug_generic_name", "drug_category"]
    ):
        group = group.sort_values("week")
        sales = group["total_sold"].values
        weeks = group["week"].values

        if len(sales) < window + 1:
            continue

        rolling_mean = pd.Series(sales).rolling(window=window, min_periods=3).mean()
        rolling_std = pd.Series(sales).rolling(window=window, min_periods=3).std()

        for i in range(window, len(sales)):
            mu = rolling_mean.iloc[i - 1]
            sigma = rolling_std.iloc[i - 1]
            if sigma == 0 or np.isnan(sigma) or np.isnan(mu):
                continue

            z = (sales[i] - mu) / sigma
            if z > threshold:
                drug_info = DRUG_CONDITION_MAP.get(drug, {})
                anomalies.append(Anomaly(
                    district=district,
                    state=state,
                    drug=drug,
                    drug_category=category,
                    anomaly_type="z_score",
                    severity=_classify_severity(z, None),
                    baseline_value=round(mu, 1),
                    actual_value=float(sales[i]),
                    z_score=round(z, 2),
                    week=int(weeks[i]),
                    date_range=_week_to_date_range(int(weeks[i])),
                    confidence=drug_info.get("confidence", "medium"),
                ))

    return anomalies


def detect_iqr(weekly: pd.DataFrame, k: float = 1.5) -> list[Anomaly]:
    """IQR detection: flag values outside k×IQR of district history."""
    anomalies = []

    for (district, state, drug, category), group in weekly.groupby(
        ["district", "state", "drug_generic_name", "drug_category"]
    ):
        sales = group["total_sold"].values
        weeks = group["week"].values

        if len(sales) < 4:
            continue

        q1, q3 = np.percentile(sales, [25, 75])
        iqr = q3 - q1
        upper = q3 + k * iqr

        if iqr == 0:
            continue

        for i, (val, week) in enumerate(zip(sales, weeks)):
            if val > upper:
                z_approx = (val - np.mean(sales)) / max(np.std(sales), 1)
                drug_info = DRUG_CONDITION_MAP.get(drug, {})
                anomalies.append(Anomaly(
                    district=district,
                    state=state,
                    drug=drug,
                    drug_category=category,
                    anomaly_type="iqr_outlier",
                    severity=_classify_severity(z_approx, None),
                    baseline_value=round(float(q3), 1),
                    actual_value=float(val),
                    z_score=round(z_approx, 2),
                    week=int(week),
                    date_range=_week_to_date_range(int(week)),
                    confidence=drug_info.get("confidence", "medium"),
                ))

    return anomalies


def detect_pct_spike(weekly: pd.DataFrame, threshold: float = 200.0) -> list[Anomaly]:
    """Percentage spike detection: flag week-over-week increases > threshold%."""
    anomalies = []

    for (district, state, drug, category), group in weekly.groupby(
        ["district", "state", "drug_generic_name", "drug_category"]
    ):
        group = group.sort_values("week")
        sales = group["total_sold"].values
        weeks = group["week"].values

        for i in range(1, len(sales)):
            prev = sales[i - 1]
            if prev == 0:
                continue
            pct = ((sales[i] - prev) / prev) * 100

            if pct > threshold:
                drug_info = DRUG_CONDITION_MAP.get(drug, {})
                anomalies.append(Anomaly(
                    district=district,
                    state=state,
                    drug=drug,
                    drug_category=category,
                    anomaly_type="pct_spike",
                    severity=_classify_severity(None, pct),
                    baseline_value=float(prev),
                    actual_value=float(sales[i]),
                    z_score=None,
                    week=int(weeks[i]),
                    date_range=_week_to_date_range(int(weeks[i])),
                    confidence=drug_info.get("confidence", "medium"),
                    pct_change=round(pct, 1),
                ))

    return anomalies


def run_all_detections(df: pd.DataFrame) -> list[Anomaly]:
    """Run all three detection methods and return deduplicated anomalies."""
    weekly = aggregate_weekly(df)

    z_anomalies = detect_zscore(weekly)
    iqr_anomalies = detect_iqr(weekly)
    pct_anomalies = detect_pct_spike(weekly)

    # Deduplicate: keep the most severe per (district, drug, week)
    all_anomalies = z_anomalies + iqr_anomalies + pct_anomalies
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    seen: dict[tuple, Anomaly] = {}
    for a in all_anomalies:
        key = (a.district, a.drug, a.week)
        if key not in seen or severity_order.get(a.severity, 4) < severity_order.get(seen[key].severity, 4):
            seen[key] = a

    return sorted(seen.values(), key=lambda a: (severity_order.get(a.severity, 4), a.week))
