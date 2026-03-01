"""
Evaluation module — compute precision/recall/F1 against synthetic ground truth.

The 4 injected anomaly scenarios in generate_synthetic.py define known ground truth
events. This module compares detected anomalies against those events.
"""

GROUND_TRUTH_EVENTS = [
    {
        "name": "Delhi Respiratory Spike",
        "districts": ["South Delhi", "North Delhi", "East Delhi"],
        "drugs": ["salbutamol", "budesonide", "montelukast"],
        "week_range": (70, 72),  # Weeks 18-20 of 2024 (offset from 2023-01-01)
        "scenario": "Air pollution event causing respiratory drug demand surge",
    },
    {
        "name": "Chennai Waterborne Outbreak",
        "districts": ["Chennai", "Kanchipuram"],
        "drugs": ["ors_sachets", "metronidazole", "loperamide"],
        "week_range": (63, 65),  # Week 12 of 2024
        "scenario": "Water contamination causing GI illness",
    },
    {
        "name": "Pune Flu Cluster",
        "districts": ["Pune", "Pimpri-Chinchwad"],
        "drugs": ["oseltamivir", "paracetamol", "cetirizine"],
        "week_range": (59, 62),  # Weeks 8-10 of 2024
        "scenario": "Influenza wave",
    },
    {
        "name": "Vizag Thyroid Anomaly",
        "districts": ["Visakhapatnam"],
        "drugs": ["levothyroxine"],
        "week_range": (61, 78),  # Weeks 9-25+ of 2024 (gradual)
        "scenario": "Industrial pollution causing thyroid disorders",
    },
]


def _anomaly_matches_event(anomaly: dict, event: dict) -> bool:
    """Check if a detected anomaly matches a ground truth event."""
    district_match = anomaly["district"] in event["districts"]
    drug_match = anomaly["drug"] in event["drugs"]
    week_match = event["week_range"][0] <= anomaly["week"] <= event["week_range"][1]
    return district_match and drug_match and week_match


def evaluate_detections(anomalies: list[dict]) -> dict:
    """Evaluate detected anomalies against ground truth events.

    Returns:
        Dict with per-event metrics and overall precision/recall/F1.
    """
    # Match each anomaly to ground truth events
    matched_anomalies = set()  # indices of anomalies that match some event
    event_results = []

    for event in GROUND_TRUTH_EVENTS:
        # Expected detections: all district x drug combinations
        expected = set()
        for d in event["districts"]:
            for drug in event["drugs"]:
                expected.add((d, drug))

        # Find matching detected anomalies
        detected = set()
        for i, a in enumerate(anomalies):
            if _anomaly_matches_event(a, event):
                detected.add((a["district"], a["drug"]))
                matched_anomalies.add(i)

        true_positives = len(expected & detected)
        false_negatives = len(expected - detected)
        false_positives_event = len(detected - expected)

        precision = true_positives / max(true_positives + false_positives_event, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)

        event_results.append({
            "event": event["name"],
            "scenario": event["scenario"],
            "expected_signals": len(expected),
            "detected_signals": len(detected),
            "true_positives": true_positives,
            "false_negatives": false_negatives,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "detected_details": sorted(detected),
            "missed_details": sorted(expected - detected),
        })

    # Overall metrics
    total_tp = sum(e["true_positives"] for e in event_results)
    total_expected = sum(e["expected_signals"] for e in event_results)
    total_fp = len(anomalies) - len(matched_anomalies)  # anomalies not matching any event

    overall_precision = total_tp / max(total_tp + total_fp, 1)
    overall_recall = total_tp / max(total_expected, 1)
    overall_f1 = 2 * overall_precision * overall_recall / max(overall_precision + overall_recall, 1e-9)

    return {
        "events": event_results,
        "overall": {
            "true_positives": total_tp,
            "false_positives": total_fp,
            "total_expected": total_expected,
            "total_detected": len(anomalies),
            "precision": round(overall_precision, 3),
            "recall": round(overall_recall, 3),
            "f1": round(overall_f1, 3),
        },
    }
