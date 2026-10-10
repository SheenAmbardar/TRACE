
from pathlib import Path

import networkx as nx
import pandas as pd


# ==========================================================
# SECTION 1: DEFINE INPUT AND OUTPUT FILES
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "data" / "detections.csv"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "root_cause_ranking.csv"


# ==========================================================
# SECTION 2: BUILD THE SERVICE DEPENDENCY GRAPH
# ==========================================================

def build_service_graph():
    """Create a graph where each service points to its dependencies."""

    graph = nx.DiGraph()

    graph.add_edges_from([
        ("api", "order"),
        ("order", "payment"),
        ("order", "inventory"),
    ])

    return graph


# ==========================================================
# SECTION 3: SUMMARIZE ANOMALIES FOR EACH SERVICE
# ==========================================================

def summarize_services(df):
    """Calculate anomaly volume, severity, and first detection time."""

    summaries = []

    for service, group in df.groupby("service"):
        anomalies = group[group["ml_anomaly"] == 1]

        if anomalies.empty:
            summaries.append({
                "service": service,
                "anomaly_count": 0,
                "mean_anomaly_score": 0.0,
                "first_anomaly_minute": None,
            })
            continue

        summaries.append({
            "service": service,
            "anomaly_count": len(anomalies),
            "mean_anomaly_score": anomalies["anomaly_score"].mean(),
            "first_anomaly_minute": anomalies["minute"].min(),
        })

    return pd.DataFrame(summaries)


# ==========================================================
# SECTION 4: CALCULATE DEPENDENCY-BASED EVIDENCE
# ==========================================================

def count_affected_services(graph, service, anomalous_services):
    """Count anomalous services that depend on this candidate service."""

    # Edges point from callers to dependencies.
    # Reverse traversal finds callers that may be affected by a dependency.
    reverse_graph = graph.reverse(copy=False)

    callers = nx.descendants(reverse_graph, service)

    return sum(
        1 for caller in callers
        if caller in anomalous_services
    )


# ==========================================================
# SECTION 5: RANK CANDIDATE ROOT CAUSES
# ==========================================================

def rank_root_causes(df):
    """Rank candidate root causes and explain the evidence."""

    graph = build_service_graph()
    summaries = summarize_services(df)

    # Identify services where the ML model detected anomalies.
    anomalous_services = set(
        summaries.loc[
            summaries["anomaly_count"] > 0,
            "service"
        ]
    )

    # Measure how many anomalous callers depend on each service.
    summaries["affected_anomalous_services"] = summaries[
        "service"
    ].apply(
        lambda service: count_affected_services(
            graph,
            service,
            anomalous_services
        )
    )

    # Evidence 1: Normalize anomaly volume to a 0–1 range.
    max_count = summaries["anomaly_count"].max()

    if max_count > 0:
        summaries["volume_score"] = (
            summaries["anomaly_count"] / max_count
        )
    else:
        summaries["volume_score"] = 0.0

    # Evidence 2: Normalize the mean anomaly score.
    low = summaries["mean_anomaly_score"].min()
    high = summaries["mean_anomaly_score"].max()

    if high > low:
        summaries["strength_score"] = (
            (summaries["mean_anomaly_score"] - low)
            / (high - low)
        )
    else:
        summaries["strength_score"] = 0.0

    # Evidence 3: Earlier first detections receive higher scores.
    first_times = summaries["first_anomaly_minute"].dropna()

    if len(first_times) > 1 and first_times.max() > first_times.min():
        summaries["temporal_score"] = (
            (first_times.max() - summaries["first_anomaly_minute"])
            / (first_times.max() - first_times.min())
        ).fillna(0.0)
    else:
        summaries["temporal_score"] = 0.0

    # Evidence 4: Normalize dependency impact.
    max_affected = summaries["affected_anomalous_services"].max()

    if max_affected > 0:
        summaries["dependency_score"] = (
            summaries["affected_anomalous_services"]
            / max_affected
        )
    else:
        summaries["dependency_score"] = 0.0

    # Combine the four evidence signals into a heuristic score.
    summaries["root_cause_score"] = (
        0.35 * summaries["strength_score"]
        + 0.25 * summaries["volume_score"]
        + 0.20 * summaries["temporal_score"]
        + 0.20 * summaries["dependency_score"]
    )

    # Rank candidates from highest to lowest score.
    summaries = summaries.sort_values(
        "root_cause_score",
        ascending=False
    ).reset_index(drop=True)

    summaries.insert(
        0,
        "rank",
        range(1, len(summaries) + 1)
    )

    # Add a readable explanation for every candidate.
    def explain_candidate(row):
        evidence = []

        if row["strength_score"] >= 0.7:
            evidence.append("relatively strong anomaly scores")

        if row["volume_score"] >= 0.7:
            evidence.append("a high number of detected anomalies")

        if row["dependency_score"] > 0:
            evidence.append(
                f'{int(row["affected_anomalous_services"])} '
                "anomalous dependent service(s)"
            )

        if pd.notna(row["first_anomaly_minute"]):
            evidence.append(
                f'first detected anomaly at minute '
                f'{int(row["first_anomaly_minute"])}'
            )

        if not evidence:
            evidence.append("limited evidence from the current signals")

        return "; ".join(evidence) + "."

    summaries["evidence_summary"] = summaries.apply(
        explain_candidate,
        axis=1
    )

    # These labels describe relative heuristic scores, not probabilities.
    def label_evidence(score):
        if score >= 0.7:
            return "Strong relative evidence"
        elif score >= 0.4:
            return "Moderate relative evidence"
        else:
            return "Limited relative evidence"

    summaries["evidence_label"] = summaries[
        "root_cause_score"
    ].apply(label_evidence)

    return summaries
# ==========================================================
# SECTION 6: LOAD DETECTIONS AND SAVE THE RANKING
# ==========================================================

def run_root_cause_analysis():
    """Run the ranking workflow and save the results."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Detection file not found: {INPUT_PATH}. "
            "Run src/anomaly_detector.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required = {
        "service",
        "minute",
        "ml_anomaly",
        "anomaly_score",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    ranking = rank_root_causes(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(OUTPUT_PATH, index=False)

    print("TRACE root-cause ranking completed!")
    print(f"Saved to: {OUTPUT_PATH}")
    print("\nRanked candidate services:")

    print(
        ranking[
            [
                "rank",
                "service",
                "anomaly_count",
                "affected_anomalous_services",
                "root_cause_score",
            ]
        ].to_string(index=False)
    )


# ==========================================================
# SECTION 7: EXECUTE THE ANALYSIS
# ==========================================================

if __name__ == "__main__":
    run_root_cause_analysis()
