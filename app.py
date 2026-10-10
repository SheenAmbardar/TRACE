
# ==========================================================
# TRACE: ROOT-CAUSE INTELLIGENCE FOR DISTRIBUTED SYSTEMS
# Displays telemetry, ML anomalies, root-cause explanations,
# and service dependencies in an interactive dashboard.
# ==========================================================


# SECTION 1: IMPORT REQUIRED LIBRARIES
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st


# SECTION 2: CONFIGURE THE DASHBOARD
st.set_page_config(
    page_title="TRACE | Root-Cause Intelligence",
    page_icon="🔎",
    layout="wide",
)

# Locate the main TRACE project directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Locate the generated results.
DETECTIONS_PATH = PROJECT_ROOT / "data" / "detections.csv"
RANKING_PATH = PROJECT_ROOT / "reports" / "root_cause_ranking.csv"


# SECTION 3: LOAD GENERATED RESULTS
@st.cache_data
def load_data():
    """Load telemetry detections and root-cause rankings."""
    detections = pd.read_csv(DETECTIONS_PATH)
    ranking = pd.read_csv(RANKING_PATH)
    return detections, ranking


# SECTION 4: DISPLAY DASHBOARD TITLE
st.title("TRACE 🔎")
st.subheader("Root-Cause Intelligence for Distributed Systems")

st.caption(
    "Analyze service telemetry, detect unusual behavior, "
    "and investigate possible root causes using ML and "
    "dependency-based evidence."
)


# SECTION 5: CHECK INPUT FILES AND LOAD RESULTS
if not DETECTIONS_PATH.exists() or not RANKING_PATH.exists():
    st.error(
        "TRACE result files are missing. "
        "Run the pipeline using: python src/pipeline.py"
    )
    st.stop()

detections, ranking = load_data()

# Validate the minimum columns required by the dashboard.
required_detection_columns = {
    "service",
    "minute",
    "ml_anomaly",
    "latency_ms",
    "error_rate",
    "cpu_usage_pct",
}

missing_detection_columns = (
    required_detection_columns - set(detections.columns)
)

if missing_detection_columns:
    st.error(
        f"Detection file is missing columns: "
        f"{sorted(missing_detection_columns)}"
    )
    st.stop()

if ranking.empty:
    st.warning("No root-cause ranking results are available.")
    st.stop()


# SECTION 6: SIDEBAR FILTERS
st.sidebar.header("Analysis Controls")

services = sorted(detections["service"].dropna().unique().tolist())

selected_services = st.sidebar.multiselect(
    "Select services",
    options=services,
    default=services,
)

filtered = detections[
    detections["service"].isin(selected_services)
].copy()

if filtered.empty:
    st.warning("Select at least one service to display results.")
    st.stop()


# SECTION 7: KEY PERFORMANCE INDICATORS
total_rows = len(filtered)
anomaly_rows = int(filtered["ml_anomaly"].sum())

anomaly_percentage = (
    100 * anomaly_rows / total_rows if total_rows else 0
)

# The global ranking is generated from all services.
top_candidate = ranking.iloc[0]["service"]
top_score = ranking.iloc[0]["root_cause_score"]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Telemetry Records", f"{total_rows:,}")
col2.metric("ML Anomalies", f"{anomaly_rows:,}")
col3.metric("Anomalies (%)", f"{anomaly_percentage:.1f}%")
col4.metric(
    "Top Candidate",
    str(top_candidate),
    f"Heuristic score: {top_score:.3f}",
)

st.caption(
    "The top candidate is based on the global root-cause ranking; "
    "the sidebar filter only changes the telemetry views."
)


# SECTION 8: ANOMALY COUNTS BY SERVICE
st.header("1. Service Anomaly Overview")

anomaly_summary = (
    filtered.groupby("service")["ml_anomaly"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(anomaly_summary)


# SECTION 9: TELEMETRY OVER TIME
st.header("2. Telemetry Over Time")

metric = st.selectbox(
    "Choose a telemetry metric",
    ["latency_ms", "error_rate", "cpu_usage_pct"],
)

time_data = filtered.pivot(
    index="minute",
    columns="service",
    values=metric,
)

st.line_chart(time_data)

st.caption(
    "The synthetic incident was injected into payment around "
    "minutes 100–110. Compare telemetry patterns across services."
)


# SECTION 10: EXPLAINABLE ROOT-CAUSE RANKINGS
st.header("3. Ranked Root-Cause Candidates")

st.write(
    "TRACE ranks candidate services using anomaly strength, "
    "anomaly volume, first-detection timing, and dependency impact."
)

# Keep the original ranking columns and include explanations
# when the updated root_cause.py has generated them.
display_columns = [
    "rank",
    "service",
    "anomaly_count",
    "mean_anomaly_score",
    "first_anomaly_minute",
    "affected_anomalous_services",
    "strength_score",
    "volume_score",
    "temporal_score",
    "dependency_score",
    "root_cause_score",
    "evidence_label",
    "evidence_summary",
]

available_columns = [
    column
    for column in display_columns
    if column in ranking.columns
]

st.dataframe(
    ranking[available_columns],
    width="stretch",
    hide_index=True,
)

# SECTION 10A: EXPLAIN THE TOP-RANKED CANDIDATE
top_row = ranking.iloc[0]

st.subheader("Why is this service ranked first?")

st.markdown(f"**Candidate service:** `{top_row['service']}`")
st.metric(
    "Root-cause heuristic score",
    f"{top_row['root_cause_score']:.3f}",
)

if "evidence_label" in ranking.columns:
    st.write(f"**Evidence assessment:** {top_row['evidence_label']}")

if "evidence_summary" in ranking.columns:
    st.info(str(top_row["evidence_summary"]))
else:
    st.warning(
        "Explanations are not present in the ranking CSV yet. "
        "Run `python src/root_cause.py` to regenerate the ranking."
    )

st.caption(
    "Evidence labels describe relative heuristic scores, not "
    "calibrated probabilities. A high score does not prove causality."
)


# SECTION 11: SERVICE DEPENDENCY GRAPH
st.header("4. Service Dependency Graph")

graph = nx.DiGraph()

graph.add_edges_from([
    ("api", "order"),
    ("order", "payment"),
    ("order", "inventory"),
])

st.write(
    "**Dependency direction:** caller → dependency. "
    "For example, `order` depends on `payment`."
)

edge_table = pd.DataFrame(
    list(graph.edges),
    columns=["Calling service", "Dependency"],
)

st.dataframe(
    edge_table,
    width="stretch",
    hide_index=True,
)


# SECTION 12: INSPECT TELEMETRY AND ANOMALY DETAILS
with st.expander("View telemetry and anomaly details"):
    st.dataframe(
        filtered.sort_values(["minute", "service"]),
        width="stretch",
        hide_index=True,
    )


# SECTION 13: PROJECT FOOTER
st.divider()

st.caption(
    "TRACE prototype | Synthetic telemetry | "
    "Root-cause scores are heuristic rankings, not proof of causality."
)
