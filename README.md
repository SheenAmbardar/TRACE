# TRACE — AI-Powered Root-Cause Intelligence

TRACE is a research-oriented machine learning project that analyzes distributed-service telemetry, detects anomalous behavior, and ranks potential root causes using anomaly evidence and service dependencies.

## Key Features

- **Synthetic telemetry generation:** Creates service-level latency, error-rate, and CPU metrics.
- **Feature engineering:** Calculates rolling statistics, Z-scores, and anomaly indicators.
- **ML anomaly detection:** Uses Isolation Forest to flag unusual service behavior.
- **Root-cause ranking:** Combines anomaly strength, anomaly volume, timing, and dependency impact.
- **Interactive dashboard:** Presents telemetry, detection results, and ranked root-cause candidates.
- **Automated tests:** Validates datasets, detection outputs, root-cause reports, and injected-incident detection.

## Architecture

Telemetry Generation → Feature Engineering → Isolation Forest → Root-Cause Ranking → Streamlit Dashboard

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, NetworkX, Streamlit, Pytest, Matplotlib.

## Current Evaluation

On the synthetic dataset of 720 service-minute records, TRACE detected all 11 deliberately injected incident records.

- Precision: 0.500
- Recall: 1.000
- F1 score: 0.667
- Automated tests: 4 passed

The current detector also produced 11 false positives. Root-cause ranking is a heuristic candidate-ranking method, not proof of causation. Evaluation uses a known synthetic incident and does not establish performance on real production data.

## Run Locally

```bash
pip install -r requirements.txt
python src/data_generator.py
python src/features.py
python src/anomaly_detector.py
python src/root_cause.py
streamlit run app.py
```

## Run Tests

```bash
python -m pytest tests/test_trace.py -v
```

## Project Status

Functional prototype with a synthetic-data pipeline, anomaly detection, root-cause candidate ranking, dashboard, and passing automated tests. Further work includes reducing false positives, validating causal rankings, and evaluating on realistic external datasets.