from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
def test_features_dataset():
    df = pd.read_csv(ROOT / "data" / "features.csv")
    assert len(df) == 720
    assert {"service", "minute"}.issubset(df.columns)
def test_detection_results():
    df = pd.read_csv(ROOT / "data" / "detections.csv")
    assert len(df) == 720
    assert "ml_anomaly" in df.columns
def test_root_cause_report():
    df = pd.read_csv(ROOT / "reports" / "root_cause_ranking.csv")
    assert not df.empty
    assert {"service", "root_cause_score"}.issubset(df.columns)
def test_injected_incident_detected():
    df = pd.read_csv(ROOT / "data" / "detections.csv")
    incident = df[
        (df["service"] == "payment") &
        (df["minute"].between(100, 110))
    ]
    assert len(incident) == 11
    assert incident["ml_anomaly"].sum() == 11
