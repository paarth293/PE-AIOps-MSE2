"""Synthetic classification model monitoring and retraining workflow.

The workflow deliberately keeps the champion and candidate artifacts separate.
Running it produces an evaluation report and a quality-gate decision without
silently changing the deployed model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["tenure_months", "monthly_spend", "support_tickets", "usage_hours"]
TARGET_COLUMN = "churned"
ID_COLUMN = "customer_id"
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_REPORT_DIR = Path("reports")


@dataclass(frozen=True)
class QualityGate:
    """The documented quality rule used to protect the champion."""

    primary_metric: str = "f1"
    minimum_f1_delta: float = -0.02
    minimum_accuracy_delta: float = -0.02


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_synthetic_data(n_rows: int, seed: int, start_id: int = 1) -> pd.DataFrame:
    """Create reproducible labelled customer records for the demonstration."""

    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_rows)
    spend = rng.normal(85, 25, n_rows).clip(15, 200)
    tickets = rng.poisson(2.2, n_rows)
    usage = rng.normal(28, 10, n_rows).clip(1, 80)
    logit = 0.1 - 0.025 * tenure + 0.28 * tickets - 0.025 * usage + 0.008 * (spend - 85)
    probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, probability)
    return pd.DataFrame(
        {
            ID_COLUMN: np.arange(start_id, start_id + n_rows),
            "tenure_months": tenure,
            "monthly_spend": np.round(spend, 2),
            "support_tickets": tickets,
            "usage_hours": np.round(usage, 2),
            TARGET_COLUMN: target,
        }
    )


def validate_batch(batch: pd.DataFrame) -> dict[str, Any]:
    """Validate schema, missing values, identifiers, and target distribution."""

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in batch.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    missing_values = batch[REQUIRED_COLUMNS].isna().sum()
    missing_report = {key: int(value) for key, value in missing_values.items() if value}
    if missing_report:
        raise ValueError(f"Missing values found: {missing_report}")
    duplicate_ids = int(batch[ID_COLUMN].duplicated().sum())
    if duplicate_ids:
        raise ValueError(f"Duplicate customer IDs found: {duplicate_ids}")
    if not set(batch[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1")
    return {
        "rows": int(len(batch)),
        "columns": list(batch.columns),
        "missing_values": missing_report,
        "duplicate_ids": duplicate_ids,
        "target_distribution": {
            str(key): int(value) for key, value in batch[TARGET_COLUMN].value_counts().sort_index().items()
        },
    }


def build_pipeline() -> Pipeline:
    """Build the same preprocessing and estimator pipeline for every run."""

    preprocessing = ColumnTransformer(
        [("numeric", StandardScaler(), FEATURE_COLUMNS)],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(data: pd.DataFrame) -> Pipeline:
    model = build_pipeline()
    model.fit(data[FEATURE_COLUMNS], data[TARGET_COLUMN])
    return model


def evaluate_model(model: Pipeline, data: pd.DataFrame) -> dict[str, float]:
    predictions = model.predict(data[FEATURE_COLUMNS])
    return {
        "accuracy": round(float(accuracy_score(data[TARGET_COLUMN], predictions)), 6),
        "precision": round(float(precision_score(data[TARGET_COLUMN], predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(data[TARGET_COLUMN], predictions, zero_division=0)), 6),
        "f1": round(float(f1_score(data[TARGET_COLUMN], predictions, zero_division=0)), 6),
    }


def initialize_project(artifact_dir: Path = DEFAULT_ARTIFACT_DIR) -> dict[str, str]:
    """Create the initial training set and champion model artifact."""

    artifact_dir.mkdir(parents=True, exist_ok=True)
    training_data = make_synthetic_data(800, seed=7)
    training_path = artifact_dir / "training_data.csv"
    champion_path = artifact_dir / "champion_model.joblib"
    training_data.to_csv(training_path, index=False)
    joblib.dump(train_model(training_data), champion_path)
    return {"training_data": str(training_path), "champion_model": str(champion_path)}


def _quality_gate(champion: dict[str, float], candidate: dict[str, float], rule: QualityGate) -> dict[str, Any]:
    f1_delta = round(candidate["f1"] - champion["f1"], 6)
    accuracy_delta = round(candidate["accuracy"] - champion["accuracy"], 6)
    eligible = (
        f1_delta >= rule.minimum_f1_delta
        and accuracy_delta >= rule.minimum_accuracy_delta
    )
    return {
        "eligible_for_promotion": eligible,
        "decision": "candidate_eligible" if eligible else "retain_champion",
        "rule": asdict(rule),
        "metric_deltas": {"f1": f1_delta, "accuracy": accuracy_delta},
        "reason": (
            "Candidate meets the F1 and accuracy non-regression thresholds."
            if eligible
            else "Candidate does not meet the F1 and accuracy thresholds; champion retained."
        ),
    }


def run_workflow(
    batch_path: Path,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    """Run validation, champion evaluation, retraining, and the quality gate."""

    batch = pd.read_csv(batch_path)
    validation = validate_batch(batch)
    training_path = artifact_dir / "training_data.csv"
    champion_path = artifact_dir / "champion_model.joblib"
    if not training_path.exists() or not champion_path.exists():
        raise FileNotFoundError("Run the init command before running monitoring.")

    training_data = pd.read_csv(training_path)
    champion = joblib.load(champion_path)
    champion_metrics = evaluate_model(champion, batch)
    combined = pd.concat([training_data, batch], ignore_index=True)
    candidate = train_model(combined)
    candidate_metrics = evaluate_model(candidate, batch)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = artifact_dir / "candidate_model.joblib"
    joblib.dump(candidate, candidate_path)
    gate = _quality_gate(champion_metrics, candidate_metrics, QualityGate())

    mlflow_status = "not_installed"
    try:
        import mlflow  # type: ignore

        with mlflow.start_run(run_name="model-monitoring-retraining"):
            mlflow.log_metrics({f"candidate_{key}": value for key, value in candidate_metrics.items()})
            mlflow.log_param("quality_gate_decision", gate["decision"])
            mlflow_status = "logged"
    except ImportError:
        pass

    report = {
        "generated_at": _utc_now(),
        "batch_path": str(batch_path),
        "validation": validation,
        "champion_metrics": champion_metrics,
        "candidate_metrics": candidate_metrics,
        "quality_gate": gate,
        "candidate_model": str(candidate_path),
        "mlflow": mlflow_status,
        "champion_overwritten": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="create synthetic training data and champion")
    init_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    batch_parser = subparsers.add_parser("generate-batch", help="create a labelled incoming synthetic batch")
    batch_parser.add_argument("output", type=Path)
    batch_parser.add_argument("--rows", type=int, default=200)
    batch_parser.add_argument("--seed", type=int, default=99)
    run_parser = subparsers.add_parser("run", help="monitor a labelled batch and retrain a candidate")
    run_parser.add_argument("batch", type=Path)
    run_parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    run_parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    if args.command == "init":
        print(json.dumps(initialize_project(args.artifact_dir), indent=2))
    elif args.command == "generate-batch":
        batch = make_synthetic_data(args.rows, seed=args.seed, start_id=801)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        batch.to_csv(args.output, index=False)
        print(json.dumps(validate_batch(batch), indent=2))
    else:
        print(json.dumps(run_workflow(args.batch, args.artifact_dir, args.report_dir), indent=2))


if __name__ == "__main__":
    main()
