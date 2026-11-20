import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml_monitoring import (
    ID_COLUMN,
    initialize_project,
    make_synthetic_data,
    run_workflow,
    validate_batch,
)


class MonitoringWorkflowTests(unittest.TestCase):
    def test_validation_reports_size_and_target_distribution(self):
        batch = make_synthetic_data(20, seed=12)
        result = validate_batch(batch)
        self.assertEqual(result["rows"], 20)
        self.assertEqual(sum(result["target_distribution"].values()), 20)
        self.assertEqual(result["duplicate_ids"], 0)

    def test_validation_rejects_duplicate_ids(self):
        batch = make_synthetic_data(5, seed=12)
        batch.loc[1, ID_COLUMN] = batch.loc[0, ID_COLUMN]
        with self.assertRaisesRegex(ValueError, "Duplicate customer IDs"):
            validate_batch(batch)

    def test_validation_rejects_missing_values(self):
        batch = make_synthetic_data(5, seed=12)
        batch.loc[0, "monthly_spend"] = None
        with self.assertRaisesRegex(ValueError, "Missing values"):
            validate_batch(batch)

    def test_workflow_writes_report_and_keeps_champion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            reports = root / "reports"
            initialize_project(artifacts)
            batch_path = root / "new_batch.csv"
            make_synthetic_data(200, seed=99, start_id=801).to_csv(batch_path, index=False)
            report = run_workflow(batch_path, artifacts, reports)
            self.assertTrue(Path(report["candidate_model"]).exists())
            self.assertTrue(Path(report["report_path"]).exists())
            self.assertFalse(report["champion_overwritten"])
            saved = json.loads(Path(report["report_path"]).read_text())
            self.assertIn("f1", saved["champion_metrics"])
            self.assertIn(report["quality_gate"]["decision"], {"candidate_eligible", "retain_champion"})


if __name__ == "__main__":
    unittest.main()
