# Project Evaluation Report

## Model-Quality Monitoring and Retraining Workflow

| Student | Details |
|---|---|
| Name | Paarth Gupta |
| Enrollment number | 202401100200221 |
| Class/Section | CSE V D |
| Report type | Complete technical project evaluation |
| Evaluation date | 20 November 2026 |

---

## 1. Executive Summary

This project implements a reproducible machine-learning monitoring and
retraining workflow for binary customer-churn classification. It validates an
incoming labelled batch, evaluates the existing **champion** model, trains a
separate **candidate** model, compares their quality, and applies a documented
quality gate before recommending promotion.

The demonstrated run completed successfully:

- Incoming batch: **200 records**
- Data validation: **Passed**
- Candidate F1 score: **0.253165**
- Champion F1 score: **0.233766**
- Candidate F1 improvement: **0.019399**
- Accuracy change: **0.000000**
- Quality-gate decision: **`candidate_eligible`**
- Champion automatically overwritten: **No**
- Automated tests: **7 passed**

The project is promotion-safe because the candidate model is saved separately
and the champion artifact is never overwritten by the workflow.

> **Scope note:** The project uses reproducible synthetic customer data. The
> results demonstrate the engineering workflow and model-governance controls;
> they are not a financial valuation or a production business-performance
> claim.

---

## 2. Project Objective

The objective is to demonstrate an end-to-end MLOps process that:

1. Validates data schema and quality before model use.
2. Measures the current champion model on new labelled data.
3. Retrains a candidate using historical and incoming data.
4. Compares champion and candidate with standard classification metrics.
5. Prevents promotion when the candidate regresses beyond defined limits.
6. Keeps champion and candidate artifacts separate.
7. Produces an auditable JSON evaluation report.
8. Runs automatically through continuous integration and scheduled MLOps
   automation.

---

## 3. Repository Structure

| Path | Purpose |
|---|---|
| [`ml_monitoring.py`](../ml_monitoring.py) | Main validation, training, evaluation, quality-gate, and CLI implementation |
| [`tests/test_monitoring.py`](../tests/test_monitoring.py) | Unit and workflow tests |
| [`artifacts/training_data.csv`](../artifacts/training_data.csv) | Historical synthetic training data |
| [`artifacts/new_batch.csv`](../artifacts/new_batch.csv) | Incoming labelled evaluation batch |
| [`artifacts/champion_model.joblib`](../artifacts/champion_model.joblib) | Existing champion model |
| [`artifacts/candidate_model.joblib`](../artifacts/candidate_model.joblib) | Separately trained candidate model |
| [`reports/evaluation_report.json`](evaluation_report.json) | Machine-readable run output |
| [`reports/valuation_report.txt`](valuation_report.txt) | Plain-text professional report |
| [`requirements.txt`](../requirements.txt) | Runtime dependency declarations |
| [`README.md`](../README.md) | Setup, workflow, and demonstration instructions |
| [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) | Continuous integration workflow |
| [`.github/workflows/mlops.yml`](../.github/workflows/mlops.yml) | Scheduled/manual monitoring and retraining workflow |

---

## 4. Technical Implementation

### 4.1 Data generation

`make_synthetic_data()` creates deterministic labelled customer records using a
controlled random seed. The generated features are:

- `tenure_months`
- `monthly_spend`
- `support_tickets`
- `usage_hours`

The target column is `churned`, and `customer_id` provides a unique record
identifier.

### 4.2 Validation

`validate_batch()` verifies:

- Required columns are present.
- Required fields contain no missing values.
- Customer IDs are not duplicated.
- The target contains only binary values, `0` and `1`.
- Row count and target distribution are recorded.

Validation errors are raised explicitly, preventing invalid data from reaching
evaluation or retraining.

### 4.3 Model pipeline

The same pipeline structure is used for the champion and candidate models:

1. `StandardScaler` standardizes numerical input features.
2. `LogisticRegression` performs binary classification.
3. `random_state=42` and `max_iter=1000` support reproducibility and reliable
   convergence.

The champion is loaded from its existing artifact. The candidate is trained
from the historical training set combined with the incoming batch and is
written to a separate file.

### 4.4 Evaluation metrics

The workflow calculates:

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** correctness of positive churn predictions.
- **Recall:** proportion of actual churn cases detected.
- **F1:** balance between precision and recall.

F1 is the primary metric because churn detection requires a balance between
finding likely churners and avoiding excessive false alerts. Accuracy is used
as a secondary non-regression check.

### 4.5 Quality gate

The candidate is eligible only when both conditions are met:

```text
candidate F1 - champion F1 >= -0.02
candidate accuracy - champion accuracy >= -0.02
```

The gate returns either:

- `candidate_eligible`, or
- `retain_champion`

The gate decision, metric deltas, rule, and reason are persisted in the JSON
report. The workflow never silently replaces the champion model.

---

## 5. End-to-End Workflow

```text
Initialize project
      |
      v
Generate labelled incoming batch
      |
      v
Validate schema, IDs, missing values, and target
      |
      v
Evaluate champion on incoming batch
      |
      v
Combine historical data and incoming batch
      |
      v
Train candidate model separately
      |
      v
Evaluate candidate and calculate metric deltas
      |
      v
Apply F1 and accuracy quality gate
      |
      +--> Candidate eligible: retain both artifacts for approved promotion
      |
      +--> Gate failed: retain champion and reject candidate promotion
```

### Reproduction commands

```bash
python -m pip install -r requirements.txt
python ml_monitoring.py init
python ml_monitoring.py generate-batch artifacts/new_batch.csv --rows 200
python ml_monitoring.py run artifacts/new_batch.csv
python -m unittest discover -s tests -v
```

For CI-style enforcement, the gate can be made command-failing when the
candidate is not eligible:

```bash
python ml_monitoring.py run artifacts/new_batch.csv --enforce-gate
```

---

## 6. Demonstrated Run Results

### 6.1 Incoming batch validation

| Validation item | Result |
|---|---:|
| Rows | 200 |
| Required columns | Present |
| Missing values | 0 |
| Duplicate customer IDs | 0 |
| Invalid target values | 0 |
| Class 0 records | 133 |
| Class 1 records | 67 |
| Validation status | **Passed** |

### 6.2 Champion and candidate comparison

| Metric | Champion | Candidate | Difference |
|---|---:|---:|---:|
| Accuracy | 0.705000 | 0.705000 | 0.000000 |
| Precision | 0.900000 | 0.833333 | -0.066667 |
| Recall | 0.134328 | 0.149254 | +0.014926 |
| F1 score | 0.233766 | 0.253165 | +0.019399 |

### 6.3 Interpretation

- The candidate maintained the champion's overall accuracy.
- The candidate detected a larger share of actual churn cases, improving
  recall from **13.43%** to **14.93%**.
- The primary F1 score increased from **0.233766** to **0.253165**.
- Precision decreased from **90.00%** to **83.33%**, so the candidate may create
  more false-positive retention alerts.
- The candidate satisfied both quality-gate thresholds.

### 6.4 Final gate decision

| Gate field | Result |
|---|---|
| F1 threshold | Candidate delta `+0.019399`, passed |
| Accuracy threshold | Candidate delta `0.000000`, passed |
| Decision | **`candidate_eligible`** |
| Champion overwritten | **False** |
| MLflow | `not_installed` |

---

## 7. Testing and Verification

The test suite was executed with:

```bash
python -m unittest discover -s tests -v
```

**Result: 7 tests passed in 0.146 seconds.**

| Test area | Verification |
|---|---|
| Batch statistics | Reports row count and target distribution |
| Missing schema | Rejects missing required columns |
| Duplicate IDs | Rejects duplicate customer identifiers |
| Missing values | Rejects incomplete records |
| Metric calculation | Returns accuracy, precision, recall, and F1 |
| Quality gate | Covers eligible and rejected decisions |
| Artifact safety | Writes candidate separately and preserves champion |

The CI workflow additionally:

- Runs on pushes to `main` and pull requests.
- Uses Python 3.11.
- Installs declared requirements.
- Compiles `ml_monitoring.py`.
- Runs the complete unit-test suite.

---

## 8. CI/CD and MLOps Automation

The project includes two GitHub Actions workflows.

### Continuous Integration

The CI workflow checks code quality at pull-request and `main`-branch level.
It installs dependencies, compiles the application, and executes all tests.

### MLOps Monitoring and Retraining

The MLOps workflow runs:

- On relevant changes to the monitoring implementation.
- Weekly using a scheduled trigger.
- Manually through `workflow_dispatch`.

It initializes the champion, generates an incoming batch, runs the enforced
quality gate, publishes a GitHub Step Summary, and uploads model/report
artifacts for 14 days. The workflow uses `if: always()` for reporting and
artifact retention, which supports investigation even when the quality gate
fails.

The design is intentionally promotion-safe: a passing gate creates an eligible
candidate artifact, while a failed gate retains the champion. No workflow step
automatically overwrites the deployed model.

---

## 9. Strengths of the Project

1. **Complete workflow:** Covers validation, evaluation, retraining, gating,
   reporting, and automation.
2. **Reproducibility:** Uses deterministic data generation and fixed model
   configuration.
3. **Artifact safety:** Separates champion and candidate model files.
4. **Explicit decisions:** Records thresholds, deltas, decision, and reason.
5. **Error visibility:** Invalid data raises clear validation errors.
6. **Test coverage:** Includes both successful paths and invalid-input cases.
7. **Operational automation:** Includes CI, schedule, manual dispatch, summary,
   and artifact upload.
8. **Auditability:** Produces a structured JSON report and human-readable
   documentation.

---

## 10. Limitations and Responsible Interpretation

- The data is synthetic and cannot establish real-world model performance.
- The demonstration evaluates the candidate on the incoming batch that also
  contributes to candidate training. Production use should include an unseen
  holdout or time-based test set.
- No probability calibration, drift detection, fairness analysis, or
  segment-level performance analysis is currently implemented.
- The precision decline should be evaluated against the business cost of
  false-positive retention outreach.
- MLflow was not installed during the demonstrated local run.
- A technical quality-gate pass is not equivalent to business approval or
  automatic deployment.

---

## 11. Recommendations

Before production promotion, the project should:

1. Evaluate on an independent, unseen, time-separated test set.
2. Add confusion-matrix and threshold analysis.
3. Define business costs for false positives and false negatives.
4. Add feature-drift, target-drift, calibration, and segment monitoring.
5. Add fairness checks for relevant customer groups.
6. Enable MLflow or an equivalent experiment-tracking service.
7. Require human approval for production promotion.
8. Retain the champion for immediate rollback and monitor post-deployment
   performance.

---

## 12. Overall Evaluation

The project satisfies the core requirements of a model-monitoring and
retraining demonstration. It provides a clear implementation, reproducible
execution path, validation safeguards, metric-based comparison, quality-gate
logic, artifact preservation, automated tests, CI/CD integration, and both
machine-readable and human-readable reports.

The demonstrated candidate is technically eligible for promotion because it
improves F1 without reducing accuracy beyond the configured tolerance.
Promotion should remain subject to independent validation and business review,
especially because precision decreased.

**Overall project status: COMPLETE AND SUCCESSFULLY VERIFIED**

---

## 13. Evidence Files

- [Source implementation](../ml_monitoring.py)
- [Automated tests](../tests/test_monitoring.py)
- [Machine-readable evaluation output](evaluation_report.json)
- [Plain-text valuation report](valuation_report.txt)
- [Project instructions](../README.md)
- [CI workflow](../.github/workflows/ci.yml)
- [MLOps workflow](../.github/workflows/mlops.yml)

