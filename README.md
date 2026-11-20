# Model-quality monitoring and retraining

This project demonstrates a complete classification monitoring workflow using
reproducible synthetic customer-churn data. It keeps the deployed **champion**
model separate from a newly trained **candidate** model.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run the workflow

```bash
python ml_monitoring.py init
python ml_monitoring.py generate-batch artifacts/new_batch.csv --rows 200
python ml_monitoring.py run artifacts/new_batch.csv
```

The commands create:

- `artifacts/training_data.csv`: the original labelled training set.
- `artifacts/champion_model.joblib`: the existing model and preprocessing pipeline.
- `artifacts/candidate_model.joblib`: the retrained model, never written over the champion.
- `reports/evaluation_report.json`: validation details, metrics, MLflow status, and
  the quality-gate decision.

### Workflow decisions

1. **Prepare data:** required columns, missing values, duplicate customer IDs,
   row count, and churn target distribution are validated and recorded.
2. **Evaluate champion:** accuracy, precision, recall, and F1 are calculated on
   the labelled incoming batch.
3. **Retrain:** the original training set and incoming batch are combined and
   fitted with the same `StandardScaler` + `LogisticRegression` pipeline.
4. **Quality gate:** the candidate is eligible only when its F1 and accuracy are
   no more than 0.02 below the champion. F1 is the primary criterion because it
   balances precision and recall for the churn class; the accuracy floor prevents
   a large overall regression. A failed gate records `retain_champion`.
5. **Promotion:** the workflow records eligibility but never automatically
   overwrites the champion.

If MLflow is installed, candidate metrics and the gate decision are logged to an
MLflow run. Without MLflow, the report explicitly records `not_installed`.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests cover batch statistics, duplicate-ID rejection, artifact creation,
report creation, and preservation of the champion artifact.

## CI/CD and MLOps automation

The repository uses two GitHub Actions workflows:

- **Continuous Integration** ([ci.yml](.github/workflows/ci.yml)) runs on every
  `main` push and pull request. It installs pinned runtime requirements,
  compiles the application, and runs the complete test suite.
- **MLOps Monitoring and Retraining** ([mlops.yml](.github/workflows/mlops.yml))
  runs on relevant `main` changes, weekly on a schedule, or manually when
  repository Actions permissions allow dispatch. It validates a new labelled
  batch, evaluates the champion, trains a candidate, enforces the quality gate,
  publishes a run summary, and retains model/report artifacts for 14 days.

The MLOps workflow is intentionally promotion-safe: a failed gate exits the
job while retaining the champion, and a passing gate produces an eligible
candidate artifact for an approved deployment process. No GitHub Action
silently replaces the champion model.

## Assessment demonstration checklist

For a live demonstration, show the terminal output from `init`,
`generate-batch`, and `run`, then open `reports/evaluation_report.json`. Point
out the validation summary, both metric sets, `candidate_eligible`, and
`champion_overwritten: false`. Finally, show both model files in `artifacts/`
and run the test command to demonstrate that validation, evaluation,
retraining, and both quality-gate decisions are covered.