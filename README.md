# ADIS — Automated Data Intelligence System

[![CI](https://github.com/Abhijeet777ui/ADIS-ML-Pipeline-App/actions/workflows/ci.yml/badge.svg)](https://github.com/Abhijeet777ui/ADIS-ML-Pipeline-App/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**An explainability-first AutoML library with built-in AI vulnerability detection.**

ADIS runs a complete data science pipeline — ingestion, cleaning, EDA, feature engineering, model benchmarking — and produces a human-readable explanation at every step. Its AI Critic then audits the entire pipeline for data leakage, metric illusions, overfitting risks, and production readiness.

---

## Quick Start

### Install

```bash
pip install adis-autoresearch
```

### Basic Usage (3 lines)

```python
from adis import ADISPipeline

pipeline = ADISPipeline(target_column="target")
results = pipeline.run("data.csv")
pipeline.save_report()   # Saves report.json + report.md + cleaned_data.csv
```

### Use Individual Modules

```python
from adis import run_ingestion, run_cleaning, run_eda, run_critic

# Just ingest and inspect
result = run_ingestion("data.csv")
print(result["column_info"])     # Per-column type detection
print(result["validation"])      # Schema issues & warnings

# Clean a DataFrame
from adis import run_cleaning
cleaned = run_cleaning(df, column_info, strategy="knn")
print(cleaned["log"])            # Every cleaning action logged

# Run the AI Critic on any pipeline results
critic = run_critic(pipeline_results)
for vuln in critic["vulnerabilities"]:
    print(f"[{vuln['severity']}] {vuln['issue']}")
```

### Use the Autonomous Agent (Experimental)

```python
from adis.agent import AutoResearchAgent

agent = AutoResearchAgent(
    filepath="data.csv",
    target_column="price",
    max_iterations=10,
)
# Requires: LLM_API_KEY env var + ADIS_ALLOW_EXEC=1
results = agent.optimize()
```

---

## What Makes ADIS Different

| Feature | Typical AutoML | ADIS |
|---------|---------------|------|
| **Explainability** | Post-hoc (SHAP/LIME) | Built into every step — `what_happened`, `why`, `impact` |
| **Safety Audit** | None | AI Critic detects leakage, metric illusions, overfitting |
| **Pipeline Report** | Metrics table | Full Markdown/JSON narrative with rationale |
| **Leakage Prevention** | Manual | Automatic — train/test split before feature engineering |
| **Target** | Best score | Best score *that's safe for production* |

---

## Pipeline Stages

```
CSV File
  │
  ▼
┌─────────────────┐
│   Ingestion     │  → Type detection, schema validation, warnings
├─────────────────┤
│   Cleaning      │  → Imputation, dedup, outlier detection, type coercion
├─────────────────┤
│   EDA           │  → Distributions, correlations, class imbalance, flags
├─────────────────┤
│   Feature Eng.  │  → Log/sqrt transforms, binning, OHE, datetime decomposition
├─────────────────┤
│   Feature Sel.  │  → Variance filter, correlation filter, mutual information
├─────────────────┤
│   Benchmarking  │  → 3-4 models + dummy baseline, full metric suite
├─────────────────┤
│   AI Critic     │  → Cross-signal vulnerability detection
└─────────────────┘
  │
  ▼
JSON/Markdown Report + Cleaned CSV
```

Each stage returns a structured result dict with:
- **`df`** — The transformed DataFrame
- **`explanation`** — Human-readable `{title, what_happened, why, impact}`
- **`step`** — Stage identifier

---

## AI Critic — Vulnerability Detection

The Critic cross-references signals from across the pipeline to flag issues that single-stage analysis would miss:

| Vulnerability | What It Catches |
|--------------|-----------------|
| **Metric Illusion** | High accuracy + low AUC on imbalanced data = model is lazy |
| **Target Leakage** | Near-perfect score driven by one dominant feature |
| **Overfitting Risk** | Complex model on tiny dataset |
| **Temporal Leakage** | Random split on time-series data |
| **Production Blockers** | Composite check — is this model safe to deploy? |

```python
critic = results["critic"]
print(critic["is_structurally_safe"])   # True/False
for v in critic["vulnerabilities"]:
    print(f"  [{v['severity']}] {v['issue']} (confidence: {v['confidence']})")
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_API_KEY` | For agent only | API key for LLM-powered research agent |
| `ADIS_ALLOW_EXEC` | For agent only | Set to `1` to enable code execution sandbox |

### Optional Dependencies

```bash
pip install -e ".[ui]"          # Streamlit dashboard
pip install -e ".[agent]"       # Autonomous research agent
pip install -e ".[imbalanced]"  # SMOTE oversampling
pip install -e ".[all]"         # Everything
pip install -e ".[dev]"         # pytest + ruff
```

---

## Streamlit Dashboard

A visual frontend is included for interactive exploration:

```bash
pip install -e ".[ui]"
streamlit run app.py
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check adis/ tests/
```

---

## Project Structure

```
adis/
├── __init__.py              # Public API: ADISPipeline + all run_* functions
├── schemas.py               # Pydantic data contracts
├── pipeline.py              # Pipeline orchestrator
├── agent.py                 # Autonomous research agent (experimental)
├── ingestion.py             # CSV loading, type detection, validation
├── cleaning.py              # Imputation, dedup, outliers, coercion
├── eda.py                   # Distributions, correlations, imbalance
├── feature_engineering.py   # Transforms, binning, encoding, datetime
├── feature_selection.py     # Variance, correlation, mutual information
├── model_recommendation.py  # Problem type detection, model ranking
├── benchmarking.py          # Multi-model training + evaluation
└── critic.py                # AI vulnerability detection
```

---

## License

MIT
