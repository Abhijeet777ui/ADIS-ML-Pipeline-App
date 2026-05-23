# ADIS Deep Guide (How Everything Works + Trade-offs)

> Audience: contributors and advanced users who want to understand what ADIS does internally, how to extend it safely, and where its design choices have trade-offs.

---

## 1) What ADIS is
**ADIS (Automated Data Intelligence System)** is an explainability-first AutoML-style library. Given a dataset (CSV) and an optional `target_column`, it runs a pipeline:

1. **Ingestion** (load CSV, detect column semantics, validate schema)
2. **Cleaning** (impute/drop/coerce + logging)
3. **EDA** (distributions, correlations, imbalance flags)
4. **Feature engineering** (transform numerics, encode categoricals, decompose datetime)
5. **Feature selection** (variance/correlation/MI style filtering)
6. **Benchmarking** (train multiple models, compare on a held-out split)
7. **AI Critic** (audit results across pipeline outputs)

It aims to return **structured dicts** for every stage plus a **human-readable report**.

---

## 2) Repository map (core modules)
Key package:

- `adis/__init__.py` – public API exports (e.g., `ADISPipeline`, stage runners)
- `adis/pipeline.py` – main orchestration class
- `adis/ingestion.py` – CSV loading + column type detection + schema validation
- `adis/cleaning.py` – missing/outlier/type coercion + action logs
- `adis/eda.py` – distributions/correlations/imbalance detection + flags
- `adis/feature_engineering.py` – transformations + encoding + datetime decomposition
- `adis/feature_selection.py` – selects features for modeling
- `adis/model_recommendation.py` – chooses problem type / candidate model strategy
- `adis/benchmarking.py` – trains and evaluates multiple models
- `adis/critic.py` – vulnerability detection across signals
- `adis/schemas.py` – Pydantic data contracts for stage outputs
- `adis/agent.py` – optional autonomous agent that can generate feature code

Tests:

- `tests/test_pipeline.py` – asserts stage output keys/types and pipeline runs

---

## 3) Public API and typical usage
### Main entry point: `ADISPipeline`
The core orchestration is:

- `pipeline = ADISPipeline(target_column="price")`
- `results = pipeline.run("data.csv")`
- `pipeline.save_report()` to write `report.json`, `report.md`, and `cleaned_data.csv`.

### Stage runners
For advanced use, modules are also exposed individually (via `adis/__init__.py`).

---

## 4) The orchestration logic (`adis/pipeline.py`)
### 4.1 Constructor
`ADISPipeline.__init__(target_column=None)` stores:
- `self.target_column`
- `self.results` (a dict storing stage outputs)
- `self.report` (narrative report structure)

### 4.2 `run(filepath)` flow
1. **Validate filepath**: requires a non-empty string.
2. **Run ingestion**: `run_ingestion(filepath)` returns:
   - `df`
   - `column_info` (semantic types + stats)
   - `validation`
   - `explanation`
3. **Target check**: if `target_column` is set but missing from `df.columns`, it raises `TargetColumnMissingError`.
4. **Cleaning**: `run_cleaning(df, col_info, outlier_action="none")`.
5. **EDA**: `run_eda(df, col_info, target_col=self.target_column)`.
6. **If no target**: skip modeling steps.
   - still runs feature engineering for reporting/EDA mode.
7. **If target exists**: modeling mode.

#### Leakage prevention design (important)
ADIS attempts a key leakage mitigation:
- it splits into `df_train` and `df_test` **before feature engineering**.
- it then recomputes EDA/distribution stats on **train only** (`train_eda`) and passes that into feature engineering so FE decisions aren’t based on test distribution.

This is a solid trade-off: safer than “EDA on full dataset”, but it reduces the amount of global information available for feature transforms.

### 4.3 Train/test splitting details
The pipeline uses:
- `train_test_split(test_size=0.2, random_state=42, stratify=...)` for classification.
- For regression, it does not stratify.

Trade-off:
- Random split is easy and fast.
- It is *not* ideal for time-series / ordered data.

---

## 5) Ingestion (`adis/ingestion.py`)
### 5.1 `load_csv`
- Ensures the file exists and ends in `.csv`
- Reads via `pd.read_csv`
- Computes metadata (file size KB, rows, columns, memory usage, etc.)

### 5.2 `detect_column_types`
Produces `column_info` keyed by column name. For each column:
- samples non-null values
- calculates:
  - `unique_count`
  - `missing_count`
  - `missing_pct`
  - `cardinality_ratio`
- calls `_infer_column_type`.

#### `_infer_column_type` heuristics
It infers semantic types based on:
- pandas dtype (`bool`, datetime keywords)
- column name keywords (`created`, `updated`, `year`, etc.)
- uniqueness and cardinality:
  - high cardinality + large average string length ⇒ `text`
  - high cardinality + very high ratio ⇒ sometimes `id`
  - low cardinality objects ⇒ `categorical`
- numeric dtype ⇒ `id` if name keyword matches + ratio is very high.

Trade-off:
- Heuristics are fast and require no external inference models.
- But they can be wrong for “messy real datasets” (e.g., IDs stored as ints without `id` in the name; or datetime columns with unusual names).

### 5.3 `validate_schema`
Flags:
- fully empty columns
- duplicate column names
- columns with >40% missing
- single-value columns
- duplicate rows
- dataset too small (<50 rows)
- more columns than rows (high-dimensional)

Outputs `validation_summary` with:
- `passed`
- `issues`
- `warnings`

---

## 6) Cleaning (`adis/cleaning.py`)
While the deep logic isn’t shown here in the earlier excerpts, tests expect that:
- `run_cleaning(...)` returns:
  - `df` cleaned
  - `log` list of actions
  - `explanation` (narrative)

Typical cleaning responsibilities implied by README/test:
- missing value handling
- deduplication
- outlier detection
- type coercion
- multiple strategies (e.g., `strategy="knn"`)

Trade-off:
- KNN imputation can preserve locality but is slower and can leak if done incorrectly.
- Simple strategies are faster and more stable.

---

## 7) EDA (`adis/eda.py`)
Tests expect:
- `distributions`
- `correlations`
- `imbalance`
- `flags`
- `explanation`

And pipeline uses EDA output as input to:
- feature engineering decisions (e.g., skewness)
- critic decisions (e.g., class imbalance ratio)

Critical leakage detail:
- pipeline recomputes train-only distributions when modeling mode.

Trade-off:
- Doing “train-only EDA for FE decisions” is safer but slightly reduces feature engineering signal.

---

## 8) Feature engineering (`adis/feature_engineering.py`)
Feature engineering is one of ADIS’s strongest differentiators because it logs transformations.

### 8.1 The `FeatureLog`
`FeatureLog` collects entries:
- `feature_name`
- `source_column`
- `transformation`
- `rationale`
- `expected_benefit`

At the end it emits:
- `feature_log`: list of all transformation entries

### 8.2 Steps performed by `run_feature_engineering`
1. **Drop ID-like columns** (`drop_id_columns`)
   - Any column where `column_info[col]["detected_type"] == "id"`
   - Adds log entries.

2. **Datetime decomposition** (`extract_datetime_features`)
   - Uses `pd.to_datetime(errors='coerce')`
   - Extracts components:
     - `__year`, `__month`, `__day`, `__weekday`, `__quarter`, `__is_weekend`
     - and optionally hour/business hour
   - Drops original datetime column

3. **Numeric transforms** (`apply_numeric_transformations`)
   - Uses skewness from `eda_results["distributions"][col]["skewness"]`
   - Creates:
     - log transform (`__log`, log1p when min == 0)
     - sqrt transform (`__sqrt`)
     - squared (`__squared`) for negative-skew cases

4. **Binning** (`apply_binning`)
   - For numeric columns with enough unique values
   - Tries quantile binning (`pd.qcut(..., q=n_bins, labels=False)`)
   - Fallback to uniform binning

5. **Categorical encoding** (`apply_one_hot_encoding`)
   - For each categorical/boolean column (excluding target):
     - if `n_unique <= max_cardinality`:
       - `pd.get_dummies(drop_first=True)`
       - drops original column
     - elif `n_unique <= 100`:
       - label encodes via categorical codes
     - else:
       - drops very high-cardinality categorical

### 8.3 Output contract
`run_feature_engineering` returns dict with:
- `df`: engineered DataFrame
- `feature_log`: list of feature transformations
- `original_columns`
- `new_features`: names created
- `dropped_columns`
- `final_columns`
- `explanation` narrative
- `step` = `feature_engineering`

Trade-offs:
- `get_dummies` can explode feature space for large low-cardinality sets.
- Label encoding can impose spurious ordinal structure for some linear models.
- Dropping high-cardinality categories can remove signal.

---

## 9) Feature selection (`adis/feature_selection.py`)
`pipeline.py` expects:
- `fs_res = run_feature_selection(df_train.copy(), target_column, problem_type=...)`
- `fs_res["df"]` transformed/filtered train df
- `fs_res["selected_features"]` list

The critic and benchmarking depend on feature selection indirectly by changing the feature set.

Trade-off:
- Feature selection can improve generalization and speed.
- But it can also remove useful weak predictors or create instability across splits.

---

## 10) Benchmarking (`adis/benchmarking.py`)
### 10.1 Core idea
- Build `X` and `y` from engineered feature set
- Train multiple models
- Evaluate metrics on `df_test`
- Rank by primary metric

### 10.2 `prepare_X_y`
- Uses all numeric columns only.
- Encodes target:
  - classification: uses pandas categorical codes
  - regression: float cast

Trade-off:
- Because `prepare_X_y` selects only numeric features, the FE step must ensure encodings are numeric.

### 10.3 Classification models
Returns a list including:
- Dummy baseline (most_frequent)
- LogisticRegression inside GridSearchCV
- RandomForestClassifier inside GridSearchCV
- GradientBoostingClassifier inside GridSearchCV

GridSearchCV introduces extra compute but provides some hyperparameter robustness.

### 10.4 Regression models
- DummyRegressor
- Ridge in GridSearchCV
- RandomForestRegressor in GridSearchCV
- GradientBoostingRegressor in GridSearchCV

### 10.5 Feature importance maps
During `train_and_evaluate`:
- extracts:
  - `feature_importances_` if available
  - else `coef_` if available
- stores:
  - `feature_importance_map`: `{feature_name: importance}` in `run_benchmarking`

This is how the critic reasons about top contributing features.

### 10.6 Output contract
`run_benchmarking` returns:
- `status`, `results`
- `best_model`: model name
- `split_info`
- `explanation`
- `step` = `benchmarking`

---

## 11) AI Critic (`adis/critic.py`)
The critic produces a vulnerability report intended to be a cross-signal audit.

### 11.1 Preflight logic
Critic checks:
- dataset size
- class imbalance ratio (from EDA)
- best model metrics (from benchmarking)
- top feature importance
- datetime-derived features

### 11.2 Vulnerability types (examples)
It can append vulnerabilities like:
- **Accuracy/ROC-AUC mismatch** under severe imbalance ("metric illusion")
- **Severe target leakage** suspicion when:
  - performance is near-perfect AND
  - a single feature dominates importance
- **Overfitting risk** when dataset is small + complex model
- **Temporal leakage risk** when engineered features include datetime component substrings (year/month/day)
- **Production readiness**: it generally won’t auto-approve production.

### 11.3 Key trade-off: heuristic vs evidence
This critic is valuable as a “rule-based audit assistant”, but:
- it may **over-trigger** (e.g., any datetime decomposition creates year/month/day features)
- it may **under-detect** leakage types that don’t fit its evidence patterns

If you want it to be trusted, you should calibrate thresholds and reduce generic triggers.

---

## 12) Agent mode (`adis/agent.py`) and code execution trade-offs
The `AutoResearchAgent` is experimental.

### 12.1 How the loop works
- Initializes baseline pipeline with ADIS
- Uses LLM (via `litellm`) to generate code string defining:
  - `build_features(df)`
- Executes generated code in Python using `exec()`
- Applies ADIS tier checks:
  - **Tier 1 preflight**: reject if strong correlation or zero variance
  - **Tier 2 proxy**: shallow tree cross-val to detect suspiciously perfect performance
  - **Tier 3**: runs more standard benchmarking (and critic audit)

### 12.2 “Sandbox” reality check
Even with:
- `ADIS_ALLOW_EXEC=1`

the execution is not OS-isolated. It can still:
- access file system (it writes to `adis_agent_sandbox/failed_attempt.py`)
- potentially consume resources

Trade-off:
- Agent mode is powerful for experimentation.
- It must be marketed carefully as “opt-in exec (not OS-isolated sandbox)” unless you add proper isolation (containers, subprocess with restrictions, timeouts, resource caps, etc.).

---

## 13) Reporting (`ADISPipeline.save_report`)
`save_report(output_dir)` writes:
- `cleaned_data.csv` (final df)
- `report.json` (structured narrative)
- `report.md` (human readable markdown)

It also includes a custom JSON encoder for numpy/pandas objects.

Trade-off:
- Great for user transparency.
- Fragile if any stage’s explanation schema changes.

---

## 14) How to extend ADIS safely (contributor notes)
### 14.1 Preserve stage contracts
Each stage should return:
- `df`
- `explanation`
- stage-specific keys (e.g., `new_features`, `selected_features`)

Critic and report generator implicitly depend on these names.

### 14.2 Add/strengthen schema enforcement
You already have Pydantic models in `adis/schemas.py`.
Recommended extension:
- validate each module output with `model_validate()` in integration tests
- validate critic output entries match `Vulnerability`

### 14.3 Avoid leakage regressions
When modifying feature engineering or EDA:
- ensure any distribution-based decisions are computed from train-only data in pipeline
- avoid using full dataset stats for transform parameters

---

## 15) Where ADIS has the biggest trade-offs today (honest summary)
1. **Heuristic critic can over-trigger**
2. **Datetime leakage heuristic should be evidence-based**, not substring-based
3. **Agent exec is powerful but not a secure sandbox**
4. **Version/API mismatches can harm credibility** (keep versions consistent)
5. **Tests verify “runs” more than “quality”** (consider adding calibrated critic tests)

---

## 16) Practical “mental model” of ADIS
Think of ADIS as:
- a deterministic data transformation pipeline with logs
- a multi-model baseline benchmarking system
- an audit layer that turns metrics + feature importance + EDA flags into human-readable risk narratives

That makes it useful for:
- learning
- building explainable baselines
- quickly identifying suspicious outcomes

It is not yet a replacement for:
- fully rigorous production MLOps validation
- leakage detection guarantees
- OS-isolated autonomous research environments

---

## 17) Suggested next improvements (if you want to mature it)
- Calibrate critic rules and reduce generic datetime warnings.
- Add “critic evidence provenance” from `feature_log` and FE operations.
- Make schemas the source of truth: validate outputs in tests.
- Align version strings across `pyproject.toml`, `__init__.py`, and report.
- Improve agent isolation (timeouts, subprocess restrictions, containerization).

---

### End of guide

