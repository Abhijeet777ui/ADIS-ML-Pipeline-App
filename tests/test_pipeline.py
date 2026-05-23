"""
Tests for the ADIS pipeline — verifies that each module returns
the expected keys and types so the UI/backend contract never drifts.
"""
import pytest
import pandas as pd
import numpy as np


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """Create a small but realistic CSV for integration testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n),
        "income": np.random.normal(50000, 15000, n).round(2),
        "category": np.random.choice(["A", "B", "C"], n),
        "is_active": np.random.choice([True, False], n),
        "target": np.random.choice([0, 1], n),
    })
    # Inject some missing values
    df.loc[0:5, "income"] = np.nan
    df.loc[10:12, "category"] = np.nan

    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_df():
    """Return a clean in-memory DataFrame for unit tests."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feat_a": np.random.normal(0, 1, n),
        "feat_b": np.random.normal(5, 2, n),
        "cat_col": np.random.choice(["x", "y", "z"], n),
        "target": np.random.choice([0, 1], n),
    })


@pytest.fixture
def regression_csv(tmp_path):
    """CSV with a continuous target for regression testing."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "sqft": np.random.randint(500, 5000, n),
        "bedrooms": np.random.randint(1, 6, n),
        "price": np.random.normal(300000, 100000, n).round(2),
    })
    path = tmp_path / "regression_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def imbalanced_csv(tmp_path):
    """CSV with severe class imbalance (95/5 split)."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, n),
        "feature_b": np.random.normal(5, 2, n),
        "target": np.concatenate([np.zeros(190), np.ones(10)]).astype(int),
    })
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    path = tmp_path / "imbalanced_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def single_feature_csv(tmp_path):
    """CSV with only one feature column and a target."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "only_feature": np.random.normal(0, 1, n),
        "target": np.random.choice([0, 1], n),
    })
    path = tmp_path / "single_feature.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def all_missing_col_csv(tmp_path):
    """CSV with one column that is entirely null."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "good_feature": np.random.normal(0, 1, n),
        "empty_col": [np.nan] * n,
        "target": np.random.choice([0, 1], n),
    })
    path = tmp_path / "empty_col_data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def multiclass_csv(tmp_path):
    """CSV with a multi-class target (5 classes)."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "f1": np.random.normal(0, 1, n),
        "f2": np.random.normal(5, 2, n),
        "f3": np.random.uniform(0, 10, n),
        "target": np.random.choice(["cat", "dog", "fish", "bird", "snake"], n),
    })
    path = tmp_path / "multiclass_data.csv"
    df.to_csv(path, index=False)
    return str(path)


# ─── Unit Tests: Individual Modules ──────────────────────────────────────────

class TestIngestion:
    def test_returns_expected_keys(self, sample_csv):
        from adis.ingestion import run_ingestion
        result = run_ingestion(sample_csv)

        assert "df" in result
        assert "metadata" in result
        assert "column_info" in result
        assert "explanation" in result
        assert isinstance(result["df"], pd.DataFrame)

    def test_column_info_has_required_fields(self, sample_csv):
        from adis.ingestion import run_ingestion
        result = run_ingestion(sample_csv)

        for col_name, info in result["column_info"].items():
            assert "detected_type" in info
            assert "pandas_dtype" in info
            assert "unique_count" in info
            assert "missing_pct" in info

    def test_explanation_structure(self, sample_csv):
        from adis.ingestion import run_ingestion
        result = run_ingestion(sample_csv)
        expl = result["explanation"]

        assert "title" in expl
        assert "what_happened" in expl
        assert "why" in expl
        assert "impact" in expl

    def test_file_not_found_raises(self):
        from adis.ingestion import run_ingestion
        with pytest.raises(FileNotFoundError):
            run_ingestion("nonexistent_file.csv")

    def test_non_csv_raises(self, tmp_path):
        from adis.ingestion import run_ingestion
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .csv file"):
            run_ingestion(str(txt_file))


class TestCleaning:
    def test_returns_expected_keys(self, sample_csv):
        from adis.ingestion import run_ingestion
        from adis.cleaning import run_cleaning

        ing = run_ingestion(sample_csv)
        result = run_cleaning(ing["df"], ing["column_info"])

        assert "df" in result
        assert "log" in result
        assert "explanation" in result
        assert isinstance(result["df"], pd.DataFrame)
        assert isinstance(result["log"], list)

    def test_no_missing_values_after_cleaning(self, sample_csv):
        from adis.ingestion import run_ingestion
        from adis.cleaning import run_cleaning

        ing = run_ingestion(sample_csv)
        result = run_cleaning(ing["df"], ing["column_info"])
        numeric_cols = [c for c, i in ing["column_info"].items() if i["detected_type"] == "numeric"]
        for col in numeric_cols:
            if col in result["df"].columns:
                assert result["df"][col].isna().sum() == 0, f"{col} still has nulls"

    def test_knn_strategy(self, sample_csv):
        from adis.ingestion import run_ingestion
        from adis.cleaning import run_cleaning

        ing = run_ingestion(sample_csv)
        result = run_cleaning(ing["df"], ing["column_info"], strategy="knn")
        assert isinstance(result["df"], pd.DataFrame)

    def test_drop_strategy(self, sample_csv):
        from adis.ingestion import run_ingestion
        from adis.cleaning import run_cleaning

        ing = run_ingestion(sample_csv)
        result = run_cleaning(ing["df"], ing["column_info"], strategy="drop")
        # Should have fewer rows than original
        assert len(result["df"]) <= len(ing["df"])


class TestEDA:
    def test_returns_expected_keys(self, sample_csv):
        from adis.ingestion import run_ingestion
        from adis.cleaning import run_cleaning
        from adis.eda import run_eda

        ing = run_ingestion(sample_csv)
        clean = run_cleaning(ing["df"], ing["column_info"])
        result = run_eda(clean["df"], ing["column_info"], target_col="target")

        assert "distributions" in result
        assert "correlations" in result
        assert "imbalance" in result
        assert "flags" in result
        assert "explanation" in result


class TestCritic:
    def test_returns_expected_keys_with_no_benchmarking(self):
        from adis.critic import run_critic
        result = run_critic({"ingestion": {"metadata": {"rows": 100, "columns": 5}}})

        assert "is_production_safe" in result
        assert "vulnerabilities" in result
        assert isinstance(result["vulnerabilities"], list)

    def test_critic_always_returns_vulnerabilities(self):
        from adis.critic import run_critic
        result = run_critic({})
        assert len(result["vulnerabilities"]) > 0


# ─── Integration Tests: Full Pipeline ────────────────────────────────────────

class TestFullPipeline:
    def test_pipeline_runs_without_crash(self, sample_csv):
        """The most important test: given a CSV + target, the pipeline completes."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="target")
        results = pipeline.run(sample_csv)

        assert "ingestion" in results
        assert "cleaning" in results
        assert "eda" in results
        assert "feature_engineering" in results
        assert "feature_selection" in results
        assert "benchmarking" in results
        assert "critic" in results
        assert "pipeline_info" in results

    def test_pipeline_without_target(self, sample_csv):
        """Pipeline should still work for pure EDA when no target is given."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column=None)
        results = pipeline.run(sample_csv)

        assert "ingestion" in results
        assert "cleaning" in results
        assert "eda" in results
        assert "feature_engineering" in results
        # Modeling steps should NOT exist
        assert "benchmarking" not in results

    def test_pipeline_report_generation(self, sample_csv, tmp_path):
        """Verify save_report() produces files without crashing."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="target")
        pipeline.run(sample_csv)
        pipeline.save_report(str(tmp_path / "output"))

        assert (tmp_path / "output" / "report.json").exists()
        assert (tmp_path / "output" / "report.md").exists()
        assert (tmp_path / "output" / "cleaned_data.csv").exists()

    def test_pipeline_regression(self, regression_csv):
        """Pipeline handles continuous regression targets."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="price")
        results = pipeline.run(regression_csv)

        assert results["pipeline_info"]["problem_type"] == "regression"
        assert "benchmarking" in results

    def test_pipeline_multiclass(self, multiclass_csv):
        """Pipeline handles multi-class string targets."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="target")
        results = pipeline.run(multiclass_csv)

        assert "multiclass" in results["pipeline_info"]["problem_type"]
        assert "benchmarking" in results

    def test_pipeline_imbalanced_data(self, imbalanced_csv):
        """Pipeline completes on severely imbalanced data and the critic flags it."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="target")
        results = pipeline.run(imbalanced_csv)

        assert "benchmarking" in results
        assert "critic" in results

    def test_pipeline_single_feature(self, single_feature_csv):
        """Pipeline doesn't crash on a dataset with only one feature."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="target")
        results = pipeline.run(single_feature_csv)

        assert "benchmarking" in results

    def test_pipeline_all_missing_column(self, all_missing_col_csv):
        """Pipeline handles a column that is entirely null."""
        from adis.pipeline import ADISPipeline

        pipeline = ADISPipeline(target_column="target")
        results = pipeline.run(all_missing_col_csv)

        assert "benchmarking" in results


# ─── Input Validation Tests ──────────────────────────────────────────────────

class TestInputValidation:
    def test_invalid_filepath_type(self):
        from adis.pipeline import ADISPipeline
        pipeline = ADISPipeline(target_column="target")
        with pytest.raises(ValueError, match="non-empty string"):
            pipeline.run("")

    def test_missing_target_column(self, sample_csv):
        from adis.pipeline import ADISPipeline
        from adis.exceptions import TargetColumnMissingError
        pipeline = ADISPipeline(target_column="nonexistent_col")
        with pytest.raises(TargetColumnMissingError, match="not found in dataset"):
            pipeline.run(sample_csv)

    def test_file_not_found(self):
        from adis.pipeline import ADISPipeline
        pipeline = ADISPipeline(target_column="target")
        with pytest.raises(FileNotFoundError):
            pipeline.run("does_not_exist.csv")


# ─── Schema Validation Tests ─────────────────────────────────────────────────

class TestSchemas:
    def test_explanation_schema(self):
        from adis.schemas import Explanation
        expl = Explanation(
            title="Test", what_happened="Stuff happened",
            why="Because", impact="Big"
        )
        assert expl.title == "Test"

    def test_explanation_allows_extra_fields(self):
        from adis.schemas import Explanation
        expl = Explanation(
            title="Test", what_happened="...", why="...", impact="...",
            key_findings=["something"],
        )
        assert expl.title == "Test"

    def test_column_info_schema(self):
        from adis.schemas import ColumnInfo
        info = ColumnInfo(
            detected_type="numeric", pandas_dtype="float64",
            unique_count=42, missing_pct=1.5
        )
        assert info.detected_type == "numeric"
        assert info.missing_pct == 1.5

    def test_vulnerability_schema(self):
        from adis.schemas import Vulnerability
        vuln = Vulnerability(
            issue="Target Leakage",
            severity="critical",
            evidence=["Accuracy: 0.999"],
            reasoning="One feature dominates",
            impact="Model will fail in production",
            fix=["Remove the feature"],
            confidence=0.95,
        )
        assert vuln.severity == "critical"
        assert vuln.confidence == 0.95

    def test_validate_column_info_from_real_output(self, sample_csv):
        """Validate that actual ingestion output matches schema."""
        from adis.ingestion import run_ingestion
        from adis.schemas import validate_column_info

        result = run_ingestion(sample_csv)
        validated = validate_column_info(result["column_info"])
        assert len(validated) == len(result["column_info"])

    def test_validate_explanation_from_real_output(self, sample_csv):
        """Validate that actual explanation output matches schema."""
        from adis.ingestion import run_ingestion
        from adis.schemas import validate_explanation

        result = run_ingestion(sample_csv)
        validated = validate_explanation(result["explanation"])
        assert validated.title == "Data Ingestion"

    def test_pipeline_info_schema(self, sample_csv):
        """Validate pipeline_info from a real pipeline run."""
        from adis.pipeline import ADISPipeline
        from adis.schemas import validate_pipeline_info

        pipeline = ADISPipeline(target_column="target")
        results = pipeline.run(sample_csv)
        validated = validate_pipeline_info(results["pipeline_info"])
        from adis import __version__
        assert validated.adis_version == __version__
        assert validated.completed_at is not None
