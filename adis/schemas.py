"""
ADIS — Pydantic schemas that define the data contract for every pipeline stage.

These schemas serve two purposes:
1. Documentation: Define exactly what each module returns.
2. Validation: Can be used to validate outputs at runtime via .model_validate().

Usage:
    from adis.schemas import Explanation, ColumnInfo, Vulnerability
    expl = Explanation.model_validate(some_dict)
"""

from pydantic import BaseModel
from typing import Dict, List, Any, Optional


# ─── Common Explanation Block ────────────────────────────────────────────────

class Explanation(BaseModel):
    """Every pipeline module returns an explanation with this shape."""
    title: str = ""
    what_happened: str = ""
    why: str = ""
    impact: str = ""

    model_config = {"extra": "allow"}  # modules may add extra keys like key_findings


# ─── Ingestion ───────────────────────────────────────────────────────────────

class ColumnInfo(BaseModel):
    """Schema for per-column type detection results."""
    detected_type: str
    pandas_dtype: str
    unique_count: int
    missing_count: Optional[int] = None
    missing_pct: float
    sample_values: Optional[List[str]] = None
    cardinality_ratio: Optional[float] = None


class IngestionMetadata(BaseModel):
    """File-level metadata from ingestion."""
    filename: str
    filepath: str
    file_size_kb: float
    rows: int
    columns: int
    column_names: List[str]
    total_cells: int
    memory_usage_kb: float


class ValidationSummary(BaseModel):
    """Schema validation results."""
    passed: bool
    total_rows: int
    total_columns: int
    duplicate_rows: int
    fully_empty_columns: List[str]
    single_value_columns: List[str]
    high_missing_columns: Dict[str, float]
    issues: List[str]
    warnings: List[str]


# ─── Cleaning ────────────────────────────────────────────────────────────────

class CleaningLogEntry(BaseModel):
    """One cleaning operation."""
    operation: str
    column: Optional[str] = None
    detail: str
    rows_affected: int = 0


# ─── Feature Engineering ─────────────────────────────────────────────────────

class FeatureLogEntry(BaseModel):
    """One feature creation/transformation."""
    feature_name: str
    source_column: str
    transformation: str
    rationale: str
    expected_benefit: str


# ─── Feature Selection ───────────────────────────────────────────────────────

class DropExplanation(BaseModel):
    """Why a feature was dropped."""
    column: str
    reason: str
    explanation: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    correlation_value: Optional[float] = None
    correlated_with: Optional[str] = None
    mi_score: Optional[float] = None

    model_config = {"extra": "allow"}


# ─── Benchmarking ────────────────────────────────────────────────────────────

class ModelMetrics(BaseModel):
    """Metrics for a single model."""
    model_config = {"extra": "allow"}


class ModelResult(BaseModel):
    """One model's full benchmarking result."""
    model_name: str
    metrics: Dict[str, Any]
    training_time_seconds: Optional[float] = None

    model_config = {"extra": "allow"}


# ─── Critic ──────────────────────────────────────────────────────────────────

class Vulnerability(BaseModel):
    """One vulnerability flagged by the AI Critic."""
    issue: str
    severity: str  # "critical", "warning", "info"
    evidence: List[str]
    reasoning: str
    impact: str
    fix: List[str]
    confidence: float


# ─── Pipeline Info ───────────────────────────────────────────────────────────

class PipelineInfo(BaseModel):
    """Metadata about the pipeline run itself."""
    filepath: str
    target_column: Optional[str] = None
    adis_version: str = "0.1.0"
    started_at: str
    completed_at: Optional[str] = None
    problem_type: Optional[str] = None


# ─── Validation Helpers ──────────────────────────────────────────────────────

def validate_explanation(data: dict) -> Explanation:
    """Validate that a dict matches the Explanation schema."""
    return Explanation.model_validate(data)


def validate_column_info(data: dict) -> Dict[str, ColumnInfo]:
    """Validate the column_info dict from ingestion."""
    return {col: ColumnInfo.model_validate(info) for col, info in data.items()}


def validate_vulnerability(data: dict) -> Vulnerability:
    """Validate that a dict matches the Vulnerability schema."""
    return Vulnerability.model_validate(data)


def validate_pipeline_info(data: dict) -> PipelineInfo:
    """Validate pipeline metadata."""
    return PipelineInfo.model_validate(data)
