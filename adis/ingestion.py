"""
ADIS - Data Ingestion Module
Handles CSV loading, column type detection, and schema validation.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def load_csv(filepath: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a CSV file and return the dataframe along with ingestion metadata.
    
    Args:
        filepath: Path to the CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Tuple of (DataFrame, metadata dict)
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    if not path.suffix.lower() == '.csv':
        raise ValueError(f"Expected a .csv file, got: {path.suffix}")

    df = pd.read_csv(filepath, **kwargs)
    
    metadata = {
        "filename": path.name,
        "filepath": str(path.resolve()),
        "file_size_kb": round(path.stat().st_size / 1024, 2),
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "total_cells": df.size,
        "memory_usage_kb": round(df.memory_usage(deep=True).sum() / 1024, 2),
    }
    
    logger.info(f"Loaded '{path.name}': {len(df)} rows × {len(df.columns)} columns")
    return df, metadata


def detect_column_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Detect column types: numeric, categorical, datetime, text, boolean, id.
    
    Returns:
        Dict mapping column name → {type, dtype, unique_count, sample_values, notes}
    """
    column_info = {}
    
    for col in df.columns:
        series = df[col].dropna()
        n_unique = df[col].nunique()
        n_total = len(df)
        n_missing = df[col].isna().sum()
        sample_vals = series.head(5).tolist()
        
        dtype = str(df[col].dtype)
        col_type = _infer_column_type(df[col], series, n_unique, n_total, dtype)
        
        column_info[col] = {
            "detected_type": col_type,
            "pandas_dtype": dtype,
            "unique_count": int(n_unique),
            "missing_count": int(n_missing),
            "missing_pct": round(n_missing / n_total * 100, 2) if n_total > 0 else 0.0,
            "sample_values": [str(v) for v in sample_vals],
            "cardinality_ratio": round(n_unique / n_total, 4) if n_total > 0 else 0.0,
        }
    
    return column_info


def _infer_column_type(col_series: pd.Series, non_null: pd.Series, 
                        n_unique: int, n_total: int, dtype: str) -> str:
    """Internal helper to infer the semantic type of a column."""
    col_name_lower = col_series.name.lower() if col_series.name else ""
    
    # Boolean detection
    if dtype == 'bool':
        return "boolean"
    if set(non_null.unique()).issubset({0, 1, True, False, 'true', 'false', 'yes', 'no', 'y', 'n'}):
        if n_unique <= 2:
            return "boolean"
    
    # Datetime detection
    if 'datetime' in dtype or 'date' in dtype:
        return "datetime"
    
    dt_keywords = ['date', 'time', 'year', 'month', 'day', 'hour', 'timestamp', 'created', 'updated']
    if any(kw in col_name_lower for kw in dt_keywords):
        if non_null.dtype == object:
            sample = non_null.head(20)
            try:
                pd.to_datetime(sample)
                return "datetime"
            except Exception:
                pass
    
    # Numeric types
    if pd.api.types.is_numeric_dtype(col_series):
        # Try to detect ID columns
        id_keywords = ['id', 'index', 'key', 'code', 'num', 'number', 'no']
        if any(kw == col_name_lower or col_name_lower.endswith(f'_{kw}') or col_name_lower.startswith(f'{kw}_')
               for kw in id_keywords):
            if n_unique / n_total > 0.9:
                return "id"
        return "numeric"
    
    # Object dtype - determine if categorical or text
    if dtype == 'object':
        cardinality_ratio = n_unique / n_total if n_total > 0 else 0
        
        # High cardinality → likely text
        if cardinality_ratio > 0.5 and n_unique > 20:
            avg_length = non_null.astype(str).str.len().mean()
            if avg_length > 50:
                return "text"
            if n_unique / n_total > 0.9:
                return "id"
        
        # Low cardinality → categorical
        return "categorical"
    
    return "other"


def validate_schema(df: pd.DataFrame, column_info: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Validate the dataset schema and flag potential issues.
    
    Returns:
        Dict containing validation results and warnings.
    """
    warnings = []
    issues = []
    
    total_rows = len(df)
    total_cols = len(df.columns)
    
    # Check for completely empty columns
    fully_empty = [c for c in df.columns if df[c].isna().all()]
    if fully_empty:
        issues.append(f"Completely empty columns found: {fully_empty}")
    
    # Check for duplicate column names
    dup_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
    if dup_cols:
        issues.append(f"Duplicate column names: {list(set(dup_cols))}")
    
    # High missing rate columns
    high_missing = {
        c: info["missing_pct"] 
        for c, info in column_info.items() 
        if info["missing_pct"] > 40
    }
    if high_missing:
        warnings.append(f"High missing rate (>40%): {high_missing}")
    
    # Single-value columns (zero variance)
    single_val = [c for c in df.columns if df[c].nunique() == 1]
    if single_val:
        warnings.append(f"Single-value columns (no information): {single_val}")
    
    # Duplicate rows
    n_dups = df.duplicated().sum()
    if n_dups > 0:
        warnings.append(f"{n_dups} duplicate rows found ({round(n_dups/total_rows*100, 2)}%)")
    
    # Very low row count
    if total_rows < 50:
        warnings.append(f"Very small dataset ({total_rows} rows). ML results may be unreliable.")
    
    # Too many columns relative to rows  
    if total_cols > total_rows:
        warnings.append(f"More columns ({total_cols}) than rows ({total_rows}) — high-dimensional problem.")
    
    validation_summary = {
        "passed": len(issues) == 0,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": int(n_dups),
        "fully_empty_columns": fully_empty,
        "single_value_columns": single_val,
        "high_missing_columns": high_missing,
        "issues": issues,
        "warnings": warnings,
    }
    
    return validation_summary


def run_ingestion(filepath: str, **csv_kwargs) -> Dict[str, Any]:
    """
    Full ingestion pipeline: load → detect types → validate schema.
    
    Returns:
        Dict with keys: df, metadata, column_info, validation
    """
    df, metadata = load_csv(filepath, **csv_kwargs)
    column_info = detect_column_types(df)
    validation = validate_schema(df, column_info)
    
    explanation = _generate_ingestion_explanation(metadata, column_info, validation)
    
    return {
        "df": df,
        "metadata": metadata,
        "column_info": column_info,
        "validation": validation,
        "explanation": explanation,
        "step": "ingestion",
    }


def _generate_ingestion_explanation(metadata: dict, column_info: dict, validation: dict) -> dict:
    """Generate a human-readable explanation of the ingestion step."""
    type_counts = {}
    for info in column_info.values():
        t = info["detected_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    
    type_summary = ", ".join(f"{v} {k}" for k, v in type_counts.items())
    
    return {
        "title": "Data Ingestion",
        "what_happened": (
            f"Loaded '{metadata['filename']}' ({metadata['file_size_kb']} KB). "
            f"Dataset has {metadata['rows']:,} rows and {metadata['columns']} columns "
            f"({metadata['memory_usage_kb']} KB in memory)."
        ),
        "why": (
            "Column types are detected automatically using heuristics: pandas dtype, "
            "cardinality ratio, column name keywords, and value sampling. This lets "
            "downstream steps apply the right transformations for each column."
        ),
        "column_type_summary": type_counts,
        "key_findings": validation["warnings"] + validation["issues"],
        "impact": (
            f"Detected {type_summary}. "
            + (f"Found {validation['duplicate_rows']} duplicate rows. " if validation['duplicate_rows'] > 0 else "")
            + (f"Issues flagged: {len(validation['issues'])}. " if validation['issues'] else "No critical schema issues.")
        ),
    }
