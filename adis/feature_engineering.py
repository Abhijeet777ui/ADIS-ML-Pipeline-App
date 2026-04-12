"""
ADIS - Feature Engineering Module
Creates meaningful transformations: log/sqrt for skewed numerics,
binning, one-hot encoding, polynomial features, and datetime decomposition.
Every transformation is documented with its rationale.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureLog:
    """Tracks every feature created or transformed."""
    
    def __init__(self):
        self.entries: List[Dict] = []
    
    def log(self, feature_name: str, source_col: str, transformation: str, 
            rationale: str, expected_benefit: str):
        self.entries.append({
            "feature_name": feature_name,
            "source_column": source_col,
            "transformation": transformation,
            "rationale": rationale,
            "expected_benefit": expected_benefit,
        })
    
    def to_list(self):
        return self.entries


def apply_numeric_transformations(
    df: pd.DataFrame,
    column_info: Dict,
    eda_results: Dict,
    log: FeatureLog,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply log, sqrt, and square transformations for skewed numeric columns.
    Returns modified df and list of new column names.
    """
    df = df.copy()
    new_cols = []
    distributions = eda_results.get("distributions", {})
    
    for col, info in column_info.items():
        if info["detected_type"] != "numeric" or col not in df.columns:
            continue
        
        dist = distributions.get(col, {})
        skewness = abs(dist.get("skewness", 0))
        col_min = df[col].min()
        
        # Log transform for highly skewed positive columns
        if skewness > 1.0 and col_min >= 0:
            offset = 1 if col_min == 0 else 0
            new_col = f"{col}__log"
            df[new_col] = np.log1p(df[col]) if offset else np.log(df[col])
            new_cols.append(new_col)
            log.log(
                new_col, col, "log1p" if offset else "log",
                f"Column '{col}' has skewness={dist.get('skewness', 0):.2f} (highly right-skewed).",
                "Log transform compresses large values and makes the distribution more Gaussian, "
                "improving linear model performance and stabilizing variance."
            )
        
        # Sqrt transform for moderately skewed columns
        elif 0.5 < skewness <= 1.0 and col_min >= 0:
            new_col = f"{col}__sqrt"
            df[new_col] = np.sqrt(df[col])
            new_cols.append(new_col)
            log.log(
                new_col, col, "sqrt",
                f"Column '{col}' has moderate skewness={dist.get('skewness', 0):.2f}.",
                "Square root transform reduces moderate skewness less aggressively than log."
            )
        
        # Square transform for left-skewed (negative skew) columns
        elif skewness > 1.0 and col_min < 0:
            new_col = f"{col}__squared"
            df[new_col] = df[col] ** 2
            new_cols.append(new_col)
            log.log(
                new_col, col, "squared",
                f"Column '{col}' has strong skewness={dist.get('skewness', 0):.2f} with negative values.",
                "Squaring can help emphasize differences in negative-skewed features."
            )
    
    return df, new_cols


def apply_binning(
    df: pd.DataFrame,
    column_info: Dict,
    eda_results: Dict,
    log: FeatureLog,
    n_bins: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Discretize continuous numeric columns into bins.
    Uses quantile-based binning (equal frequency).
    """
    df = df.copy()
    new_cols = []
    distributions = eda_results.get("distributions", {})
    
    for col, info in column_info.items():
        if info["detected_type"] != "numeric" or col not in df.columns:
            continue
        
        dist = distributions.get(col, {})
        n_unique = info["unique_count"]
        
        # Only bin if there's enough unique values and it makes sense
        if n_unique < n_bins * 2:
            continue
        
        # Skip if already a count-like column (few values)
        new_col = f"{col}__bin"
        try:
            df[new_col] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
            new_cols.append(new_col)
            log.log(
                new_col, col, f"quantile_binning(n_bins={n_bins})",
                f"Column '{col}' is continuous with {n_unique} unique values.",
                "Binning creates ordinal categories that capture non-linear relationships "
                "and are robust to outliers."
            )
        except Exception:
            # Fallback to uniform binning
            try:
                df[new_col] = pd.cut(df[col], bins=n_bins, labels=False)
                new_cols.append(new_col)
                log.log(
                    new_col, col, f"uniform_binning(n_bins={n_bins})",
                    f"Column '{col}' uses uniform binning (quantile failed due to repeated values).",
                    "Uniform bins divide the value range into equal-width intervals."
                )
            except Exception:
                pass
    
    return df, new_cols


def apply_one_hot_encoding(
    df: pd.DataFrame,
    column_info: Dict,
    log: FeatureLog,
    max_cardinality: int = 15,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    One-hot encode low-cardinality categorical columns.
    High-cardinality columns are label encoded instead.
    """
    df = df.copy()
    new_cols = []
    dropped_cols = []
    
    for col, info in column_info.items():
        if info["detected_type"] not in ("categorical", "boolean") or col not in df.columns:
            continue
        
        n_unique = info["unique_count"]
        
        if n_unique <= max_cardinality:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True, dtype=int)
            # Remove the original column
            df = df.drop(columns=[col])
            dropped_cols.append(col)
            for dc in dummies.columns:
                df[dc] = dummies[dc]
                new_cols.append(str(dc))
            
            log.log(
                f"{col}__OHE_*", col, "one_hot_encoding",
                f"Column '{col}' has {n_unique} unique values (≤ {max_cardinality}).",
                f"Created {len(dummies.columns)} binary indicator columns. "
                "One-hot encoding makes categorical data usable by all ML algorithms."
            )
        elif n_unique <= 100:
            # Label encode high-cardinality
            df[f"{col}__label"] = df[col].astype('category').cat.codes
            new_cols.append(f"{col}__label")
            df = df.drop(columns=[col])
            dropped_cols.append(col)
            
            log.log(
                f"{col}__label", col, "label_encoding",
                f"Column '{col}' has {n_unique} unique values (> {max_cardinality}, ≤ 100). "
                "OHE would create too many columns.",
                "Label encoding maps categories to integers. Use with tree-based models."
            )
        else:
            # Drop very high cardinality text-like categoricals
            df = df.drop(columns=[col])
            dropped_cols.append(col)
            log.log(
                f"{col} (DROPPED)", col, "dropped_high_cardinality",
                f"Column '{col}' has {n_unique} unique values. Too many for encoding.",
                "Extremely high cardinality categorical columns are usually IDs or free text "
                "and add noise without value. They are dropped from the feature set."
            )
    
    return df, new_cols, dropped_cols


def extract_datetime_features(
    df: pd.DataFrame,
    column_info: Dict,
    log: FeatureLog,
) -> Tuple[pd.DataFrame, List[str]]:
    """Decompose datetime columns into year, month, day, hour, weekday, etc."""
    df = df.copy()
    new_cols = []
    
    for col, info in column_info.items():
        if info["detected_type"] != "datetime" or col not in df.columns:
            continue
        
        try:
            dt_col = pd.to_datetime(df[col], errors='coerce')
        except Exception:
            continue
        
        # Extract components
        components = {
            f"{col}__year": dt_col.dt.year,
            f"{col}__month": dt_col.dt.month,
            f"{col}__day": dt_col.dt.day,
            f"{col}__weekday": dt_col.dt.weekday,
            f"{col}__quarter": dt_col.dt.quarter,
            f"{col}__is_weekend": (dt_col.dt.weekday >= 5).astype(int),
        }
        
        if dt_col.dt.hour.any():
            components[f"{col}__hour"] = dt_col.dt.hour
            components[f"{col}__is_business_hour"] = (
                (dt_col.dt.hour >= 9) & (dt_col.dt.hour <= 17)
            ).astype(int)
        
        for feat_name, feat_vals in components.items():
            df[feat_name] = feat_vals
            new_cols.append(feat_name)
        
        # Drop original datetime column
        df = df.drop(columns=[col])
        
        log.log(
            ", ".join(list(components.keys())[:3]) + "...", col, "datetime_decomposition",
            f"Column '{col}' is a datetime type.",
            f"Extracted {len(components)} temporal features (year, month, day, weekday, etc.). "
            "Models cannot use raw datetime objects; decomposing reveals seasonal and cyclical patterns."
        )
    
    return df, new_cols


def drop_id_columns(
    df: pd.DataFrame,
    column_info: Dict,
    log: FeatureLog,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove ID/index-like columns that have no predictive value."""
    df = df.copy()
    dropped = []
    
    for col, info in column_info.items():
        if info["detected_type"] == "id" and col in df.columns:
            df = df.drop(columns=[col])
            dropped.append(col)
            log.log(
                f"{col} (DROPPED)", col, "drop_id_column",
                f"Column '{col}' detected as an ID/index with high cardinality ({info['unique_count']} unique values).",
                "ID columns are unique per row and provide no predictive signal. "
                "Including them would cause overfitting."
            )
    
    return df, dropped


def run_feature_engineering(
    df: pd.DataFrame,
    column_info: Dict,
    eda_results: Dict,
    target_col: Optional[str] = None,
    enable_binning: bool = True,
    enable_log_transforms: bool = True,
    max_ohe_cardinality: int = 15,
) -> Dict[str, Any]:
    """
    Full feature engineering pipeline.
    
    Returns:
        Dict with engineered df, feature log, new columns list, and explanation.
    """
    log = FeatureLog()
    original_cols = set(df.columns)
    new_feature_cols = []
    dropped_feature_cols = []
    
    # Step 1: Drop ID columns
    df, dropped_ids = drop_id_columns(df, column_info, log)
    dropped_feature_cols.extend(dropped_ids)
    
    # Step 2: Extract datetime features
    df, new_dt = extract_datetime_features(df, column_info, log)
    new_feature_cols.extend(new_dt)
    
    # Step 3: Apply numeric transforms (only if not target)
    if enable_log_transforms:
        ci_no_target = {c: v for c, v in column_info.items() if c != target_col}
        df, new_num = apply_numeric_transformations(df, ci_no_target, eda_results, log)
        new_feature_cols.extend(new_num)
    
    # Step 4: Binning
    if enable_binning:
        ci_no_target = {c: v for c, v in column_info.items() if c != target_col}
        df, new_bins = apply_binning(df, ci_no_target, eda_results, log)
        new_feature_cols.extend(new_bins)
    
    # Step 5: Categorical encoding (exclude target)
    ci_no_target = {c: v for c, v in column_info.items() if c != target_col}
    df, new_ohe, dropped_cats = apply_one_hot_encoding(
        df, ci_no_target, log, max_cardinality=max_ohe_cardinality
    )
    new_feature_cols.extend(new_ohe)
    dropped_feature_cols.extend(dropped_cats)
    
    final_cols = list(df.columns)
    explanation = _generate_fe_explanation(log, original_cols, new_feature_cols, dropped_feature_cols, final_cols)
    
    return {
        "df": df,
        "feature_log": log.to_list(),
        "original_columns": list(original_cols),
        "new_features": new_feature_cols,
        "dropped_columns": dropped_feature_cols,
        "final_columns": final_cols,
        "explanation": explanation,
        "step": "feature_engineering",
    }


def _generate_fe_explanation(log, original_cols, new_features, dropped_cols, final_cols) -> dict:
    """Generate human-readable feature engineering explanation."""
    op_types = {}
    for entry in log.to_list():
        t = entry["transformation"]
        category = (
            "numeric_transform" if any(x in t for x in ["log", "sqrt", "squared"]) else
            "encoding" if "encoding" in t else
            "binning" if "binning" in t else
            "datetime" if "datetime" in t else
            "drop"
        )
        op_types[category] = op_types.get(category, 0) + 1
    
    return {
        "title": "Feature Engineering",
        "what_happened": (
            f"Started with {len(original_cols)} columns → "
            f"ended with {len(final_cols)} features. "
            f"Created {len(new_features)} new features, dropped {len(dropped_cols)} columns."
        ),
        "why": (
            "Raw data rarely has the right representation for ML algorithms. "
            "Feature engineering transforms raw columns into signals that models can use effectively: "
            "linearizing skewed distributions, converting categories to numbers, "
            "and extracting temporal patterns from dates."
        ),
        "operations_summary": op_types,
        "new_features_sample": new_features[:10],
        "dropped_columns": dropped_cols,
        "all_transformations": log.to_list(),
        "impact": (
            f"Feature space changed: {len(original_cols)} → {len(final_cols)} columns. "
            f"Key operations: {', '.join(op_types.keys())}."
        ),
    }
