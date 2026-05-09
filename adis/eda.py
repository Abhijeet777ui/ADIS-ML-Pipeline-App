"""
ADIS - Exploratory Data Analysis Module
Auto-generates statistical insights, flags distribution issues, 
correlations, class imbalance, and skewness.
"""
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy import stats as scipy_stats
import logging

logger = logging.getLogger(__name__)


def analyze_distributions(df: pd.DataFrame, column_info: Dict) -> Dict[str, Dict]:
    """Compute distribution statistics for each column."""
    dist_info = {}
    
    for col, info in column_info.items():
        if col not in df.columns:
            continue
        
        col_type = info["detected_type"]
        series = df[col].dropna()
        
        if len(series) == 0:
            dist_info[col] = {"type": col_type, "error": "all values missing"}
            continue
        
        if col_type == "numeric":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            skewness = float(series.skew())
            kurtosis = float(series.kurtosis())
            
            # Normality test (if small enough)
            normality_p = None
            if len(series) <= 5000:
                try:
                    _, normality_p = scipy_stats.shapiro(series.sample(min(len(series), 5000)))
                    normality_p = float(normality_p)
                except Exception:
                    pass
            
            dist_info[col] = {
                "type": "numeric",
                "count": int(len(series)),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "normality_pvalue": round(normality_p, 4) if normality_p is not None else None,
                "is_normal": normality_p > 0.05 if normality_p is not None else None,
                "skewness_label": _label_skewness(skewness),
            }
        
        elif col_type in ("categorical", "boolean"):
            value_counts = series.value_counts()
            total = len(series)
            top_5 = {str(k): int(v) for k, v in value_counts.head(5).items()}
            entropy = float(scipy_stats.entropy(value_counts / total)) if total > 0 else 0.0
            
            dist_info[col] = {
                "type": "categorical",
                "unique_values": int(series.nunique()),
                "top_categories": top_5,
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_pct": round(value_counts.iloc[0] / total * 100, 2) if len(value_counts) > 0 else 0,
                "entropy": round(entropy, 4),
                "is_high_cardinality": series.nunique() > 20,
            }
        
        elif col_type == "datetime":
            try:
                dt_series = pd.to_datetime(series, errors='coerce').dropna()
                dist_info[col] = {
                    "type": "datetime",
                    "min_date": str(dt_series.min()),
                    "max_date": str(dt_series.max()),
                    "date_range_days": int((dt_series.max() - dt_series.min()).days),
                    "count": int(len(dt_series)),
                }
            except Exception:
                dist_info[col] = {"type": "datetime", "error": "could not parse dates"}
        
        else:
            dist_info[col] = {
                "type": col_type,
                "count": int(len(series)),
                "sample": [str(v) for v in series.head(5).tolist()],
            }
    
    return dist_info


def _label_skewness(skewness: float) -> str:
    if abs(skewness) < 0.5:
        return "approximately symmetric"
    elif abs(skewness) < 1.0:
        return "moderately skewed"
    else:
        direction = "right (positive)" if skewness > 0 else "left (negative)"
        return f"highly {direction} skewed"


def analyze_correlations(df: pd.DataFrame, column_info: Dict, threshold: float = 0.8) -> Dict[str, Any]:
    """Compute correlation matrix and flag highly correlated pairs."""
    numeric_cols = [c for c, info in column_info.items() 
                    if info["detected_type"] == "numeric" and c in df.columns]
    
    if len(numeric_cols) < 2:
        return {"message": "Fewer than 2 numeric columns; correlation not applicable."}
    
    corr_matrix = df[numeric_cols].corr()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1, c2 = numeric_cols[i], numeric_cols[j]
            r = corr_matrix.loc[c1, c2]
            if abs(r) >= threshold:
                high_corr_pairs.append({
                    "col1": c1, "col2": c2, 
                    "correlation": round(float(r), 4),
                    "direction": "positive" if r > 0 else "negative",
                })
    
    return {
        "numeric_columns": numeric_cols,
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "high_correlation_pairs": high_corr_pairs,
        "high_correlation_threshold": threshold,
        "n_high_corr_pairs": len(high_corr_pairs),
    }


def detect_class_imbalance(df: pd.DataFrame, column_info: Dict, 
                            target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect class imbalance in potential target columns.
    Checks all low-cardinality categorical columns.
    """
    
    imbalance_results = {}
    
    # Determine which columns to check
    candidate_cols = []
    if target_col and target_col in df.columns:
        candidate_cols = [target_col]
    else:
        candidate_cols = [
            c for c, info in column_info.items()
            if info["detected_type"] in ("categorical", "boolean") 
            and info["unique_count"] <= 20
            and c in df.columns
        ]
    
    for col in candidate_cols:
        vc = df[col].value_counts()
        total = vc.sum()
        
        if len(vc) < 2:
            continue
        
        majority = vc.iloc[0]
        minority = vc.iloc[-1]
        imbalance_ratio = majority / minority if minority > 0 else float('inf')
        majority_pct = majority / total * 100
        
        is_imbalanced = imbalance_ratio >= 3.0 or majority_pct >= 80
        
        imbalance_results[col] = {
            "class_distribution": {str(k): int(v) for k, v in vc.items()},
            "n_classes": int(len(vc)),
            "imbalance_ratio": round(float(imbalance_ratio), 2),
            "majority_class": str(vc.index[0]),
            "majority_pct": round(float(majority_pct), 2),
            "minority_class": str(vc.index[-1]),
            "minority_pct": round(float(minority / total * 100), 2),
            "is_imbalanced": is_imbalanced,
            "severity": _classify_imbalance(imbalance_ratio),
            "recommendation": _imbalance_recommendation(imbalance_ratio),
        }
    
    return imbalance_results


def _classify_imbalance(ratio: float) -> str:
    if ratio < 2:
        return "balanced"
    elif ratio < 5:
        return "mild"
    elif ratio < 10:
        return "moderate"
    else:
        return "severe"


def _imbalance_recommendation(ratio: float) -> str:
    if ratio < 2:
        return "No special handling needed."
    elif ratio < 5:
        return "Consider using class_weight='balanced' in models."
    elif ratio < 10:
        return "Use SMOTE or class_weight='balanced'. Weight F1-score over accuracy."
    else:
        return "Apply SMOTE/ADASYN for oversampling; use precision-recall AUC instead of ROC-AUC."


def flag_data_issues(df: pd.DataFrame, distributions: Dict, correlations: Dict) -> List[Dict]:
    """Compile a list of flagged issues from EDA findings."""
    flags = []
    
    # Skewness flags
    for col, dist in distributions.items():
        if dist.get("type") == "numeric":
            skew = abs(dist.get("skewness", 0))
            if skew > 1.0:
                flags.append({
                    "column": col,
                    "issue": "high_skewness",
                    "detail": f"Skewness = {dist['skewness']:.2f}. Consider log/sqrt transformation.",
                    "severity": "medium",
                })
            
            # Check for potential near-zero variance
            std = dist.get("std", 1)
            mean = abs(dist.get("mean", 1))
            if mean > 0 and std / mean < 0.01:
                flags.append({
                    "column": col,
                    "issue": "near_zero_variance",
                    "detail": f"CV = {std/mean:.4f}. Column has very low variance (near-constant).",
                    "severity": "low",
                })
    
    # Categorical flags
    for col, dist in distributions.items():
        if dist.get("type") == "categorical":
            if dist.get("is_high_cardinality"):
                flags.append({
                    "column": col,
                    "issue": "high_cardinality",
                    "detail": f"{dist['unique_values']} unique values. One-hot encoding may create too many features.",
                    "severity": "medium",
                })
            if dist.get("most_frequent_pct", 0) > 95:
                flags.append({
                    "column": col,
                    "issue": "dominant_category",
                    "detail": f"'{dist['most_frequent']}' appears in {dist['most_frequent_pct']:.1f}% of rows.",
                    "severity": "low",
                })
    
    # Correlation flags
    for pair in correlations.get("high_correlation_pairs", []):
        flags.append({
            "column": f"{pair['col1']} ↔ {pair['col2']}",
            "issue": "high_correlation",
            "detail": f"Correlation = {pair['correlation']}. One may be redundant for modeling.",
            "severity": "medium",
        })
    
    return flags


def run_eda(df: pd.DataFrame, column_info: Dict, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Full EDA pipeline.
    
    Returns:
        Dict with distributions, correlations, imbalance, flags, and explanation.
    """
    
    distributions = analyze_distributions(df, column_info)
    correlations = analyze_correlations(df, column_info)
    imbalance = detect_class_imbalance(df, column_info, target_col)
    flags = flag_data_issues(df, distributions, correlations)
    
    explanation = _generate_eda_explanation(distributions, correlations, imbalance, flags)
    
    return {
        "distributions": distributions,
        "correlations": correlations,
        "imbalance": imbalance,
        "flags": flags,
        "explanation": explanation,
        "step": "eda",
    }


def _generate_eda_explanation(distributions, correlations, imbalance, flags) -> dict:
    """Generate human-readable EDA explanation."""
    numeric_count = sum(1 for d in distributions.values() if d.get("type") == "numeric")
    skewed_cols = [c for c, d in distributions.items() 
                   if d.get("type") == "numeric" and abs(d.get("skewness", 0)) > 1]
    
    high_corr = correlations.get("high_correlation_pairs", [])
    imbalanced = [c for c, info in imbalance.items() if info.get("is_imbalanced")]
    
    severity_counts = {}
    for f in flags:
        s = f["severity"]
        severity_counts[s] = severity_counts.get(s, 0) + 1
    
    return {
        "title": "Exploratory Data Analysis",
        "what_happened": (
            f"Analyzed {len(distributions)} columns. "
            f"Found {numeric_count} numeric, {len(distributions) - numeric_count} categorical/other. "
            f"Generated {len(flags)} data quality flags."
        ),
        "why": (
            "EDA reveals hidden patterns, data quality issues, and informs every subsequent decision: "
            "which transformations to apply, which models to try, and how to evaluate them."
        ),
        "key_findings": {
            "skewed_columns": skewed_cols,
            "highly_correlated_pairs": len(high_corr),
            "imbalanced_targets": imbalanced,
            "flags_by_severity": severity_counts,
        },
        "impact": (
            f"{len(skewed_cols)} columns are highly skewed (log/sqrt transforms recommended). "
            f"{len(high_corr)} highly correlated pairs may cause multicollinearity. "
            + (f"Class imbalance detected in: {imbalanced}." if imbalanced else "No class imbalance detected.")
        ),
    }



