"""
ADIS - Feature Selection Module
Applies correlation filtering, variance threshold, and mutual information
to identify the most informative features. Documents every dropped feature.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def variance_threshold_filter(
    df: pd.DataFrame,
    target_col: Optional[str],
    threshold: float = 0.01,
) -> Tuple[pd.DataFrame, List[str], List[Dict]]:
    """Remove features with variance below the threshold."""
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    dropped = []
    explanations = []
    
    for col in numeric_features:
        var = float(df[col].var())
        if var < threshold:
            dropped.append(col)
            explanations.append({
                "column": col,
                "reason": "variance_threshold",
                "value": round(var, 6),
                "threshold": threshold,
                "explanation": (
                    f"'{col}' has variance {var:.6f} (below threshold {threshold}). "
                    "Near-constant features add noise without predictive value."
                ),
            })
    
    df = df.drop(columns=dropped)
    return df, dropped, explanations


def correlation_filter(
    df: pd.DataFrame,
    target_col: Optional[str],
    threshold: float = 0.95,
) -> Tuple[pd.DataFrame, List[str], List[Dict]]:
    """
    Remove features with pairwise correlation above threshold (keep one of each pair).
    The feature with lower mean absolute correlation to other features is kept.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) < 2:
        return df, [], []
    
    corr_matrix = df[numeric_features].corr().abs()
    
    # Upper triangle mask
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation > threshold
    to_drop = set()
    explanations = []
    
    for col in upper.columns:
        correlated_with = upper[col][upper[col] > threshold].index.tolist()
        for corr_col in correlated_with:
            if corr_col not in to_drop and col not in to_drop:
                # Keep the one with more variation (higher variance)
                if df[col].var() > df[corr_col].var():
                    dropped_col = corr_col
                    kept_col = col
                else:
                    dropped_col = col
                    kept_col = corr_col
                
                to_drop.add(dropped_col)
                r = corr_matrix.loc[col, corr_col]
                explanations.append({
                    "column": dropped_col,
                    "reason": "high_correlation",
                    "correlated_with": kept_col,
                    "correlation_value": round(float(r), 4),
                    "threshold": threshold,
                    "explanation": (
                        f"'{dropped_col}' is {r:.4f} correlated with '{kept_col}' "
                        f"(threshold: {threshold}). Keeping '{kept_col}' (higher variance). "
                        "Highly correlated features are redundant and can cause multicollinearity."
                    ),
                })
    
    dropped_list = list(to_drop)
    df = df.drop(columns=dropped_list)
    return df, dropped_list, explanations


def mutual_information_selection(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    top_k: Optional[int] = None,
    min_mi_score: float = 0.001,
) -> Tuple[pd.DataFrame, List[str], List[Dict], Dict[str, float]]:
    """
    Use mutual information to rank features and optionally drop low-MI features.
    
    Args:
        problem_type: 'classification' or 'regression'
        top_k: Keep only top K features (if None, keep all above min_mi_score)
        min_mi_score: Minimum MI score to keep (if top_k is None)
    """
    if target_col not in df.columns:
        return df, [], [], {}
    
    feature_cols = [c for c in df.columns if c != target_col]
    numeric_features = [c for c in feature_cols 
                        if pd.api.types.is_numeric_dtype(df[c])]
    
    if not numeric_features or len(numeric_features) < 2:
        return df, [], [], {}
    
    X = df[numeric_features].fillna(0)
    y = df[target_col]
    
    try:
        if problem_type in ("binary_classification", "multiclass_classification", "classification"):
            from sklearn.feature_selection import mutual_info_classif
            y_encoded = pd.Categorical(y).codes
            mi_scores = mutual_info_classif(X, y_encoded, random_state=42)
        else:
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(X, y, random_state=42)
    except Exception as e:
        logger.warning(f"Mutual information failed: {e}")
        return df, [], [], {}
    
    mi_dict = {col: float(score) for col, score in zip(numeric_features, mi_scores)}
    mi_series = pd.Series(mi_dict).sort_values(ascending=False)
    
    # Determine which to drop
    if top_k is not None:
        keep_cols = set(mi_series.head(top_k).index)
        drop_candidates = [c for c in numeric_features if c not in keep_cols]
    else:
        keep_cols = set(mi_series[mi_series >= min_mi_score].index)
        drop_candidates = [c for c in numeric_features if mi_dict.get(c, 0) < min_mi_score]
    
    explanations = []
    for col in drop_candidates:
        score = mi_dict.get(col, 0)
        explanations.append({
            "column": col,
            "reason": "low_mutual_information",
            "mi_score": round(score, 6),
            "threshold": top_k if top_k else min_mi_score,
            "explanation": (
                f"'{col}' has mutual information score {score:.6f} — "
                f"very little information about the target '{target_col}'. "
                "Low MI indicates the feature is largely uninformative for this task."
            ),
        })
    
    df = df.drop(columns=drop_candidates)
    return df, drop_candidates, explanations, mi_dict


def run_feature_selection(
    df: pd.DataFrame,
    target_col: Optional[str],
    problem_type: str = "classification",
    variance_thresh: float = 0.01,
    correlation_thresh: float = 0.95,
    use_mutual_info: bool = True,
    top_k_features: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full feature selection pipeline.
    
    Returns:
        Dict with selected df, all dropped features, importance scores, and explanation.
    """
    original_cols = list(df.columns)
    all_dropped = []
    all_explanations = []
    mi_scores = {}
    
    # Step 1: Variance threshold
    df, dropped_var, expl_var = variance_threshold_filter(df, target_col, variance_thresh)
    all_dropped.extend(dropped_var)
    all_explanations.extend(expl_var)
    
    # Step 2: Correlation filter
    df, dropped_corr, expl_corr = correlation_filter(df, target_col, correlation_thresh)
    all_dropped.extend(dropped_corr)
    all_explanations.extend(expl_corr)
    
    # Step 3: Mutual information (if target is available)
    if use_mutual_info and target_col and target_col in df.columns:
        df, dropped_mi, expl_mi, mi_scores = mutual_information_selection(
            df, target_col, problem_type, top_k=top_k_features
        )
        all_dropped.extend(dropped_mi)
        all_explanations.extend(expl_mi)
    
    final_cols = list(df.columns)
    feature_cols = [c for c in final_cols if c != target_col]
    
    explanation = _generate_fs_explanation(
        original_cols, all_dropped, final_cols, all_explanations, mi_scores, target_col
    )
    
    return {
        "df": df,
        "original_columns": original_cols,
        "dropped_features": all_dropped,
        "selected_features": feature_cols,
        "final_columns": final_cols,
        "drop_explanations": all_explanations,
        "mutual_information_scores": mi_scores,
        "explanation": explanation,
        "step": "feature_selection",
    }


def _generate_fs_explanation(original_cols, dropped, final_cols, explanations, mi_scores, target_col) -> dict:
    """Generate human-readable feature selection explanation."""
    by_reason = {}
    for e in explanations:
        r = e["reason"]
        by_reason[r] = by_reason.get(r, 0) + 1
    
    feature_count = len([c for c in final_cols if c != target_col])
    
    # Top features by MI
    top_features = (
        sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        if mi_scores else []
    )
    
    return {
        "title": "Feature Selection",
        "what_happened": (
            f"Reduced from {len(original_cols)} to {len(final_cols)} columns. "
            f"Dropped {len(dropped)} features across {len(set(e['reason'] for e in explanations))} criteria."
        ),
        "why": (
            "Not all features help models — many are redundant, noisy, or near-constant. "
            "Feature selection improves generalization, reduces training time, and avoids the curse of dimensionality."
        ),
        "dropped_by_reason": by_reason,
        "top_features_by_mi": [{"feature": f, "mi_score": round(s, 4)} for f, s in top_features],
        "selected_feature_count": feature_count,
        "all_drop_decisions": explanations,
        "impact": (
            f"Selected {feature_count} features for modeling. "
            f"Removed: {by_reason.get('variance_threshold', 0)} low-variance, "
            f"{by_reason.get('high_correlation', 0)} correlated, "
            f"{by_reason.get('low_mutual_information', 0)} low-MI features."
        ),
    }
