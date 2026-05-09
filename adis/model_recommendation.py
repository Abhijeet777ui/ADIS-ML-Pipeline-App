"""
ADIS - Model Recommendation Module
Auto-detects problem type (binary/multi-class/regression) and
recommends the most appropriate models based on data characteristics.
"""
import pandas as pd
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def detect_problem_type(
    df: pd.DataFrame,
    target_col: str,
    column_info: Dict,
) -> Dict[str, Any]:
    """
    Automatically detect whether this is a regression, binary classification,
    or multi-class classification problem.
    """
    if target_col not in df.columns:
        return {"problem_type": "unknown", "reason": f"Target column '{target_col}' not found."}
    
    target = df[target_col].dropna()
    n_unique = target.nunique()
    dtype = str(target.dtype)
    detected_type = column_info.get(target_col, {}).get("detected_type", "unknown")
    
    # Continuous numeric → regression
    if detected_type == "numeric" and n_unique > 20:
        problem_type = "regression"
        reason = (
            f"Target '{target_col}' is numeric with {n_unique} unique values. "
            "This suggests a continuous regression task."
        )
    elif detected_type in ("categorical", "boolean") or n_unique <= 2:
        if n_unique == 2:
            problem_type = "binary_classification"
            reason = (
                f"Target '{target_col}' has exactly 2 unique values "
                f"({target.unique().tolist()[:2]}). Binary classification."
            )
        elif 2 < n_unique <= 20:
            problem_type = "multiclass_classification"
            reason = (
                f"Target '{target_col}' has {n_unique} unique values (3–20). "
                "Multi-class classification task."
            )
        else:
            problem_type = "regression"
            reason = (
                f"Target '{target_col}' has {n_unique} unique values — treating as regression."
            )
    elif n_unique <= 2:
        problem_type = "binary_classification"
        reason = f"Target has only {n_unique} values."
    elif 2 < n_unique <= 20:
        problem_type = "multiclass_classification"
        reason = f"Target has {n_unique} discrete values."
    else:
        problem_type = "regression"
        reason = f"Target has {n_unique} unique values — treating as regression."
    
    return {
        "problem_type": problem_type,
        "target_column": target_col,
        "target_dtype": dtype,
        "target_unique_values": int(n_unique),
        "target_value_range": (
            [float(target.min()), float(target.max())] if pd.api.types.is_numeric_dtype(target)
            else target.unique().tolist()[:10]
        ),
        "reason": reason,
    }


def recommend_models(
    problem_type: str,
    n_samples: int,
    n_features: int,
    has_imbalance: bool = False,
    has_missing: bool = False,
    has_categoricals: bool = False,
) -> List[Dict[str, Any]]:
    """
    Recommend models based on problem type and dataset characteristics.
    
    Returns ordered list of model recommendations with rationale.
    """
    recommendations = []
    
    if problem_type in ("binary_classification", "multiclass_classification"):
        recommendations = _classification_recommendations(
            n_samples, n_features, has_imbalance, has_missing
        )
    elif problem_type == "regression":
        recommendations = _regression_recommendations(n_samples, n_features, has_missing)
    else:
        recommendations = [
            {
                "model": "GradientBoostingClassifier",
                "library": "sklearn",
                "priority": 1,
                "rationale": "General-purpose fallback.",
                "pros": ["Handles mixed data", "Robust"],
                "cons": ["Slower training"],
                "hyperparameter_hints": {},
            }
        ]
    
    return recommendations


def _classification_recommendations(n_samples, n_features, has_imbalance, has_missing):
    """Return classification model recommendations."""
    recs = []
    
    # Gradient Boosting - almost always excellent
    recs.append({
        "model": "GradientBoostingClassifier",
        "library": "sklearn",
        "priority": 1,
        "rationale": (
            "Gradient boosting is typically the best-performing ensemble for tabular data. "
            "Handles mixed features, missing data (partially), and imbalanced classes well."
        ),
        "pros": ["High accuracy", "Feature importance", "Handles mixed scales"],
        "cons": ["Slower to train", "Memory intensive for large datasets"],
        "hyperparameter_hints": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.1,
            "class_weight": "balanced" if has_imbalance else None,
        },
        "when_to_use": "Almost always a strong choice for tabular classification.",
    })
    
    # Random Forest
    recs.append({
        "model": "RandomForestClassifier",
        "library": "sklearn",
        "priority": 2,
        "rationale": (
            "Random forest is robust, fast to train, and rarely overfits. "
            "Parallelizable and provides reliable feature importances."
        ),
        "pros": ["Fast to train", "Parallelizable", "Low overfitting risk"],
        "cons": ["Less accurate than boosting on clean data", "Large memory footprint"],
        "hyperparameter_hints": {
            "n_estimators": 100,
            "max_depth": None,
            "class_weight": "balanced" if has_imbalance else None,
        },
        "when_to_use": "Good baseline; use when training speed matters.",
    })
    
    # Logistic Regression for smaller, lower-dimensional datasets
    if n_samples <= 50000 and n_features <= 100:
        recs.append({
            "model": "LogisticRegression",
            "library": "sklearn",
            "priority": 3,
            "rationale": (
                "Logistic regression is simple, fast, and highly interpretable. "
                "Works well on small datasets with well-scaled features."
            ),
            "pros": ["Interpretable", "Very fast", "Good calibrated probabilities"],
            "cons": ["Assumes linear decision boundary", "Requires feature scaling"],
            "hyperparameter_hints": {
                "C": 1.0,
                "solver": "lbfgs",
                "max_iter": 1000,
                "class_weight": "balanced" if has_imbalance else None,
            },
            "when_to_use": "When interpretability is important or as a fast baseline.",
        })
    
    # SVM for medium datasets
    if n_samples <= 10000:
        recs.append({
            "model": "SVC",
            "library": "sklearn",
            "priority": 4,
            "rationale": (
                "SVM with RBF kernel can find complex decision boundaries. "
                "Best for small-to-medium datasets with normalized features."
            ),
            "pros": ["Powerful non-linear classifier", "Memory efficient (kernels)"],
            "cons": ["Slow on large datasets", "Requires feature scaling", "Sensitive to C/gamma"],
            "hyperparameter_hints": {
                "C": 1.0,
                "kernel": "rbf",
                "probability": True,
                "class_weight": "balanced" if has_imbalance else None,
            },
            "when_to_use": "Small datasets (<10k rows) with rich feature interactions.",
        })
    
    return recs


def _regression_recommendations(n_samples, n_features, has_missing):
    """Return regression model recommendations."""
    recs = []
    
    recs.append({
        "model": "GradientBoostingRegressor",
        "library": "sklearn",
        "priority": 1,
        "rationale": (
            "Gradient boosting consistently achieves top regression performance on tabular data. "
            "Handles non-linear relationships and feature interactions automatically."
        ),
        "pros": ["Top accuracy", "Handles non-linearity", "Built-in feature importance"],
        "cons": ["Slower training", "Many hyperparameters"],
        "hyperparameter_hints": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "loss": "squared_error",
        },
        "when_to_use": "Primary choice for regression; outperforms linear methods when data is non-linear.",
    })
    
    recs.append({
        "model": "RandomForestRegressor",
        "library": "sklearn",
        "priority": 2,
        "rationale": (
            "Random forest for regression is fast, robust, and rarely overfits. "
            "Reliable when you need a quick, strong baseline."
        ),
        "pros": ["Parallelizable", "Robust", "No scaling needed"],
        "cons": ["Biased toward mean prediction", "Less accurate than boosting"],
        "hyperparameter_hints": {"n_estimators": 100, "max_depth": None},
        "when_to_use": "Fast, reliable baseline for regression.",
    })
    
    if n_features > 50:
        recs.append({
            "model": "Ridge",
            "library": "sklearn",
            "priority": 3,
            "rationale": (
                "Ridge regression handles high dimensionality via L2 regularization. "
                "Fast, interpretable, and works well when the true relationship is linear."
            ),
            "pros": ["Very fast", "Handles high dimensions", "Interpretable coefficients"],
            "cons": ["Assumes linearity", "Requires scaling"],
            "hyperparameter_hints": {"alpha": 1.0},
            "when_to_use": "Many features, linear relationship expected.",
        })
    else:
        recs.append({
            "model": "LinearRegression",
            "library": "sklearn",
            "priority": 3,
            "rationale": (
                "Ordinary least squares regression — the simplest interpretable baseline. "
                "Use to establish a linear benchmark."
            ),
            "pros": ["Highly interpretable", "Instant training"],
            "cons": ["No regularization", "Sensitive to outliers"],
            "hyperparameter_hints": {},
            "when_to_use": "Sanity check baseline for regression.",
        })
    
    return recs


def generate_recommendation_report(
    problem_type_info: Dict,
    model_recommendations: List[Dict],
    data_characteristics: Dict,
) -> Dict[str, Any]:
    """Generate the full model recommendation explanation."""
    pt = problem_type_info["problem_type"]
    
    evaluation_metrics = {
        "binary_classification": [
            "Accuracy", "ROC-AUC", "F1-Score (weighted)", 
            "Precision", "Recall", "Confusion Matrix"
        ],
        "multiclass_classification": [
            "Accuracy", "Macro F1-Score", "Weighted F1-Score",
            "Precision (per class)", "Confusion Matrix"
        ],
        "regression": [
            "RMSE (Root Mean Squared Error)", "MAE (Mean Absolute Error)",
            "R² Score", "MAPE (Mean Absolute Percentage Error)"
        ],
    }.get(pt, ["Accuracy"])
    
    return {
        "title": "Model Recommendation",
        "what_happened": (
            f"Detected problem type: {pt.replace('_', ' ').title()}. "
            f"Recommended {len(model_recommendations)} models in priority order."
        ),
        "why": (
            f"Problem type was determined by the target column's data type and cardinality. "
            f"Model selection considers dataset size ({data_characteristics.get('n_samples', '?')} rows), "
            f"dimensionality ({data_characteristics.get('n_features', '?')} features), "
            f"class imbalance, and interpretability needs."
        ),
        "problem_type_details": problem_type_info,
        "data_characteristics": data_characteristics,
        "top_recommendation": model_recommendations[0]["model"] if model_recommendations else None,
        "all_recommendations": model_recommendations,
        "suggested_evaluation_metrics": evaluation_metrics,
        "impact": (
            f"Top recommended model: {model_recommendations[0]['model'] if model_recommendations else 'N/A'}. "
            f"Key metrics to track: {', '.join(evaluation_metrics[:3])}."
        ),
    }


def run_model_recommendation(
    df: pd.DataFrame,
    target_col: str,
    column_info: Dict,
    eda_results: Dict,
) -> Dict[str, Any]:
    """
    Full model recommendation pipeline.
    """
    problem_type_info = detect_problem_type(df, target_col, column_info)
    problem_type = problem_type_info["problem_type"]
    
    feature_cols = [c for c in df.columns if c != target_col]
    n_samples = len(df)
    n_features = len(feature_cols)
    
    # Check characteristics
    has_imbalance = any(
        info.get("is_imbalanced", False) 
        for info in eda_results.get("imbalance", {}).values()
    )
    has_missing = df.isna().any().any()
    has_categoricals = any(
        info.get("detected_type") in ("categorical", "boolean")
        for info in column_info.values()
    )
    
    data_characteristics = {
        "n_samples": n_samples,
        "n_features": n_features,
        "has_imbalance": has_imbalance,
        "has_missing": has_missing,
        "has_categoricals": has_categoricals,
        "dataset_size_category": (
            "small" if n_samples < 1000 else
            "medium" if n_samples < 50000 else
            "large"
        ),
    }
    
    model_recommendations = recommend_models(
        problem_type, n_samples, n_features, has_imbalance, has_missing, has_categoricals
    )
    
    report = generate_recommendation_report(problem_type_info, model_recommendations, data_characteristics)
    
    return {
        "problem_type": problem_type,
        "problem_type_info": problem_type_info,
        "data_characteristics": data_characteristics,
        "model_recommendations": model_recommendations,
        "explanation": report,
        "step": "model_recommendation",
    }
