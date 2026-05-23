"""
ADIS - Model Benchmarking Module
Trains 3-4 models + dummy baseline, computes metrics, and compiles
a comparative benchmarking report with full explanations.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


def prepare_X_y(
    df: pd.DataFrame,
    target_col: str,
    problem_type: str,
    categories: Optional[Any] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str], Any]:
    """Prepare feature matrix X and target vector y for training."""
    if target_col not in df.columns:
        return None, None, [], None
    
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Keep only numeric features (encoding already done in FE step)
    numeric_features = [c for c in feature_cols 
                        if pd.api.types.is_numeric_dtype(df[c])]
    
    if not numeric_features:
        return None, None, [], None
    
    X = df[numeric_features].fillna(0).values
    y_raw = df[target_col]
    
    # Encode target for classification
    if problem_type in ("binary_classification", "multiclass_classification"):
        cat_y = pd.Categorical(y_raw, categories=categories)
        y_encoded = cat_y.codes
        label_encoder = cat_y.categories
    else:
        y_encoded = y_raw.values.astype(float)
        label_encoder = None
    
    return X, y_encoded, numeric_features, label_encoder


def _get_classification_models(n_samples: int, has_imbalance: bool) -> List[Tuple[str, Any]]:
    """Return list of (name, model) tuples for classification."""
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import GridSearchCV
    
    cw = "balanced" if has_imbalance else None
    cv_folds = 5 if n_samples > 100 else 3
    
    models = [
        ("DummyClassifier (Baseline)", DummyClassifier(strategy="most_frequent")),
        ("LogisticRegression", GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight=cw, random_state=42),
            param_grid={'C': [0.1, 1.0, 10.0]}, cv=cv_folds, n_jobs=-1
        )),
        ("RandomForestClassifier", GridSearchCV(
            RandomForestClassifier(class_weight=cw, random_state=42),
            param_grid={'n_estimators': [50, 100, 200], 'max_depth': [None, 10]}, cv=cv_folds, n_jobs=-1
        )),
        ("GradientBoostingClassifier", GridSearchCV(
            GradientBoostingClassifier(n_estimators=100, random_state=42),
            param_grid={'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}, cv=cv_folds, n_jobs=-1
        )),
    ]
    
    return models


def _get_regression_models(n_samples: int) -> List[Tuple[str, Any]]:
    """Return list of (name, model) tuples for regression."""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GridSearchCV
    
    cv_folds = 5 if n_samples > 100 else 3
    
    models = [
        ("DummyRegressor (Baseline)", DummyRegressor(strategy="mean")),
        ("Ridge", GridSearchCV(
            Ridge(),
            param_grid={'alpha': [0.1, 1.0, 10.0]}, cv=cv_folds, n_jobs=-1
        )),
        ("RandomForestRegressor", GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid={'n_estimators': [50, 100, 200], 'max_depth': [None, 10]}, cv=cv_folds, n_jobs=-1
        )),
        ("GradientBoostingRegressor", GridSearchCV(
            GradientBoostingRegressor(n_estimators=100, random_state=42),
            param_grid={'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}, cv=cv_folds, n_jobs=-1
        )),
    ]
    
    return models


def compute_classification_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """Compute a suite of classification metrics."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score
    )
    
    n_classes = len(np.unique(y_true))
    avg = "binary" if n_classes == 2 else "weighted"
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
    }
    
    if y_prob is not None:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            metrics["roc_auc"] = round(float(auc), 4)
        except Exception:
            pass
    
    return metrics


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute a suite of regression metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    
    # MAPE (avoid division by zero)
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None
    
    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2_score": round(r2, 4),
        "mape_pct": round(mape, 2) if mape is not None else None,
    }


def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    model_name: str,
    model,
    problem_type: str,
) -> Dict[str, Any]:
    """Train a single model and evaluate it."""
    result = {"model_name": model_name, "status": "success"}
    
    try:
        # Train
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 3)
        
        # Predict
        y_pred = model.predict(X_test)
        
        if problem_type in ("binary_classification", "multiclass_classification"):
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)
                except Exception:
                    pass
            metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        else:
            metrics = compute_regression_metrics(y_test, y_pred)
        
        # Feature importances (handle GridSearchCV)
        feature_importances = None
        base_model = model.best_estimator_ if hasattr(model, "best_estimator_") else model
        
        if hasattr(base_model, "feature_importances_"):
            feature_importances = base_model.feature_importances_.tolist()
        elif hasattr(base_model, "coef_"):
            coefs = base_model.coef_
            if coefs.ndim > 1:
                coefs = np.abs(coefs).mean(axis=0)
            feature_importances = np.abs(coefs).tolist()
        
        result.update({
            "metrics": metrics,
            "training_time_seconds": train_time,
            "feature_importances": feature_importances,
        })
        
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
        result["metrics"] = {}
        logger.error(f"Model '{model_name}' failed: {e}")
    
    return result


def run_benchmarking(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    problem_type: str,
    data_characteristics: Dict,
    model_recommendations: List[Dict],
    scale_features: bool = True,
) -> Dict[str, Any]:
    """
    Full benchmarking pipeline: train models on train set → compare metrics on test set.
    """
    from sklearn.preprocessing import StandardScaler
    
    X_train, y_train, feature_names, label_encoder = prepare_X_y(df_train, target_col, problem_type)
    X_test, y_test, _, _ = prepare_X_y(df_test, target_col, problem_type, categories=label_encoder)
    
    if X_train is None or len(X_train) < 10:
        return {
            "status": "skipped",
            "reason": f"Insufficient samples for benchmarking (got {0 if X_train is None else len(X_train)}).",
            "results": [],
            "explanation": {
                "title": "Benchmarking",
                "what_happened": "Skipped — insufficient data.",
                "why": "Need more rows to train models.",
                "impact": "No benchmark results available.",
            },
            "step": "benchmarking",
        }
    
    # Feature scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    has_imbalance = data_characteristics.get("has_imbalance", False)
    
    # SMOTE for Imbalanced Data (Flaw 6)
    smote_applied = False
    if has_imbalance and problem_type in ("binary_classification", "multiclass_classification"):
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            smote_applied = True
            logger.info("Applied SMOTE to training data.")
        except ImportError:
            logger.warning("imbalanced-learn not installed. Cannot apply SMOTE. Using class weights.")
    
    # Get models
    if problem_type in ("binary_classification", "multiclass_classification"):
        models = _get_classification_models(len(X_train), has_imbalance)
    else:
        models = _get_regression_models(len(X_train))
    
    # Train each model
    all_results = []
    for model_name, model in models:
        logger.info(f"Training {model_name}...")
        result = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            model_name, model, problem_type
        )
        # Attach feature importances with names
        if result.get("feature_importances") and feature_names:
            result["feature_importance_map"] = {
                name: round(float(imp), 4)
                for name, imp in zip(feature_names, result["feature_importances"])
                if len(feature_names) == len(result["feature_importances"])
            }
        all_results.append(result)
    
    # Rank results
    ranked = _rank_results(all_results, problem_type)
    
    # Best model info
    best = ranked[0] if ranked else None
    
    # Split info
    split_info = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "n_features_used": len(feature_names),
        "feature_names": feature_names,
        "feature_scaling": "StandardScaler" if scale_features else "None",
        "smote_applied": smote_applied,
    }
    
    explanation = _generate_benchmark_explanation(ranked, problem_type, split_info, best)
    
    return {
        "status": "success",
        "results": ranked,
        "best_model": best["model_name"] if best else None,
        "split_info": split_info,
        "explanation": explanation,
        "step": "benchmarking",
    }


def _rank_results(results: List[Dict], problem_type: str) -> List[Dict]:
    """Sort results by primary metric."""
    primary_metric = {
        "binary_classification": "roc_auc",
        "multiclass_classification": "f1_score",
        "regression": "r2_score",
    }.get(problem_type, "accuracy")
    
    def sort_key(r):
        metrics = r.get("metrics", {})
        val = metrics.get(primary_metric, metrics.get("accuracy", 0))
        return val if val is not None else 0
    
    # For regression R2, higher is better. For RMSE, lower is better.
    reverse = problem_type != "regression" or primary_metric != "rmse"
    
    return sorted(
        [r for r in results if r.get("status") == "success"],
        key=sort_key,
        reverse=reverse
    )


def _generate_benchmark_explanation(results, problem_type, split_info, best) -> dict:
    """Generate human-readable benchmarking explanation."""
    primary_metric = {
        "binary_classification": "ROC-AUC",
        "multiclass_classification": "F1-Score",
        "regression": "R² Score",
    }.get(problem_type, "Accuracy")
    
    model_summary = []
    for r in results:
        m = r.get("metrics", {})
        model_summary.append({
            "model": r["model_name"],
            "primary_metric": m.get(
                "roc_auc" if problem_type == "binary_classification" else
                "f1_score" if "classification" in problem_type else "r2_score", 
                m.get("accuracy", "N/A")
            ),
            "training_time": r.get("training_time_seconds", "N/A"),
        })
    
    return {
        "title": "Model Benchmarking",
        "what_happened": (
            f"Trained {len(results)} models on {split_info['train_samples']:,} training samples "
            f"and evaluated on {split_info['test_samples']:,} test samples. "
            f"Best model: {best['model_name'] if best else 'N/A'}."
        ),
        "why": (
            "Benchmarking multiple algorithms with consistent train/test splits ensures fair comparison. "
            "The dummy baseline shows what performance would be with no model at all — "
            "any real model should significantly outperform it."
        ),
        "primary_metric": primary_metric,
        "split_info": split_info,
        "model_comparison": model_summary,
        "best_model_metrics": best.get("metrics", {}) if best else {},
        "impact": (
            f"Best performer: {best['model_name'] if best else 'N/A'} "
            f"({primary_metric}: {model_summary[0]['primary_metric'] if model_summary else 'N/A'} "
            f"vs baseline: {model_summary[-1]['primary_metric'] if len(model_summary) > 1 else 'N/A'})."
        ),
    }
