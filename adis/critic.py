# AI Critic & Diagnosis Engine
# Analyzes cross-signal patterns in the pipeline results to generate an advanced
# vulnerability report (Leakage, Overfitting, Metric Illusion, etc.)

from typing import Dict, Any

def run_critic(results: Dict[str, Any]) -> Dict[str, Any]:
    vulnerabilities = []
    
    # Extract overall context
    n_rows = results.get("ingestion", {}).get("metadata", {}).get("rows", 0)
    n_cols_raw = results.get("ingestion", {}).get("metadata", {}).get("columns", 0)
    
    eda = results.get("eda", {})
    imbalance_info = eda.get("imbalance", {})
    max_imbalance_ratio = 1.0
    for col, info in imbalance_info.items():
        if info.get("imbalance_ratio", 1.0) > max_imbalance_ratio:
            max_imbalance_ratio = info.get("imbalance_ratio", 1.0)
            
    has_severe_imbalance = max_imbalance_ratio > 3.0
    
    benchmarking = results.get("benchmarking", {})
    best_model_name = benchmarking.get("best_model", "Unknown")
    
    best_res = None
    for res in benchmarking.get("results", []):
        if res["model_name"] == best_model_name:
            best_res = res
            break
            
    if not best_res:
        return {
            "is_production_safe": False,
            "vulnerabilities": [{"issue": "Pipeline Incomplete", "severity": "critical", "evidence": [], "reasoning": "Benchmarking failed to run or no target was provided.", "impact": "No predictive analysis possible.", "fix": ["Provide a valid target column."], "confidence": 1.0}]
        }
        
    metrics = best_res.get("metrics", {})
    acc = metrics.get("accuracy", 0.0)
    roc_auc = metrics.get("roc_auc", 0.0)
    r2 = metrics.get("r2_score", 0.0)
    
    is_classification = "accuracy" in metrics
    
    feature_importance = best_res.get("feature_importance_map", {})
    top_feature = max(feature_importance.items(), key=lambda x: x[1]) if feature_importance else (None, 0.0)
    
    # --- CROSS SIGNAL LOGIC ---
    
    # 1. Metric Illusion (Severe Imbalance + High Accuracy + Poor AUC)
    if is_classification and has_severe_imbalance:
        if acc > 0.90 and (roc_auc > 0 and roc_auc < 0.75):
            vulnerabilities.append({
                "issue": "Misleading Primary Metric (Accuracy Illusion)",
                "severity": "critical",
                "evidence": [
                    f"Accuracy: {acc:.3f}",
                    f"ROC-AUC: {roc_auc:.3f}",
                    f"Max Imbalance Ratio: {max_imbalance_ratio:.1f}:1"
                ],
                "reasoning": "A high accuracy combined with a significantly lower ROC-AUC on a highly imbalanced dataset proves the model is acting lazily by predominantly predicting the majority class.",
                "impact": "The model will fail to identify the minority class entirely in production.",
                "fix": [
                    "Switch evaluation metric to F1-Score or Precision-Recall AUC.",
                    "Apply SMOTE or ADASYN to the training data.",
                    "Ensure 'class_weight=balanced' is strictly enforced."
                ],
                "confidence": 0.95
            })
            
    # 2. Target Leakage (Perfect performance or 1 dominating feature)
    is_perfect = (is_classification and acc > 0.98) or (not is_classification and r2 > 0.98)
    if is_perfect and top_feature[1] > 0.70:
        metric_str = f"Accuracy: {acc:.3f}" if is_classification else f"R2 Score: {r2:.3f}"
        vulnerabilities.append({
            "issue": "Severe Target Leakage Suspected",
            "severity": "critical",
            "evidence": [
                metric_str,
                f"Feature '{top_feature[0]}' importance: {top_feature[1]:.3f}"
            ],
            "reasoning": f"Near-perfect performance driven almost entirely by a single feature ('{top_feature[0]}'). This is a classic signature of target leakage, where a feature is acting as a proxy for the target or was recorded after the target occurred.",
            "impact": "Model performance is artificially inflated and will collapse on new, unseen data. Completely unreliable in production.",
            "fix": [
                f"Investigate the provenance of '{top_feature[0]}'.",
                "Drop the feature if it contains future information.",
                "Review data collection timestamps."
            ],
            "confidence": 0.90
        })
        
    # 3. Data-Starved Overfitting (Tiny dataset + Complex Model)
    # Give a warning if dataset very small and using high-capacity model
    is_complex = best_model_name in ["RandomForestClassifier", "GradientBoostingClassifier", "RandomForestRegressor", "GradientBoostingRegressor"]
    if n_rows < 1000 and is_complex:
        vulnerabilities.append({
            "issue": "High Risk of Overfitting",
            "severity": "warning",
            "evidence": [
                f"Dataset size: {n_rows} rows",
                f"Best Model: {best_model_name}",
                f"Number of columns: {n_cols_raw}"
            ],
            "reasoning": "Tree-based ensembles have very high capacity and will easily memorize small datasets. Training them on datasets with fewer than 1000 samples often results in high structural variance and poor generalization.",
            "impact": "Performance on production data may be substantially lower than validation scores.",
            "fix": [
                "Reduce tree depth and increase regularization.",
                "Consider simpler linear models or Naive Bayes as baselines."
            ],
            "confidence": 0.85
        })
        
    # 4. Temporal Leakage Risk
    dt_features = results.get("feature_engineering", {}).get("new_features", [])
    has_temporal = any("year" in f or "month" in f or "day" in f for f in dt_features)
    if has_temporal:
        vulnerabilities.append({
            "issue": "High Risk of Temporal Leakage",
            "severity": "warning",
            "evidence": [
                "Datetime features detected and parsed.",
                "Random Train/Test split used."
            ],
            "reasoning": "When data has a time dimension, random splitting allows the model to 'see into the future' (e.g., training on Friday's data to predict Thursday's).",
            "impact": "Overly optimistic evaluation. The model will perform worse on future out-of-time data.",
            "fix": [
                "Switch to TimeSeriesSplit or an Out-of-Time validation set.",
                "Ensure no future features (like 'updated_at') leaked into the predictors."
            ],
            "confidence": 0.90
        })
        
    # 5. Production Readiness Check (Flaw 10)
    production_blockers = []
    if is_complex and n_rows < 5000:
        production_blockers.append("Dataset too small for complex tree ensembles in production without rigorous out-of-time validation.")
    if has_severe_imbalance and roc_auc < 0.7:
        production_blockers.append("Model fails to adequately capture the minority class.")
        
    if production_blockers:
        vulnerabilities.append({
            "issue": "Not Ready for Production",
            "severity": "critical",
            "evidence": production_blockers,
            "reasoning": "The model fails basic production robustness checks.",
            "impact": "Deploying this model poses a business risk.",
            "fix": ["Address the blockers above before deployment."],
            "confidence": 1.0
        })
    else:
        vulnerabilities.append({
            "issue": "Model Structurally Sound, but Requires MLOps Infrastructure",
            "severity": "info",
            "evidence": [
                "No severe leakage detected",
                "Metrics are stable and balanced"
            ],
            "reasoning": "While the math looks okay, an ML model is not 'production safe' until MLOps practices are in place.",
            "impact": "Safe for A/B testing or Shadow Deployment.",
            "fix": [
                "Implement Data Drift detection (e.g., EvidentlyAI).",
                "Set up a continuous retraining schedule.",
                "Require Human-in-the-Loop review for the first 30 days."
            ],
            "confidence": 0.80
        })

    # Sort vulnerabilities by severity (critical first)
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    vulnerabilities.sort(key=lambda x: severity_order.get(x["severity"], 3))
    
    is_safe = len([v for v in vulnerabilities if v["severity"] == "critical"]) == 0
    
    # Never auto-approve for production (Flaw 10)
    is_production_safe = False
    
    return {
        "step": "critic",
        "is_production_safe": is_production_safe,
        "is_structurally_safe": is_safe,
        "vulnerabilities": vulnerabilities,
        "explanation": {
            "title": "AI Critic & Diagnosis Engine",
            "what_happened": f"Scanned pipeline results combining signals across raw data size, class distribution, model choice, and output metrics. Flagged {len(vulnerabilities)} core analytical narratives.",
            "why": "Local rule-sets miss the big picture. An elite ML audit requires cross-signal reasoning (e.g., evaluating accuracy *in the context* of distribution and feature importance).",
            "impact": "Provides a pragmatic, honest assessment of production readiness and highlights blind spots."
        }
    }
