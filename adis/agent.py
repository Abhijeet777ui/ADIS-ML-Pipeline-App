import logging
import json
import os
import time
from typing import Dict, Any, List, Optional
from adis.pipeline import ADISPipeline
from adis.critic import run_critic

logger = logging.getLogger(__name__)

class AutoResearchAgent:
    """
    Autonomous ML Research Agent powered by ADIS Constraints and LLMs.
    """
    
    def __init__(
        self, 
        data_path: str, 
        target_col: str, 
        time_col: Optional[str] = None,
        model_name: str = "gpt-4o", # Model agnostic: Can be claude-3-5-sonnet, ollama/llama3, etc.
        api_key: Optional[str] = None
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.time_col = time_col
        self.model_name = model_name
        self.api_key = api_key
        
        self.learning_log: List[Dict[str, Any]] = []
        self.best_score = -float('inf')
        self.best_code = ""
        
        # We will use litellm for model-agnostic routing
        try:
            import litellm
            self.litellm = litellm
        except ImportError:
            logger.warning("litellm not installed. You will need it to run the agent loop.")
            self.litellm = None

        self.base_data = None
        self._initialize_baseline()

    def _initialize_baseline(self):
        """Run ADIS to clean data and establish safety constraints/EDA context."""
        import pandas as pd
        logger.info("Initializing ADIS Baseline Pipeline...")
        self.pipeline = ADISPipeline(target_column=self.target_col)
        
        # Run ADIS on the data to generate the initial context report
        self.pipeline_results = self.pipeline.run(self.data_path)
        
        # Initialize best_score from the baseline result
        baseline_results = self.pipeline_results.get("benchmarking", {}).get("results", [])
        if baseline_results:
            m = baseline_results[0].get("metrics", {})
            self.best_score = m.get("roc_auc", m.get("accuracy", m.get("r2_score", 0.0)))
            logger.info(f"Baseline Score established: {self.best_score:.4f}")

        # Keep the cleaned base data for the sandbox
        self.base_data = self.pipeline_results.get("cleaning", {}).get("df", pd.read_csv(self.data_path))
        
        # Extract the EDA context to feed to the LLM
        self.context_report = self.pipeline_results.get("eda", {}).get("explanation", {})
        logger.info("Baseline established. Context report generated.")

    def _generate_hypothesis(self, iteration: int = 1) -> Dict[str, str]:
        """
        Ask the LLM to propose new Python code for features or modeling.
        """
        if not self.litellm:
            raise ImportError("litellm is required to run the agent.")
            
        base_rules = f"""
        RULES:
        1. You must return a JSON object with two keys: "hypothesis" (string) and "code" (string). For Iteration 2+, include a "reflection" key.
        2. The "code" must define a function `build_features(df)`.
        3. IMPORTANT: The `build_features(df)` function MUST return the dataframe with the target column `{self.target_col}` intact.
        4. Use pandas (pd), numpy (np), itertools, math, re, and sklearn tools. They are all pre-imported.
        5. DO NOT use `pd.get_dummies(df)` on the whole dataframe. Always specify columns: `pd.get_dummies(df, columns=['col'])`.
        6. Avoid complex loops over millions of rows; prefer vectorized operations.
        7. Ensure new features don't have NaN values.
        """
        
        if iteration == 1:
            prompt = f"""
            You are an autonomous ML research agent. Your goal is to improve the model's {self.target_col} prediction.
            
            ADIS ANALYSIS:
            The pipeline has detected this is a {self.pipeline_results.get('pipeline_info', {}).get('problem_type', 'classification')} problem.
            EDA context: {json.dumps(self.context_report, indent=2)}
            
            {base_rules}
            """
        else:
            prompt = f"""
            You are an autonomous ML research agent. Your goal is to improve the model's {self.target_col} prediction.
            
            ADIS ANALYSIS:
            The pipeline has detected this is a {self.pipeline_results.get('pipeline_info', {}).get('problem_type', 'classification')} problem.
            EDA context: {json.dumps(self.context_report, indent=2)}
            
            ===== PREVIOUS ITERATION RESULTS =====
            {json.dumps(self.learning_log[-3:], indent=2)}
            
            ===== MANDATORY REFLECTION SECTION =====
            Before you propose new feature engineering code, you MUST complete the following analysis in the "reflection" key:
            1. FAILURE ROOT CAUSE: Why did previous features fail (e.g. leakage, variance, timeout)?
            2. LESSON LEARNED: What feature engineering approaches are risky here?
            3. NEW HYPOTHESIS: How will this iteration explicitly avoid the previous failures?
            
            {base_rules}
            """
        
        logger.info(f"Querying {self.model_name} for next experiment...")
        response = self.litellm.completion(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            api_key=self.api_key,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)

    def _evaluate_standard(self, df) -> Dict[str, Any]:
        """Evaluate using standard ADIS Benchmarking if no time_col is present."""
        from adis.benchmarking import run_benchmarking
        from sklearn.model_selection import train_test_split
        
        # Simple split for evaluation
        stratify = df[self.target_col] if df[self.target_col].nunique() < 20 else None
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify)
        
        problem_type = self.pipeline_results.get("pipeline_info", {}).get("problem_type", "binary_classification")
        res = run_benchmarking(
            df_train, df_test, self.target_col, problem_type=problem_type,
            data_characteristics={"has_imbalance": False}, model_recommendations=[]
        )
        
        return res

    def _execute_sandbox(self, code_str: str) -> Dict[str, Any]:
        """
        Safely execute the LLM-generated code in memory and evaluate it.
        """
        import numpy as np
        import pandas as pd
        logger.info("Executing generated code in memory sandbox...")
        
        # 1. Compile the code string in a local namespace
        # We provide common data science tools in the globals
        import math
        import re
        import datetime
        import scipy
        import scipy.stats as stats
        from sklearn.preprocessing import (
            OneHotEncoder, StandardScaler, MinMaxScaler, 
            PolynomialFeatures, LabelEncoder, PowerTransformer, QuantileTransformer
        )
        from sklearn.impute import SimpleImputer
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.compose import ColumnTransformer
        from itertools import combinations
        import itertools
        
        exec_globals = {
            "pd": pd, 
            "np": np, 
            "math": math,
            "re": re,
            "datetime": datetime,
            "scipy": scipy,
            "stats": stats,
            "itertools": itertools,
            "combinations": combinations,
            "OneHotEncoder": OneHotEncoder,
            "StandardScaler": StandardScaler,
            "MinMaxScaler": MinMaxScaler,
            "PolynomialFeatures": PolynomialFeatures,
            "LabelEncoder": LabelEncoder,
            "SimpleImputer": SimpleImputer,
            "PCA": PCA,
            "PowerTransformer": PowerTransformer,
            "QuantileTransformer": QuantileTransformer,
            "Pipeline": Pipeline,
            "make_pipeline": make_pipeline,
            "ColumnTransformer": ColumnTransformer
        } 
        local_vars = {}
        # Security Gate: Only allow exec if explicitly opted-in via env var
        if os.getenv("ADIS_ALLOW_EXEC", "0") != "1":
            msg = "Security: exec() is disabled by default. Set ADIS_ALLOW_EXEC=1 environment variable to enable agentic code execution."
            logger.error(msg)
            return {"error": msg, "pessimistic_score": 0.0}

        # Create sandbox directory if it doesn't exist
        os.makedirs("adis_agent_sandbox", exist_ok=True)

        try:
            exec(code_str, exec_globals, local_vars)
            build_features = local_vars.get("build_features")
            if not callable(build_features):
                raise ValueError("Generated code must define a function `build_features(df)`")
        except Exception as e:
            logger.error(f"Sandbox compilation error: {e}")
            with open("adis_agent_sandbox/failed_attempt.py", "w") as f:
                f.write(f"# FAILED COMPILATION: {e}\n" + code_str)
            return {"error": str(e), "pessimistic_score": 0.0}
            
        # 2. Apply feature engineering to the baseline data
        try:
            df_engineered = build_features(self.base_data.copy())
            if self.target_col not in df_engineered.columns:
                raise ValueError(f"The target column '{self.target_col}' was dropped by your code. You must preserve it.")
        except Exception as e:
            logger.error(f"Sandbox execution error: {e}")
            with open("adis_agent_sandbox/failed_attempt.py", "w") as f:
                f.write(f"# FAILED EXECUTION: {e}\n" + code_str)
            return {"error": str(e), "pessimistic_score": 0.0, "status": "SANDBOX_ERROR"}
            
        from adis.critic import run_preflight, run_proxy
        
        # Tier 1: Pre-Flight
        preflight = run_preflight(df_engineered, self.target_col, list(self.base_data.columns))
        if not preflight["is_safe"]:
            logger.warning(f"Tier 1 (Pre-Flight) REJECTED in {preflight['elapsed_seconds']:.2f}s")
            return {"error": "PREFLIGHT_REJECTED", "flags": preflight["flags"], "pessimistic_score": 0.0, "status": "TIER_1_FAIL"}
            
        # Tier 2: Proxy
        task_type = self.pipeline_results.get("pipeline_info", {}).get("problem_type", "classification")
        proxy = run_proxy(df_engineered, self.target_col, task_type=task_type)
        if proxy["is_suspicious"]:
            logger.warning(f"Tier 2 (Proxy) REJECTED in {proxy['elapsed_seconds']:.2f}s: {proxy['reason']}")
            return {"error": "PROXY_SUSPICIOUS", "flags": [{"reason": proxy["reason"]}], "pessimistic_score": proxy["proxy_score"], "status": "TIER_2_FAIL"}
            
        # 3. Evaluate the Engineered Data (Tier 3)
        bench_res = self._evaluate_standard(df_engineered)
        
        # 4. Compute Pessimistic Score (mean - 1 std)
        # For agent loop, we simplify pessimistic score to the best model's performance
        # unless we implement cross-validation inside the sandbox.
        best_metric = 0.0
        if bench_res.get("status") == "success" and bench_res.get("results"):
            best_model = bench_res["results"][0]
            m = best_model.get("metrics", {})
            best_metric = m.get("roc_auc", m.get("accuracy", m.get("r2_score", 0.0)))

        pessimistic_score = best_metric # Simplification for now
        
        logger.info(f"Sandbox Evaluation: Score={best_metric:.4f}")
        
        # 5. Merge with baseline for Critic Audit
        # The Critic needs ingestion and eda context to detect leakage/overfitting
        full_results = self.pipeline_results.copy()
        full_results.update({
            "benchmarking": bench_res,
            "feature_engineering": {"new_features": list(df_engineered.columns)},
            "pessimistic_score": pessimistic_score,
            "is_production_safe": True 
        })
        
        return full_results

    def optimize(self, iterations: int = 5):
        """The main autonomous loop."""
        for i in range(iterations):
            logger.info(f"--- Iteration {i+1}/{iterations} ---")
            
            # 1. Form Hypothesis
            try:
                proposal = self._generate_hypothesis(iteration=i+1)
                logger.info(f"Hypothesis: {proposal.get('hypothesis')}")
                if "reflection" in proposal:
                    logger.info(f"Reflection: {proposal.get('reflection')}")
            except Exception as e:
                logger.error(f"LLM Generation failed: {e}")
                continue
            
            # 2. Run Experiment in Memory
            results = self._execute_sandbox(proposal.get("code", ""))
            
            if results.get("status") in ["SANDBOX_ERROR", "TIER_1_FAIL", "TIER_2_FAIL"]:
                log_entry = {
                    "iteration": i + 1,
                    "hypothesis": proposal.get("hypothesis"),
                    "status": results.get("status"),
                    "vulnerabilities": [f.get("reason", "Unknown") for f in results.get("flags", [])] if "flags" in results else [results.get("error")]
                }
                self.learning_log.append(log_entry)
                logger.warning(f"REJECTED: {results.get('status')}. Feedback: {log_entry['vulnerabilities']}")
            else:
                # 3. Audit with ADIS Critic (Tier 3)
                critic_report = run_critic(results)
                
                # 4. Update Learning Log with results AND critic feedback
                log_entry = {
                    "iteration": i + 1,
                    "hypothesis": proposal.get("hypothesis"),
                    "score": results.get("pessimistic_score", 0),
                    "is_safe": critic_report.get("is_structurally_safe"),
                    "vulnerabilities": [v.get("issue", "Unknown Issue") for v in critic_report.get("vulnerabilities", [])]
                }
                self.learning_log.append(log_entry)
                
                if critic_report.get("is_structurally_safe") and results.get("pessimistic_score", 0) > self.best_score:
                    self.best_score = results["pessimistic_score"]
                    self.best_code = proposal.get("code", "")
                    logger.info(f"NEW BEST SCORE: {self.best_score:.4f}")
                else:
                    reason = "Lower score" if results.get("pessimistic_score", 0) <= self.best_score else "Rejected by Critic"
                    logger.warning(f"REJECTED: {reason}. Feedback: {log_entry['vulnerabilities']}")
                
            # 5. Rate Limit Safety (Free Tier)
            if i < iterations - 1:
                logger.info("Waiting 30s to respect API rate limits...")
                time.sleep(30)
                
        logger.info(f"Optimization complete. Best Score: {self.best_score}")
        return self.best_code

if __name__ == "__main__":
    # Example usage:
    # agent = AutoResearchAgent(data_path="temp_data.csv", target_col="target", model_name="gpt-4o")
    # best_features_code = agent.optimize(iterations=3)
    pass
