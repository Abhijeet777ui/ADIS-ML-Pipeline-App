# main pipeline coordinator
# puts all the ML puzzle pieces together into a single workflow.
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from adis.exceptions import TargetColumnMissingError
from adis.ingestion import run_ingestion
from adis.cleaning import run_cleaning
from adis.eda import run_eda
from adis.feature_engineering import run_feature_engineering
from adis.feature_selection import run_feature_selection
from adis.model_recommendation import run_model_recommendation
from adis.benchmarking import run_benchmarking
from adis.critic import run_critic

logger = logging.getLogger(__name__)

# main wrapper around all the ML logic
class ADISPipeline:
    def __init__(self, target_column: Optional[str] = None):
        self.target_column = target_column
        self.results = {}
        self.report = {
            "title": "ADIS Intelligence Report",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "steps": []
        }

    def run(self, filepath: str) -> Dict[str, Any]:
        """
        Runs the full ADIS pipeline on a CSV file.

        Args:
            filepath: Path to a CSV file.

        Returns:
            Dict containing results from every pipeline stage.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If target_column is set but not found in the data.
        """
        logger.info(f"Starting ADIS pipeline for {filepath}...")

        # --- Input validation ---
        if not isinstance(filepath, str) or not filepath.strip():
            raise ValueError("filepath must be a non-empty string pointing to a CSV file.")

        # Store pipeline metadata
        self.results["pipeline_info"] = {
            "filepath": filepath,
            "target_column": self.target_column,
            "adis_version": "0.1.1",
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        total_steps = 7 if self.target_column else 4
        pbar = tqdm(total=total_steps, desc="ADIS Pipeline", unit="stage")

        # 1. Ingestion
        pbar.set_postfix_str("Ingestion")
        ingestion_res = run_ingestion(filepath)
        self.results["ingestion"] = ingestion_res
        self.report["steps"].append(ingestion_res["explanation"])
        df = ingestion_res["df"]
        col_info = ingestion_res["column_info"]
        pbar.update(1)

        # Validate target column exists in the data
        if self.target_column and self.target_column not in df.columns:
            pbar.close()
            raise TargetColumnMissingError(
                f"Target column '{self.target_column}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )
        
        # 2. Cleaning
        pbar.set_postfix_str("Cleaning")
        cleaning_res = run_cleaning(df, col_info, outlier_action="none")
        self.results["cleaning"] = cleaning_res
        self.report["steps"].append(cleaning_res["explanation"])
        df = cleaning_res["df"]
        pbar.update(1)
        
        # 3. EDA (full dataset — for reporting purposes)
        pbar.set_postfix_str("EDA")
        eda_res = run_eda(df, col_info, target_col=self.target_column)
        self.results["eda"] = eda_res
        self.report["steps"].append(eda_res["explanation"])
        pbar.update(1)
        
        # 4. Model Recommendation & Problem Type Detection
        if not self.target_column:
            logger.warning("No target column specified. Skipping modeling steps.")
            self.report["steps"].append({
                "title": "Modeling Steps Skipped",
                "what_happened": "Model Recommendation and Benchmarking were skipped because no target column was specified.",
                "why": "Supervised learning requires a labeled target variable.",
                "impact": "The system could not recommend or train models."
            })

            # Still run FE on full data for EDA-only mode
            pbar.set_postfix_str("Feature Engineering")
            fe_res = run_feature_engineering(df, col_info, eda_res, target_col=self.target_column)
            self.results["feature_engineering"] = fe_res
            self.report["steps"].append(fe_res["explanation"])
            df = fe_res["df"]
            pbar.update(1)

        else:
            # 4a. Detect problem type
            pbar.set_postfix_str("Model Recommendation")
            rec_res = run_model_recommendation(df, self.target_column, col_info, eda_res)
            self.results["model_recommendation"] = rec_res
            self.report["steps"].append(rec_res["explanation"])
            problem_type = rec_res["problem_type"]
            self.results["pipeline_info"]["problem_type"] = problem_type
            logger.info(f"Detected problem type: {problem_type}")

            # --- LEAKAGE FIX: Train/Test Split BEFORE Feature Engineering ---
            from sklearn.model_selection import train_test_split
            stratify_col = df[self.target_column] if problem_type != "regression" else None
            df_train, df_test = train_test_split(
                df, test_size=0.2, random_state=42, stratify=stratify_col
            )

            # 4b. Run EDA on TRAIN DATA ONLY for FE decisions (prevents distributional leakage)
            from adis.eda import analyze_distributions
            train_eda = dict(eda_res)  # copy the full EDA for structure
            train_eda["distributions"] = analyze_distributions(df_train, col_info)
            logger.info("Computed train-only distributions for feature engineering.")

            # 4c. Feature Engineering — uses train-only distribution stats
            pbar.set_postfix_str("Feature Engineering")
            fe_res_train = run_feature_engineering(df_train, col_info, train_eda, target_col=self.target_column)
            self.results["feature_engineering"] = fe_res_train
            self.report["steps"].append(fe_res_train["explanation"])
            df_train = fe_res_train["df"]

            # Apply the same transforms to test data (uses train EDA stats for decisions)
            fe_res_test = run_feature_engineering(df_test, col_info, train_eda, target_col=self.target_column)
            df_test = fe_res_test["df"]

            # Align columns: test may be missing columns that train created (or vice versa)
            common_cols = [c for c in df_train.columns if c in df_test.columns]
            df_train = df_train[common_cols]
            df_test = df_test[common_cols]
            pbar.update(1)
            
            # 5. Feature Selection — fit on train only
            pbar.set_postfix_str("Feature Selection")
            fs_res = run_feature_selection(df_train.copy(), self.target_column, problem_type=problem_type)
            self.results["feature_selection"] = fs_res
            self.report["steps"].append(fs_res["explanation"])
            df_train = fs_res["df"]
            
            # Transform test set (keep only selected features + target)
            selected_features = fs_res["selected_features"]
            keep_cols = selected_features + [self.target_column]
            keep_cols = [c for c in keep_cols if c in df_test.columns]
            df_test = df_test[keep_cols]
            pbar.update(1)

            # 6. Benchmarking
            pbar.set_postfix_str("Benchmarking")
            bench_res = run_benchmarking(
                df_train, df_test, self.target_column, problem_type,
                rec_res["data_characteristics"],
                rec_res["model_recommendations"]
            )
            self.results["benchmarking"] = bench_res
            self.report["steps"].append(bench_res["explanation"])
            pbar.update(1)
            
            # 7. AI Critic & Diagnosis
            pbar.set_postfix_str("AI Critic")
            critic_res = run_critic(self.results)
            self.results["critic"] = critic_res
            self.report["steps"].append(critic_res["explanation"])
            pbar.update(1)
            
        pbar.close()
        self.results["final_df"] = df
        self.results["pipeline_info"]["completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.results

    def save_report(self, output_dir: str = "adis_output"):
        """Saves the cleaned data and the human-readable report."""
        out_path = Path(output_dir)
        out_path.mkdir(exist_ok=True)
        
        # Save Clean CSV
        if "final_df" in self.results:
            self.results["final_df"].to_csv(out_path / "cleaned_data.csv", index=False)
            logger.info(f"Cleaned data saved to {out_path / 'cleaned_data.csv'}")
            
        # Custom JSON Encoder for numpy types
        class ADISEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, (datetime, pd.Timestamp)):
                    return obj.isoformat()
                return super(ADISEncoder, self).default(obj)

        # Save JSON Report
        try:
            with open(out_path / "report.json", "w", encoding='utf-8') as f:
                json.dump(self.report, f, indent=4, cls=ADISEncoder)
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
            # Fallback for extreme cases
            with open(out_path / "report.json", "w", encoding='utf-8') as f:
                json.dump({"error": "Serialization failed", "msg": str(e)}, f)
        
        # Generate Markdown Report
        md_report = f"# {self.report['title']}\n"
        md_report += f"**Generated At:** {self.report['generated_at']}\n\n"
        
        for step in self.report["steps"]:
            md_report += f"## {step['title']}\n"
            md_report += f"**What Happened:** {step['what_happened']}\n\n"
            md_report += f"**Rationale:** {step['why']}\n\n"
            if "impact" in step:
                md_report += f"**Impact:** {step['impact']}\n\n"
            md_report += "---\n\n"
            
        with open(out_path / "report.md", "w", encoding='utf-8') as f:
            f.write(md_report)
            
        logger.info(f"Reports saved to {out_path}")

if __name__ == "__main__":
    # Example usage
    # pipeline = ADISPipeline(target_column="target")
    # pipeline.run("data.csv")
    # pipeline.save_report()
    pass
