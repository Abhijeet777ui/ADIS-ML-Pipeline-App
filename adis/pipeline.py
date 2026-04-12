# main pipeline coordinator
# puts all the ML puzzle pieces together into a single workflow.
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from adis.ingestion import run_ingestion
from adis.cleaning import run_cleaning
from adis.eda import run_eda
from adis.feature_engineering import run_feature_engineering
from adis.feature_selection import run_feature_selection
from adis.model_recommendation import run_model_recommendation
from adis.benchmarking import run_benchmarking

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        """Runs the full ADIS pipeline on a CSV file."""
        logger.info(f"kicking off the pipeline for {filepath}...")
        
        # 1. Ingestion
        ingestion_res = run_ingestion(filepath)
        self.results["ingestion"] = ingestion_res
        self.report["steps"].append(ingestion_res["explanation"])
        df = ingestion_res["df"]
        col_info = ingestion_res["column_info"]
        
        # 2. Cleaning
        cleaning_res = run_cleaning(df, col_info)
        self.results["cleaning"] = cleaning_res
        self.report["steps"].append(cleaning_res["explanation"])
        df = cleaning_res["df"]
        
        # 3. EDA
        eda_res = run_eda(df, col_info, target_col=self.target_column)
        self.results["eda"] = eda_res
        self.report["steps"].append(eda_res["explanation"])
        
        # 4. Feature Engineering
        fe_res = run_feature_engineering(df, col_info, eda_res, target_col=self.target_column)
        self.results["feature_engineering"] = fe_res
        self.report["steps"].append(fe_res["explanation"])
        df = fe_res["df"]
        
        # 5. Model Recommendation & Problem Type Detection
        # Run this FIRST so we know the correct problem_type before feature selection.
        if not self.target_column:
            logger.warning("No target column specified. Skipping model recommendation and benchmarking.")
            self.report["steps"].append({
                "title": "Modeling Steps Skipped",
                "what_happened": "Model Recommendation and Benchmarking were skipped because no target column was specified.",
                "why": "Supervised learning requires a labeled target variable.",
                "impact": "The system could not recommend or train models."
            })
        else:
            # 5a. Detect problem type and get model recommendations first
            rec_res = run_model_recommendation(df, self.target_column, col_info, eda_res)
            self.results["model_recommendation"] = rec_res
            self.report["steps"].append(rec_res["explanation"])
            problem_type = rec_res["problem_type"]
            logger.info(f"Detected problem type: {problem_type}")

            # 5b. Feature Selection — now uses the correctly detected problem_type
            fs_res = run_feature_selection(df, self.target_column, problem_type=problem_type)
            self.results["feature_selection"] = fs_res
            self.report["steps"].append(fs_res["explanation"])
            df = fs_res["df"]

            # 6. Benchmarking
            bench_res = run_benchmarking(df, self.target_column, problem_type,
                                        rec_res["data_characteristics"],
                                        rec_res["model_recommendations"])
            self.results["benchmarking"] = bench_res
            self.report["steps"].append(bench_res["explanation"])
        
        self.results["final_df"] = df
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
