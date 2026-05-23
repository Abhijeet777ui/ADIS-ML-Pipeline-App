"""
ADIS AutoML - Simple Python Library Usage Example
This script demonstrates how to import and run the ADIS pipeline in your custom Python scripts.
"""

from adis.pipeline import ADISPipeline

def main():
    # 1. Initialize the ADIS Pipeline and specify the target column
    # For credit_sample.csv, the target column is 'class'
    target_column = "class"
    pipeline = ADISPipeline(target_column=target_column)

    # 2. Run the full pipeline on a CSV dataset
    print("Running ADIS AutoML Pipeline...")
    results = pipeline.run("credit_sample.csv")

    # 3. Access results programmatically from the returning dictionary
    print("\n--- Pipeline Completed Successfully ---")
    problem_type = results["pipeline_info"]["problem_type"]
    print(f"Problem Type: {problem_type}")
    
    # 4. View Best Performing Model
    best_model = results.get("benchmarking", {}).get("best_model", "None")
    print(f"Best Model Found: {best_model}")
    
    # 5. Extract and print evaluation metrics for the best model
    bench_results = results.get("benchmarking", {}).get("results", [])
    if bench_results:
        best_metrics = bench_results[0].get("metrics", {})
        print("\nBest Model Metrics:")
        for metric_name, val in best_metrics.items():
            print(f"  - {metric_name}: {val}")

    # 6. Save the structured JSON/Markdown reports and clean data to disk
    print("\nSaving reports to 'adis_output' directory...")
    pipeline.save_report("adis_output")
    print("Done! Check 'adis_output/' for report.json, report.md, and cleaned_data.csv")

if __name__ == "__main__":
    main()
