# ADIS (Automated Data Intelligence System)

Welcome to **ADIS**, a humble yet robust automated machine learning tool designed to simplify the end-to-end data pipeline. Whether you are a beginner looking to understand your data or an experienced data scientist wanting a quick baseline, ADIS takes care of the tedious parts of data preparation, cleaning, analysis, and model benchmarking. Use the app using this link https://adis-ml-pipeline-app-cvdooqf5gwnhkldurvjzuu.streamlit.app/

## What is it?
ADIS is a Streamlit-based web application that accepts any standard CSV dataset and automatically performs an end-to-end data science workflow. At each step, the system not only applies data transformations but also provides a **clear, plain-English explanation** of what it did, why it did it, and the impact it had on your data.

## Pipeline Steps

1. **Ingestion & Summary**: Safely loads your dataset, infers logical data types, and profiles memory usage.
2. **Data Cleaning**: Intelligently handles missing values, clips outliers, deduplicates rows, and coerces messy string formats.
3. **Exploratory Data Analysis (EDA)**: Analyzes distributions, computes correlations, flags skewness, and detects dataset structure problems.
4. **Feature Engineering & Selection**: Generates new insightful features, removes redundant or highly correlated ones, and handles categorical encoding safely.
5. **Model Recommendation & Benchmarking**: Recommends appropriate algorithms based on problem type (regression, binary, or multiclass classification) and dataset size, and directly benchmarks their performance.

## System Outputs & MLOps Integration

While the Streamlit UI aggregates everything for visual consumption, ADIS natively acts as a decoupled backend pipeline. When parsing data, it generates structured artifacts:
* **`report.json`**: Acts as an API-ready payload containing every mathematical decision, feature drop, and metric result. This makes ADIS perfectly positioned to plug into automated enterprise logging tools like MLflow, Weights & Biases, or any external UI/frontend framework.
* **`report.md`**: A localized, markdown-rendered version of the entire decision-making process for physical documentation.
* **`cleaned_data.csv`**: A ready-to-use version of the data after imputation and feature engineering.

## Edge Cases Handled

During development, we paid special attention to several real-world data edge cases to ensure the pipeline doesn't break easily:

* **Unspecified or Missing Target Column**: If a user uploads data simply for analysis without specifying a target to predict, ADIS gracefully defaults to purely analytical behavior, skipping the modeling components without crashing.
* **Configurable Missing Value Strategies**: The cleaning module supports multiple imputation strategies — `auto` (median for numeric, mode for categorical), `mean`, `mode`, `knn`, and `drop`. KNN imputation is available as an explicit strategy option, not an automatic fallback. The default `auto` strategy uses median/mode, which is robust and safe for most datasets.
* **Severe Target Imbalance**: Automatically detected during EDA. The system flags minority class issues and recommends strategies like `class_weight='balanced'` or SMOTE to prevent the model from blindly guessing the majority class.
* **Messy Data Formatting**: Intelligently strips currency formats (e.g., "$1,234.56"), interprets various boolean synonyms ('y', 'yes', 'True', '1'), and converts them efficiently to computational types.
* **High Cardinality & Near-Zero Variance**: Discovers and flags categorical columns with too many unique values (which would explode dimensionality through one-hot encoding) and features with near-zero variance that provide no predictive power.
* **Outlier Distortions**: Identifies extreme values using Interquartile Range (IQR) or Z-score methods and clips them safely, preventing algorithms from skewing their decision boundaries.
* **Multicollinearity Flagging**: Identifies and logs variable pairs with extremely high correlation (e.g., Pearson > 0.8) to prevent redundant feature weights.

## Known Limitations & Unaddressed Edge Cases

As a foundational and basic ML tool, there are a few edge cases and limitations that are outside the current scope of ADIS:

* **Massive Datasets (OOM)**: ADIS processes data entirely in memory using Pandas. If a user uploads an extremely large CSV file (e.g., multiple gigabytes) that exceeds the server's available RAM, the application will crash. It does not currently support chunking, Dask, or Spark for out-of-core processing.
* **Complex NLP or Free Text**: If a row is full of sentences or unstructured text, ADIS might attempt to treat it as a standard categorical variable. While we flag high cardinality, it might still erroneously attempt to one-hot encode it or crash your session, and it currently lacks native Natural Language Processing (NLP) tokenization capabilities.
* **Advanced Machine Learning Needs**: ADIS successfully covers strong classical ML algorithms (like Random Forests and basic Gradient Boosting via scikit-learn). However, it does not integrate state-of-the-art external implementations like XGBoost, LightGBM, CatBoost, or Deep Learning. It also does not perform exhaustive hyperparameter optimization.
* **No Direct Report Download Option**: Currently, the system saves the generated JSON and Markdown execution reports locally to the `adis_output` directory, but the Streamlit user interface lacks a dedicated "Download Report" button to fetch these artifacts directly from the browser.
* **Time Series & Non-Tabular Data**: This tool is strictly designed for standard cross-sectional tabular data. It does not support unstructured data (images/audio) or time series forecasting data.

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed. It's recommended to run this in a virtual environment.

```bash
# 1. Clone the repository (or download the files)
git clone <repository_url>
cd Basic_ML

# 2. Create and activate a Virtual Environment
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Mac/Linux
source .venv/bin/activate

# 3. Install Requirements
pip install -r requirements.txt

# 4. Run the Streamlit Application
streamlit run app.py
```

## Tech Stack

* **Frontend UI**: [Streamlit](https://streamlit.io/)
* **Data Processing**: Pandas, NumPy
* **Data Visualization**: Plotly
* **Machine Learning**: Scikit-Learn, SciPy

## Disclaimer

This project is intended as a helpful utility and a learning tool. While ADIS is thorough, truly intelligent pipelines require domain expertise and an understanding of contextual limitations. Always sanity-check automated ML solutions in production!

---
*Created with care to simplify the data intelligence journey.*
