"""
ADIS - Automated Data Intelligence System
A comprehensive, explainability-first pipeline for automated data analysis,
cleaning, feature engineering, model benchmarking, and vulnerability detection.

Usage:
    from adis import ADISPipeline
    from adis.agent import AutoResearchAgent

    pipeline = ADISPipeline(target_column="price")
    results = pipeline.run("data.csv")
    pipeline.save_report()
"""

from adis.version import __version__
__author__ = "Abhijeet Baug"

from adis.pipeline import ADISPipeline
from adis.critic import run_critic

# Individual stage functions for advanced users
from adis.ingestion import run_ingestion
from adis.cleaning import run_cleaning
from adis.eda import run_eda
from adis.feature_engineering import run_feature_engineering
from adis.feature_selection import run_feature_selection
from adis.model_recommendation import run_model_recommendation
from adis.benchmarking import run_benchmarking

__all__ = [
    "ADISPipeline",
    "run_critic",
    "run_ingestion",
    "run_cleaning",
    "run_eda",
    "run_feature_engineering",
    "run_feature_selection",
    "run_model_recommendation",
    "run_benchmarking",
]
