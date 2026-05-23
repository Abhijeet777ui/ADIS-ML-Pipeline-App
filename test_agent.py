import os
from adis.agent import AutoResearchAgent
import logging

# Set up logging so we can watch the agent "think"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. SET YOUR API KEY
# Ensure GROQ_API_KEY is set in your environment variables.
# e.g., export GROQ_API_KEY="your_key" in terminal before running.
if "GROQ_API_KEY" not in os.environ:
    logging.warning("GROQ_API_KEY is not set. The agent will likely fail.")

def main():
    # 1. SET YOUR API KEY
    os.environ["GROQ_API_KEY"] = "your_key_here"
    os.environ["ADIS_ALLOW_EXEC"] = "1"

    # 2. Initialize the Agent
    # We'll use the existing sample of the data to keep the research loop fast
    sample_path = "credit_sample.csv"

    agent = AutoResearchAgent(
        data_path=sample_path,
        target_col="Class", # The column we are predicting
        model_name="groq/llama-3.1-8b-instant", # Active Groq model
    )
    
    # 3. Start the Autonomous Loop
    # We'll run it for 5 iterations.
    print("\n[START] Starting Autonomous Research Agent...")
    best_features_code = agent.optimize(iterations=5)
    
    print("\n[SUCCESS] Research Complete!")
    print("Here is the best feature engineering code the agent generated:")
    print("--------------------------------------------------")
    print(best_features_code)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
