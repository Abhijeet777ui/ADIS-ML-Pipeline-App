import os
from adis.agent import AutoResearchAgent
import logging

# Set up logging so we can watch the agent "think"
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 1. SET YOUR API KEY
# Ensure GEMINI_API_KEY is set in your environment variables.
# e.g., export GEMINI_API_KEY="your_key" in terminal before running.
if "GEMINI_API_KEY" not in os.environ:
    logging.warning("GEMINI_API_KEY is not set. The agent will likely fail.")

def main():
    # 2. Initialize the Agent
    # We point it to the dataset and specify the Gemini model.
    agent = AutoResearchAgent(
        data_path="temp_data.csv",
        target_col="Performance_Score", # The column we are predicting
        model_name="gemini/gemini-2.5-flash", # Use the free-tier available 2.5 Flash model
    )
    
    # 3. Start the Autonomous Loop
    # We'll run it for just 2 iterations to see how it works.
    print("\n[START] Starting Autonomous Research Agent...")
    best_features_code = agent.optimize(iterations=5)
    
    print("\n[SUCCESS] Research Complete!")
    print("Here is the best feature engineering code the agent generated:")
    print("--------------------------------------------------")
    print(best_features_code)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
