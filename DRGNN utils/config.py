# utils/config.py

import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Results directories
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
FINE_TUNING_DIR = os.path.join(RESULTS_DIR, "fine_tuning")

# Utils, Notebooks, and Documentation directories
UTILS_DIR = os.path.join(BASE_DIR, "utils")
DOCS_DIR = os.path.join(BASE_DIR, "docs")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Function to ensure a directory exists
def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Ensure all required directories exist
required_dirs = [
    DATA_DIR, INPUT_DIR, PROCESSED_DIR,
    RESULTS_DIR, MODELS_DIR, EVALUATION_DIR, ANALYSIS_DIR, FINE_TUNING_DIR,
    UTILS_DIR, DOCS_DIR, NOTEBOOKS_DIR
]

for directory in required_dirs:
    ensure_dir_exists(directory)

# Print paths for debugging (optional, can be removed in production)
if __name__ == "__main__":
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"PROCESSED_DIR: {PROCESSED_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"EVALUATION_DIR: {EVALUATION_DIR}")
    print(f"ANALYSIS_DIR: {ANALYSIS_DIR}")
    print(f"FINE_TUNING_DIR: {FINE_TUNING_DIR}")
    print(f"UTILS_DIR: {UTILS_DIR}")
    print(f"DOCS_DIR: {DOCS_DIR}")
    print(f"NOTEBOOKS_DIR: {NOTEBOOKS_DIR}")
