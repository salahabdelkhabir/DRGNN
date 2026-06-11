import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_DIR = os.path.join(DATA_DIR, "input")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
EVALUATION_DIR = os.path.join(RESULTS_DIR, "evaluation")
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
FINE_TUNING_DIR = os.path.join(RESULTS_DIR, "fine_tuning")


def ensure_dir_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


required_dirs = [
    DATA_DIR, INPUT_DIR, PROCESSED_DIR,
    RESULTS_DIR, MODELS_DIR, EVALUATION_DIR, ANALYSIS_DIR, FINE_TUNING_DIR,
]

for directory in required_dirs:
    ensure_dir_exists(directory)
