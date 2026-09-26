import os

# Base paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STUDENT_RESOURCE_DIR = os.path.join(PROJECT_ROOT, "6ab10eb3b23ba_student_resource", "student_resource")
DATASET_DIR = os.path.join(STUDENT_RESOURCE_DIR, "dataset")

TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Model artifacts directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Processing parameters based on empirical audit
MAX_CANDIDATES_PER_ENTITY = 25
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)

# Default F0.5 decision threshold (tuned on holdout validation)
DEFAULT_DECISION_THRESHOLD = 0.80
