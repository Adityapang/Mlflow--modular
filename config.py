from utils.hardware import detect_gpu

EXPERIMENT_NAME = "Tamil_Nadu_project"
MLFLOW_URI = "http://127.0.0.1:5000/"

ROLLING_WINDOW_MONTHS = None  # Keep last 24 months, set to None for all data
N_OPTUNA_TRIALS = 30
N_CV_SPLITS = 5

MODEL_NAME = "CatBoostRegressor"
TRAINING_RUN_TYPE = "training"
EVALUATION_RUN_TYPE = "evaluation"
CV_TYPE = "TimeSeriesSplit"
REGISTERED_MODEL_NAME = f"{EXPERIMENT_NAME}_{MODEL_NAME}"

FILEPATH = "re_data_5_jan_new.csv"
DATE_COL = "week_start"
CUTOFF_WEEK = 16
TARGET_COL = "Case_Count_next_week"
DISTRICT_COL = "dist_name"
TEST_META_COLS = [DATE_COL, DISTRICT_COL, TARGET_COL]

TRAIN_PATH = "train_dataset.csv"
TEST_PATH = "test_dataset.csv"
FEATURE_IMPORTANCE_PATH = "feature_importance.csv"
PREDICTIONS_PATH = "predictions.csv"



GPU_INFO = detect_gpu()
USE_GPU = GPU_INFO["available"]

