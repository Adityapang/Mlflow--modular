from utils.hardware import detect_gpu

EXPERIMENT_NAME = "Tamil_Nadu_project"
MLFLOW_URI = "http://127.0.0.1:5000/"

ROLLING_WINDOW_MONTHS = None  # Keep last 24 months, set to None for all data
N_OPTUNA_TRIALS = 1
N_CV_SPLITS = 5

PREPROCESSOR_NAME = "tabular_v1"

MODEL_NAME = "CatBoostRegressor"

TRAINING_RUN_TYPE = "training"
EVALUATION_RUN_TYPE = "evaluation"
CV_TYPE = "TimeSeriesSplit"


FILEPATH = "re_data_5_jan_new.csv"
DATE_COL = "week_start"
CUTOFF_WEEK = 16
TARGET_COL = "Case_Count_next_week"
DISTRICT_COL = "dist_name"
TEST_META_COLS = [DATE_COL, DISTRICT_COL, TARGET_COL]

TRAIN_PATH = "train_dataset.parquet"
TEST_PATH = "test_dataset.parquet"
FEATURE_IMPORTANCE_PATH = "feature_importance.parquet"
PREDICTIONS_PATH = "predictions.parquet"
SHAP_VALUES_PATH = "shap_values.parquet"
SHAP_SUMMARY_PATH =  "shap_summary.png"

GPU_INFO = detect_gpu()
USE_GPU = GPU_INFO["available"]

