import pandas as pd
import mlflow
from config import *
from data.data_loader import save_dataframe_to_csv
from data.prediction_builder import build_prediction_data
from utils.artifact_logger import log_parquet

def run_evaluation_pipeline( X_test: pd.DataFrame, y_test: pd.Series,
                             X_test_meta: pd.DataFrame,
                             model,
                             metric_fn,
                             best_cv_rmse: float,
                             predictions_path: str,
                             model_name: str,
                             run_type: str,
                             pipeline_root_run_id: str):
    with mlflow.start_run(run_name = f"{model_name}_{run_type}", nested=True):
        mlflow.set_tags(
            {
                "model_name": model_name,
                "run_name": run_type,
                "pipeline_root_run_id": pipeline_root_run_id
            }
        )

        predictions = model.predict(X_test)

        test_rmse = metric_fn(y_test, predictions)
        mlflow.log_metric("test_rmse", test_rmse)

        predictions_df = build_prediction_data(predictions,
                                               X_test_meta,
                                               best_cv_rmse,
                                               test_rmse)
        log_parquet(df=predictions_df, filename=predictions_path, artifact_path="predictions")

        return test_rmse