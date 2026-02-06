import pandas as pd
import mlflow
from config import *
from data.prediction_builder import build_prediction_data
from utils.artifact_logger import log_parquet
from training.metric_factory import MetricSpec

def run_evaluation_pipeline(
            X_test: pd.DataFrame,
            y_test: pd.Series,
            X_test_meta: pd.DataFrame,
            model,
            eval_metrics: list[MetricSpec],
            best_cv_score: float,
            predictions_path: str,
            model_name: str,
            run_type: str,
            pipeline_root_run_id: str,
    ):
    with mlflow.start_run(run_name=f"{model_name}_{run_type}", nested=True):

        mlflow.set_tags({
            "model_name": model_name,
            "run_name": run_type,
            "pipeline_root_run_id": pipeline_root_run_id,
        })

        predictions = model.predict(X_test)

        metric_results = {}
        for metric in eval_metrics:
            value = metric.fn(y_test, predictions)
            mlflow.log_metric(f"test_{metric.name}", value)
            metric_results[metric.name] = value

        predictions_df = build_prediction_data(
            predictions=predictions,
            X_test_meta=X_test_meta,
            best_cv_score=best_cv_score,
            test_metrics=metric_results,
        )

        log_parquet(
            df=predictions_df,
            filename=predictions_path,
            artifact_path="predictions",
        )

        return metric_results
