from data.data_loader import load_raw_data, sort_data, get_data_hash, extract_data_metadata, temporal_train_test_split, split_features_target, drop_null_values, save_dataframe_to_csv
from preprocessing.factory import PreprocessorFactory
from utils.mlflow_helpers import start_mlflow_experiment, register_model_with_data_tags
from pipelines.train_pipeline import run_training_pipeline
from pipelines.evaluation_pipeline import run_evaluation_pipeline
from training.search_space_factory import SearchSpaceFactory
from training.metric_factory import MetricFactory
from training.trainer import TimeSeriesTrainer
from training.optimizer import OptunaOptimizer
from models.factory import ModelFactory
from datetime import datetime
import mlflow
from utils.explainability import log_shap_summary
from utils.artifact_logger import log_parquet
from sklearn.model_selection import TimeSeriesSplit
from config import *

def main():
    df = load_raw_data(path = FILEPATH, date_col = DATE_COL)

    train_df, test_df = temporal_train_test_split(df, date_col=DATE_COL, cutoff_week=CUTOFF_WEEK)
    train_df, test_df = sort_data(train_df, test_df, date_col=DATE_COL)

    train_df, test_df = drop_null_values(train_df=train_df,
                                         test_df=test_df,
                                         target_col=TARGET_COL)



    train_data_hash = get_data_hash(train_df)
    test_data_hash = get_data_hash(test_df)

    train_metadata = extract_data_metadata(train_df, date_col=DATE_COL, train=True)
    test_metadata = extract_data_metadata(test_df, date_col=DATE_COL)

    X_train, y_train = split_features_target(train_df, TARGET_COL, DATE_COL)
    X_test, y_test, X_test_meta = split_features_target(test_df,
                                                        TARGET_COL,
                                                        DATE_COL,
                                                        return_meta=True,
                                                        test_meta_cols=TEST_META_COLS)


    pre = PreprocessorFactory.create(PREPROCESSOR_NAME)
    pre.fit(X_train)
    X_train_preprocessed = pre.transform(X_train)

    X_test_preprocessed = pre.transform(X_test)

    feature_names = pre.get_feature_names()
    cat_feature_indices = pre.get_cat_feature_indices()

    experiment = start_mlflow_experiment(mlflow_uri=MLFLOW_URI,
                                         experiment_name=EXPERIMENT_NAME)
    today_date = datetime.now().strftime("%Y/%m/%d")

    with mlflow.start_run(run_name = f"{EXPERIMENT_NAME}_pipeline_root_{today_date}") as pipeline_root:
        pipeline_root_run_id = pipeline_root.info.run_id
        mlflow.set_tags(
            {
                "preprocessor_name": PREPROCESSOR_NAME,
                "train_data_hash": train_data_hash,
                "test_data_hash": test_data_hash,
                "train_date_min": train_metadata["train_date_min"],
                "train_date_max": train_metadata["train_date_max"],
                "test_date_min": test_metadata["test_date_min"],
                "test_date_max": test_metadata["test_date_max"]

            }
        )


        log_parquet(df = train_df, filename=TRAIN_PATH, artifact_path="data")
        log_parquet(df=test_df, filename=TEST_PATH, artifact_path="data")

        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

        metric_fn = MetricFactory.get_metric(MODEL_NAME)

        param_space_function = SearchSpaceFactory.get_search_space(
            model_name=MODEL_NAME,
            use_gpu=USE_GPU
        )

        model_cls, model_kwargs = ModelFactory.get_model(
            model_name=MODEL_NAME,
            cat_feature_indices=cat_feature_indices
        )

        final_model, best_cv_rmse, best_params, training_run_id = run_training_pipeline(X_train=X_train_preprocessed,
                                                                  y_train=y_train,
                                                                  param_space_fn=param_space_function,
                                                                  model_cls=model_cls,
                                                                  model_kwargs=model_kwargs,
                                                                  cv=tscv,
                                                                  trainer_cls=TimeSeriesTrainer,
                                                                  optimizer_cls=OptunaOptimizer,
                                                                  metric_fn=metric_fn,
                                                                  model_name=MODEL_NAME,
                                                                  run_type=TRAINING_RUN_TYPE,
                                                                  cv_name=CV_TYPE,
                                                                  pipeline_root_run_id=pipeline_root_run_id)


        if final_model.has_feature_importance():
            importance_df = final_model.get_feature_importance(feature_names=feature_names)
            importance_df = importance_df[importance_df["feature"] != "Case_Count"]
            log_parquet(df=importance_df, filename=FEATURE_IMPORTANCE_PATH, artifact_path="feature_importance")


        test_rmse = run_evaluation_pipeline(X_test=X_test_preprocessed,
                                y_test = y_test,
                                X_test_meta=X_test_meta,
                                model = final_model,
                                metric_fn=metric_fn,
                                best_cv_rmse=best_cv_rmse,
                                predictions_path=PREDICTIONS_PATH,
                                model_name=MODEL_NAME,
                                run_type=EVALUATION_RUN_TYPE,
                                pipeline_root_run_id=pipeline_root_run_id)

        REGISTERED_MODEL_NAME = f"{EXPERIMENT_NAME}_{MODEL_NAME}"
        register_model_with_data_tags(
            training_run_id=training_run_id,
            model_name=REGISTERED_MODEL_NAME,
            train_data_hash=train_data_hash,
            test_data_hash=test_data_hash,
            pipeline_root_run_id=pipeline_root_run_id,
            preprocessor_name=PREPROCESSOR_NAME,
        )


        shap_summary_path, shap_df= log_shap_summary(model_wrapper=final_model,
                                                     X_sample=X_test_preprocessed,
                                                     feature_names=feature_names,
                                                     shap_summary_path=SHAP_SUMMARY_PATH)

        log_parquet(df=shap_df, filename=SHAP_VALUES_PATH, artifact_path="explainability")
        mlflow.log_artifact(shap_summary_path, artifact_path="explainability")


if __name__ == "__main__":
    main()
