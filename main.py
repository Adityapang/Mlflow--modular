from data.data_loader import load_raw_data, sort_data, get_data_hash, extract_data_metadata, temporal_train_test_split, split_features_target, drop_null_values, save_dataframe_to_csv
from preprocessing.factory import PreprocessorFactory
from utils.mlflow_helpers import start_mlflow_experiment, register_model_with_data_tags
from pipelines.train_pipeline import run_training_pipeline
from pipelines.evaluation_pipeline import run_evaluation_pipeline
from training.search_space import CatBoostSearchSpace
from training.evaluator import rmse
from training.trainer import TimeSeriesTrainer
from training.optimizer import OptunaOptimizer
from models.catboost_model import CatBoostModel
from datetime import datetime
import mlflow
from sklearn.model_selection import TimeSeriesSplit
from config import *

def main():
    df = load_raw_data(path = FILEPATH, date_col = DATE_COL)

    train_df, test_df = temporal_train_test_split(df, date_col=DATE_COL, cutoff_week=CUTOFF_WEEK)
    train_df, test_df = sort_data(train_df, test_df, date_col=DATE_COL)

    train_df, test_df = drop_null_values(train_df=train_df,
                                         test_df=test_df,
                                         target_col=TARGET_COL)

    save_dataframe_to_csv(train_df, TRAIN_PATH)
    save_dataframe_to_csv(test_df, TEST_PATH)

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

    for preprocessor_name in PREPROCESSOR_LIST:
        pre = PreprocessorFactory.create(preprocessor_name)
        pre.fit(X_train)
        X_train_preprocessed = pre.transform(X_train)

        X_test_preprocessed = pre.transform(X_test)

        features_names = pre.get_feature_names()
        cat_feature_indices = pre.get_cat_feature_indices()

        experiment = start_mlflow_experiment(mlflow_uri=MLFLOW_URI,
                                             experiment_name=EXPERIMENT_NAME)
        today_date = datetime.now().strftime("%Y/%m/%d")

        with mlflow.start_run(run_name = f"{EXPERIMENT_NAME}_pipeline_root_{today_date}") as pipeline_root:
            pipeline_root_run_id = pipeline_root.info.run_id
            mlflow.set_tags(
                {
                    "preprocessor_name": preprocessor_name,
                    "train_data_hash": train_data_hash,
                    "test_data_hash": test_data_hash,
                    "train_date_min": train_metadata["train_date_min"],
                    "train_date_max": train_metadata["train_date_max"],
                    "test_date_min": test_metadata["test_date_min"],
                    "test_date_max": test_metadata["test_date_max"]

                }
            )


            mlflow.log_artifact(TRAIN_PATH, artifact_path="data")
            mlflow.log_artifact(TEST_PATH, artifact_path="data")

            tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

            param_space_function = lambda trial: CatBoostSearchSpace.optuna(trial = trial,
                                                                            use_gpu = USE_GPU)

            final_model, best_cv_rmse, best_params, training_run_id = run_training_pipeline(X_train=X_train_preprocessed,
                                                                      y_train=y_train,
                                                                      param_space_fn=param_space_function,
                                                                      model_cls=CatBoostModel,
                                                                      model_kwargs={
                                                                          "cat_feature_indices": cat_feature_indices
                                                                      },
                                                                      cv=tscv,
                                                                      trainer_cls=TimeSeriesTrainer,
                                                                      optimizer_cls=OptunaOptimizer,
                                                                      metric_fn=rmse,
                                                                      model_name=MODEL_NAME,
                                                                      run_type=TRAINING_RUN_TYPE,
                                                                      cv_name=CV_TYPE,
                                                                      pipeline_root_run_id=pipeline_root_run_id)


            if final_model.has_feature_importance():
                importance_df = final_model.get_feature_importance(feature_names=features_names)
                save_dataframe_to_csv(importance_df, FEATURE_IMPORTANCE_PATH)
                mlflow.log_artifact(FEATURE_IMPORTANCE_PATH, artifact_path="feature_importance")

            test_rmse = run_evaluation_pipeline(X_test=X_test_preprocessed,
                                    y_test = y_test,
                                    X_test_meta=X_test_meta,
                                    model = final_model,
                                    metric_fn=rmse,
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
                preprocessor_name=preprocessor_name,
            )




if __name__ == "__main__":
    main()