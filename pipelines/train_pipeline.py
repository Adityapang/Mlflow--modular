import mlflow
import abc
import pandas as pd
import sklearn
from config import *

def run_training_pipeline(X_train: pd.DataFrame, y_train: pd.Series,
                        param_space_fn,
                        model_cls: abc.ABCMeta,
                        model_kwargs: dict,
                        cv: sklearn,
                        trainer_cls,
                        optimizer_cls,
                        metric_fn,
                        model_name: str,
                        cv_name: str,
                        run_type: str,
                        pipeline_root_run_id: str):

    with mlflow.start_run(run_name=f"{model_name}_{run_type}", nested = True):

        mlflow.set_tags(
            {
                "model_name": model_name,
                "run_name": run_type,
                "cv_name": cv_name,
                "pipeline_root_run_id": pipeline_root_run_id
            }
        )

        mlflow.log_params({
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_cv_splits": N_CV_SPLITS,
            "rolling_window_months": ROLLING_WINDOW_MONTHS,
            "gpu_used": USE_GPU
        })

        trainer_cls = trainer_cls(model_cls, cv, metric_fn)
        optimizer = optimizer_cls(
            trainer=trainer_cls,
            n_trials=N_OPTUNA_TRIALS,
            param_space_fn=param_space_fn,
            direction="minimize"
        )

        study = optimizer.optimize(
            X=X_train, y=y_train, **model_kwargs
        )

        best_cv_rmse = study.best_value
        best_params = study.best_params

        mlflow.log_metric("best_cv_rmse", best_cv_rmse)
        mlflow.log_params(best_params)


        final_model = model_cls(
            params = study.best_params,
            **model_kwargs
        )
        final_model.train(X_train, y_train)

        final_model.log_to_mlflow()

        training_run_id = mlflow.active_run().info.run_id

        return final_model, best_cv_rmse, best_params, training_run_id