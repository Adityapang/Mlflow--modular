import mlflow
import abc
import pandas as pd
from config import *
from training.metric_factory import MetricSpec

def run_training_pipeline(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_space_fn,
        model_cls: abc.ABCMeta,
        model_kwargs: dict,
        cv,
        trainer_cls,
        optimizer_cls,
        optimization_metric: MetricSpec,
        model_name: str,
        cv_name: str,
        run_type: str,
        pipeline_root_run_id: str,
    ):
    with mlflow.start_run(run_name=f"{model_name}_{run_type}", nested=True):

        mlflow.set_tags({
            "model_name": model_name,
            "run_name": run_type,
            "cv_name": cv_name,
            "pipeline_root_run_id": pipeline_root_run_id,
            "optimization_metric": optimization_metric.name,
        })

        mlflow.log_params({
            "n_optuna_trials": N_OPTUNA_TRIALS,
            "n_cv_splits": N_CV_SPLITS,
            "rolling_window_months": ROLLING_WINDOW_MONTHS,
            "gpu_used": USE_GPU,
        })

        # Trainer knows CV + metric
        trainer = trainer_cls(
            model_cls=model_cls,
            cv=cv,
            metric_spec=optimization_metric,
        )

        optimizer = optimizer_cls(
            trainer=trainer,
            param_space_fn=param_space_fn,
            n_trials=N_OPTUNA_TRIALS,
            direction=optimization_metric.direction,
        )

        study = optimizer.optimize(X=X_train, y=y_train, **model_kwargs)

        best_score = study.best_value
        best_params = study.best_params

        mlflow.log_metric(f"best_cv_{optimization_metric.name}", best_score)
        mlflow.log_params(best_params)

        # Train final model on full training set
        final_model = model_cls.from_params(best_params, **model_kwargs)
        final_model.fit(X_train, y_train)

        final_model.log_to_mlflow()

        training_run_id = mlflow.active_run().info.run_id

        return final_model, best_score, best_params, training_run_id
