import optuna
import mlflow
from trainer import TimeSeriesTrainer

class OptunaOptimizer:
    def __init__(self, trainer: TimeSeriesTrainer, param_space_fn, n_trials: int):
        self.trainer = trainer
        self.param_space_fn = param_space_fn
        self.n_trials = n_trials

    def optimize(self, X, y, **model_kwargs):
        metric = self.trainer.metric

        def objective(trial):
            params = self.param_space_fn(trial)

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)

                score = self.trainer.evaluate_params(X, y, params, **model_kwargs)

                mlflow.log_metric(f"cv_{metric.name}", score)

            return score

        study = optuna.create_study(direction=metric.direction)
        study.optimize(objective, n_trials=self.n_trials)
        return study
