import optuna
import mlflow


class OptunaOptimizer:
    def __init__(self, trainer, param_space_fn, direction: str, n_trials: int):
        self.trainer = trainer
        self.direction = direction
        self.n_trials = n_trials
        self.param_space_fn = param_space_fn

    def optimize(self, X, y,  cat_feature_indices):
        def objective(trial):
            model_params = self.param_space_fn(trial)

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested = True):
                mlflow.log_params(model_params)
                score = self.trainer.cross_validate(X, y, model_params,  cat_feature_indices)
                mlflow.log_metric("cv_rmse", score)

            return score

        study = optuna.create_study(direction=self.direction)
        study.optimize(objective, n_trials=self.n_trials)
        return study
