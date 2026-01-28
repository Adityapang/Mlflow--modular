class CatBoostSearchSpace:

    @staticmethod
    def optuna(trial, use_gpu: bool):
        return {
            "iterations": trial.suggest_int("iterations", 300, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "random_strength": trial.suggest_float("random_strength", 0, 10),
        }