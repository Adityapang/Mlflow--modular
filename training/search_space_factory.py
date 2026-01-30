from training.search_space import CatBoostSearchSpace

class SearchSpaceFactory:

    REGISTRY = {
        "CatBoostRegressor": CatBoostSearchSpace.optuna,
        # we’ll add more later
    }

    @classmethod
    def get_search_space(cls, model_name: str, use_gpu: bool = False):
        if model_name not in cls.REGISTRY:
            raise ValueError(f"No search space registered for model: {model_name}")

        search_fn = cls.REGISTRY[model_name]

        # Return a function that Optuna can call
        return lambda trial: search_fn(trial=trial, use_gpu=use_gpu)