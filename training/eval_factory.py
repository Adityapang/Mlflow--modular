from training.evaluator import rmse

class MetricFactory:

    REGISTRY = {
        "CatBoostRegressor": rmse,
        # we’ll add more later
    }

    @classmethod
    def get_metric(cls, model_name: str):
        if model_name not in cls.REGISTRY:
            raise ValueError(f"No metric defined for model: {model_name}")
        return cls.REGISTRY[model_name]