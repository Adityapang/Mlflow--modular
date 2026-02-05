from models.catboost_model import CatBoostModel
from training.search_space import CatBoostSearchSpace
from config import USE_GPU
from typing import List

def build_model_registry(cat_feature_indices: list) -> List[dict]:
    model_registry = [
        {
            "name": "CatBoost",
            "model_cls": CatBoostModel,
            "param_space_fn": lambda trial: CatBoostSearchSpace.optuna(
                trial=trial,
                use_gpu=USE_GPU
            ),
            "model_kwargs": {
                "cat_feature_indices": cat_feature_indices
            }
        }
    ]
    return model_registry