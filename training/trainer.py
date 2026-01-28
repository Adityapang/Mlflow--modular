import numpy as np

class TimeSeriesTrainer:
    def __init__(self, model_cls, cv, metric_fn):
        self.model_cls = model_cls
        self.cv = cv
        self.metric_fn = metric_fn

    def cross_validate(self, X, y, model_params, cat_features_indices):
        scores = []

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            model = self.model_cls(
                params=model_params,
                cat_feature_indices=cat_features_indices
            )

            model.train(
                X[train_idx],
                y.iloc[train_idx],
            )

            preds = model.predict(X[val_idx])
            score = self.metric_fn(y.iloc[val_idx], preds)

            scores.append(score)

        return float(np.mean(scores))

    def train_final_model(self, X, y, best_params, cat_features_indices):
        model = self.model_cls(**best_params)
        model.fit(X, y, cat_features=cat_features_indices)
        return model