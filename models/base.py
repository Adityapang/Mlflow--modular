from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def has_feature_importance(self):
        return False

    @abstractmethod
    def get_feature_importance(self):
        raise NotImplementedError

