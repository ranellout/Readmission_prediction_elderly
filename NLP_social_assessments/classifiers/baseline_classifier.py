import numpy as np
from sklearn.base import BaseEstimator as BaseEstimatorSklearn
from sklearn.metrics import precision_score


class BaselineClassifier(BaseEstimatorSklearn):
    def __init__(self, class_guess = None):
        self.class_guess = class_guess

    def fit(self, X, y = None):
        self.X_ = X
        self.y_ = y
        self.majority_class = np.bincount(y).argmax()
        self.minority_class = np.bincount(y).argmin()
        return self

    def predict(self, X):
        if self.class_guess is None:
            return np.ones(X.shape[0]) * self.majority_class
        else:
            return np.ones(X.shape[0]) * self.class_guess

    def score(self, X, y = None):
        preds = self.predict(X)
        return precision_score(y, preds)