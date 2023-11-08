import sys
sys.path.append("C:\\Python39\\lib\\site-packages")
from sklearn.base import BaseEstimator as BaseEstimatorSklearn
from sklearn.base import TransformerMixin as BaseTransformerSklearn


class IdentityTransformer(BaseEstimatorSklearn, BaseTransformerSklearn):
    def __init__(self):
        pass

    def fit(self, X, y = None):
        self.X_ = X
        return self

    def transform(self, X, y = None):
        return X

    def fit_transform(self, X, y=None, **fit_params):
        return X

    def fit_resample(self, X, y = None):
        return X, y