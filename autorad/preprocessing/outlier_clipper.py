from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower_percentile=1, upper_percentile=99):
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.lower_clip = None
        self.upper_clip = None

    def fit(self, X, y=None):
        # Compute lower and upper clips based on percentiles of the input data
        self.lower_clip = np.percentile(X, self.lower_percentile, axis=0)
        self.upper_clip = np.percentile(X, self.upper_percentile, axis=0)
        return self

    def transform(self, X):
        # Clip outliers
        X_clipped = np.maximum(X, self.lower_clip)
        X_clipped = np.minimum(X_clipped, self.upper_clip)
        return X_clipped