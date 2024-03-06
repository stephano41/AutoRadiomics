from __future__ import annotations

import abc
import logging
import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
import math

from autorad.config import config

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)
log = logging.getLogger(__name__)


class CoreSelector(abc.ABC):
    """Template for feature selection methods"""

    def __init__(self):
        self._selected_features: list[str] | None = None

    @abc.abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> list[int]:
        """fit method should update self._selected_features.
        If no features are selected, it should raise
        NoFeaturesSelectedError.
        """
        pass

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X, y)

    def transform(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> pd.DataFrame:
        return X[self.selected_features]

    @property
    def selected_features(self):
        if self._selected_features is None:
            raise ValueError(
                "No features selected!" "Call fit() first before transforming."
            )
        return self._selected_features


class AnovaSelector(CoreSelector):
    def __init__(self, alpha=0.05):

        self.alpha = alpha
        # self.model = SelectKBest(f_classif, k=self.n_features)
        super().__init__()

    def fit(self, X, y):
        indices = self.run_anova(X, y)
        self._selected_features = X.columns[indices].tolist()

    def run_anova(self, X, y, pass_through=False):
        _, p_value = f_classif(X, y)
        indices = np.where(p_value<self.alpha)[0]
        if len(indices<=0):
            if pass_through:
                return np.arange(len(X))
            log.info("ANOVA failed to select features, selecting the top sqrt of X instead")
            fail_model = SelectKBest(f_classif, k=int(math.sqrt(len(X))))
            fail_model.fit(X, y)
            support = fail_model.get_support(indices=True)
            selected_columns = support.tolist()
            self._selected_features = X.columns[selected_columns].tolist()

        return indices
    


class LinearSVCSelector(AnovaSelector):
    def __init__(self):
        self.model = SelectFromModel(LinearSVC(dual='auto', penalty='l1'))
        super().__init__()

    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X, _y = X.iloc[indices], y.iloc[indices]

        self.model.fit(_X, _y)

        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("LinearSVC selector failed to select features")
        selected_columns = support.tolist()
        self._selected_features = _X.columns[selected_columns].tolist()


class TreeSelector(AnovaSelector):
    def __init__(self, n_estimators=50):
        self.n_estimators=n_estimators
        self.model = SelectFromModel(ExtraTreesClassifier(n_estimators=n_estimators))
        super().__init__()
    
    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X, _y = X.iloc[indices], y.iloc[indices]

        self.model.fit(_X, _y)
        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("Tree selector failed to select features")
        selected_columns = support.tolist()
        self._selected_features = _X.columns[selected_columns].tolist()


class LassoSelector(AnovaSelector):
    def __init__(self, alpha=0.002):
        self.model = SelectFromModel(Lasso(random_state=config.SEED, alpha=alpha, max_iter=10000))
        super().__init__()

    def optimize_params(self, X, y, verbose=0):
        search = GridSearchCV(
            self.model,
            {"alpha": np.logspace(-5, 1, num=100)},
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=verbose,
        )
        search.fit(X, y)
        best_params = search.best_params_
        log.info(f"Best params for Lasso: {best_params}")
        self.model = self.model.set_params(**best_params)

    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X, _y = X.iloc[indices], y.iloc[indices]
        self.optimize_params(_X, _y)
        self.model.fit(_X, _y)

        support = self.model.get_support(indices=True)
        if support is None:
            log.info("LASSO failed to select features.")
            self._selected_features = _X.columns.tolist()
        else:
            selected_columns = support.tolist()
            self._selected_features = _X.columns[selected_columns].tolist()

    def params_to_optimize(self):
        return {"alpha": np.logspace(-5, 1, num=100)}


class BorutaSelector(CoreSelector):
    def fit(self, X, y, verbose=0):
        model = BorutaPy(
            RandomForestClassifier(
                max_depth=5, n_jobs=-1, random_state=config.SEED
            ),
            n_estimators="auto",
            verbose=verbose,
            random_state=config.SEED,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X.to_numpy(), y.to_numpy())
        selected_columns = np.where(model.support_)[0].tolist()
        if not selected_columns:
            raise ValueError("Boruta failed to select features.")
        self._selected_features = X.columns[selected_columns].tolist()


class FeatureSelectorFactory:
    def __init__(self):
        self.selectors = {
            "anova": AnovaSelector,
            "lasso": LassoSelector,
            "boruta": BorutaSelector,
            "linear_svc": LinearSVCSelector,
            "tree": TreeSelector
        }

    def register_selector(self, name, selector):
        self.selectors[name] = selector

    def get_selector(self, name, *args, **kwargs):
        selector = self.selectors[name]
        if not selector:
            raise ValueError(f"Unknown feature selection ({name}).")
        return selector(*args, **kwargs)


def create_feature_selector(
    method: str = "anova",
    *args,
    **kwargs,
):
    selector = FeatureSelectorFactory().get_selector(method, *args, **kwargs)
    return selector


class FailoverSelectorWrapper(CoreSelector):
    """
    Wrapper for FeatureSelectors which doesn't raise 'NoFeaturesSelectedError'
    but instead returns all features.
    """

    def __init__(self, selector):
        self.selector = selector
        super().__init__()

    def fit(self, X, y):
        try:
            self.selector.fit(X, y)
            self._selected_features = self.selector._selected_features
        except ValueError:
            self._selected_features = X.columns.tolist()
