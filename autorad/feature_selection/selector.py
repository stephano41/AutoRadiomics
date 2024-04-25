from __future__ import annotations

import abc
import logging
import warnings

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
import math
from collections.abc import Mapping
from ast import literal_eval
from mrmr import mrmr_classif
from sklearn.decomposition import PCA


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
    def __init__(self, alpha=0.05, select_top_best='auto'):
        self.select_top_best=select_top_best

        self.alpha = alpha
        # self.model = SelectKBest(f_classif, k=self.n_features)
        super().__init__()

    def fit(self, X, y):
        indices = self.run_anova(X, y)
        self._selected_features = X.columns[indices].tolist()

    def run_anova(self, X, y, pass_through=False):
        _, p_value = f_classif(X, y)
        support = np.where(p_value < self.alpha)[0]

        if not isinstance(self.select_top_best, bool):
            if self.select_top_best=='auto':
                n_features = int(math.sqrt(len(X)))
            elif isinstance(self.select_top_best, int):
                n_features = self.select_top_best
            else:
                raise ValueError("Invalid select_top_best variable type!")
            
            model = SelectKBest(f_classif, k=n_features)
            model.fit(X, y)
            support = model.get_support(indices=True)

        if support is None:
            if pass_through:
                return np.arange(len(X))
            raise ValueError("ANOVA failed to select features, selecting the top sqrt of X instead")

        return support
    


class LinearSVCSelector(AnovaSelector):
    def __init__(self):
        self.model = SelectFromModel(LinearSVC(dual='auto', penalty='l1'))
        super().__init__(select_top_best=False)

    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]
        
        self.model.set_params(**{'max_features': int(math.sqrt(len(X)))})
        self.model.fit(_X, y)

        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("LinearSVC selector failed to select features")
        selected_columns = support.tolist()
        self._selected_features = _X.columns[selected_columns].tolist()


class TreeSelector(AnovaSelector):
    def __init__(self):
        super().__init__(select_top_best=False)
    
    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]

        n_features =int(math.sqrt(len(X)))
        model = SelectFromModel(ExtraTreesClassifier(n_estimators=min(len(_X), n_features)), max_features=n_features)
        model.fit(_X, y)
        support = model.get_support(indices=True)
        if support is None:
            raise ValueError("Tree selector failed to select features")
        selected_columns = support.tolist()
        self._selected_features = _X.columns[selected_columns].tolist()


class LassoSelector(AnovaSelector):
    def __init__(self, alpha=0.002, n_jobs=None):
        self.alpha=alpha
        self.n_jobs=n_jobs
        self.model = Lasso(random_state=config.SEED, alpha=alpha, max_iter=10000)
        super().__init__(select_top_best=False)

    def optimize_params(self, X, y, verbose=0):
        search = GridSearchCV(
            self.model,
            {"alpha": np.logspace(-5, 1, num=100)},
            cv=5,
            scoring="neg_mean_squared_error",
            verbose=verbose,
            n_jobs=self.n_jobs
        )
        search.fit(X, y)
        best_params = search.best_params_
        log.info(f"Best params for Lasso: {best_params}")
        self.model = self.model.set_params(**best_params)

    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]
        self.optimize_params(_X, y)

        n_features =int(math.sqrt(len(X)))
        selector = SelectFromModel(self.model, max_features=n_features)
        selector.fit(_X, y)
        support = selector.get_support(indices=True)
        if support is None:
            log.info("LASSO failed to select features.")
            self._selected_features = _X.columns.tolist()
        else:
            selected_columns = support.tolist()
            self._selected_features = _X.columns[selected_columns].tolist()

    def params_to_optimize(self):
        return {"alpha": np.logspace(-5, 1, num=100)}
    

class SFSelector(AnovaSelector):
    def __init__(self, direction='forward', scoring='roc_auc', n_jobs=None, n_features_to_select='auto',tol=0.1):
        self.n_jobs=n_jobs
        self.direction=direction
        self.scoring=scoring
        self.n_features_to_select=n_features_to_select
        self.tol=tol
        self.model = SequentialFeatureSelector(LogisticRegression(), 
                                               direction=direction, 
                                               scoring=scoring, 
                                               n_jobs=n_jobs, 
                                               n_features_to_select=n_features_to_select,
                                               tol=tol)
        super().__init__(select_top_best=False)

    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]

        self.model.fit(_X,y)
        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("SFSelector failed to select features")
        selected_columns = support.tolist()
        self._selected_features=_X.columns[selected_columns].tolist()


class RFESelector(AnovaSelector):
    def __init__(self, min_features=2, scoring='roc_auc', n_jobs=None):
        self.n_jobs=n_jobs
        self.min_features=min_features
        self.scoring=scoring
        self.model = RFECV(LogisticRegression(), min_features_to_select=min_features, scoring=scoring, n_jobs=n_jobs)
        super().__init__(select_top_best=False)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:, indices]
        
        self.model.set_params(n_features_to_select=int(math.sqrt(len(X))))
        self.model.fit(_X,y)
        support = self.model.get_support(indices=True)
        if support is None:
            raise ValueError("RFESelector failed to select features")
        selected_columns = support.tolist()
        self._selected_features=_X.columns[selected_columns].tolist()


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


class MRMRSelector(AnovaSelector):
    def __init__(self, K=None):
        self.K = K
        
        super().__init__(select_top_best=False)
    
    def fit(self, X, y):
        indices = self.run_anova(X, y, True)
        _X = X.iloc[:,indices]

        if self.K is None:
            num_features = int(math.sqrt(len(X)))
        else:
            num_features = self.K
        
        self._selected_features = mrmr_classif(X=_X, y=y, K=num_features)


class PCASelector(AnovaSelector):
    def __init__(self, n_components=None):
        self.n_components=n_components
        self.model = PCA(n_components).set_output(transform='pandas')

        super().__init__(select_top_best=False)

    
    def fit(self, X, y=None):
        _X = self.model.fit_transform(X)
        indices = self.run_anova(_X, y, True)
        
        self._selected_features = _X.columns[indices].tolist()
    
    def transform(self, X, y=None):
        _X = self.model.transform(X)

        return _X[self.selected_features]


class FeatureSelectorFactory:
    def __init__(self):
        self.selectors = {
            "anova": AnovaSelector,
            "lasso": LassoSelector,
            "boruta": BorutaSelector,
            "linear_svc": LinearSVCSelector,
            "tree": TreeSelector,
            "sf": SFSelector,
            "rfe": RFESelector,
            "mrmr": MRMRSelector,
            "pca": PCASelector
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
    
    processed_method_param = process_dict(method)
    if isinstance(processed_method_param, Mapping):
        kwarg_dict = {k:v for k, v in processed_method_param.items() if k!='_method_'}
        kwargs.update(kwarg_dict)
        selector = FeatureSelectorFactory().get_selector(processed_method_param['_method_'],*args, **kwargs)
    elif isinstance(processed_method_param, str):
        selector = FeatureSelectorFactory().get_selector(processed_method_param, *args, **kwargs)
    else:
        raise TypeError(f"method is not a recognised datatype, got {type(processed_method_param)}")
    return selector


def process_dict(maybe_dict):
    try:
        eval_dict = literal_eval(maybe_dict)
        if isinstance(eval_dict, Mapping):
            return eval_dict
    except ValueError:
        return maybe_dict
    return maybe_dict


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
