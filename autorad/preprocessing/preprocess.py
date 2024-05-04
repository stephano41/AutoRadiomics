from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any
import inspect
import joblib
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from autorad.config import config
from autorad.data import TrainingData, TrainingInput, TrainingLabels
from autorad.feature_selection import create_feature_selector
from autorad.preprocessing import oversample_utils
from autorad.preprocessing.outlier_clipper import OutlierClipper

log = logging.getLogger(__name__)


class Preprocessor:
    def __init__(
        self,
        standardize: bool = True,
        feature_selection_method: str | None = None,
        oversampling_method: str | None = None,
        random_state: int = config.SEED,
        feature_first=True
    ):
        """Performs preprocessing, including:
        1. standardization
        2. feature selection
        3. oversampling

        Args:
            standardize: whether to standardize features to mean 0 and std 1
            feature_selection_method: algorithm to select key features,
                if None, don't perform selection and leave all features
            oversampling_method: minority class oversampling method,
                if None, no oversampling
            random_state: seed
            feature_first (bool): A flag indicating whether feature selection should be performed before oversampling.
                Defaults to True.
        """
        self.standardize = standardize
        self.feature_selection_method = feature_selection_method
        self.oversampling_method = oversampling_method
        self.random_state = random_state
        self.feature_first = feature_first
        self.pipeline = self._build_pipeline()

    def fit_transform_data(self, data: TrainingData) -> TrainingData:
        X, y = data.X, data.y
        _data = dataclasses.replace(data)
        _data.X, _data.y = self.fit_transform(X, y)
        return _data

    def fit_transform(
        self, X: TrainingInput, y: TrainingLabels
    ) -> tuple[TrainingInput, TrainingLabels]:

        result_X = {}
        result_y = {}

        result_X["train"], result_y["train"] = self._fit_transform(X.train, y.train)
        # if self.oversampling_method is not None:
        #     result_X["train"], result_y["train"] = self.pipeline.fit_resample(X.train, y.train)
        # else:
        #     result_X["train"] = self.pipeline.fit_transform(X.train, y.train)
        #     result_y["train"] = y.train

        # allow for empty test set

        result_X["test"] = None
        result_y["test"] = None
        
        if X.test is not None:
            if not X.test.empty:
                result_X["test"] = self._transform(X.test)
                result_y["test"] = y.test
           
        if X.val is not None:
            result_X["val"] = self._transform(X.val)
            result_y["val"] = y.val
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_y["train_folds"],
                result_X["val_folds"],
                result_y["val_folds"],
            ) = self._fit_transform_cv_folds(X, y)
        X_preprocessed = TrainingInput(**result_X)
        y_preprocessed = TrainingLabels(**result_y)
        return X_preprocessed, y_preprocessed
    
    def _transform(self, X):
        Xt = X
        for _, _, transform in self.pipeline._iter(with_final=True, filter_resample=False):
            if hasattr(transform, "fit_resample"):
                continue
            Xt = transform.transform(Xt)
        return Xt

    def _fit_transform(self, X, y):
        Xt = X
        yt = y
        for _, _, transform in self.pipeline._iter(with_final=True, filter_resample=False):
            if hasattr(transform, "fit_resample"):
        #             print("i got called!")
                Xt, yt = transform.fit_resample(Xt,yt)
            else:
                transform.fit(Xt, yt)
                Xt = transform.transform(Xt)
        return Xt, yt


    def _fit_transform_cv_folds(
        self, X: TrainingInput, y: TrainingLabels
    ) -> tuple[
        list[pd.DataFrame],
        list[pd.Series],
        list[pd.DataFrame],
        list[pd.Series],
    ]:
        if (
            X.train_folds is None
            or y.train_folds is None
            or X.val_folds is None
            or y.val_folds is None
        ):
            raise AttributeError("Folds are not set")
        (
            result_X_train_folds,
            result_y_train_folds,
            result_X_val_folds,
            result_y_val_folds,
        ) = ([], [], [], [])
        for X_train, y_train, X_val in zip(
            X.train_folds,
            y.train_folds,
            X.val_folds,
        ):
            # reinstantiate pipeline
            self.pipeline = self._build_pipeline()

            result_df_X_train, result_y_train = self._fit_transform(X_train, y_train)
            # if self.oversampling_method is not None:
            #     result_df_X_train, result_y_train = self.pipeline.fit_resample(X_train, y_train)
            # else:
            #     result_df_X_train = self.pipeline.fit_transform(X_train, y_train)
            #     result_y_train = y_train

            result_df_X_val = self._transform(X_val)

            result_X_train_folds.append(result_df_X_train)
            result_y_train_folds.append(result_y_train)
            result_X_val_folds.append(result_df_X_val)
        result_y_val_folds = y.val_folds
        return (
            result_X_train_folds,
            result_y_train_folds,
            result_X_val_folds,
            result_y_val_folds,
        )

    def transform_df(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)

    def transform(self, X: TrainingInput):
        result_X = {}
        result_X["train"] = self._transform(X.train)
        # allow for empty test set
        result_X["test"] = None

        if X.test is not None:
            if not X.test.empty:
                result_X["test"] = self._transform(X.test)
        
        if X.val is not None:
            result_X["val"] = self._transform(X.val)
        if X.train_folds is not None and X.val_folds is not None:
            (
                result_X["train_folds"],
                result_X["val_folds"],
            ) = self._transform_cv_folds(X)
        X_preprocessed = TrainingInput(**result_X)
        return X_preprocessed

    def _transform_cv_folds(
        self, X: TrainingInput
    ) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        if X.train_folds is None or X.val_folds is None:
            raise AttributeError("Folds are not set")
        (
            result_X_train_folds,
            result_X_val_folds,
        ) = ([], [])
        for X_train, X_val in zip(
            X.train_folds,
            X.val_folds,
        ):
            result_df_X_train = self._transform(X_train)
            result_df_X_val = self._transform(X_val)
            result_X_train_folds.append(result_df_X_train)
            result_X_val_folds.append(result_df_X_val)
        return (
            result_X_train_folds,
            result_X_val_folds,
        )

    def _build_pipeline(self):
        steps = []
        if self.standardize:
            # steps.append(
            #     (
            #         "outlier_clipper",
            #         OutlierClipper()
            #     )
            # )
            steps.append(
                (
                    "standardize",
                    StandardScaler().set_output(transform="pandas"),
                )
            )
        if self.feature_first:
            if self.feature_selection_method is not None and self.feature_selection_method != "None":
                steps.append(
                    ("select", create_feature_selector(method=self.feature_selection_method))
                )
            
            if self.oversampling_method is not None and self.oversampling_method != "None":
                steps.append(
                    ("oversample", oversample_utils.create_oversampling_model(
                            method=self.oversampling_method,
                            random_state=self.random_state))
                )
        else:
            if self.oversampling_method is not None and self.oversampling_method != "None":
                steps.append(
                    ("oversample", oversample_utils.create_oversampling_model(
                            method=self.oversampling_method,
                            random_state=self.random_state))
                )
            if self.feature_selection_method is not None and self.feature_selection_method != "None":
                steps.append(
                    ("select", create_feature_selector(method=self.feature_selection_method))
                )


        pipeline = Pipeline(steps)
        return pipeline
    
    def get_params(self, deep=None):
        return {key: getattr(self, key) for key in inspect.signature(self.__init__).parameters.keys() if
                key != "self"}


def run_auto_preprocessing(
    data: TrainingData,
    result_dir: Path,
    use_oversampling: bool = True,
    use_feature_selection: bool = True,
    oversampling_methods: list[str] | None = None,
    feature_selection_methods: list[str] | None = None,
    feature_first=True,
    preprocessor_cls=Preprocessor,
    **kwargs
):
    """Run preprocessing with a variety of feature selection and oversampling methods.

    Args:
    - data (TrainingData): The training data to preprocess.
    - result_dir (Path): The directory path where the preprocessed data will be saved.
    - use_oversampling (bool): A flag indicating whether to use oversampling. If True and
      oversampling_methods is not provided, all methods in the config.OVERSAMPLING_METHODS
      list will be used.
    - use_feature_selection (bool): A flag indicating whether to use feature selection. If True and
      feature_selection_methods is not provided, all methods in the config.FEATURE_SELECTION_METHODS
      list will be used.
    - oversampling_methods (list[str] | None): A list of oversampling methods to use. If not provided,
      all methods in the config.OVERSAMPLING_METHODS list will be used.
    - feature_selection_methods (list[str] | None): A list of feature selection methods to use. If not
      provided, all methods in the config.FEATURE_SELECTION_METHODS list will be used.
    - feature_first (bool): A flag indicating whether feature selection should be performed before oversampling.
      Defaults to True.
    - preprocessor_cls: The class to be used for preprocessing.
    - **kwargs: Additional keyword arguments to be passed to the preprocessor.

    Returns:
    - None. The preprocessed data will be saved to the result_dir directory.
    """
    if use_oversampling:
        if oversampling_methods is None:
            oversampling_methods = config.OVERSAMPLING_METHODS
    else:
        oversampling_methods = [None]

    if use_feature_selection:
        if feature_selection_methods is None:
            feature_selection_methods = config.FEATURE_SELECTION_METHODS
    else:
        feature_selection_methods = [None]

    if feature_first:
        preprocessed_feature_selection = {}
        for selection_method in feature_selection_methods:
            fs_preprocessor = preprocessor_cls(standardize=True, feature_selection_method=selection_method, 
                                               oversampling_method=None, 
                                               feature_first=feature_first,
                                               **kwargs)
            log.info(f'preprocessing with {selection_method}')
            preprocessed_feature_selection[str(selection_method)]=fs_preprocessor.fit_transform_data(data)
            if not preprocessed_feature_selection[str(selection_method)]:
                del preprocessed_feature_selection[str(selection_method)]
        preprocessed={}
        for selection_method, selected_data in preprocessed_feature_selection.items():
            preprocessed[str(selection_method)]= {}
            for oversampling_method in oversampling_methods:
                log.info(f'preprocessing with {selection_method} with {oversampling_method}')

                preprocessor = preprocessor_cls(standardize=False, feature_selection_method=None, 
                                                oversampling_method=oversampling_method, 
                                                feature_first=feature_first,
                                                **kwargs)
                preprocessed[str(selection_method)][str(oversampling_method)] = preprocessor.fit_transform_data(selected_data)
    else:
        preprocessed_over_sampling = {}
        for oversampling_method in oversampling_methods:
            log.info(f'preprocessing with {oversampling_method}')
            oversampling_preprocessor = preprocessor_cls(standardize=True, feature_selection_method=None, 
                                                         oversampling_method=oversampling_method, feature_first=feature_first,
                                                         **kwargs)
            preprocessed_over_sampling[str(oversampling_method)] = oversampling_preprocessor.fit_transform_data(data)
        preprocessed={}
        for selection_method in feature_selection_methods:
            preprocessed[str(selection_method)] = {}
            for oversampling_method, oversampled_data in preprocessed_over_sampling.items():
                log.info(f'preprocessing with {selection_method} with {oversampling_method}')
                preprocessor = preprocessor_cls(standardize=False, feature_selection_method=selection_method, 
                                                oversampling_method=None, 
                                                feature_first=feature_first,
                                                **kwargs)
                preprocessed[str(selection_method)][str(oversampling_method)] = preprocessor.fit_transform_data(oversampled_data)
            if not preprocessed[str(selection_method)]:
                del preprocessed[str(selection_method)]

    final_preprocessor = preprocessor_cls(standardize=True, feature_selection_method=selection_method, 
                                          oversampling_method=oversampling_method, 
                                          feature_first=feature_first,
                                          **kwargs)
    with open(Path(result_dir) / "preprocessed.pkl", "wb") as f:
        joblib.dump((preprocessed, final_preprocessor.get_params()), f)