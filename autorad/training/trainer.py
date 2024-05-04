import logging
from functools import partial
from pathlib import Path
from typing import Sequence
import multiprocessing as mp

import joblib
import mlflow
import numpy as np
from optuna.trial import Trial
from autorad.config.type_definitions import PathLike
from autorad.data import FeatureDataset, TrainingData
from autorad.models import MLClassifier
from autorad.preprocessing import Preprocessor
from autorad.training import OptunaOptimizer, train_utils
from autorad.utils import io, mlflow_utils
from autorad.preprocessing import Preprocessor
from autorad.metrics import roc_auc, pr_auc

log = logging.getLogger(__name__)


class Trainer:
    """
    Runs the experiment that optimizes the hyperparameters
    for all the models, given the dataset with extracted features.
    """

    def __init__(
        self,
        dataset: FeatureDataset,
        models: Sequence[MLClassifier],
        result_dir: PathLike,
        seed: int = 42,
        multi_class="raise",
        labels=None,
        average="macro",
        metric='roc_auc',
        preprocessor=Preprocessor,
        n_jobs=1
    ):
        self.dataset = dataset
        self.models = models
        self.result_dir = Path(result_dir)
        self.seed = seed
        self.multi_class = multi_class
        self.preprocessor = preprocessor
        self.n_jobs = n_jobs
        self._optimizer = None
        self.auto_preprocessing = False

        if metric == 'roc_auc':
            self.get_auc = partial(roc_auc, average=average, multi_class=multi_class, labels=labels)
        elif metric == 'pr_auc':
            self.get_auc = partial(pr_auc, average=average, multi_class=multi_class, labels=labels)
        else:
            raise ValueError(f'metric not implemented, got {metric}')
        

    def set_optimizer(self, optimizer: str, n_trials: int = 100):
        if optimizer == "optuna":
            self._optimizer = OptunaOptimizer(
                n_trials=n_trials, seed=self.seed
            )
        else:
            raise ValueError("Optimizer not recognized.")

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise ValueError("Optimizer is not set!")
        return self._optimizer

    def set_optuna_params(self, model: MLClassifier, trial: Trial):
        params = model.param_fn(trial)
        model.set_params(**params)
        return model

    def save_best_preprocessor(self, best_trial_params: dict):
        feature_selection = best_trial_params["feature_selection_method"]
        oversampling = best_trial_params["oversampling_method"]

        _, preprocessor_kwargs = self.get_preprocessed_pickle()

        preprocessor_kwargs.update({
            'feature_selection_method': feature_selection,
            'oversampling_method': oversampling
        })

        preprocessor = self.preprocessor(**preprocessor_kwargs)

        fitted_X, fitted_y = preprocessor._fit_transform(self.dataset.X, self.dataset.y)
        mlflow.sklearn.log_model(preprocessor, "preprocessor")

        return fitted_X, fitted_y
    
    def get_best_trial(self, study):
        study_df = study.trials_dataframe(attrs=('number', 'user_attrs'))
        unique_auc = np.unique(study_df['user_attrs_AUC_val'])
        cutoff = unique_auc[-int(len(unique_auc) * 0.025)]

        selected_trials = study_df[study_df['user_attrs_AUC_val'] >= cutoff]

        min_std_row = selected_trials[
            selected_trials['user_attrs_std_AUC_val'] == selected_trials['user_attrs_std_AUC_val'].min()]

        best_trial_number = min_std_row['number'].iloc[0]

        for trial in study.trials:
            if trial.number == best_trial_number:
                log.info(
                    f'Best trial was number {best_trial_number}, {trial.params} with AUC: {trial.user_attrs["AUC_val"]} and standard deviation: {trial.user_attrs["std_AUC_val"]} ')
                return trial
        raise ValueError("trial number of best trial not found in study dataframe!")

    def run(
        self,
        auto_preprocess: bool = False,
        experiment_name="model_training",
        mlflow_start_kwargs=None
    ):
        """
        Run hyperparameter optimization for all the models.
        """
        if mlflow_start_kwargs is None:
            mlflow_start_kwargs = {}
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(experiment_name)
        else:
            log.warn("Running training in existing experiment.")
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(**mlflow_start_kwargs):
            study = self.optimizer.create_study(
                study_name=experiment_name,
            )

            study.optimize(
                lambda trial: self._objective(
                    trial, auto_preprocess=auto_preprocess
                ),
                n_trials=self.optimizer.n_trials,
                callbacks=[_save_model_callback],
            )
            self.log_to_mlflow(study=study)

    def log_to_mlflow(self, study):
        best_trial = self.get_best_trial(study)
        best_auc = best_trial.user_attrs["AUC_val"]
        mlflow.log_metric("AUC_val", best_auc)

        train_utils.log_optuna(study)

        best_params = study.best_trial.params
        self.save_params(best_params)
        fitted_X, fitted_y = self.save_best_preprocessor(best_params)

        best_model = best_trial.user_attrs["model"]
        best_model.fit(fitted_X, fitted_y)
        best_model.save_to_mlflow()

        self.copy_extraction_artifacts()
        train_utils.log_dataset(self.dataset)
        train_utils.log_splits(self.dataset.splits)

        data_preprocessed = study.user_attrs["data_preprocessed"]
        self.log_train_auc(best_model, data_preprocessed)
        train_utils.log_preprocessed_data(data_preprocessed)

    def log_train_auc(self, model: MLClassifier, data: TrainingData):
        y_true = data.y.train
        X_train = data.X.train

        model.fit(X_train, y_true)

        y_pred_proba = model.predict_proba(X_train)

        train_auc = self.get_auc(y_true, y_pred_proba)
        mlflow.log_metric("AUC_train", float(train_auc))

    def copy_extraction_artifacts(self):
        try:
            extraction_run_id = self.dataset.df["extraction_ID"].iloc[0]
            mlflow_utils.copy_artifacts_from(extraction_run_id)
        except KeyError:
            log.warn(
                "Copying of feature extraction params failed! "
                "No extraction_id column found in feature table. "
                "This will cause problems with inference from images."
            )
        except mlflow.exceptions.MlflowException:
            log.warn(
                "Copying of feature extraction params failed! "
                "No feature extraction artifact included in the run. "
                "This will cause problems with inference from images."
            )

    def save_params(self, params: dict):
        mlflow.log_params(params)
        io.save_json(params, (self.result_dir / "best_params.json"))

    def get_best_preprocessed_dataset(self, trial: Trial) -> TrainingData:
        """ "
        Get preprocessed dataset with preprocessing method that performed
        best in the training.
        """
        preprocessed, _ = self.get_preprocessed_pickle()
        feature_selection_method = trial.suggest_categorical(
            "feature_selection_method", preprocessed.keys()
        )
        oversampling_method = trial.suggest_categorical(
            "oversampling_method",
            preprocessed[feature_selection_method].keys(),
        )
        result = preprocessed[feature_selection_method][oversampling_method]

        return result

    def get_trial_data(
        self, trial: Trial, auto_preprocess: bool = False
    ) -> TrainingData:
        """
        Get the data for the trial, either from the preprocessed data
        or from the original dataset.
        """
        if auto_preprocess:
            data = self.get_best_preprocessed_dataset(trial)
        else:
            data = self.dataset.data
        return data

    def _objective(self, trial: Trial, auto_preprocess=False) -> float:
        """Get params from optuna trial, return the metric."""
        data = self.get_trial_data(trial, auto_preprocess=auto_preprocess)

        model_name = trial.suggest_categorical(
            "model", [m.name for m in self.models]
        )
        model = train_utils.get_model_by_name(model_name, self.models)
        model = self.set_optuna_params(model=model, trial=trial)
        aucs = []
        try:
            if len(data.X.train_folds) > self.n_jobs:
                mp.set_start_method('spawn', force=True)
                with mp.Pool(processes=self.n_jobs) as pool:
                    for auc in pool.map(self._fit_and_evaluate, [(model, X_train, y_train, X_val, y_val) for
                                                              X_train, y_train, _, X_val, y_val, _ in
                                                              data.iter_training()]):
                        aucs.append(auc)

            else:
                for X_train, y_train, _, X_val, y_val, _ in data.iter_training():
                    auc_val = self._fit_and_evaluate((model, X_train, y_train, X_val, y_val))
                    aucs.append(auc_val)
        except Exception as e:
            log.warning(f"training {trial.params} failed")
            raise e

        auc_val = float(np.nanmean(aucs))
        trial.set_user_attr("AUC_val", auc_val)
        trial.set_user_attr("std_AUC_val", float(np.nanstd(aucs)))
        trial.set_user_attr("model", model)
        trial.set_user_attr("data_preprocessed", data)

        return auc_val
    

    def _fit_and_evaluate(self, args):
        """Fit the model and evaluate on validation data."""
        model, X_train, y_train, X_val, y_val = args

        _X_train = X_train.copy()
        _y_train = y_train.copy()

        model.fit(_X_train, _y_train)

        y_pred = model.predict_proba(X_val)

        auc_val = self.get_auc(y_val, y_pred)

        return auc_val
    
    def get_preprocessed_pickle(self):
        pkl_path = self.result_dir / "preprocessed.pkl"
        with open(pkl_path, "rb") as f:
            preprocessed_data, preprocessor_kwargs = joblib.load(f)
        return preprocessed_data, preprocessor_kwargs


def _save_model_callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="AUC_val", value=trial.user_attrs["AUC_val"])
        study.set_user_attr(key="model", value=trial.user_attrs["model"])
        study.set_user_attr(
            key="data_preprocessed",
            value=trial.user_attrs["data_preprocessed"],
        )
