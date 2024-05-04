import tempfile
from pathlib import Path
from typing import Sequence

import mlflow
import pandas as pd
import shap
from optuna.study import Study
import optuna
import pickle
from autorad.data import FeatureDataset
from autorad.models import MLClassifier
from autorad.utils import io, mlflow_utils


def get_model_by_name(name: str, models: Sequence[MLClassifier]) -> MLClassifier:
    for model in models:
        if model.name == name:
            return MLClassifier(type(model.model)(**model.model.get_params()), name, model.params)
    raise ValueError(f"Model with name {name} not found")


def log_splits(splits: dict):
    mlflow_utils.log_dict_as_artifact(splits, "splits")


def log_shap(model: MLClassifier, X_train: pd.DataFrame):
    explainer = shap.Explainer(model.predict_proba_binary, X_train)
    mlflow.shap.log_explainer(explainer, "shap-explainer")


def log_mlflow_params(params):
    for param_name, param_value in params.items():
        mlflow.log_param(param_name, param_value)


def log_dataset(dataset: FeatureDataset):
    dataset_config = {
        "target": dataset.target,
        "ID_colname": dataset.ID_colname,
        "additional_features": dataset.additional_features
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / "feature_dataset"
        save_dir.mkdir(exist_ok=True)
        io.save_yaml(dataset_config, save_dir / "dataset_config.yaml")
        dataset.df.to_csv(save_dir / "df.csv", index=False)
        mlflow.log_artifacts(str(save_dir), "feature_dataset")


def log_optuna(study: Study):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / "hyperparameter_study"
        save_dir.mkdir(exist_ok=True)
        study.trials_dataframe().to_csv(save_dir / "study_df.csv")
        mlflow.log_artifacts(str(save_dir), "hyperparameter_study")
    
    optimisation_history_plot = optuna.visualization.plot_optimization_history(study)
    mlflow.log_figure(optimisation_history_plot,'hyperparameter_study/optimisation_history.html')

    parallel_coordinate_plot = optuna.visualization.plot_parallel_coordinate(study, params=['oversampling_method', 'feature_selection_method','model'])
    mlflow.log_figure(parallel_coordinate_plot,'hyperparameter_study/parallel_coordinate.html')


def log_preprocessed_data(data):
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) /  'preprocessed_data.pkl'
        with open(save_dir, 'wb') as f:
            pickle.dump(data, f)
        mlflow.log_artifact(str(save_dir), 'feature_dataset')
