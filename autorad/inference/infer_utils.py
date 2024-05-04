import logging

import mlflow
import pandas as pd
import shap

from autorad.data import FeatureDataset
from autorad.models.classifier import MLClassifier
from autorad.utils import io, mlflow_utils
from imblearn.pipeline import Pipeline

log = logging.getLogger(__name__)


def get_artifacts_from_best_run(experiment_name="model_training"):
    best_run = get_best_run_from_experiment_name(experiment_name)
    artifacts = load_pipeline_artifacts(best_run)

    return artifacts


def get_best_run_from_experiment_name(experiment_name):
    experiment_id = mlflow_utils.get_experiment_id_from_name(experiment_name)
    best_run = mlflow_utils.get_best_run(experiment_id)

    return best_run


def load_pipeline_artifacts(run):
    uri = run["artifact_uri"]
    model = MLClassifier.load_from_mlflow(f"{uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{uri}/preprocessor")
    explainer = mlflow.shap.load_explainer(f"{uri}/shap-explainer")
    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "explainer": explainer,
    }
    try:
        extraction_config = io.load_yaml(
            f"{uri.removeprefix('file://')}/feature_extraction/extraction_config.yaml"
        )
        artifacts["extraction_config"] = extraction_config
    except FileNotFoundError:
        log.warn("Feature extraction config not found.")
    return artifacts


def get_pipeline_from_run(run) -> Pipeline:
    try:
        artifact_uri = run['artifact_uri']
    except IndexError:
        artifact_uri = run.info.artifact_uri

    model = MLClassifier.load_from_mlflow(f"{artifact_uri}/model")
    preprocessor = mlflow.sklearn.load_model(f"{artifact_uri}/preprocessor")

    pipeline = preprocessor.pipeline
    pipeline.steps.append(['estimator', model])
    return pipeline

def get_run_info_as_series(run_id):
    """
    Fetches the MLflow run information for the specified run ID and returns it as a pandas Series.

    Parameters:
        run_id (str): The ID of the run in MLflow.

    Returns:
        pd.Series: A series object containing the run information.

    Raises:
        Exception: If the run ID is not found or an error occurs in fetching the run information.
    """
    try:
        run = mlflow.get_run(run_id)
        run_info = pd.Series(dict(run.info))
        return run_info
    except Exception as e:
        raise Exception(f"Failed to fetch run information for run ID {run_id}: {str(e)}")


def get_pipeline_from_last_experiment_name(experiment_name="model_training") -> Pipeline:
    run = get_last_run_from_experiment_name(experiment_name)

    return get_pipeline_from_run(run)


def load_dataset_artifacts(run):
    uri = run["artifact_uri"]
    splits = io.load_yaml(f"{uri.removeprefix('file://')}/splits.yaml")
    df = pd.read_csv(f"{uri.removeprefix('file://')}/feature_dataset/df.csv")
    dataset_config = io.load_yaml(
        f"{uri.removeprefix('file://')}/feature_dataset/dataset_config.yaml"
    )
    artifacts = {
        "df": df,
        "dataset_config": dataset_config,
        "splits": splits,
    }
    return artifacts


def load_feature_dataset(feature_df, dataset_config, splits) -> FeatureDataset:
    dataset = FeatureDataset(
        dataframe=feature_df,
        **dataset_config
    )
    dataset.load_splits(splits)

    return dataset


def plot_shap_waterfall(explainer, X_preprocessed, max_display=10):
    shap_values = explainer(
        X_preprocessed, max_evals=2 * len(X_preprocessed.columns) + 1
    )
    shap_fig = shap.plots.waterfall(
        shap_values[0], max_display=max_display, show=True
    )
    return shap_fig

def get_last_run_from_experiment_name(experiment_name="model_training"):
    experiment_id = mlflow_utils.get_experiment_id_from_name(experiment_name)
    all_runs = mlflow.search_runs(experiment_ids=experiment_id)
    try:
        last_run = all_runs.iloc[0]
    except IndexError:
        raise IndexError(
            "No trained models found. Please run the training first."
        )

    return last_run


def get_artifacts_from_last_run(experiment_name="model_training"):
    last_run = get_last_run_from_experiment_name(experiment_name)
    artifacts = load_pipeline_artifacts(last_run)

    return artifacts
