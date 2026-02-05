import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
from typing import Dict

client = MlflowClient()

def start_mlflow_experiment(mlflow_uri: str,
                            experiment_name: str) -> mlflow.entities.experiment.Experiment:
    mlflow.set_tracking_uri(uri=mlflow_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment

def register_model_with_data_tags(training_run_id: str,
                                  model_name: str,
                                  train_data_hash: str,
                                  test_data_hash: str,
                                  pipeline_root_run_id: str,
                                  preprocessor_name: str,
                                  ) -> mlflow.entities.model_registry.model_version.ModelVersion:

    model_uri = f"runs:/{training_run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)

    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="train_data_hash",
        value=train_data_hash
    )

    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="test_data_hash",
        value=test_data_hash
    )

    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="pipeline_root_run_id",
        value=pipeline_root_run_id
    )

    client.set_model_version_tag(
        name=model_name,
        version=mv.version,
        key="preprocessor_name",
        value=preprocessor_name
    )
    return mv

def load_model_from_registry(
    model_name: str,
    stage: str | None = None,
    version: int | None = None) -> mlflow.pyfunc.PyFuncModel:

    if stage:
        model_uri = f"models:/{model_name}/{stage}"
    elif version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        raise ValueError("Provide either stage or version")

    model = mlflow.pyfunc.load_model(model_uri)
    return model

def get_training_context(model_name: str, version: int) -> dict:
    mv = client.get_model_version(model_name, version)
    run = client.get_run(mv.run_id)

    context = {
        "training_run_id": mv.run_id,
        "pipeline_root_run_id": mv.tags.get("pipeline_root_run_id"),
        "experiment_id": run.info.experiment_id,
        "train_data_hash": mv.tags.get("train_data_hash"),
        "test_data_hash": mv.tags.get("test_data_hash"),
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
    }
    return context

def load_train_test_data(model_name: str, version: int) -> dict:
    training_context = get_training_context(model_name, version)
    pipeline_root_run_id = training_context["pipeline_root_run_id"]

    local_path = mlflow.artifacts.download_artifacts(
        run_id=pipeline_root_run_id,
        artifact_path="data"
    )

    return {
        file.name: pd.read_parquet(file)
        for file in Path(local_path).glob("*.parquet")
    }

def load_predictions(model_name: str, version: int) -> Dict[str, pd.DataFrame]:
    training_context = get_training_context(model_name, version)
    pipeline_root_run_id = training_context["pipeline_root_run_id"]
    experiment_id = training_context["experiment_id"]

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.pipeline_root_run_id = '{pipeline_root_run_id}' "
                      f"and tags.model_name = '{model_name}' "
                      f"and tags.run_name = 'evaluation'",
        order_by=["start_time DESC"],
        max_results=1
         )
    run_id = runs[0].info.run_id
    local_path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="predictions"
        )
    return {
        file.name: pd.read_parquet(file)
        for file in Path(local_path).glob("*.parquet")
    }
