import json
import logging
import numpy as np
import pandas as pd
import requests
from zenml.steps import step
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from pipelines.utils import get_data_for_test
from steps.clean_data import clean_data
from steps.evaluate_model import evaluation
from steps.ingest_data import ingest_data
from steps.train_model import train_model

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=True)
def dynamic_importer() -> str:
    data = get_data_for_test()
    return data

@step(enable_cache=True)
def deployment_trigger(
    accuracy: float,
    min_accuracy: float = 0.2,
) -> bool:
    return accuracy >= min_accuracy

@step(enable_cache=True)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )
    
    if not existing_services:
        raise RuntimeError(
            f"No MLFlow deployment service found for pipeline '{pipeline_name}'"
        )
    return existing_services[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns", None)
    data.pop("index", None)
    
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_data = df.to_json(orient="split")

    # Send a POST request to the prediction URL
    response = requests.post(service.prediction_url, json={"data": json_data})

    if response.status_code == 200:
        prediction = np.array(response.json()["predictions"])
        return prediction
    else:
        logging.error(f"Prediction request failed with status code: {response.status_code}")
        raise RuntimeError("Prediction request failed.")

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.29,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_data()
    X_train, X_test, y_train, y_test = clean_data(df)

    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluation(model, X_test, y_test)

    deploy_decision = deployment_trigger(accuracy=r2_score, min_accuracy=min_accuracy)
    
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=workers,
        timeout=timeout
    )

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def inference_pipeline(pipeline_name: str, pipeline_step_name: str):
    data = dynamic_importer() 
    service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=True
    )
    prediction = predictor(service=service, data=data)
    return prediction
