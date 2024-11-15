import click
from rich import print
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from pipelines.deployment_pipelines import continuous_deployment_pipeline, inference_pipeline


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice(["deploy", "predict", "deploy_and_predict"]),
    default="deploy_and_predict",
    help="Run deployment, prediction, or both.",
)
@click.option(
    "--min-accuracy",
    default=0.2,
    help="Minimum accuracy required to deploy the model",
)
def run_deployment(config: str, min_accuracy: float):
    deploy = config in ["deploy", "deploy_and_predict"]
    predict = config in ["predict", "deploy_and_predict"]

    # Run deployment pipeline if requested
    if deploy:
        continuous_deployment_pipeline(min_accuracy=min_accuracy, workers=3, timeout=60)

    # Check for an active model server
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
        running=True,
    )

    # Retry deployment if no service found
    if not services:
        print("No MLflow service found. Retrying deployment.")
        continuous_deployment_pipeline(min_accuracy=min_accuracy, workers=3, timeout=60)
        services = mlflow_model_deployer_component.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            model_name="model",
            running=True,
        )

    if services:
        service = services[0]
        print(f"Service is running at {service.prediction_url}")

        # Run prediction pipeline if requested
        if predict:
            inference_pipeline(
                pipeline_name="continuous_deployment_pipeline",
                pipeline_step_name="mlflow_model_deployer_step"
            )
    else:
        print("Deployment failed. No MLflow service found after retry.")

if __name__ == "__main__":
    run_deployment()
