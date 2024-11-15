import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
# from zenml.integrations.mlflow.mlflow_experiment_tracker import MLFlowExperimentTracker
from zenml.client import Client
import mlflow

from src.model_development import RandomForestModel,LightGBMModel,XGBoostModel,LinearRegressionModel,HyperparameterTuner
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> RegressorMixin:
    try:
        model = None
        tuner = None
        user_input = "linear_regression"  # Change this to select different models
        if user_input == "lightgbm":
            mlflow.sklearn.autolog()
            model = LightGBMModel()
        elif user_input == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif user_input == "xgboost":
            mlflow.sklearn.autolog()
            model = XGBoostModel()
        elif user_input == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            raise ValueError("Model name not supported")

        # Optional: Perform hyperparameter tuning
        # tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
        # best_params = tuner.optimize(n_trials=50)
        # trained_model = model.train(x_train, y_train, **best_params)

        # For now, train without hyperparameter tuning
        
        trained_model = model.train(x_train, y_train)

        return trained_model
    except Exception as e:
        logging.error(e)
        raise e

