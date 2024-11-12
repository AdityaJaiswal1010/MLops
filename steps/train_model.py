import logging
import pandas as pd
from sklearn.base import RegressorMixin
from zenml import step
from src.model_development import RandomForestModel,LightGBMModel,XGBoostModel,LinearRegressionModel,HyperparameterTuner
@step

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
            model = LightGBMModel()
        elif user_input == "randomforest":
            model = RandomForestModel()
        elif user_input == "xgboost":
            model = XGBoostModel()
        elif user_input == "linear_regression":
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

