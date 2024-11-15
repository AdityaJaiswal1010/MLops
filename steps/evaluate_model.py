import logging

import numpy as np
import pandas as pd
from src.evaluation import MSE, RMSE, R2Score
from sklearn.base import RegressorMixin
# from sklearn.metrics import accuracy_score 
from typing_extensions import Annotated
from zenml import step
from typing import Tuple
from zenml.client import Client
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from zenml.client import Client
experiment_tracker=Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin, x_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:

    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        

        prediction = model.predict(x_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)

        # Using the R2Score class for R2 score calculation
        r2_class = R2Score()
        r2_score = r2_class.calculate_score(y_test, prediction)


        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("r2_score",r2_score)
        mlflow.log_metric("rmse",rmse)

        # accuracy = accuracy_score(y_test, prediction)
        # print("Accuracy of model - "+accuracy)
        return r2_score, rmse
    except Exception as e:
        logging.error(e)
        raise e