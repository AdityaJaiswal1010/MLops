import logging
import numpy as np
import pandas as pd
from zenml import step
from src.data_cleaning import DataStrategy, DataCleaning, DataDivideToTrainAndTest, DataPreProcessing
from typing_extensions import Annotated
from typing import Tuple

# clean the data
@step
def clean_data(df: pd.DataFrame) ->Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    try:
        process_strategy= DataPreProcessing()
        data_cleaning= DataCleaning(df,process_strategy)
        processed_data= data_cleaning.handle_data()

        divide_data = DataDivideToTrainAndTest()
        data_cleaning = DataCleaning(processed_data, divide_data)
        X_train,X_test,y_train,y_test=data_cleaning.handle_data()
        logging.info('Data Cleaning Done!')
        return X_train,X_test,y_train,y_test
    except Exception as e:
        logging.info('Some Error - {e}')
        raise e