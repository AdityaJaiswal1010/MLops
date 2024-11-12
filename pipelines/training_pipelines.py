from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluate_model import evaluation
from steps.train_model import train_model
import logging
@pipeline(enable_cache=False)
def training_pipeline():
    df = ingest_data()
    # Add this line to debug the columns
    X_train, X_test, y_train, y_test = clean_data(df)

    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluation(model, X_test, y_test)