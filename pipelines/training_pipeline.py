from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
import logging
from src.data_cleaning import DataPreprocessStrategy,DataCleaning,DataDivideStrategy
from steps.ingest_data import ingest_data
from steps.cleaning_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation
import pandas as pd
from src.config import ModelNameConfig

# Define Docker settings with required MLflow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])


from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.cleaning_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluation

# Define Docker settings with required MLflow integration
docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True, settings={"docker": docker_settings})
def train_pipeline():
    """
    Training pipeline for a machine learning workflow.

    Steps:
        1. Data ingestion
        2. Data cleaning and splitting
        3. Model training
        4. Model evaluation
    """
    # Step 1: Data ingestion
    data = ingest_data()


    x_train, x_test, y_train, y_test = clean_data(
        data=data,
    )

    # Step 3: Model training
    model = train_model(x_train, x_test, y_train, y_test,ModelNameConfig)

    # Step 4: Model evaluation
    mse, rmse = evaluation(model, x_test, y_test)

    return mse, rmse
