import logging

import mlflow
import pandas as pd
from model.model_dev import (
    HyperparameterTuner,
    LightGBMModel,
    LinearRegressionModel,
    RandomForestModel,
    XGBoostModel,
)
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client

from .config import ModelNameConfig

# Get the active experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a machine learning model with optional hyperparameter tuning.

    Args:
        x_train (pd.DataFrame): Features for training the model.
        x_test (pd.DataFrame): Features for testing the model.
        y_train (pd.Series): Target values for training.
        y_test (pd.Series): Target values for testing.
        config (ModelNameConfig): Configuration object containing model details.

    Returns:
        RegressorMixin: The trained model instance.
    """
    logging.info("Initializing model training...")

    # Input validation
    if x_train.empty or x_test.empty or y_train.empty or y_test.empty:
        logging.error("Input data contains empty DataFrame or Series.")
        raise ValueError("Training or testing data cannot be empty.")
    
    if not isinstance(config.model_name, str):
        logging.error("Model name in config is not a string.")
        raise ValueError("Model name must be a string.")

    model = None
    tuner = None

    try:
        # Model selection
        logging.info(f"Selected model: {config.model_name}")
        if config.model_name == "lightgbm":
            mlflow.lightgbm.autolog()
            model = LightGBMModel()
        elif config.model_name == "randomforest":
            mlflow.sklearn.autolog()
            model = RandomForestModel()
        elif config.model_name == "xgboost":
            mlflow.xgboost.autolog()
            model = XGBoostModel()
        elif config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
        else:
            logging.error(f"Unsupported model name: {config.model_name}")
            raise ValueError(f"Unsupported model name: {config.model_name}")

        # Hyperparameter tuning
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
        if config.fine_tuning:
            logging.info("Performing hyperparameter tuning...")
            best_params = tuner.optimize()
            logging.info(f"Best parameters found: {best_params}")
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            logging.info("Training model without fine-tuning...")
            trained_model = model.train(x_train, y_train)

        logging.info("Model training completed successfully.")
        return trained_model
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise e
