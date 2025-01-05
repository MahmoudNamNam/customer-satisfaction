import logging
import time
import mlflow
import pandas as pd
from src.model_dev import HyperparameterTuner, RandomForestModel,LightGBMModel,XGBoostModel,LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml import step
from zenml.client import Client
from src.config import ModelNameConfig
from typing import Union


# Configure logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
        Union[RegressorMixin, None]: The trained model instance or None if an error occurs.
    """
    start_time = time.time()
    logger.info("Initializing model training...")

    # Input validation
    if x_train.empty or x_test.empty or y_train.empty or y_test.empty:
        logger.error("Training or testing data cannot be empty.")
        raise ValueError("Training or testing data cannot be empty.")
    
    allowed_models = {"lightgbm", "randomforest", "xgboost", "linear_regression"}
    if config.model_name not in allowed_models:
        logger.error(f"Unsupported model name: {config.model_name}. Allowed values are: {allowed_models}")
        raise ValueError(f"Unsupported model name: {config.model_name}. Choose one of: {allowed_models}")

    model = None

    try:
        # Model selection
        logger.info(f"Selected model: {config.model_name}")
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

        # Hyperparameter tuning
        tuner = HyperparameterTuner(model, x_train, y_train, x_test, y_test)
        if config.fine_tuning:
            logger.info("Performing hyperparameter tuning...")
            best_params = tuner.optimize()
            logger.info(f"Best parameters found: {best_params}")
            trained_model = model.train(x_train, y_train, **best_params)
        else:
            logger.info("Training model without fine-tuning...")
            trained_model = model.train(x_train, y_train)

        elapsed_time = time.time() - start_time
        logger.info(f"Model training completed successfully in {elapsed_time:.2f} seconds.")
        return trained_model
    except ValueError as ve:
        logger.error(f"ValueError during training: {ve}")
        raise ve
    except Exception as e:
        logger.error(f"An unexpected error occurred during model training: {e}")
        raise e
