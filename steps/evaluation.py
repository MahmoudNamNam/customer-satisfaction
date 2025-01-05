import logging
import mlflow
import numpy as np
import pandas as pd
from src.evaluation import MSE, RMSE, R2Score, MAE, MAPE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client
from typing import Tuple

# Get the active experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluation(
    model: RegressorMixin, 
    x_test: pd.DataFrame, 
    y_test: pd.Series
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    """
    Evaluates the performance of a trained model on the test data.

    Args:
        model (RegressorMixin): The trained regression model to be evaluated.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values for the test set.

    Returns:
        Tuple[float, float]: The R2 score and RMSE for the model on the test set.
    """
    try:
        logging.info("Starting model evaluation...")

        # Ensure inputs are valid
        if x_test.empty or y_test.empty:
            raise ValueError("Test data (x_test or y_test) is empty. Please provide valid data.")
        
        # Generate predictions
        logging.info("Generating predictions on the test data...")
        prediction = model.predict(x_test)

        # Define metrics to calculate
        metrics = {
            "mse": MSE(),
            "r2_score": R2Score(),
            "rmse": RMSE(),
            "mae": MAE(),
            "mape": MAPE(),
        }

        # Calculate and log metrics
        results = {}
        for metric_name, metric_instance in metrics.items():
            logging.info(f"Calculating {metric_name.upper()}...")
            score = metric_instance.calculate_score(y_test, prediction)
            mlflow.log_metric(metric_name, score)
            results[metric_name] = score
            logging.info(f"{metric_name.upper()} value: {score}")

        logging.info("Model evaluation completed successfully.")
        return results["r2_score"], results["rmse"]
    except Exception as e:
        logging.error(f"An error occurred during model evaluation: {e}")
        raise e
