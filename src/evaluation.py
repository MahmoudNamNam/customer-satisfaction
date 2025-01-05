import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class Evaluation(ABC):
    """
    Abstract Class defining the strategy for evaluating model performance.
    """
    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass


class MSE(Evaluation):
    """
    Evaluation strategy that uses Mean Squared Error (MSE).
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the MSE class")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("The mean squared error value is: " + str(mse))
            return mse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MSE class. Exception message: " + str(e)
            )
            raise e


class R2Score(Evaluation):
    """
    Evaluation strategy that uses R2 Score.
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the R2Score class")
            r2 = r2_score(y_true, y_pred)
            logging.info("The r2 score value is: " + str(r2))
            return r2
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the R2Score class. Exception message: " + str(e)
            )
            raise e


class RMSE(Evaluation):
    """
    Evaluation strategy that uses Root Mean Squared Error (RMSE).
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the RMSE class")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info("The root mean squared error value is: " + str(rmse))
            return rmse
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the RMSE class. Exception message: " + str(e)
            )
            raise e


class MAE(Evaluation):
    """
    Evaluation strategy that uses Mean Absolute Error (MAE).
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the MAE class")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info("The mean absolute error value is: " + str(mae))
            return mae
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MAE class. Exception message: " + str(e)
            )
            raise e


class MAPE(Evaluation):
    """
    Evaluation strategy that uses Mean Absolute Percentage Error (MAPE).
    """
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Entered the calculate_score method of the MAPE class")
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            logging.info("The mean absolute percentage error value is: " + str(mape))
            return mape
        except Exception as e:
            logging.error(
                "Exception occurred in calculate_score method of the MAPE class. Exception message: " + str(e)
            )
            raise e
