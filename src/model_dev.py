import logging
from abc import ABC, abstractmethod

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, x_train, y_train):
        pass

    @abstractmethod
    def optimize(self, trial, x_train, y_train, x_test, y_test):
        pass


class RandomForestModel(Model):
    """
    RandomForestModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = RandomForestRegressor(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            logging.error(f"Training error in RandomForestModel: {e}")
            raise

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        try:
            n_estimators = trial.suggest_int("n_estimators", 1, 200)
            max_depth = trial.suggest_int("max_depth", 1, 20)
            min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

            reg = self.train(x_train, y_train, n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
            predictions = reg.predict(x_test)
            return -mean_squared_error(y_test, predictions)
        except Exception as e:
            logging.error(f"Optimization error in RandomForestModel: {e}")
            raise


class LightGBMModel(Model):
    """
    LightGBMModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LGBMRegressor(**kwargs)
            reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric="rmse", early_stopping_rounds=10, verbose=False)
            return reg
        except Exception as e:
            logging.error(f"Training error in LightGBMModel: {e}")
            raise

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        try:
            n_estimators = trial.suggest_int("n_estimators", 1, 200)
            max_depth = trial.suggest_int("max_depth", 1, 20)
            learning_rate = trial.suggest_uniform("learning_rate", 0.01, 0.99)

            reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            predictions = reg.predict(x_test)
            return -mean_squared_error(y_test, predictions)
        except Exception as e:
            logging.error(f"Optimization error in LightGBMModel: {e}")
            raise


class XGBoostModel(Model):
    """
    XGBoostModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = xgb.XGBRegressor(**kwargs)
            reg.fit(x_train, y_train, eval_set=[(x_train, y_train)], eval_metric="rmse", early_stopping_rounds=10, verbose=False)
            return reg
        except Exception as e:
            logging.error(f"Training error in XGBoostModel: {e}")
            raise

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        try:
            n_estimators = trial.suggest_int("n_estimators", 1, 200)
            max_depth = trial.suggest_int("max_depth", 1, 30)
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)

            reg = self.train(x_train, y_train, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            predictions = reg.predict(x_test)
            return -mean_squared_error(y_test, predictions)
        except Exception as e:
            logging.error(f"Optimization error in XGBoostModel: {e}")
            raise


class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, x_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            return reg
        except Exception as e:
            logging.error(f"Training error in LinearRegressionModel: {e}")
            raise

    def optimize(self, trial, x_train, y_train, x_test, y_test):
        try:
            reg = self.train(x_train, y_train)
            predictions = reg.predict(x_test)
            return -mean_squared_error(y_test, predictions)
        except Exception as e:
            logging.error(f"Optimization error in LinearRegressionModel: {e}")
            raise


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def optimize(self, n_trials=100):
        logging.info(f"Starting hyperparameter tuning for {self.model.__class__.__name__}")
        try:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.model.optimize(trial, self.x_train, self.y_train, self.x_test, self.y_test), n_trials=n_trials)
            logging.info(f"Best trial: {study.best_trial.params}")
            return study.best_trial.params
        except Exception as e:
            logging.error(f"Hyperparameter tuning error: {e}")
            raise
