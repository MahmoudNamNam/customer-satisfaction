import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)
from typing_extensions import Annotated

# from zenml.steps import Output, step
from zenml import step


@step
def clean_data(
    data: pd.DataFrame,
    preprocess_strategy: DataPreprocessStrategy,
    divide_strategy: DataDivideStrategy,
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Cleans and splits data into train and test sets using the provided strategies.

    Args:
        data (pd.DataFrame): Input data.
        preprocess_strategy (DataPreprocessStrategy): Strategy for data preprocessing.
        divide_strategy (DataDivideStrategy): Strategy for data division.

    Returns:
        Tuple containing train/test splits for features and target variables.
    """
    try:
        logging.info("Starting data preprocessing...")
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        logging.info("Dividing data into training and testing sets...")
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data()
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error during data cleaning and splitting: {e}")
        raise e
