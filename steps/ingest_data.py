import logging
import os
import pandas as pd
from zenml import step


class DataIngestionError(Exception):
    """
    Custom exception for errors in the data ingestion process.
    """
    pass


class IngestData:
    """
    Data ingestion class to load data from a specified source.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initialize the data ingestion class with a file path.

        Args:
            file_path (str): Path to the data file.
        """
        self.file_path = file_path

    def get_data(self) -> pd.DataFrame:
        """
        Load data from the specified CSV file.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.
        """
        if not os.path.exists(self.file_path):
            raise DataIngestionError(f"File not found: {self.file_path}")
        
        try:
            df = pd.read_csv(self.file_path)
            if df.empty:
                raise DataIngestionError("The data file is empty.")
            
            logging.info(f"Successfully loaded data from {self.file_path}")
            return df
        except pd.errors.ParserError as e:
            raise DataIngestionError(f"Error parsing the CSV file: {e}")
        except Exception as e:
            raise DataIngestionError(f"Unexpected error: {e}")


@step
def ingest_data(file_path: str = "./data/data.csv") -> pd.DataFrame:
    """
    ZenML step to ingest data.

    Args:
        file_path (str): Path to the data file. Default is './data/data.csv'.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        logging.info("Starting data ingestion step.")
        ingest_data = IngestData(file_path)
        df = ingest_data.get_data()
        return df
    except DataIngestionError as e:
        logging.error(f"Data ingestion error: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error during data ingestion: {e}")
        raise
