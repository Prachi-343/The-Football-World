import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException

class DataIngestionConfig:
    RAW_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "raw.csv")
    TRAIN_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "train.csv")
    TEST_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()

    def _create_match_outcome(self, row):
        """Assigns match outcome based on scores."""
        if row['home_score'] > row['away_score']:
            return 1  # Home team win
        elif row['home_score'] < row['away_score']:
            return 0  # Away team win
        else:
            return 2  # Draw

    def initiate_data_ingestion(self):
        """Initiates the data ingestion process: loading, splitting, and saving the data."""
        try:
            logging.info("Data ingestion started")

            # Load the dataset
            data_path = os.path.join(os.getcwd(), "notebooks", "data", "results.csv")
            data = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully from {data_path}")

            # Create match_outcome column
            data['match_outcome'] = data.apply(self._create_match_outcome, axis=1)
            logging.info("match_outcome column added to the dataset")

            # Save raw data
            os.makedirs(os.path.dirname(self.data_ingestion_config.RAW_DATA_PATH), exist_ok=True)
            data.to_csv(self.data_ingestion_config.RAW_DATA_PATH, index=False)
            logging.info(f"Raw data saved at {self.data_ingestion_config.RAW_DATA_PATH}")

            # Split data
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.TRAIN_DATA_PATH, index=False)
            test_set.to_csv(self.data_ingestion_config.TEST_DATA_PATH, index=False)
            logging.info("Train and test data saved")

            return self.data_ingestion_config.TRAIN_DATA_PATH, self.data_ingestion_config.TEST_DATA_PATH
        except Exception as e:
            raise CustomException(e, sys)
