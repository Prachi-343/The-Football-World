import os
import sys
import numpy as np
import pandas as pd
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    RAW_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "raw.csv")
    TRAIN_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "train.csv")
    TEST_DATA_PATH = os.path.join(os.getcwd(), "artifacts", "test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.data_ingestion_config = DataIngestionConfig()
    
    def initate_data_ingestion(self):
        """Initiates the data ingestion process: loading, splitting, and saving the data."""
        try:
            logging.info("Data ingestion started")
            
            # Load the dataset from the specified path
            data_path = os.path.join(os.getcwd(), "notebooks", "data", "results.csv")
            data = pd.read_csv(data_path)
            logging.info(f"Data loaded successfully from {data_path}")
            
            # Create match_outcome column
            def match_outcome(row):
                if row['home_score'] > row['away_score']:
                    return 1  # Home team win
                elif row['home_score'] < row['away_score']:
                    return 0  # Away team win
                else:
                    return 2  # Draw
            
            # Add the new column
            data['match_outcome'] = data.apply(match_outcome, axis=1)
            logging.info("match_outcome column added to the dataset")

            # Save the raw data into the artifacts directory
            os.makedirs(os.path.dirname(self.data_ingestion_config.RAW_DATA_PATH), exist_ok=True)
            data.to_csv(self.data_ingestion_config.RAW_DATA_PATH, index=False)
            logging.info(f"Raw data saved at {self.data_ingestion_config.RAW_DATA_PATH}")
            
            # Split the data into train and test sets
            train_data, test_data = train_test_split(data, test_size=0.25, random_state=500)
            logging.info("Train-test split completed")

            # Save the train and test data into the artifacts directory
            train_data.to_csv(self.data_ingestion_config.TRAIN_DATA_PATH, index=False)
            test_data.to_csv(self.data_ingestion_config.TEST_DATA_PATH, index=False)
            logging.info(f"Train data saved at {self.data_ingestion_config.TRAIN_DATA_PATH}")
            logging.info(f"Test data saved at {self.data_ingestion_config.TEST_DATA_PATH}")
            
            # Return the file paths for train and test data
            return (self.data_ingestion_config.TRAIN_DATA_PATH, 
                    self.data_ingestion_config.TEST_DATA_PATH)
        
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise CustomException(e, sys)

# a=DataIngestion()
# a.initate_data_ingestion()