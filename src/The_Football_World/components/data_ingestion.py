import os
import sys
import numpy as np
import pandas as pd
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import customexception

from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    RAW_DATA_PATH = os.path.join(os.getcwd(), "artifact", "raw.csv")
    TRAIN_DATA_PATH = os.path.join(os.getcwd(), "artifact", "train.csv")
    TEST_DATA_PATH = os.path.join(os.getcwd(), "artifact", "test.csv")

class DataIngestion:
    def __init__(self) -> None:
        pass
    
    def initate_data_ingestion(self):
        try:
            logging.info("Data ingestion started")
            data_path = os.path.join(os.getcwd(), "data", "football_data.csv")
            df = pd.read_csv(data_path)
            logging.info("Data ingestion completed")
            return df
        except Exception as e:
            logging.error(f"Error occurred during data ingestion: {str(e)}")
            raise customexception("Data ingestion failed")