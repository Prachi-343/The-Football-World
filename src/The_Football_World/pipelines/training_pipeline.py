from src.The_Football_World.components.data_ingestion import DataIngestion
from src.The_Football_World.components.data_transformation import DataTransformation
from src.The_Football_World.components.model_trainer import ModelTrainer

import os
import sys
import numpy as np
import pandas as pd
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException


class TrainingPipeline:
    """Class to manage the entire training pipeline: data ingestion, transformation, and model training."""
    
    def start_data_ingestion(self) -> tuple:
        """
        Initiates the data ingestion process.
        Returns:
            tuple: Paths to train and test data.
        """
        try:
            logging.info("Starting data ingestion process...")
            data_ingest = DataIngestion()
            train_data_path, test_data_path = data_ingest.initate_data_ingestion()
            logging.info(f"Data ingestion complete. Train data: {train_data_path}, Test data: {test_data_path}")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)
        
    def start_data_transformation(self, train_data_path: str, test_data_path: str) -> tuple:
        """
        Initiates the data transformation process.
        Args:
            train_data_path (str): Path to the training data.
            test_data_path (str): Path to the test data.
        Returns:
            tuple: Transformed train and test data arrays.
        """
        try:
            logging.info("Starting data transformation process...")
            data_transformation = DataTransformation()
            Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df = data_transformation.initate_data_transformation(train_data_path, test_data_path)
            logging.info("Data transformation complete.")
            return Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)
        
    def start_model_training(self,Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df):
        """
        Initiates the model training process.
        Args:
            train_arr (np.ndarray): Transformed training data.
            test_arr (np.ndarray): Transformed test data.
        """
        try:
            logging.info("Starting model training process...")
            model_trainer = ModelTrainer()
            train_accuracy,test_accuracy=model_trainer.initiate_model_training(Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df)
            print(train_accuracy)
            print(test_accuracy)
            logging.info("Model training complete.")
            
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

    def start_training(self):
        """
        Orchestrates the entire training pipeline, including data ingestion, 
        transformation, and model training.
        """
        try:
            logging.info("Training pipeline initiated...")
            train_data_path, test_data_path = self.start_data_ingestion()
            Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df = self.start_data_transformation(train_data_path, test_data_path)
            self.start_model_training(Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df)
            logging.info("Training pipeline completed successfully.")
        except Exception as e:
            logging.error(f"Error in the training pipeline: {str(e)}")
            raise CustomException(e, sys)


# Entry point for the pipeline execution
if __name__ == "__main__":
    training_obj = TrainingPipeline()
    training_obj.start_training()
