import os
import sys
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException
from src.The_Football_World.components.data_ingestion import DataIngestion
from src.The_Football_World.components.data_transformation import DataTransformation
from src.The_Football_World.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_data_ingestion(self):
        """Handles the data ingestion process."""
        try:
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed: {train_data_path}, {test_data_path}")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error(f"Error during data ingestion: {str(e)}")
            raise CustomException(e, sys)

    def start_data_transformation(self, train_data_path, test_data_path):
        """Handles the data transformation process."""
        try:
            input_features_train, target_feature_train, input_features_test, target_feature_test = self.data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info("Data transformation completed")
            return input_features_train, target_feature_train, input_features_test, target_feature_test
        except Exception as e:
            logging.error(f"Error during data transformation: {str(e)}")
            raise CustomException(e, sys)

    def start_model_training(self, input_features_train, target_feature_train, input_features_test, target_feature_test):
        """Handles the model training and evaluation process."""
        try:
            accuracy, precision, recall, f1 = self.model_trainer.initiate_model_training(input_features_train, target_feature_train, input_features_test, target_feature_test)
            logging.info(f"Model training completed with accuracy: {accuracy}, precision: {precision}, recall: {recall}, F1-score: {f1}")
        except Exception as e:
            logging.error(f"Error during model training: {str(e)}")
            raise CustomException(e, sys)

    def start_training(self):
        """Orchestrates the entire training pipeline: ingestion, transformation, and training."""
        try:
            logging.info("Training pipeline initiated")
            
            # Step 1: Data Ingestion
            train_data_path, test_data_path = self.start_data_ingestion()
            
            # Step 2: Data Transformation
            input_features_train, target_feature_train, input_features_test, target_feature_test = self.start_data_transformation(train_data_path, test_data_path)
            
            # Step 3: Model Training
            self.start_model_training(input_features_train, target_feature_train, input_features_test, target_feature_test)
            
            logging.info("Training pipeline completed successfully")
        except Exception as e:
            logging.error(f"Error in the training pipeline: {str(e)}")
            raise CustomException(e, sys)

# Entry point for the pipeline execution
if __name__ == "__main__":
    training_obj = TrainingPipeline()
    training_obj.start_training()
