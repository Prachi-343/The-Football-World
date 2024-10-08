import os
import sys
import numpy as np
import pandas as pd
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException
from src.The_Football_World.utils.utils import save_object, load_object

from xgboost import XGBClassifier

class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self,Preprocessor_obj_filePath, X_train, y_train):
        """Train the XGBoost model on the training dataset."""
        try:
            logging.info("Starting model training...")

            # Initialize the XGBoost classifier
            model = load_object(Preprocessor_obj_filePath)

            # Train the model
            model.fit(X_train, y_train)
            
            logging.info("Model training completed.")

            return model
        except Exception as e:
            logging.info("Error in training the model", e)
            raise CustomException(e, sys)

    def initiate_model_training(self, Preprocessor_obj_filePath,input_feature_train_df,target_feature_train_df):
        """Main function to initiate the model training and evaluation pipeline."""
        try:
            logging.info("Model training pipeline started...")
            x_train=input_feature_train_df
            y_train=target_feature_train_df
            # Train the model
            model = self.train_model(Preprocessor_obj_filePath,x_train, y_train)

            # Evaluate the model
            model.best_estimator_
            train_accuracy = model.best_score_
            # logging.info("accuracy =",train_accuracy)
            
            test_accuracy = model.score(x_train,y_train)
            # logging.info("accuracy =",test_accuracy)
            
            # Save the trained model to a file
            save_object(file_path=self.model_trainer_config.model_file_path, obj=model)

            logging.info("Model training pipeline completed successfully.")

            return train_accuracy,test_accuracy
        except Exception as e:
            logging.info("Error in model training pipeline", e)
            raise CustomException(e, sys)
