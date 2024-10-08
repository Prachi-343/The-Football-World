import os
import sys
import numpy as np
import pandas as pd
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException
from src.The_Football_World.utils.utils import save_object, evaluate_model

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

class ModelTrainerConfig:
    model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def train_model(self, X_train, y_train):
        """Train the XGBoost model on the training dataset."""
        try:
            logging.info("Starting model training...")

            # Initialize the XGBoost classifier
            model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)

            # Train the model
            model.fit(X_train, y_train)
            
            logging.info("Model training completed.")

            return model
        except Exception as e:
            logging.info("Error in training the model", e)
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the trained model on the test data."""
        try:
            logging.info("Evaluating model performance...")

            # Make predictions on test data
            y_pred = model.predict(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"Model Accuracy: {accuracy}")

            # Generate and log the classification report and confusion matrix
            report = classification_report(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            logging.info(f"Classification Report:\n {report}")
            logging.info(f"Confusion Matrix:\n {cm}")

            return accuracy
        except Exception as e:
            logging.info("Error during model evaluation", e)
            raise CustomException(e, sys)

    def initiate_model_training(self, train_arr, test_arr):
        """Main function to initiate the model training and evaluation pipeline."""
        try:
            logging.info("Model training pipeline started...")

            # Separate input features and target labels
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            # Train the model
            model = self.train_model(X_train, y_train)

            # Evaluate the model
            accuracy = self.evaluate_model(model, X_test, y_test)

            # Save the trained model to a file
            save_object(file_path=self.model_trainer_config.model_file_path, obj=model)

            logging.info("Model training pipeline completed successfully.")

            return accuracy
        except Exception as e:
            logging.info("Error in model training pipeline", e)
            raise CustomException(e, sys)
