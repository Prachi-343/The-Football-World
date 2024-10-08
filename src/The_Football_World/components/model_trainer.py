import numpy as np
import os
import sys
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from src.The_Football_World.logger import logging
from src.The_Football_World.exception import CustomException

class ModelTrainerConfig:
    MODEL_FILE_PATH = os.path.join(os.getcwd(), "artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        """Trains and tunes the model."""
        try:
            logging.info("Model training started")

            # Define the XGBoost model with hyperparameter tuning
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            param_dist = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 5, 7],
                'subsample': [0.5, 0.7, 1.0]
            }

            # Perform randomized search for hyperparameter tuning
            randomized_search = RandomizedSearchCV(xgb, param_distributions=param_dist, 
                                                   n_iter=10, scoring='accuracy', 
                                                   cv=3, verbose=1, random_state=42)
            randomized_search.fit(X_train, y_train)

            best_model = randomized_search.best_estimator_

            # Cross-validation to check performance
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
            logging.info(f"Cross-validation accuracy: {cv_scores.mean()}")

            # Train the best model on the full training data
            best_model.fit(X_train, y_train)

            # Make predictions and evaluate
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            logging.info(f"Test accuracy: {accuracy}")
            logging.info(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

            # Save the trained model
            joblib.dump(best_model, self.model_trainer_config.MODEL_FILE_PATH)
            logging.info(f"Model saved at {self.model_trainer_config.MODEL_FILE_PATH}")

            return accuracy, precision, recall, f1
        except Exception as e:
            raise CustomException(e, sys)
