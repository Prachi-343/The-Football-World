import os
import sys
import pandas as pd
import numpy as np

from src.The_Football_World.exception import CustomException
from src.The_Football_World.logger import logging
from src.The_Football_World.utils.utils import load_object
import joblib

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        """Predicts the match outcome based on input features."""
        model = joblib.load(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)
        transformed_features = preprocessor.transform(features)
        predictions = model.predict(transformed_features)
        return predictions

    def predict_proba(self, features):
        """Predicts the probabilities for each outcome (home win, away win, draw)."""
        model = joblib.load(self.model_path)
        preprocessor = joblib.load(self.preprocessor_path)
        transformed_features = preprocessor.transform(features)
        probabilities = model.predict_proba(transformed_features)
        return probabilities


class CustomData:
    def __init__(self, home_team: str, away_team: str, tournament: str, neutral: bool, city: str, country: str, home_score=np.nan, away_score=np.nan):
        """Initializes the custom data inputs."""
        self.home_team = home_team
        self.away_team = away_team
        self.tournament = tournament
        self.neutral = neutral
        self.city = city
        self.country = country
        self.home_score = home_score
        self.away_score = away_score

    def get_data_as_dataframe(self):
        """Converts the input data into a DataFrame for prediction."""
        try:
            custom_data_input_dict = {
                'home_team': [self.home_team],
                'away_team': [self.away_team],
                'tournament': [self.tournament],
                'neutral': [self.neutral],
                'city': [self.city],
                'country': [self.country],
                'home_score': [self.home_score],
                'away_score': [self.away_score]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Dataframe created for prediction")
            return df
        except Exception as e:
            logging.error(f"Exception occurred while creating DataFrame: {str(e)}")
            raise CustomException(e, sys)


