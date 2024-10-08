import os
import sys
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer 
from src.The_Football_World.exception import CustomException
from src.The_Football_World.logger import logging
from src.The_Football_World.utils.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "Model.pkl")
            
            # Load the preprocessor and the trained model separately
            model = load_object(model_path)  # This should be the BayesSearchCV object
            
            # Extract the fitted pipeline from BayesSearchCV
            # model.best_estimator_  # This is your pipeline
            
            # Preprocess the input features using the pipeline's 'preprocessor' step
            transformed_data = SimpleImputer.fit(features)
            
            # Use the model to make predictions on the preprocessed data
            pred = model.predict(transformed_data)
            
            # Map predictions to the specific outcomes
            outcome_mapping = {0: 'Away Team Wins', 1: 'Home Team Wins', 2: 'Draw/Penalty'}
            final_predictions = [outcome_mapping.get(p, "Unknown") for p in pred]
            
            return final_predictions
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
class CustomData:
    def __init__(self, home_team: str, away_team: str, tournament: str, home_score=np.nan ,away_score=np.nan):
        # Input features related to the match
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score
        self.tournament = tournament
        
    def get_data_as_dataframe(self):
        try:
            # Create a dictionary of input data
            custom_data_input_dict = {
                'home_team': [self.home_team],
                'away_team': [self.away_team],
                'home_score': [self.home_score],
                'away_score': [self.away_score],
                'tournament': [self.tournament]
            }
            # Convert to DataFrame
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe created for prediction')
            return df
        except Exception as e:
            logging.info('Exception Occurred while creating DataFrame for prediction')
            raise CustomException(e, sys)



user_input = CustomData(
    home_team='Scotland',
    away_team='England',
    tournament='Friendly'
)

final_data=user_input.get_data_as_dataframe()
predictor = PredictPipeline()

prediction = predictor.predict(final_data)
print(f'Predicted Outcome: {prediction}')