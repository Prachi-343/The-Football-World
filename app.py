import os
import pandas as pd
from src.The_Football_World.pipelines.prediction_pipeline import CustomData, PredictPipeline
from src.The_Football_World.exception import CustomException
from src.The_Football_World.logger import logging

# Load the dataset containing match history
MATCH_HISTORY_PATH = os.path.join("notebooks", "data", "results.csv")
match_history = pd.read_csv(MATCH_HISTORY_PATH)

def get_previous_matches(team_name, num_matches=5):
    """Fetches the previous 'num_matches' matches of the given team."""
    team_matches = match_history[(match_history['home_team'] == team_name) | (match_history['away_team'] == team_name)]
    team_matches = team_matches.sort_values(by='date', ascending=False).head(num_matches)
    return team_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']]

def calculate_probabilities(prediction_probs):
    """Maps the model output probabilities to winning chances for both teams."""
    try:
        # Ensure we have at least three values for home, away, and draw.
        if len(prediction_probs[0]) == 3:
            home_win_prob = prediction_probs[0][1] * 100  # Home win probability
            away_win_prob = prediction_probs[0][0] * 100  # Away win probability
            draw_prob = prediction_probs[0][2] * 100  # Draw probability
        else:
            raise ValueError("Unexpected probability output format")
        
        return home_win_prob, away_win_prob, draw_prob
    except Exception as e:
        logging.error(f"Error calculating probabilities: {str(e)}")
        return 0, 0, 0

# User input for home team and away team
home_team = 'Scotland'
away_team = 'England'

# Get the previous 5 matches for both teams
home_team_previous_matches = get_previous_matches(home_team)
away_team_previous_matches = get_previous_matches(away_team)


print("\n")
print("\n")
print(f"Previous 5 matches of {home_team}:")
print(home_team_previous_matches)
print("\n")
print("\n")
print("\n")
print(f"Previous 5 matches of {away_team}:")
print(away_team_previous_matches)

# Prepare the data for prediction
user_input = CustomData(
    home_team=home_team,
    away_team=away_team,
    tournament='Friendly',
    neutral=False,
    city='Glasgow',
    country='Scotland'
)

# Convert input data to DataFrame
final_data = user_input.get_data_as_dataframe()

# Initialize prediction pipeline
predictor = PredictPipeline()

# Get predictions
prediction = predictor.predict(final_data)

# Try to get prediction probabilities, if supported
try:
    prediction_probs = predictor.predict_proba(final_data)
    home_win_prob, away_win_prob, draw_prob = calculate_probabilities(prediction_probs)
    print("\n")
    print("\n")
    print(f"Winning Chances - Home Team ({home_team}): {home_win_prob:.2f}%")
    print("\n")
    print(f"Winning Chances - Away Team ({away_team}): {away_win_prob:.2f}%")
    print("\n")
    print(f"Chances of Draw: {draw_prob:.2f}%")
except AttributeError:
    print(f"Model does not support probability predictions. Predicted Outcome: {prediction[0]}")
