from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from src.The_Football_World.pipelines.prediction_pipeline import CustomData, PredictPipeline
import os

app = Flask(__name__)

# Function to map numerical outcomes to human-readable results
def map_outcome_label(outcome):
    if outcome == 1:
        return "Home Team Wins"
    elif outcome == 0:
        return "Away Team Wins"
    elif outcome == 2:
        return "Draw"
    else:
        return "Unknown"

# Function to get the previous 5 matches of a team from raw.csv
def get_previous_matches(team_name):
    try:
        # Path to the raw.csv file in the artifacts folder
        csv_file_path = os.path.join("artifacts", "raw.csv")

        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)

        # Filter matches where the team played as home or away
        team_matches = df[(df['home_team'] == team_name) | (df['away_team'] == team_name)]

        # Sort matches by date (assuming 'date' column exists in the format YYYY-MM-DD)
        team_matches['date'] = pd.to_datetime(team_matches['date'])
        team_matches = team_matches.sort_values(by='date', ascending=False)

        # Select the last 5 matches
        last_5_matches = team_matches.head(5)

        # Convert the matches to a list of dictionaries to pass to the HTML template
        matches_list = last_5_matches[['date', 'home_team', 'away_team', 'home_score', 'away_score']].to_dict(orient='records')

        return matches_list

    except Exception as e:
        print(f"Error occurred while fetching previous matches: {str(e)}")
        return []

# Home Route ('/')
@app.route('/')
def home():
    return render_template('index.html')

# Predict Route ('/predict')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the user
        home_team = request.form['home_team']
        away_team = request.form['away_team']
        tournament = request.form['tournament']
        neutral = request.form.get('neutral', 'false') == 'true'
        city = request.form['city']
        country = request.form['country']

        # Create a CustomData instance with the form data
        user_input = CustomData(
            home_team=home_team,
            away_team=away_team,
            tournament=tournament,
            neutral=neutral,
            city=city,
            country=country
        )

        # Convert the input to a DataFrame
        final_data = user_input.get_data_as_dataframe()

        # Initialize the prediction pipeline
        predictor = PredictPipeline()

        # Get predictions
        prediction = predictor.predict(final_data)

        # Get prediction probabilities (optional)
        prediction_probs = predictor.predict_proba(final_data)

        # Get the last 5 matches for the home team
        last_5_matches = get_previous_matches(home_team)

        # Map the numerical outcome to a human-readable label
        mapped_outcome = map_outcome_label(prediction[0])

        # Round the probabilities to two decimal places
        home_win_prob = round(prediction_probs[0][1] * 100, 2)
        away_win_prob = round(prediction_probs[0][0] * 100, 2)
        draw_prob = round(prediction_probs[0][2] * 100, 2)

        # Pass the last 5 matches to the result page
        return render_template('result.html', outcome=mapped_outcome, 
                               home_win_prob=home_win_prob, 
                               away_win_prob=away_win_prob, 
                               draw_prob=draw_prob, 
                               last_5_matches=last_5_matches)
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
