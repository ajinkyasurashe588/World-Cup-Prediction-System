from flask import Flask, render_template, request
import joblib

# Create Flask app
app = Flask(__name__)

# Load trained model
model = joblib.load("models/random_forest_model.pkl")

# Load encoders
home_encoder = joblib.load("models/home_encoder.pkl")
away_encoder = joblib.load("models/away_encoder.pkl")
stage_encoder = joblib.load("models/stage_encoder.pkl")
result_encoder = joblib.load("models/result_encoder.pkl")


@app.route("/")
def home():

    # Get all teams from the encoders
    home_teams = home_encoder.classes_.tolist()
    away_teams = away_encoder.classes_.tolist()

    # Get all match stages
    stages = stage_encoder.classes_.tolist()

    return render_template(
        "index.html",
        home_teams=home_teams,
        away_teams=away_teams,
        stages=stages
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get form data
        home_team = request.form["home_team"]
        away_team = request.form["away_team"]
        stage = request.form["stage"].strip().lower()

        # Check same team
        if home_team == away_team:
            return render_template(
                "result.html",
                error="Home Team and Away Team cannot be the same."
            )

        # Encode input values
        home = home_encoder.transform([home_team])[0]
        away = away_encoder.transform([away_team])[0]
        stage_encoded = stage_encoder.transform([stage])[0]

        # Print encoded values in terminal
        print("Home:", home, type(home))
        print("Away:", away, type(away))
        print("Stage:", stage_encoded, type(stage_encoded))

        # Prepare input for the model
        match_data = [[home, away, stage_encoded]]

        # Make prediction
        prediction = model.predict(match_data)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(match_data)[0]

        # Convert prediction back to result name
        result = result_encoder.inverse_transform(
            [prediction]
        )[0]

        # Create probability dictionary
        probability_data = {}

        for class_value, probability in zip(
            model.classes_,
            probabilities
        ):
            class_name = result_encoder.inverse_transform(
                [class_value]
            )[0]

            probability_data[class_name] = round(
                probability * 100,
                2
            )

        # Get individual probabilities
        home_win_probability = probability_data.get(
            "home team win",
            0
        )

        draw_probability = probability_data.get(
            "draw",
            0
        )

        away_win_probability = probability_data.get(
            "away team win",
            0
        )

        # Convert result into actual team name
        if result == "home team win":

            predicted_winner = home_team
            prediction_text = f"{home_team} is predicted to win."

        elif result == "away team win":

            predicted_winner = away_team
            prediction_text = f"{away_team} is predicted to win."

        else:

            predicted_winner = "Draw"
            prediction_text = "The match is predicted to end in a draw."

        # Display stage nicely
        display_stage = stage.replace("-", " ").title()

        return render_template(
            "result.html",
            home_team=home_team,
            away_team=away_team,
            stage=display_stage,
            result=result,
            predicted_winner=predicted_winner,
            prediction_text=prediction_text,
            home_win_probability=home_win_probability,
            draw_probability=draw_probability,
            away_win_probability=away_win_probability
        )

    except Exception as e:

        return render_template(
            "result.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)