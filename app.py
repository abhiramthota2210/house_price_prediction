from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))  # if used

# Ocean encoding map (same as training)
ocean_map = {
    "INLAND": [1,0,0,0,0],
    "NEAR BAY": [0,1,0,0,0],
    "NEAR OCEAN": [0,0,1,0,0],
    "ISLAND": [0,0,0,1,0],
    "1H OCEAN": [0,0,0,0,1]
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    # Get values from form
    longitude = float(request.form['longitude'])
    latitude = float(request.form['latitude'])
    housing_median_age = float(request.form['housing_median_age'])
    total_rooms = float(request.form['total_rooms'])
    total_bedrooms = float(request.form['total_bedrooms'])
    population = float(request.form['population'])
    households = float(request.form['households'])
    median_income = float(request.form['median_income'])
    ocean = request.form['ocean_proximity']

    # Numerical features
    features = [
        longitude, latitude, housing_median_age,
        total_rooms, total_bedrooms,
        population, households, median_income
    ]

    # Scale numerical data
    features_scaled = scaler.transform([features])

    # Encode ocean_proximity
    ocean_encoded = ocean_map[ocean]

    # Combine features
    final_input = np.concatenate([features_scaled[0], ocean_encoded]).reshape(1, -1)

    # Predict
    prediction = model.predict(final_input)[0]

    return render_template("index.html", prediction_text=f"Predicted Price: {prediction:.2f}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)