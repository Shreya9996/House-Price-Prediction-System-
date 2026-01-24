from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
data = pd.read_csv("house_price_prediction_550_rows.csv")

X = data.drop("Price", axis=1)
y = data["Price"]

# =========================
# TRAIN TEST SPLIT
# =========================
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# SCALING
# =========================
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_train = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test = pd.DataFrame(x_test_scaled, columns=x_test.columns)

# =========================
# MODEL TRAINING
# =========================
model = LinearRegression()
model.fit(x_train, y_train)

# =========================
# HOME PAGE
# =========================
@app.route("/")
def home():
    return render_template("index.html")

# =========================
# PREDICTION ROUTE
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    try:
        Area_sqft = float(request.form["Area_sqft"])
        Bedrooms = int(request.form["Bedrooms"])
        Bathrooms = int(request.form["Bathrooms"])
        Floors = int(request.form["Floors"])
        Parking = int(request.form["Parking"])
        Age_of_House = int(request.form["Age_of_House"])
        Distance_from_City_km = float(request.form["Distance_from_City_km"])

        input_df = pd.DataFrame({
            "Area_sqft": [Area_sqft],
            "Bedrooms": [Bedrooms],
            "Bathrooms": [Bathrooms],
            "Floors": [Floors],
            "Parking": [Parking],
            "Age_of_House": [Age_of_House],
            "Distance_from_City_km": [Distance_from_City_km]
        })

        scaled_input = scaler.transform(input_df)
        scaled_input = pd.DataFrame(scaled_input, columns=x_train.columns)

        prediction = model.predict(scaled_input)[0]

        return render_template(
            "index.html",
            prediction_text=f"🏠 Estimated House Price: ₹ {round(prediction, 2)}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="❌ Please enter valid inputs"
        )

# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
