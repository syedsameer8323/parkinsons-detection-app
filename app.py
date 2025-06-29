from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")


# Landing Page
@app.route("/")
def home():
    return render_template("home.html")

# Form Page
@app.route("/form")
def form():
    return render_template("form.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        probability = model.predict_proba(scaled)[0][1] * 100  # Class 1 = Parkinson's

        output = "🧠 Parkinson's Detected" if prediction[0] == 1 else "✔️ No Parkinson's Detected"
        confidence = f"Confidence: {probability:.2f}%"
        return render_template("result.html", result=output + " — " + confidence)
    except Exception as e:
        return render_template("form.html", result="⚠️ Error: " + str(e))


if __name__ == "__main__":
    app.run(debug=True)
