from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)


# -----------------------------------------
# Load Non-Linear SVM Model
# -----------------------------------------

with open("NonLinearSVMModel.pkl", "rb") as file:
    model = pickle.load(file)


# Load Scaler
with open("NonLinearScaler.pkl", "rb") as file:
    scaler = pickle.load(file)


# Same features used during model training
features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean concavity"
]


# -----------------------------------------
# Home Page
# -----------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------------
# Prediction
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Get user input from HTML form

        radius = float(request.form["radius"])
        texture = float(request.form["texture"])
        perimeter = float(request.form["perimeter"])
        area = float(request.form["area"])
        concavity = float(request.form["concavity"])


        # Convert input into DataFrame

        input_data = pd.DataFrame(
            [[
                radius,
                texture,
                perimeter,
                area,
                concavity
            ]],
            columns=features
        )


        # Scale input data

        input_scaled = scaler.transform(input_data)


        # Make prediction

        prediction = model.predict(input_scaled)[0]


        # Convert result

        if prediction == 0:
            result = "Malignant"

        else:
            result = "Benign"


        # Send prediction to HTML

        return render_template(
            "index.html",
            prediction_text=result,
            radius=radius,
            texture=texture,
            perimeter=perimeter,
            area=area,
            concavity=concavity
        )


    except Exception as e:

        print("Error:", e)

        return render_template(
            "index.html",
            error_text="Please enter valid input values."
        )


# -----------------------------------------
# Run Flask Application
# -----------------------------------------

if __name__ == "__main__":

    app.run(debug=True)