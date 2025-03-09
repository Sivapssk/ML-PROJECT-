from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = "best_loan_status_model.pkl"  # Assuming it's in the same directory as app.py


# Check if the model file exists before loading
try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print("❌ Error loading model:", e)

# Define prediction function
def predict_loan_status(data):
    if model is None:
        return "Model not available, please check the server."

    df = pd.DataFrame([data])
    result = model.predict(df)[0]
    return "Loan Approved ✅" if result == 1 else "Loan Not Approved ❌"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if model is None:
            return render_template("index.html", prediction="Model not available, please check the server.")

        # Get form data
        data = {  # FIXED INDENTATION
            'Gender': int(request.form['Gender']),
            'Married': int(request.form['Married']),
            'Dependents': int(request.form['Dependents']),
            'Education': int(request.form['Education']),
            'Self_Employed': int(request.form['Self_Employed']),
            'ApplicantIncome': float(request.form['ApplicantIncome']),
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'LoanAmount': float(request.form['LoanAmount']),
            'Loan_Amount_Term': float(request.form['Loan_Amount_Term']),
            'Credit_History': int(request.form['Credit_History']),
            'Property_Area': int(request.form['Property_Area']),

            # 🔹 Add missing features (use default values if needed)
            'Feature_12': 0,  
            'Feature_13': 0,
            'Feature_14': 0,
            'Feature_15': 0,
            'Feature_16': 0,
            'Feature_17': 0,
            'Feature_18': 0,
            'Feature_19': 0,
            'Feature_20': 0
        }

        # Get prediction
        result = predict_loan_status(data)
        return render_template("index.html", prediction=result)

    return render_template("index.html", prediction="")

if __name__ == "__main__":
    app.run(debug=True)
