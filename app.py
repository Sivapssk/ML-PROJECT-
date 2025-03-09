from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained model
model_path = "best_loan_status_model.pkl"

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

    try:
        df = pd.DataFrame([data])
        result = model.predict(df)[0]
        return "Loan Approved ✅" if result == 1 else "Loan Not Approved ❌"
    except Exception as e:
        return f"❌ Prediction Error: {str(e)}"

# Home route (for form submission)
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if model is None:
            return render_template("index.html", prediction="Model not available, please check the server.")

        try:
            # Extracting form data
            data = {
                'Gender': int(request.form.get('Gender', 0)),
                'Married': int(request.form.get('Married', 0)),
                'Dependents': int(request.form.get('Dependents', 0)),
                'Education': int(request.form.get('Education', 0)),
                'Self_Employed': int(request.form.get('Self_Employed', 0)),
                'ApplicantIncome': float(request.form.get('ApplicantIncome', 0)),
                'CoapplicantIncome': float(request.form.get('CoapplicantIncome', 0)),
                'LoanAmount': float(request.form.get('LoanAmount', 0)),
                'Loan_Amount_Term': float(request.form.get('Loan_Amount_Term', 0)),
                'Credit_History': int(request.form.get('Credit_History', 0)),
                'Property_Area': int(request.form.get('Property_Area', 0)),

                # Placeholder for missing features
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

        except Exception as e:
            return render_template("index.html", prediction=f"❌ Form Error: {str(e)}")

    return render_template("index.html", prediction="")

# API Endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available, please check the server."})

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        result = predict_loan_status(data)
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": f"❌ API Error: {str(e)}"}), 500

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
