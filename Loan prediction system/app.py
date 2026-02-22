"""
Loan Prediction web app: serves frontend and /api/predict.
Run: python app.py then open http://127.0.0.1:5000
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

FEATURE_LABELS = {
    "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount requested",
    "Loan_Amount_Term": "Loan term",
    "Credit_History": "Credit history",
    "LoanAmount_log": "Loan size (log scale)",
    "Married_No": "Marital status (single)",
    "Married_Yes": "Marital status (married)",
    "Education_Graduate": "Graduate education",
    "Education_Not Graduate": "Not graduate",
    "Property_Area_Rural": "Property in rural area",
    "Property_Area_Semiurban": "Property in semiurban area",
    "Property_Area_Urban": "Property in urban area",
}

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

model = None
feature_columns = None


def load_model():
    global model, feature_columns
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(FEATURES_PATH):
        return False
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        feature_columns = json.load(f)
    return True


def preprocess(body):
    """Build one row with same features as training."""
    df = pd.DataFrame([{
        "Married": body.get("Married", "No"),
        "Education": body.get("Education", "Graduate"),
        "ApplicantIncome": float(body.get("ApplicantIncome", 0)),
        "CoapplicantIncome": float(body.get("CoapplicantIncome", 0)),
        "LoanAmount": float(body.get("LoanAmount", 0)),
        "Loan_Amount_Term": float(body.get("Loan_Amount_Term", 360)),
        "Credit_History": float(body.get("Credit_History", 1)),
        "Property_Area": body.get("Property_Area", "Urban"),
    }])
    df["LoanAmount_log"] = np.log(df["LoanAmount"].clip(lower=1))
    df = pd.get_dummies(df)
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    return df[feature_columns]


def explain_prediction(X):
    if not hasattr(model, "coef_"):
        return {"factors_that_helped": [], "factors_that_reduced": []}
    coef = model.coef_[0]
    row = X.iloc[0]
    contributions = []
    for i, name in enumerate(feature_columns):
        val = row.iloc[i]
        contrib = coef[i] * val
        if abs(contrib) < 1e-6:
            continue
        label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
        contributions.append({"label": label, "contribution": float(contrib)})
    contributions.sort(key=lambda x: -abs(x["contribution"]))
    helped = [c["label"] for c in contributions if c["contribution"] > 0][:6]
    reduced = [c["label"] for c in contributions if c["contribution"] < 0][:6]
    return {"factors_that_helped": helped, "factors_that_reduced": reduced}


@app.route("/")
def index():
    return send_from_directory(APP_DIR, "index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run: python train_and_save_model.py"}), 503
    try:
        data = request.get_json() or {}
        X = preprocess(data)
        pred = int(model.predict(X)[0])
        explanation = explain_prediction(X)
        return jsonify({"prediction": pred, "Loan_Status": "Y" if pred == 1 else "N", "explanation": explanation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    if load_model():
        print("Model loaded. Open http://127.0.0.1:5000")
    else:
        print("Run first: python train_and_save_model.py")
    app.run(host="0.0.0.0", port=5000, debug=False)
