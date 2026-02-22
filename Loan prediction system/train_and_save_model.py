"""
Train the loan prediction model and save it for the web app.
Run once: python train_and_save_model.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
MODEL_DIR = os.path.join(SCRIPT_DIR, "model_artifacts")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    train["Loan_Status"] = train["Loan_Status"].replace("N", 0).replace("Y", 1)

    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]:
        train[col].fillna(train[col].mode()[0], inplace=True)
    train["LoanAmount"].fillna(train["LoanAmount"].median(), inplace=True)

    train["LoanAmount_log"] = np.log(train["LoanAmount"])
    train = train.drop("Loan_ID", axis=1)
    for col in ["Gender", "Dependents", "Self_Employed"]:
        train = train.drop(col, axis=1)

    x = train.drop("Loan_Status", axis=1)
    y = train["Loan_Status"]
    x = pd.get_dummies(x)
    feature_columns = list(x.columns)

    x_train, x_cv, y_train, y_cv = train_test_split(x, y, train_size=0.75, random_state=0)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    acc = accuracy_score(y_cv, model.predict(x_cv))
    print(f"Validation accuracy: {acc:.4f}")

    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Model and features saved to {MODEL_DIR}")

if __name__ == "__main__":
    main()
