"""
Run Loan Prediction pipeline (same logic as Loan Prediction.ipynb).
Usage: python run_loan_prediction.py
Reads Data/train.csv, Data/test.csv and writes submission CSV and prints accuracy.
"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Paths relative to script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "Data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

def main():
    print("Loading data...")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    test_original = test.copy()

    # Encode target
    train["Loan_Status"] = train["Loan_Status"].replace("N", 0).replace("Y", 1)

    # Missing value imputation - categorical: mode, numerical: median
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History", "Loan_Amount_Term"]:
        train[col].fillna(train[col].mode()[0], inplace=True)
        if col in test.columns:
            test[col].fillna(test[col].mode()[0], inplace=True)
    train["LoanAmount"].fillna(train["LoanAmount"].median(), inplace=True)
    test["LoanAmount"].fillna(test["LoanAmount"].median(), inplace=True)

    # Log transform
    train["LoanAmount_log"] = np.log(train["LoanAmount"])
    test["LoanAmount_log"] = np.log(test["LoanAmount"])

    # Drop columns not used in model
    train = train.drop("Loan_ID", axis=1)
    test = test.drop("Loan_ID", axis=1)
    for col in ["Gender", "Dependents", "Self_Employed"]:
        train = train.drop(col, axis=1)
        test = test.drop(col, axis=1)

    # Features and target
    x = train.drop("Loan_Status", axis=1)
    y = train["Loan_Status"]

    # One-hot encode; align test to train columns
    x = pd.get_dummies(x)
    test = pd.get_dummies(test)
    # Ensure test has same columns as x (missing dummies -> 0)
    for c in x.columns:
        if c not in test.columns:
            test[c] = 0
    test = test[x.columns]

    # Train/validation split
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, train_size=0.75, random_state=0)

    # Train model
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    pred_cv = model.predict(x_cv)
    acc = accuracy_score(y_cv, pred_cv)
    print(f"Validation accuracy: {acc:.4f}")

    # Predict on test
    pred_test = model.predict(test)

    # Build submission
    submission = pd.DataFrame({
        "Loan_ID": test_original["Loan_ID"],
        "Loan_Status": pred_test
    })
    submission["Loan_Status"] = submission["Loan_Status"].replace(0, "N").replace(1, "Y")

    out_path = os.path.join(SCRIPT_DIR, "submission.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")
    print(submission.head(10))
    print("\nDone.")

if __name__ == "__main__":
    main()
