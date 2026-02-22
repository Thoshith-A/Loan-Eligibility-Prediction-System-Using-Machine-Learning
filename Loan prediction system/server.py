"""
Loan Prediction web app using only stdlib HTTP (no Flask).
Run: python server.py then open http://127.0.0.1:5000
"""
import os
import json
import urllib.parse
import warnings
warnings.filterwarnings("ignore")

from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import pandas as pd
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

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


# Human-readable labels for explanation
FEATURE_LABELS = {
    "ApplicantIncome": "Applicant income",
    "CoapplicantIncome": "Co-applicant income",
    "LoanAmount": "Loan amount (₹ thousands)",
    "Loan_Amount_Term": "Loan term (months)",
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

# Limits consistent with training data (train.csv: LoanAmount ~17-349 thousands, Income monthly)
LOAN_AMOUNT_THOUSANDS_MIN = 1.0
LOAN_AMOUNT_THOUSANDS_MAX = 500.0   # ₹50 lakh max
INCOME_MIN = 0
INCOME_MAX = 5_000_000
LOAN_TERM_MIN = 6
LOAN_TERM_MAX = 480


def _to_float(x, default=0.0):
    """Parse to float; return default on failure."""
    if x is None or x == "":
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def preprocess(body):
    """
    Build one row for the model. Units must match training data:
    - ApplicantIncome, CoapplicantIncome: ₹/month
    - LoanAmount: in THOUSANDS of ₹ (e.g. 120 = ₹1,20,000)
    - Loan_Amount_Term: months
    - Credit_History: 0 or 1
    """
    # 1) Loan amount: accept rupees or thousands
    loan_rupees = _to_float(body.get("LoanAmountRupees"), None)
    loan_thousands_raw = _to_float(body.get("LoanAmount"), None)
    if loan_rupees is not None and loan_rupees > 0:
        loan_thousands = loan_rupees / 1000.0
    elif loan_thousands_raw is not None and loan_thousands_raw > 0:
        loan_thousands = loan_thousands_raw
    else:
        loan_thousands = 100.0
    loan_thousands = np.clip(loan_thousands, LOAN_AMOUNT_THOUSANDS_MIN, LOAN_AMOUNT_THOUSANDS_MAX)

    applicant_income = np.clip(_to_float(body.get("ApplicantIncome"), 0), INCOME_MIN, INCOME_MAX)
    coapplicant_income = np.clip(_to_float(body.get("CoapplicantIncome"), 0), INCOME_MIN, INCOME_MAX)
    loan_term = np.clip(_to_float(body.get("Loan_Amount_Term"), 360), LOAN_TERM_MIN, LOAN_TERM_MAX)
    credit = 1.0 if str(body.get("Credit_History", "1")).strip() == "1" else 0.0

    married = "Yes" if str(body.get("Married", "No")).strip().lower() in ("yes", "1", "true") else "No"
    education = str(body.get("Education", "Graduate")).strip()
    if education not in ("Graduate", "Not Graduate"):
        education = "Graduate"
    property_area = str(body.get("Property_Area", "Urban")).strip()
    if property_area not in ("Urban", "Semiurban", "Rural"):
        property_area = "Urban"

    # 2) Same feature construction as training (train_and_save_model.py)
    loan_amount_log = np.log(max(loan_thousands, 1e-6))

    df = pd.DataFrame([{
        "Married": married,
        "Education": education,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_thousands,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit,
        "Property_Area": property_area,
    }])
    df["LoanAmount_log"] = loan_amount_log
    df = pd.get_dummies(df)
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    return df[feature_columns]


def explain_prediction(X):
    """Return factors that helped (positive) and reduced (negative) eligibility."""
    if not hasattr(model, "coef_") or model.coef_ is None:
        return {"factors_that_helped": [], "factors_that_reduced": []}
    coef = np.atleast_1d(model.coef_).flatten()
    if len(coef) != len(feature_columns):
        return {"factors_that_helped": [], "factors_that_reduced": []}
    # Ensure X has one row and columns match feature_columns
    if X.shape[0] == 0:
        return {"factors_that_helped": [], "factors_that_reduced": []}
    row = X.iloc[0]
    contributions = []
    for i, name in enumerate(feature_columns):
        if name not in row.index:
            continue
        val = row[name]
        try:
            val = float(val)
        except (TypeError, ValueError):
            continue
        contrib = float(coef[i]) * val
        if np.isnan(contrib):
            continue
        label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
        contributions.append({"label": label, "contribution": contrib})
    # Fallback: use positional indexing if name lookup gave nothing
    if not contributions and len(feature_columns) == len(coef):
        for i, name in enumerate(feature_columns):
            try:
                val = float(X.iloc[0].iloc[i])
            except (TypeError, ValueError, IndexError):
                continue
            contrib = float(coef[i]) * val
            if np.isnan(contrib):
                continue
            label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
            contributions.append({"label": label, "contribution": contrib})
    contributions.sort(key=lambda x: -abs(x["contribution"]))
    helped = [c["label"] for c in contributions if c["contribution"] > 0][:6]
    reduced = [c["label"] for c in contributions if c["contribution"] < 0][:6]
    return {"factors_that_helped": helped, "factors_that_reduced": reduced}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open(os.path.join(APP_DIR, "index.html"), "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/predict":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode("utf-8") if length else "{}"
            try:
                data = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Invalid JSON"})
                return
            if model is None:
                self._send_json(503, {"error": "Model not loaded. Run: python train_and_save_model.py"})
                return
            try:
                X = preprocess(data)
                pred = int(model.predict(X)[0])
                prob = None
                if hasattr(model, "predict_proba"):
                    prob = float(model.predict_proba(X)[0][1])  # P(Eligible)
                explanation = explain_prediction(X)
                # Report values actually used (loan in thousands for model consistency)
                loan_rupees = _to_float(data.get("LoanAmountRupees"), None)
                loan_th = _to_float(data.get("LoanAmount"), None)
                if loan_rupees is not None and loan_rupees > 0:
                    used_loan_display = "₹" + str(int(loan_rupees)) + " → " + str(round(loan_rupees / 1000.0, 1)) + " thousands"
                elif loan_th is not None and loan_th > 0:
                    used_loan_display = str(loan_th) + " thousands"
                else:
                    used_loan_display = str(data.get("LoanAmount", "—"))
                inputs_used = {
                    "ApplicantIncome": data.get("ApplicantIncome"),
                    "CoapplicantIncome": data.get("CoapplicantIncome"),
                    "LoanAmount": used_loan_display,
                    "Loan_Amount_Term": data.get("Loan_Amount_Term"),
                    "Credit_History": data.get("Credit_History"),
                    "Property_Area": data.get("Property_Area"),
                    "Married": data.get("Married"),
                    "Education": data.get("Education"),
                }
                self._send_json(200, {
                    "prediction": pred,
                    "Loan_Status": "Y" if pred == 1 else "N",
                    "probability": prob,
                    "explanation": explanation,
                    "inputs_used": inputs_used,
                })
            except Exception as e:
                self._send_json(400, {"error": str(e)})
        else:
            self.send_error(404)

    def _send_json(self, status, obj):
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def log_message(self, format, *args):
        print("[%s] %s" % (self.log_date_time_string(), format % args))


if __name__ == "__main__":
    if load_model():
        print("Model loaded. Open http://127.0.0.1:5000")
    else:
        print("Run first: python train_and_save_model.py")
    port = 5000
    try:
        server = HTTPServer(("127.0.0.1", port), Handler)
    except PermissionError:
        port = 8765
        server = HTTPServer(("127.0.0.1", port), Handler)
    print("Serving at http://127.0.0.1:%s" % port)
    server.serve_forever()
