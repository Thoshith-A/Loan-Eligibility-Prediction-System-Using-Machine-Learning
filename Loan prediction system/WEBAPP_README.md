# Loan Prediction Web App

A simple, classy frontend for the Loan Prediction model.

## Quick start

1. **Install dependencies** (if not already):
   ```bash
   pip install -r requirements.txt
   ```
   If you see a Jinja2 `escape` error when running the app, run: `pip install "jinja2<3.1"`.

2. **Train and save the model** (once):
   ```bash
   python train_and_save_model.py
   ```

3. **Run the web app**:
   ```bash
   python app.py
   ```

4. Open **http://127.0.0.1:5000** in your browser.

## What you get

- **Single-page form**: Marital status, education, incomes, loan amount & term, credit history, property area.
- **Instant prediction**: “Eligible” or “Not eligible” with a clear, animated result card.
- **Design**: Dark theme, Plus Jakarta Sans font, amber accent, glass-style card, subtle grid background.

## Files

- `app.py` — Flask server (serves `index.html` and `/api/predict`)
- `index.html` — Frontend (form + result UI)
- `train_and_save_model.py` — Trains model and saves to `model_artifacts/`
- `model_artifacts/model.joblib` — Trained model (created by step 2)
- `model_artifacts/feature_columns.json` — Feature list for preprocessing (created by step 2)
