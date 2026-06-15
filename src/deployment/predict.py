import joblib
import pandas as pd
from src.utils.config import ARTIFACTS

model = joblib.load(ARTIFACTS / "xgboost_model.joblib")
preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
explainer = joblib.load(ARTIFACTS / "shap_explainer.joblib")


def predict_risk(input_data: dict):
    df = pd.DataFrame([input_data])
    X_prep = preprocessor.transform(df)
    proba = model.predict_proba(X_prep)[:, 1][0]
    prediction = 1 if proba > 0.5 else 0  # Adjust threshold

    # SHAP explanation
    shap_vals = explainer.shap_values(X_prep)[0]
    return {
        "probability_of_default": float(proba),
        "risk_prediction": "High Risk" if prediction == 1 else "Low Risk",
        "shap_explanation": dict(zip(df.columns, shap_vals.tolist()))
    }