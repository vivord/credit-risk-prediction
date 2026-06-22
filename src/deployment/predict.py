import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import joblib
import pandas as pd
import shap
from src.utils.config import ARTIFACTS
from src.features.preprocessing import get_feature_lists

# Lazy load artifacts
_model = None
_preprocessor = None
_explainer = None


def load_artifacts():
    global _model, _preprocessor, _explainer
    if _model is None:
        _model = joblib.load(ARTIFACTS / "xgboost_model.joblib")
        _preprocessor = joblib.load(ARTIFACTS / "preprocessor.joblib")
        _explainer = joblib.load(ARTIFACTS / "shap_explainer.joblib")


def predict_risk(input_data: dict):
    load_artifacts()

    # Get expected columns
    cat_features, num_features = get_feature_lists()
    expected_cols = num_features + cat_features

    # Create DataFrame and fill missing columns with defaults
    df = pd.DataFrame([input_data])

    for col in expected_cols:
        if col not in df.columns:
            if col in num_features:
                df[col] = 0.0
            else:
                df[col] = 'unknown'

    # Reorder columns exactly as during training
    df = df[expected_cols]

    # Transform and predict
    X_prep = _preprocessor.transform(df)
    proba = _model.predict_proba(X_prep)[:, 1][0]
    prediction = 1 if proba > 0.5 else 0

    # SHAP values
    shap_vals = _explainer.shap_values(X_prep)[0]

    return {
        "probability_of_default": round(float(proba), 4),
        "risk_prediction": "High Risk (Default Likely)" if prediction == 1 else "Low Risk (Good)",
        "shap_explanation": {col: round(float(val), 4) for col, val in zip(df.columns, shap_vals)}
    }