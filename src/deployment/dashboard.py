import streamlit as st
import pandas as pd
from src.deployment.predict import predict_risk

st.title("Credit Risk Predictor - Banking Dashboard")

# Input form (expand with more fields)
duration = st.number_input("Loan Duration (months)", 1, 72, 12)
amount = st.number_input("Credit Amount", 1000, 20000, 5000)
age = st.number_input("Age", 18, 75, 35)

if st.button("Predict Risk"):
    input_data = {"Duration_in_month": duration, "Credit_amount": amount, "Age_in_years": age}
    result = predict_risk(input_data)
    st.success(f"Risk: {result['risk_prediction']} (PD: {result['probability_of_default']:.2%})")
    st.json(result["shap_explanation"])