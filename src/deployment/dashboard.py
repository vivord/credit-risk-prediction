import streamlit as st
import sys
import os
import pandas as pd
# from src.deployment.predict import predict_risk
# from .predict import predict_risk
from src.deployment.predict import predict_risk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("🏦 Credit Risk Predictor - Core Banking Integration")

st.sidebar.header("Loan Application Details")

# Input fields
duration = st.sidebar.number_input("Loan Duration (months)", min_value=1, max_value=72, value=12)
amount = st.sidebar.number_input("Credit Amount", min_value=1000.0, max_value=20000.0, value=5000.0)
age = st.sidebar.number_input("Applicant Age", min_value=18, max_value=75, value=35)

# More fields can be added here
status_account = st.sidebar.selectbox("Status of Existing Account", ["A11", "A12", "A13", "A14"], index=0)
credit_history = st.sidebar.selectbox("Credit History", ["A30", "A31", "A32", "A33", "A34"], index=2)
purpose = st.sidebar.selectbox("Purpose", ["A40", "A41", "A42", "A43", "A49"], index=3)

if st.sidebar.button("🔍 Predict Risk", type="primary"):
    input_data = {
        "Duration_in_month": duration,
        "Credit_amount": amount,
        "Age_in_years": age,
        "Status_of_existing_account": status_account,
        "Credit_history": credit_history,
        "Purpose": purpose,
        # Add more fields matching your preprocessor as you expand
    }

    with st.spinner("Predicting..."):
        result = predict_risk(input_data)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.success(f"**Risk Assessment:** {result['risk_prediction']}")
        st.metric("Probability of Default (PD)", f"{result['probability_of_default']:.2%}")

    with col2:
        st.subheader("SHAP Explanation")
        st.json(result["shap_explanation"])

    st.info("🔍 SHAP values show how each feature contributed to the prediction (positive = increases risk).")