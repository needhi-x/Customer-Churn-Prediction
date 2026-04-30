import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# ================= LOAD MODEL =================
model = pickle.load(open("models/churn_model.pkl", "rb"))

with open("models/columns.json") as f:
    columns = json.load(f)

# ================= PAGE SETUP =================
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Predict whether a customer will stay or leave the service")

# ================= SIDEBAR INPUTS =================
st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# ================= ENCODING =================
contract_map = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

payment_map = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer": 2,
    "Credit card": 3
}

# ================= INPUT DATAFRAME =================
input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

# Fill numeric values
input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = total_charges

# Fill encoded categorical values
input_data["Contract"] = contract_map[contract]
input_data["PaymentMethod"] = payment_map[payment]

# ================= PREDICTION =================
st.markdown("---")

if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Result")

        if prediction[0] == 1:
            st.error("⚠️ Customer WILL CHURN")
        else:
            st.success("✅ Customer WILL STAY")

    with col2:
        st.subheader("Model Insight")
        st.info("Prediction based on trained ML model (Random Forest / Logistic Regression)")