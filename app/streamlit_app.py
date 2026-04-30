import streamlit as st
import requests

st.title("Customers Churn Prediction System")

# --- Already provided by you ---
SeniorCitizen = st.radio("SeniorCitizen", [0, 1])
Tenure = st.number_input("Tenure", min_value=0, max_value=72)
Gender = st.selectbox("Gender", ["Male", "Female"])

# --- Remaining input fields ---
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])

PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])

InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
PaymentMethod = st.selectbox("PaymentMethod", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, max_value=200.0, step=0.05)
TotalCharges = st.number_input("TotalCharges", min_value=0.0, max_value=10000.0, step=0.05)


if st.button("Predict"):
    payload = { "gender": Gender,
                "SeniorCitizen": SeniorCitizen,
                "Partner": Partner, 
                "Dependents": Dependents,
                "tenure": Tenure, 
                "PhoneService": PhoneService, 
                "MultipleLines": MultipleLines,          
                "InternetService": InternetService, 
                "OnlineSecurity": OnlineSecurity, 
                "OnlineBackup": OnlineBackup, 
                "DeviceProtection": DeviceProtection, 
                "TechSupport": TechSupport,                                 
                "StreamingTV": StreamingTV, 
                "StreamingMovies": StreamingMovies, 
                "Contract": Contract,                 
                "PaperlessBilling": PaperlessBilling, 
                "PaymentMethod": PaymentMethod, 
                "MonthlyCharges": MonthlyCharges, 
                "TotalCharges": TotalCharges }
    response = requests.post("http://127.0.0.1:8000/predict",json=payload)
 
    results =  response.json()
    st.write(results["churn_prediction"])
    st.write(results["churn_probability"])

    if results["churn_prediction"] == 1:
        st.error(f"⚠️ Customer is likely to churn! {results["churn_probability"]*100:.2f}%")
    else:
        st.success(f"✅ Customer is likely to stay. {results["churn_probability"]*100:.2f}%")


