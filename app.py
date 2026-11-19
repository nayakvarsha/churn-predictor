import streamlit as st, pandas as pd, joblib
from features import add_features                  # same feature code as training

art = joblib.load("model.pkl")                     # model and threshold
st.title("Customer Churn Predictor")

c1, c2 = st.columns(2)
tenure = c1.slider("Tenure (months)", 0, 72, 12)
monthly = c1.number_input("Monthly charges", 18.0, 120.0, 70.0)
contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
payment = c1.selectbox("Payment method", ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"])
paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
senior = c2.selectbox("Senior citizen", [0, 1])
partner = c2.selectbox("Partner", ["Yes", "No"])
dependents = c2.selectbox("Dependents", ["Yes", "No"])
security = c2.selectbox("Online security", ["Yes", "No"])
support = c2.selectbox("Tech support", ["Yes", "No"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "gender": "Male", "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": support, "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure,          # estimate, since users won't know it
    }])
    row = add_features(row)
    p = art["model"].predict_proba(row)[0, 1]
    st.metric("Churn probability", f"{p:.1%}")
    if p >= art["threshold"]:
        st.error("High risk: offer a retention deal")
    else:
        st.success("Low risk")
        