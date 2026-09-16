import json
import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "churn_pipeline.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
IMPORTANCE_PATH = BASE_DIR / "model" / "feature_importance.csv"

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}

@st.cache_data
def load_importance():
    if IMPORTANCE_PATH.exists():
        return pd.read_csv(IMPORTANCE_PATH)
    return pd.DataFrame()

st.title("Telco Customer Churn Prediction")
st.caption("Interactive UI for the saved Decision Tree churn pipeline")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

metrics = load_metrics()
importance = load_importance()

with st.sidebar:
    st.header("Customer profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)

    st.header("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.header("Contract & billing")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.35, step=1.0, format="%.2f")
    total = st.number_input("Total Charges", min_value=0.0, value=844.20, step=10.0, format="%.2f")

customer = {
    "gender": gender,
    "SeniorCitizen": senior,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Prediction")
    if st.button("Predict Churn", type="primary", use_container_width=True):
        try:
            row = pd.DataFrame([customer])
            probability = float(model.predict_proba(row)[0, 1])
            prediction = "Yes" if probability >= 0.5 else "No"
            st.session_state["prediction"] = prediction
            st.session_state["probability"] = probability
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]
        if prediction == "Yes":
            st.warning(f"Predicted churn: **Yes**")
        else:
            st.success(f"Predicted churn: **No**")
        st.metric("Churn probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("Probability is produced by the saved pipeline. A 0.50 threshold is used for the Yes/No prediction.")

with col2:
    st.subheader("Model performance")
    final = metrics.get("Decision Tree - tuned + balanced", {})
    if final:
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{final.get('accuracy', 0):.1%}")
        b.metric("Precision", f"{final.get('precision', 0):.1%}")
        c.metric("Recall", f"{final.get('recall', 0):.1%}")
        d.metric("F1", f"{final.get('f1', 0):.1%}")
    else:
        st.info("Metrics artifact not found.")

st.divider()

with st.expander("Model interpretation", expanded=True):
    if not importance.empty:
        st.write("Top model features by importance")
        top = importance.head(10).copy()
        top = top.sort_values("importance", ascending=True).set_index("feature")
        st.bar_chart(top["importance"])
    else:
        st.info("Feature-importance artifact not found.")

with st.expander("Submitted sample payload"):
    st.json(customer)

st.caption("FastAPI REST API remains available separately through app.py at POST /predict.")
