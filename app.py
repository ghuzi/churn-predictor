import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Make sure relative file paths (like churn_schema.json) work on any machine
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

from validation import validate_customer, ChurnInputError

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

# ---------- Custom look: Forest Green -> Electric Purple gradient ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            180deg,
            #1F3D2B 0%,
            #1F3D2B 35%,
            #4C2F8F 75%,
            #8B5CF6 100%
        );
        background-attachment: fixed;
    }
    h1, h2, h3 {
        color: #E8DCC4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    # Load the saved pipeline (preprocessing + model) only once
    return joblib.load(BASE_DIR / "churn_model.joblib")


def risk_level(prob):
    # Same thresholds as in the notebook and the model card
    if prob >= 0.70:
        return "High"
    if prob >= 0.40:
        return "Medium"
    return "Low"


model = load_model()

# ---------- Header ----------
st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a telecom customer's details to see how likely they are to leave, "
    "and get suggested actions to keep them."
)

# ---------- Inputs ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer profile")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has a partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])

with col2:
    st.subheader("Account")
    tenure = st.slider("Tenure (months with the company)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly = st.number_input("Monthly charges", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
    total = round(tenure * monthly, 2)
    st.caption(f"Total charges are estimated as tenure x monthly charges: {total:,.2f}")

with col3:
    st.subheader("Services")
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "Yes":
        lines = st.selectbox("Multiple lines", ["No", "Yes"])
    else:
        lines = "No phone service"
        st.caption("Multiple lines: No phone service")

    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    addon_names = {
        "OnlineSecurity": "Online security",
        "OnlineBackup": "Online backup",
        "DeviceProtection": "Device protection",
        "TechSupport": "Tech support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming movies",
    }
    addons = {}
    if internet == "No":
        for key in addon_names:
            addons[key] = "No internet service"
        st.caption("Internet add-ons: No internet service")
    else:
        for key, label in addon_names.items():
            addons[key] = st.selectbox(label, ["No", "Yes"], key=key)

customer = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": lines,
    "InternetService": internet,
    **addons,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
}

# ---------- Prediction ----------
if st.button("Predict churn risk", type="primary"):
    try:
        clean, warnings = validate_customer(customer)
    except ChurnInputError as e:
        st.error("Please fix the following problems:\n\n" + str(e))
        st.stop()

    for w in warnings:
        st.warning(w)

    row = pd.DataFrame([clean])
    prob = float(model.predict_proba(row)[0, 1])
    level = risk_level(prob)

    st.divider()
    left, right = st.columns([1, 2])

    with left:
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))
        if level == "High":
            st.error("Risk level: HIGH")
        elif level == "Medium":
            st.warning("Risk level: MEDIUM")
        else:
            st.success("Risk level: LOW")
        st.caption("The score is best used to rank customers, not as an exact percentage.")

    with right:
        st.subheader("Suggested actions")
        actions = []
        if clean["Contract"] == "Month-to-month":
            actions.append("Offer a discount or free upgrade to move this customer to a one-year or two-year contract. "
                           "Month-to-month customers churn at 43%, compared with 3% on two-year contracts.")
        if clean["tenure"] <= 12:
            actions.append("Run an early check-in (welcome call after 1 to 3 months). "
                           "Customers in their first year churn at 47%.")
        if clean["PaymentMethod"] == "Electronic check":
            actions.append("Offer a small bill credit for switching to automatic payment. "
                           "Electronic check users churn at 45%.")
        if 70 <= clean["MonthlyCharges"] < 90:
            actions.append("Review the plan price. Customers paying 70 to 90 a month have the highest churn rate (38%).")

        if level == "Low" and not actions:
            st.write("No action needed. Keep monitoring this customer.")
        elif level == "Low":
            st.write("Risk is low. Nothing urgent, but these small steps can keep it that way:")
        if level != "Low" and not actions:
            st.write("Reach out personally and ask what would make the service better.")
        for a in actions:
            st.write("- " + a)

    # ---------- Chart: how the contract type changes this customer's risk ----------
    st.subheader("What if this customer switched contract?")
    options = ["Month-to-month", "One year", "Two year"]
    what_if = pd.DataFrame([{**clean, "Contract": c} for c in options])
    probs = model.predict_proba(what_if)[:, 1]
    chart_df = pd.DataFrame({"Churn probability": probs}, index=options)
    st.bar_chart(chart_df)
    st.caption("Same customer, only the contract type changed. All other details stay the same.")
    