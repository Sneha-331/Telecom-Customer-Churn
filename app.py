
import streamlit as st
import pandas as pd
import joblib
import os

# 1. Page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# 2. Dashboard title
st.title("📊 Telecom Customer Churn Prediction System")
st.write(
    "Predict customer churn risk using Machine Learning "
    "and prioritize customer retention."
)

# 3. Load saved models and preprocessing object
@st.cache_resource
def load_models():
    data = joblib.load("models/preprocessed_data.pkl")
    logistic_model = joblib.load("models/logistic_regression.pkl")
    rf_model = joblib.load("models/best_random_forest_compressed.pkl")

    return data["preprocessor"], logistic_model, rf_model

try:
    preprocessor, logistic_model, rf_model = load_models()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# 4. Model selection
st.sidebar.header("Model Settings")

model_choice = st.sidebar.selectbox(
    "Select Machine Learning Model",
    ["Logistic Regression", "Tuned Random Forest"]
)

model = (
    logistic_model
    if model_choice == "Logistic Regression"
    else rf_model
)

# 5. Customer input form
st.header("👤 Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with col2:
        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0
        )
        total = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True
    )

# 6. Prediction
if submitted:

    customer = pd.DataFrame([{
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
        "TotalCharges": total
    }])

    try:
        processed_customer = preprocessor.transform(customer)
        probability = model.predict_proba(processed_customer)[0][1]
        risk = probability * 100

        st.divider()
        st.header("📈 Churn Prediction Result")

        c1, c2 = st.columns(2)

        with c1:
            st.metric("Churn Probability", f"{risk:.2f}%")

        with c2:
            if risk < 30:
                st.success("LOW RISK")
                st.write("Customer is less likely to leave.")
            elif risk < 60:
                st.warning("MEDIUM RISK")
                st.write("Customer needs attention.")
            else:
                st.error("HIGH RISK")
                st.write("Customer has a high predicted churn risk.")

        st.progress(int(risk))

        st.subheader("💡 Retention Recommendation")

        if risk >= 60:
            st.write(
                "- Contact the customer immediately.\n"
                "- Offer a suitable retention discount.\n"
                "- Provide personalized support."
            )
        elif risk >= 30:
            st.write(
                "- Monitor customer satisfaction.\n"
                "- Share suitable service offers.\n"
                "- Follow up with the customer."
            )
        else:
            st.write(
                "- Maintain good customer service.\n"
                "- Continue regular engagement."
            )

        st.caption(
            "Risk categories use project-defined thresholds, "
            "not calibrated business cut-offs."
        )

    except Exception as e:
        st.error(f"Prediction error: {e}")

# 7. Footer
st.divider()
st.caption("CSE252 AI/ML Group Project | Telecom Customer Churn Prediction")
