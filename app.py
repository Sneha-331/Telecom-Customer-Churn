


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# 2. LIGHT MODE CUSTOM CSS
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #F8FAFC;
    color: #1E293B;
}

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

[data-testid="stHeader"] {
    background: #F8FAFC;
}

h1, h2, h3 {
    color: #172554 !important;
}

p, label, span {
    color: #334155;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetricValue"] {
    color: #4F46E5;
    font-weight: 700;
}

div.stButton > button {
    background-color: #4F46E5;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
}

div.stButton > button:hover {
    background-color: #3730A3;
    color: white;
}

[data-testid="stDataFrame"] {
    background: white;
}

hr {
    border-color: #E2E8F0;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

MODEL_DIR = BASE_DIR / "models"

LR_PATH = MODEL_DIR / "logistic_regression.pkl"

RF_PATH = MODEL_DIR / "best_random_forest_compressed.pkl"

PREPROCESSED_PATH = MODEL_DIR / "preprocessed_data.pkl"


# =====================================================
# 4. LOAD DATASET AND MODELS
# =====================================================

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges into numeric values
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        df["TotalCharges"] = df["TotalCharges"].fillna(
            df["TotalCharges"].median()
        )

    return df


@st.cache_resource
def load_models():

    models = {}

    if LR_PATH.exists():
        models["Logistic Regression"] = joblib.load(LR_PATH)

    if RF_PATH.exists():
        models["Tuned Random Forest"] = joblib.load(RF_PATH)

    return models


@st.cache_data
def load_preprocessed_data():

    if PREPROCESSED_PATH.exists():
        return joblib.load(PREPROCESSED_PATH)

    return None


try:
    df = load_dataset()
    models = load_models()
    preprocessed_data = load_preprocessed_data()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


if not models:
    st.error("No trained models found in the models folder.")
    st.stop()


if "Churn" not in df.columns:
    st.error("The dataset must contain a Churn column.")
    st.stop()


# =====================================================
# 5. DATA PREPARATION
# =====================================================

df = df.copy()

df["Churn"] = df["Churn"].astype(str).str.strip()

df["Churn_Status"] = df["Churn"].map({
    "Yes": "Churned",
    "No": "Retained"
}).fillna(df["Churn"])

total_customers = len(df)

churned_customers = (df["Churn"] == "Yes").sum()

retained_customers = (df["Churn"] == "No").sum()

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers else 0
)

retention_rate = (
    retained_customers / total_customers * 100
    if total_customers else 0
)


# =====================================================
# 6. MODEL INPUT PREPARATION
# =====================================================

def get_model_features(model):

    """
    Use the exact feature order saved with the model.
    Fall back to preprocessed data columns when available.
    """

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if isinstance(preprocessed_data, pd.DataFrame):
        columns = list(preprocessed_data.columns)

        columns = [
            col for col in columns
            if col not in [
                "Churn", "customerID", "Churn_Status"
            ]
        ]

        return columns

    return None


def prepare_features(data, model):

    """
    Supports models trained on one-hot encoded features
    or models trained directly on raw feature columns.
    """

    features = data.copy()

    features = features.drop(
        columns=["customerID", "Churn", "Churn_Status"],
        errors="ignore"
    )

    features = features.replace({
        "Yes": 1,
        "No": 0
    })

    expected_features = get_model_features(model)

    if expected_features is None:
        features = pd.get_dummies(features, drop_first=True)
    else:
        raw_columns = set(features.columns)

        # Check whether the model expects raw columns
        if set(expected_features).issubset(raw_columns):
            features = features[expected_features]

        else:
            features = pd.get_dummies(
                features,
                drop_first=True
            )

            features = features.reindex(
                columns=expected_features,
                fill_value=0
            )

    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)

    return features


# =====================================================
# 7. PREDICTION FUNCTIONS
# =====================================================

def predict_churn(data, model):

    features = prepare_features(data, model)

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(features)

        classes = list(model.classes_)

        if "Yes" in classes:
            churn_index = classes.index("Yes")
        elif 1 in classes:
            churn_index = classes.index(1)
        else:
            churn_index = len(classes) - 1

        return probabilities[:, churn_index]

    if hasattr(model, "decision_function"):

        scores = model.decision_function(features)

        return 1 / (1 + np.exp(-scores))

    raise ValueError(
        "The selected model does not support probability prediction."
    )


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

st.sidebar.markdown("# 📊 Telecom Analytics")

st.sidebar.caption(
    "Customer Churn Risk Classification System"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Customer Analysis",
        "Churn Prediction",
        "Retention Recommendations",
        "Model Performance",
        "About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")

st.sidebar.info(
    """
    **Machine Learning Models**

    • Logistic Regression

    • Tuned Random Forest

    **Techniques**

    • EDA

    • SMOTE

    • Grid Search

    • Classification
    """
)

st.sidebar.caption("Telecom Customer Churn Project")


# =====================================================
# 9. DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.title("📊 Telecom Customer Churn Dashboard")

    st.write(
        "Monitor customer behavior, churn patterns "
        "and retention performance."
    )

    st.divider()

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    col3.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # CHURN DISTRIBUTION

    st.subheader("Customer Churn Overview")

    col1, col2 = st.columns(2)

    churn_counts = df["Churn_Status"].value_counts().reset_index()

    churn_counts.columns = ["Status", "Customers"]

    with col1:

        fig = px.pie(
            churn_counts,
            names="Status",
            values="Customers",
            hole=0.55,
            title="Churn vs Retained Customers",
            color_discrete_sequence=[
                "#EF4444", "#4F46E5"
            ]
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # CONTRACT ANALYSIS

    with col2:

        if "Contract" in df.columns:

            contract_data = (
                df.groupby(["Contract", "Churn_Status"])
                .size()
                .reset_index(name="Customers")
            )

            fig = px.bar(
                contract_data,
                x="Contract",
                y="Customers",
                color="Churn_Status",
                barmode="group",
                title="Churn by Contract Type",
                color_discrete_sequence=[
                    "#EF4444", "#4F46E5"
                ]
            )

            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.divider()

    # MONTHLY CHARGES

    col1, col2 = st.columns(2)

    with col1:

        if "MonthlyCharges" in df.columns:

            fig = px.histogram(
                df,
                x="MonthlyCharges",
                color="Churn_Status",
                nbins=30,
                title="Monthly Charges Distribution",
                color_discrete_sequence=[
                    "#EF4444", "#4F46E5"
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if "tenure" in df.columns:

            fig = px.box(
                df,
                x="Churn_Status",
                y="tenure",
                color="Churn_Status",
                title="Customer Tenure vs Churn",
                color_discrete_sequence=[
                    "#EF4444", "#4F46E5"
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # HIGH RISK CUSTOMER TABLE

    st.divider()

    st.subheader("Customer Churn Risk Analysis")

    selected_model_name = st.selectbox(
        "Select prediction model",
        list(models.keys()),
        key="dashboard_model"
    )

    selected_model = models[selected_model_name]

    try:

        probabilities = predict_churn(
            df,
            selected_model
        )

        risk_df = df.copy()

        risk_df["Churn_Probability"] = probabilities * 100

        risk_df["Risk_Level"] = np.select(
            [
                risk_df["Churn_Probability"] >= 70,
                risk_df["Churn_Probability"] >= 40
            ],
            [
                "High Risk",
                "Medium Risk"
            ],
            default="Low Risk"
        )

        high_risk = risk_df[
            risk_df["Risk_Level"] == "High Risk"
        ]

        st.metric(
            "Predicted High-Risk Customers",
            f"{len(high_risk):,}"
        )

        display_columns = [
            col for col in [
                "customerID",
                "tenure",
                "Contract",
                "MonthlyCharges",
                "Churn_Probability",
                "Risk_Level"
            ]
            if col in risk_df.columns
        ]

        st.dataframe(
            risk_df.sort_values(
                "Churn_Probability",
                ascending=False
            )[display_columns].head(15),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.warning(
            f"Risk analysis unavailable: {e}"
        )


# =====================================================
# 10. CUSTOMER ANALYSIS PAGE
# =====================================================

elif page == "Customer Analysis":

    st.title("👥 Customer Analysis")

    st.write(
        "Explore customer demographics, services "
        "and subscription behavior."
    )

    st.divider()

    # FILTERS

    filtered_df = df.copy()

    col1, col2 = st.columns(2)

    with col1:

        if "gender" in df.columns:

            gender = st.selectbox(
                "Select Gender",
                ["All"] + sorted(
                    df["gender"].dropna().unique().tolist()
                )
            )

            if gender != "All":
                filtered_df = filtered_df[
                    filtered_df["gender"] == gender
                ]

    with col2:

        if "Contract" in df.columns:

            contract = st.selectbox(
                "Select Contract",
                ["All"] + sorted(
                    df["Contract"].dropna().unique().tolist()
                )
            )

            if contract != "All":
                filtered_df = filtered_df[
                    filtered_df["Contract"] == contract
                ]

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Filtered Customers",
        len(filtered_df)
    )

    col2.metric(
        "Churned",
        int((filtered_df["Churn"] == "Yes").sum())
    )

    filtered_rate = (
        (filtered_df["Churn"] == "Yes").mean() * 100
        if len(filtered_df) else 0
    )

    col3.metric(
        "Filtered Churn Rate",
        f"{filtered_rate:.2f}%"
    )

    st.divider()

    # INTERNET SERVICE

    col1, col2 = st.columns(2)

    with col1:

        if "InternetService" in filtered_df.columns:

            service_data = (
                filtered_df.groupby(
                    ["InternetService", "Churn_Status"]
                )
                .size()
                .reset_index(name="Customers")
            )

            fig = px.bar(
                service_data,
                x="InternetService",
                y="Customers",
                color="Churn_Status",
                barmode="group",
                title="Internet Service Analysis"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        if "PaymentMethod" in filtered_df.columns:

            payment_data = (
                filtered_df.groupby(
                    ["PaymentMethod", "Churn_Status"]
                )
                .size()
                .reset_index(name="Customers")
            )

            fig = px.bar(
                payment_data,
                x="PaymentMethod",
                y="Customers",
                color="Churn_Status",
                title="Payment Method Analysis"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    st.subheader("Customer Records")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# 11. CHURN PREDICTION PAGE
# =====================================================

elif page == "Churn Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Enter customer details to estimate "
        "the probability of customer churn."
    )

    st.divider()

    selected_model_name = st.selectbox(
        "Choose ML Model",
        list(models.keys()),
        key="prediction_model"
    )

    selected_model = models[selected_model_name]

    st.subheader("Customer Information")

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=72,
                value=12
            )

        with col2:

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                ["No phone service", "No", "Yes"]
            )

            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            online_security = st.selectbox(
                "Online Security",
                ["Yes", "No", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["Yes", "No", "No internet service"]
            )

        with col3:

            device_protection = st.selectbox(
                "Device Protection",
                ["Yes", "No", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["Yes", "No", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["Yes", "No", "No internet service"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

        with col5:

            payment = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col6:

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                max_value=500.0,
                value=70.0
            )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(tenure * monthly_charges)
        )

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            use_container_width=True
        )

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
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])

        try:

            probability = predict_churn(
                customer,
                selected_model
            )[0] * 100

            st.divider()

            st.subheader("Prediction Result")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Churn Probability",
                    f"{probability:.2f}%"
                )

            with col2:

                if probability >= 70:

                    st.error("HIGH CHURN RISK")

                elif probability >= 40:

                    st.warning("MEDIUM CHURN RISK")

                else:

                    st.success("LOW CHURN RISK")

            fig = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability,
                    title={"text": "Churn Probability"},
                    number={"suffix": "%"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "steps": [
                            {
                                "range": [0, 40],
                                "color": "#DCFCE7"
                            },
                            {
                                "range": [40, 70],
                                "color": "#FEF3C7"
                            },
                            {
                                "range": [70, 100],
                                "color": "#FEE2E2"
                            }
                        ],
                        "bar": {"color": "#4F46E5"}
                    }
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            if probability >= 70:

                st.info(
                    "Recommendation: Contact this customer "
                    "with a personalized retention offer, "
                    "review their service experience and "
                    "consider a suitable contract incentive."
                )

            elif probability >= 40:

                st.info(
                    "Recommendation: Monitor this customer "
                    "and offer additional support or "
                    "a suitable service package."
                )

            else:

                st.success(
                    "Recommendation: Maintain service quality "
                    "and continue regular customer engagement."
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )

            st.caption(
                "Check that the model's saved preprocessing "
                "and input feature format match this application."
            )


# =====================================================
# 12. RETENTION RECOMMENDATIONS PAGE
# =====================================================

elif page == "Retention Recommendations":

    st.title("🎯 Retention Recommendations")

    st.write(
        "Identify customer groups that may need "
        "additional attention."
    )

    st.divider()

    if "Contract" in df.columns:

        st.subheader("Contract-Based Retention")

        contract_churn = (
            df.groupby("Contract")["Churn"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index(name="Churn Rate")
        )

        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            text_auto=".2f",
            title="Churn Rate by Contract Type",
            color="Churn Rate",
            color_continuous_scale="Reds"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Recommended Retention Actions")

    recommendations = [
        (
            "Month-to-month customers",
            "Offer annual contract discounts or loyalty benefits."
        ),
        (
            "New customers",
            "Provide onboarding assistance and early engagement."
        ),
        (
            "High monthly charges",
            "Review pricing and offer suitable packages."
        ),
        (
            "Customers without support services",
            "Promote technical support and service assistance."
        ),
        (
            "Electronic check users",
            "Encourage convenient automatic payment options."
        )
    ]

    for title, description in recommendations:

        with st.container(border=True):

            st.markdown(f"### {title}")

            st.write(description)


# =====================================================
# 13. MODEL PERFORMANCE PAGE
# =====================================================

elif page == "Model Performance":

    st.title("📈 Machine Learning Model Performance")

    st.write(
        "Compare Logistic Regression, Random Forest "
        "and Tuned Random Forest."
    )

    st.divider()

    # Previously evaluated test-set metrics.
    metrics = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Tuned Random Forest"
        ],
        "Accuracy": [
            73.74, 77.36, 77.00
        ],
        "Precision": [
            50.34, 57.75, 57.02
        ],
        "Recall": [
            79.41, 54.81, 54.28
        ],
        "F1 Score": [
            61.62, 56.24, 55.62
        ],
        "ROC-AUC": [
            84.01, 81.50, 81.74
        ]
    })

    st.subheader("Model Comparison Table")

    st.dataframe(
        metrics.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Performance Comparison")

    selected_metric = st.selectbox(
        "Choose Evaluation Metric",
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ]
    )

    fig = px.bar(
        metrics,
        x="Model",
        y=selected_metric,
        text=selected_metric,
        color="Model",
        title=f"{selected_metric} Comparison",
        color_discrete_sequence=[
            "#4F46E5",
            "#0EA5E9",
            "#8B5CF6"
        ]
    )

    fig.update_layout(
        yaxis_title=f"{selected_metric} (%)",
        showlegend=False,
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        """
        **Project Evaluation:**

        Logistic Regression achieved the highest recall,
        F1-score and ROC-AUC among the three evaluated models.

        Random Forest achieved the highest accuracy
        and precision.

        Model selection should consider the business cost
        of missing customers who are likely to churn.
        """
    )

    # FEATURE IMPORTANCE

    st.divider()

    st.subheader("Random Forest Feature Importance")

    rf_model = models.get("Tuned Random Forest")

    if rf_model is not None:

        importance_model = rf_model

        if hasattr(importance_model, "named_steps"):
            importance_model = list(
                importance_model.named_steps.values()
            )[-1]

        if hasattr(importance_model, "feature_importances_"):

            feature_names = get_model_features(rf_model)

            importances = importance_model.feature_importances_

            if feature_names and len(feature_names) == len(importances):

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                })

            else:

                importance_df = pd.DataFrame({
                    "Feature": [
                        f"Feature {i+1}"
                        for i in range(len(importances))
                    ],
                    "Importance": importances
                })

            importance_df = importance_df.sort_values(
                "Importance",
                ascending=False
            ).head(15)

            fig = px.bar(
                importance_df.sort_values(
                    "Importance",
                    ascending=True
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Important Features",
                color="Importance",
                color_continuous_scale="Blues"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Feature importance is not directly available "
                "for this saved model."
            )

    else:

        st.warning("Tuned Random Forest model not found.")


# =====================================================
# 14. ABOUT PROJECT PAGE
# =====================================================

elif page == "About Project":

    st.title("ℹ️ About the Project")

    st.write(
        "Telecom Customer Churn Risk Classification "
        "and Retention Prioritization System"
    )

    st.divider()

    st.subheader("Project Objective")

    st.write(
        """
        The main objective is to identify customers who
        are likely to discontinue telecom services.

        The system uses machine learning models to estimate
        churn risk and help telecom companies take
        preventive retention actions.
        """
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", f"{len(df):,}")

    col2.metric("Total Features", f"{df.shape[1]}")

    col3.metric("Target Variable", "Churn")

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - SMOTE (Imbalanced-learn)
    - Logistic Regression
    - Random Forest
    - Grid Search
    - Streamlit
    - Plotly
    """)

    st.subheader("Machine Learning Workflow")

    st.markdown("""
    1. Data Collection
    2. Data Cleaning
    3. Exploratory Data Analysis
    4. Data Preprocessing and Encoding
    5. Train-Test Split
    6. SMOTE for Training Data
    7. Logistic Regression
    8. Random Forest
    9. Hyperparameter Tuning using Grid Search
    10. Model Evaluation
    11. Customer Churn Prediction
    12. Retention Prioritization
    """)

    st.divider()

    st.caption(
        "Telecom Customer Churn Risk Classification "
        "and Retention Prioritization System | 2026"
    )


# =====================================================
# 15. FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748B;">
        Telecom Customer Churn Analytics Dashboard
        <br>
        Built with Python, Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
