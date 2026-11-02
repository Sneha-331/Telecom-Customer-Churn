
"""import streamlit as st
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

# 3. Load saved models and preprocessing object

@st.cache_resource
def load_file(path):
    try:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

        return joblib.load(path)

    except Exception as e:
        raise RuntimeError(
            f"Problem loading {path}: {type(e).__name__}: {e}"
        ) from e


@st.cache_resource
def load_models():
    data = load_file("models/preprocessed_data.pkl")
    logistic_model = load_file("models/logistic_regression.pkl")
    rf_model = load_file("models/best_random_forest_compressed.pkl")

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
st.caption("CSE252 AI/ML Group Project | Telecom Customer Churn Prediction")"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111c30;
        border-right: 1px solid #263750;
    }

    h1, h2, h3 {
        color: #e2e8f0;
    }

    .hero {
        background: linear-gradient(120deg, #142c4d, #123b4b);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #28516b;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #ffffff;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 16px;
    }

    .info-card {
        background: #142238;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #293e5a;
        min-height: 120px;
    }

    .info-card h3 {
        color: #38bdf8;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .info-card p {
        color: #f8fafc;
        font-size: 23px;
        font-weight: 700;
        margin: 0;
    }

    .small-note {
        color: #94a3b8;
        font-size: 13px;
    }

    div[data-testid="stMetric"] {
        background-color: #142238;
        border: 1px solid #293e5a;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        background-color: #111c30;
        padding: 20px;
        border: 1px solid #293e5a;
        border-radius: 12px;
    }

    .result-card {
        background: #142238;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #293e5a;
        text-align: center;
    }

    .result-number {
        font-size: 38px;
        font-weight: 800;
        color: #38bdf8;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 20px;
        font-size: 13px;
    }

    hr {
        border-color: #293e5a;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. FILE PATHS
# =========================================================

DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"

PREPROCESSOR_PATH = "models/preprocessed_data.pkl"
LOGISTIC_PATH = "models/logistic_regression.pkl"
RF_PATH = "models/best_random_forest_compressed.pkl"


# =========================================================
# 4. LOAD MACHINE LEARNING MODELS
# =========================================================

@st.cache_resource
def load_models():

    required_files = [
        PREPROCESSOR_PATH,
        LOGISTIC_PATH,
        RF_PATH
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Required model file not found: {file}"
            )

    data = joblib.load(PREPROCESSOR_PATH)

    logistic_model = joblib.load(LOGISTIC_PATH)
    rf_model = joblib.load(RF_PATH)

    return (
        data["preprocessor"],
        logistic_model,
        rf_model
    )


try:
    preprocessor, logistic_model, rf_model = load_models()

except Exception as e:
    st.error(f"Unable to load machine learning models: {e}")
    st.stop()


# =========================================================
# 5. LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df


df = load_dataset()


# =========================================================
# 6. SIDEBAR NAVIGATION
# =========================================================

st.sidebar.markdown("## 📡 Churn Intelligence")
st.sidebar.caption("Machine Learning Analytics System")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Overview",
        "📊 Dataset Analysis",
        "🤖 Model Performance",
        "🎯 Predict Churn",
        "📘 About Project"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")
st.sidebar.write("**Course:** CSE252")
st.sidebar.write("**Domain:** Artificial Intelligence & ML")
st.sidebar.write("**Dataset:** IBM Telco Customer Churn")

st.sidebar.divider()
st.sidebar.caption("Developed using Python and Streamlit")


# =========================================================
# 7. HELPER FUNCTIONS
# =========================================================

def page_header(title, description):

    st.markdown(
        f"""
        <div class="hero">
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def make_chart():

    fig, ax = plt.subplots(figsize=(9, 4))

    fig.patch.set_facecolor("#142238")
    ax.set_facecolor("#142238")

    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#ffffff")

    for spine in ax.spines.values():
        spine.set_color("#40516a")

    return fig, ax


def risk_category(probability):

    if probability < 30:
        return "LOW RISK"

    elif probability < 60:
        return "MEDIUM RISK"

    return "HIGH RISK"


# =========================================================
# 8. OVERVIEW PAGE
# =========================================================

if page == "🏠 Overview":

    page_header(
        "Telecom Customer Churn Intelligence",
        "An AI-powered system for understanding customer behaviour "
        "and predicting customer churn risk."
    )

    st.subheader("📌 Project Overview")

    st.write(
        """
        Customer churn occurs when a customer stops using the services
        of a telecom company.

        This project applies Supervised Machine Learning classification
        algorithms to historical customer data to estimate the probability
        of churn and help prioritize customer-retention activities.
        """
    )

    st.divider()

    st.subheader("📊 Dataset at a Glance")

    if df is not None:

        total_customers = len(df)

        churn_count = (df["Churn"] == "Yes").sum()
        retained_count = (df["Churn"] == "No").sum()

        churn_rate = churn_count / total_customers * 100

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Customers", f"{total_customers:,}")
        c2.metric("Churned Customers", f"{churn_count:,}")
        c3.metric("Retained Customers", f"{retained_count:,}")
        c4.metric("Churn Rate", f"{churn_rate:.2f}%")

        st.divider()

        left, right = st.columns(2)

        with left:

            st.markdown("### Customer Churn Distribution")

            churn_data = df["Churn"].value_counts()

            fig, ax = make_chart()

            ax.pie(
                churn_data.values,
                labels=churn_data.index,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": "white"},
                wedgeprops={"edgecolor": "#142238"}
            )

            ax.set_title("Churn vs Retained Customers")

            st.pyplot(fig)
            plt.close(fig)

        with right:

            st.markdown("### Customers by Contract")

            contract_data = pd.crosstab(
                df["Contract"],
                df["Churn"]
            )

            fig, ax = make_chart()

            contract_data.plot(
                kind="bar",
                ax=ax
            )

            ax.set_title("Churn by Contract Type")
            ax.set_xlabel("Contract")
            ax.set_ylabel("Number of Customers")
            ax.tick_params(axis="x", rotation=0)

            ax.legend(
                facecolor="#142238",
                labelcolor="white"
            )

            st.pyplot(fig)
            plt.close(fig)

    else:

        st.warning(
            "Dataset not found. Check the dataset folder and filename."
        )

    st.divider()

    st.subheader("🧠 Machine Learning Models")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="info-card">
                <h3>MODEL 01</h3>
                <p>Logistic Regression</p>
                <span class="small-note">
                Supervised classification algorithm
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="info-card">
                <h3>MODEL 02</h3>
                <p>Tuned Random Forest</p>
                <span class="small-note">
                Ensemble classification algorithm
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 9. DATASET ANALYSIS PAGE
# =========================================================

elif page == "📊 Dataset Analysis":

    page_header(
        "Exploratory Data Analysis",
        "Explore customer behaviour, churn patterns and important "
        "telecom service characteristics."
    )

    if df is None:

        st.error("Dataset not found. Please check the dataset folder.")

    else:

        st.subheader("Dataset Preview")

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

        st.write(
            f"Dataset Shape: **{df.shape[0]} rows × {df.shape[1]} columns**"
        )

        st.divider()

        st.subheader("1. Churn Distribution")

        churn_counts = df["Churn"].value_counts()

        fig, ax = make_chart()

        churn_counts.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Number of Churned and Retained Customers")
        ax.set_xlabel("Customer Status")
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis="x", rotation=0)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "This graph shows the distribution of customers who left "
            "the telecom service and those who stayed."
        )

        st.divider()

        st.subheader("2. Churn According to Contract Type")

        contract_data = pd.crosstab(
            df["Contract"],
            df["Churn"]
        )

        fig, ax = make_chart()

        contract_data.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Contract Type vs Customer Churn")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis="x", rotation=0)

        ax.legend(
            facecolor="#142238",
            labelcolor="white"
        )

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "This graph helps compare churn behaviour among different "
            "contract types."
        )

        st.divider()

        st.subheader("3. Customer Tenure Analysis")

        fig, ax = make_chart()

        ax.hist(
            df["tenure"],
            bins=20,
            edgecolor="white"
        )

        ax.set_title("Distribution of Customer Tenure")
        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Number of Customers")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Tenure represents the number of months a customer has "
            "used the telecom service."
        )

        st.divider()

        st.subheader("4. Monthly Charges Analysis")

        fig, ax = make_chart()

        ax.hist(
            df["MonthlyCharges"],
            bins=25,
            edgecolor="white"
        )

        ax.set_title("Monthly Charges Distribution")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Number of Customers")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "This graph shows how monthly charges are distributed "
            "across customers."
        )

        st.divider()

        st.subheader("5. Churn by Internet Service")

        internet_data = pd.crosstab(
            df["InternetService"],
            df["Churn"]
        )

        fig, ax = make_chart()

        internet_data.plot(
            kind="bar",
            ax=ax
        )

        ax.set_title("Internet Service vs Churn")
        ax.set_xlabel("Internet Service")
        ax.set_ylabel("Number of Customers")
        ax.tick_params(axis="x", rotation=0)

        ax.legend(
            facecolor="#142238",
            labelcolor="white"
        )

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "This comparison shows churn counts for different "
            "internet service categories."
        )

        st.divider()

        st.subheader("6. Dataset Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.markdown("**Missing Values**")

            missing = df.isnull().sum()

            st.dataframe(
                missing[missing > 0].rename("Missing Values"),
                use_container_width=True
            )

        with info_col2:

            st.markdown("**Numerical Summary**")

            st.dataframe(
                df.describe(),
                use_container_width=True
            )


# =========================================================
# 10. MODEL PERFORMANCE PAGE
# =========================================================

elif page == "🤖 Model Performance":

    page_header(
        "Machine Learning Model Evaluation",
        "Compare classification algorithms and understand "
        "how well they predict customer churn."
    )

    st.subheader("Model Performance Comparison")

    # Actual results from the completed project

    results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "Tuned Random Forest"
        ],

        "Accuracy": [73.74, 77.36, 77.00],
        "Precision": [50.34, 57.75, 57.02],
        "Recall": [79.41, 54.81, 54.28],
        "F1-Score": [61.62, 56.24, 55.62],
        "ROC-AUC": [84.01, 81.50, 81.74]

    })

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }),
        use_container_width=True
    )

    st.divider()

    st.subheader("1. Model Accuracy Comparison")

    fig, ax = make_chart()

    ax.bar(
        results["Model"],
        results["Accuracy"]
    )

    ax.set_title("Accuracy Comparison")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=10)

    st.pyplot(fig)
    plt.close(fig)

    st.divider()

    st.subheader("2. Evaluation Metrics Comparison")

    selected_metrics = st.multiselect(
        "Select metrics",
        ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        default=["Accuracy", "Precision", "Recall"]
    )

    if selected_metrics:

        fig, ax = make_chart()

        x = np.arange(len(results))
        width = 0.8 / len(selected_metrics)

        for i, metric in enumerate(selected_metrics):

            ax.bar(
                x + i * width,
                results[metric],
                width,
                label=metric
            )

        ax.set_xticks(
            x + width * (len(selected_metrics) - 1) / 2
        )

        ax.set_xticklabels(
            results["Model"],
            rotation=10
        )

        ax.set_ylim(0, 100)
        ax.set_ylabel("Score (%)")
        ax.set_title("Performance Metrics")

        ax.legend(
            facecolor="#142238",
            labelcolor="white"
        )

        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    st.subheader("3. Important Features")

    feature_names = [
        "Tenure",
        "TotalCharges",
        "MonthlyCharges",
        "Month-to-month Contract",
        "OnlineSecurity_No"
    ]

    feature_values = [
        0.111710,
        0.108634,
        0.096639,
        0.077914,
        0.055608
    ]

    feature_data = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_values
    }).sort_values("Importance")

    fig, ax = make_chart()

    ax.barh(
        feature_data["Feature"],
        feature_data["Importance"]
    )

    ax.set_title("Top Random Forest Feature Importances")
    ax.set_xlabel("Importance")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "These are the recorded top five feature-importance values "
        "from the Random Forest model."
    )

    st.divider()

    st.subheader("4. Key Findings")

    c1, c2, c3 = st.columns(3)

    c1.metric("Highest Accuracy", "77.36%", "Random Forest")
    c2.metric("Highest Recall", "79.41%", "Logistic Regression")
    c3.metric("Highest ROC-AUC", "84.01%", "Logistic Regression")

    st.info(
        """
        **Interpretation:**

        Logistic Regression achieved the highest recall. This means it
        identified a larger proportion of the actual churn customers
        in the test set.

        Recall is important in customer retention because failing to
        identify a customer who may leave can mean losing an opportunity
        to take preventive action.
        """
    )

    # Display saved graphs if available

    st.divider()

    st.subheader("Saved Evaluation Graphs")

    graph_paths = [
        "models/roc_curve_comparison.png",
        "models/logistic_regression_confusion_matrix.png",
        "models/random_forest_confusion_matrix.png",
        "models/tuned_random_forest_confusion_matrix.png",
        "models/feature_importance.png"
    ]

    for graph_path in graph_paths:

        if os.path.exists(graph_path):

            st.markdown(
                f"**{os.path.basename(graph_path).replace('_', ' ').replace('.png', '').title()}**"
            )

            st.image(
                graph_path,
                use_container_width=True
            )


# =========================================================
# 11. CUSTOMER CHURN PREDICTION PAGE
# =========================================================

elif page == "🎯 Predict Churn":

    page_header(
        "Customer Churn Prediction",
        "Enter customer details and use your trained machine learning "
        "model to estimate churn probability."
    )

    st.subheader("⚙️ Select Machine Learning Model")

    model_choice = st.selectbox(
        "Choose prediction model",
        [
            "Logistic Regression",
            "Tuned Random Forest"
        ]
    )

    model = (
        logistic_model
        if model_choice == "Logistic Regression"
        else rf_model
    )

    st.caption(f"Selected Model: {model_choice}")

    st.divider()

    st.subheader("👤 Customer Information")

    with st.form("customer_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown("#### Personal Details")

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            tenure = st.slider(
                "Tenure (Months)",
                0,
                72,
                12
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

            multiple = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:

            st.markdown("#### Internet & Services")

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

            st.markdown("#### Billing & Contract")

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
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
            "🚀 Predict Customer Churn",
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

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

            probability = model.predict_proba(
                processed_customer
            )[0][1]

            risk = probability * 100

            risk_label = risk_category(risk)

            st.divider()

            st.subheader("📈 Prediction Results")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Churn Probability",
                f"{risk:.2f}%"
            )

            c2.metric(
                "Risk Category",
                risk_label
            )

            c3.metric(
                "Selected Model",
                model_choice
            )

            # Probability visualization

            st.markdown("### Churn Probability Meter")

            fig, ax = make_chart()

            ax.barh(
                ["Churn Probability"],
                [risk],
                color="#38bdf8"
            )

            ax.barh(
                ["Churn Probability"],
                [100 - risk],
                left=[risk],
                color="#334155"
            )

            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)")
            ax.set_title("Predicted Churn Probability")

            st.pyplot(fig)
            plt.close(fig)

            st.progress(
                min(int(round(risk)), 100)
            )

            if risk < 30:

                st.success(
                    "🟢 LOW RISK – Customer is less likely to churn."
                )

            elif risk < 60:

                st.warning(
                    "🟡 MEDIUM RISK – Customer may need additional attention."
                )

            else:

                st.error(
                    "🔴 HIGH RISK – Customer has a high predicted "
                    "churn probability."
                )

            # Retention recommendations

            st.divider()

            st.subheader("💡 Retention Recommendations")

            recommendations = []

            if risk >= 60:

                recommendations.extend([
                    "Contact the customer and understand their concerns.",
                    "Review whether a suitable retention offer is appropriate.",
                    "Provide personalized customer support."
                ])

            elif risk >= 30:

                recommendations.extend([
                    "Monitor customer satisfaction.",
                    "Share suitable service offers.",
                    "Follow up with the customer."
                ])

            else:

                recommendations.extend([
                    "Maintain good customer service.",
                    "Continue regular customer engagement."
                ])

            if contract == "Month-to-month":

                recommendations.append(
                    "Consider offering a suitable long-term contract option."
                )

            if tenure < 12:

                recommendations.append(
                    "Provide additional support during the early customer relationship."
                )

            if security == "No" and internet != "No":

                recommendations.append(
                    "Consider explaining available online security services."
                )

            for item in recommendations:

                st.write(f"• {item}")

            # Customer profile

            st.divider()

            st.subheader("Customer Profile Summary")

            st.dataframe(
                customer.T.rename(columns={0: "Customer Value"}),
                use_container_width=True
            )

            st.caption(
                "Risk thresholds are project-defined categories. "
                "They are not calibrated business cut-offs. "
                "Probability is the model's estimated churn probability, "
                "not a guarantee that the customer will leave."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =========================================================
# 12. ABOUT THE PROJECT PAGE
# =========================================================

elif page == "📘 About Project":

    page_header(
        "Understanding the Project",
        "Learn the concepts, algorithms and workflow behind "
        "the Telecom Customer Churn Prediction System."
    )

    st.subheader("📖 What is Customer Churn?")

    st.write(
        """
        Customer churn occurs when a customer stops using the services
        provided by a company.

        In the telecom industry, it means that a customer discontinues
        their current telecom service.
        """
    )

    st.divider()

    st.subheader("🧠 AI and ML Concepts Used")

    with st.expander("1. Artificial Intelligence"):

        st.write(
            """
            Artificial Intelligence enables computer systems to perform
            tasks that normally require intelligent decision-making.

            In this project, an ML-based system analyzes customer
            information and supports retention decisions.
            """
        )

    with st.expander("2. Supervised Machine Learning"):

        st.write(
            """
            Supervised Learning is a machine learning approach in which
            a model learns from labelled examples.

            Our dataset contains the target column Churn, with two labels:
            Yes and No.

            The model learns from these examples and predicts churn
            for new customer records.
            """
        )

    with st.expander("3. Classification"):

        st.write(
            """
            Classification is a supervised learning task that predicts
            a category or class.

            This project uses Logistic Regression and Random Forest
            to classify customers as Churn or No Churn.
            """
        )

    with st.expander("4. SMOTE"):

        st.write(
            """
            SMOTE stands for Synthetic Minority Over-sampling Technique.

            It creates synthetic examples of the minority class to
            address class imbalance in the training dataset.

            In this project, SMOTE was applied to the training data.
            """
        )

    with st.expander("5. Model Evaluation"):

        st.write(
            """
            Accuracy measures overall correct predictions.

            Precision measures how many predicted churn customers
            actually churned.

            Recall measures how many actual churn customers
            were correctly identified.

            F1-score balances precision and recall.

            ROC-AUC measures the model's ability to distinguish
            between churn and non-churn customers.
            """
        )

    st.divider()

    st.subheader("⚙️ Algorithms Used")

    st.markdown("**Logistic Regression**")

    st.write(
        """
        A supervised classification algorithm that estimates the
        probability of a customer belonging to a particular class.
        """
    )

    st.markdown("**Random Forest**")

    st.write(
        """
        An ensemble learning algorithm that combines multiple
        decision trees to produce a classification result.
        """
    )

    st.markdown("**GridSearchCV**")

    st.write(
        """
        A hyperparameter tuning method used to evaluate different
        parameter combinations and identify a suitable configuration.
        """
    )

    st.divider()

    st.subheader("🔄 Project Workflow")

    workflow = [
        "1. Historical Telecom Customer Dataset",
        "2. Exploratory Data Analysis",
        "3. Data Cleaning and Preprocessing",
        "4. Train-Test Split and SMOTE",
        "5. Model Training",
        "6. Model Evaluation",
        "7. Customer Churn Prediction",
        "8. Risk Classification and Retention Suggestions"
    ]

    for step in workflow:

        st.info(step)

    st.divider()

    st.subheader("🛠️ Technologies Used")

    st.write(
        """
        - Python – Main programming language.
        - Pandas and NumPy – Data manipulation.
        - Scikit-learn – Machine learning algorithms.
        - Imbalanced-learn – SMOTE.
        - Matplotlib – Data visualization.
        - Joblib – Model saving and loading.
        - Streamlit – Interactive dashboard.
        """
    )

    st.divider()

    st.subheader("🔗 Project Resources")

    st.markdown(
        "[GitHub Repository](https://github.com/Sneha-331/Telecom-Customer-Churn)"
    )

    st.markdown(
        "[Live Streamlit Application](https://telecom-customer-churn-kftqexctp5zjg5cweqjcgx.streamlit.app/)"
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Telecom Customer Churn Risk Classification and Retention
        Prioritization System<br>
        CSE252 | Introduction to Artificial Intelligence and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
