


"""import streamlit as st
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
)"""



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
