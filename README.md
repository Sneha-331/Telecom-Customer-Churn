# 📊 Telecom Customer Churn Risk Classification & Retention Prioritization System

An end-to-end Machine Learning project that predicts customer churn in the telecom industry and helps businesses identify customers who are likely to discontinue their services.

The system uses Machine Learning classification models to estimate churn probability, categorize customers according to risk level, and provide data-driven retention recommendations through an interactive Streamlit dashboard.

## 🚀 Live Demo

🌐 **[Launch Telecom Customer Churn Dashboard](https://telecom-customer-churn-kftqexctp5zjg5cweqjcgx.streamlit.app/)**

📁 **[GitHub Repository](https://github.com/Sneha-331/Telecom-Customer-Churn)**

## 🎯 Project Objectives

- Predict whether a telecom customer is likely to leave the company.
- Identify high-risk customers before they discontinue their services.
- Analyze customer behavior and subscription patterns.
- Compare different Machine Learning classification algorithms.
- Prioritize customers for targeted retention strategies.
- Build an interactive dashboard for churn prediction.

## 🧠 Project Overview

Customer churn is a major challenge for telecommunication companies. Losing existing customers can affect revenue and business growth.

This project uses historical customer data to train Machine Learning models that predict the probability of customer churn.

Based on the predicted probability, customers are classified into three risk categories:

| Risk Level | Churn Probability |
|---|---|
| 🟢 Low Risk | Below 30% |
| 🟡 Medium Risk | 30%–60% |
| 🔴 High Risk | 60% and above |

The dashboard also provides retention suggestions to help businesses make informed decisions.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine Learning models |
| Imbalanced-learn | Handling imbalanced data using SMOTE |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web dashboard |
| VS Code | Development environment |
| GitHub | Version control and project hosting |

## 📂 Dataset

**Dataset:** IBM Telco Customer Churn Dataset

- Total records: 7,043 customers
- Total features: 21 columns
- Target variable: `Churn`
- Prediction classes: Yes / No

The dataset contains information about customer demographics, account details, subscribed services, contract types, monthly charges, and customer tenure.

## ⚙️ Machine Learning Workflow

1. **Data Collection:** Load the IBM Telco Customer Churn dataset.
2. **Exploratory Data Analysis:** Analyze customer distribution, churn patterns, contracts, tenure, and monthly charges.
3. **Data Preprocessing:** Handle missing values, convert data types, encode categorical variables, and scale numerical features.
4. **Class Imbalance Handling:** Apply SMOTE to balance the training dataset.
5. **Model Training:** Train Logistic Regression and Random Forest classifiers.
6. **Hyperparameter Tuning:** Use GridSearchCV to optimize Random Forest parameters.
7. **Model Evaluation:** Compare models using Accuracy, Precision, Recall, F1-score, and ROC-AUC.
8. **Deployment:** Integrate the trained models into an interactive Streamlit dashboard.

## 🤖 Machine Learning Models

Two classification algorithms were implemented and compared:

- **Logistic Regression**
- **Random Forest Classifier**

A GridSearchCV-based hyperparameter tuning process was also applied to Random Forest.

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 73.74% | 50.34% | 79.41% | 61.62% | 84.01% |
| Random Forest | 77.36% | 57.75% | 54.81% | 56.24% | 81.50% |
| Tuned Random Forest | 77.00% | 57.02% | 54.28% | 55.62% | 81.74% |

**Key Observation:** Logistic Regression achieved the highest Recall (79.41%) and F1-score (61.62%), making it useful for identifying a larger proportion of customers who actually churn. Random Forest achieved the highest Accuracy (77.36%).

## 📊 Dashboard Features

- Interactive customer information input.
- Churn probability prediction.
- Low, Medium, and High risk classification.
- Model selection between Logistic Regression and Tuned Random Forest.
- Customer churn risk analysis.
- Retention recommendations based on customer risk.

## 📁 Project Structure

```text
Telecom-Customer-Churn/
│
├── dataset/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── preprocessed_data.pkl
│   ├── logistic_regression.pkl
│   └── best_random_forest_compressed.pkl
│
├── app.py
├── eda.py
├── preprocessing.py
├── model_training.py
├── model_evaluation.py
├── train_model.py
├── compress_model.py
├── requirements.txt
│
├── churn_distribution.png
├── churn_by_contract.png
├── tenure_distribution.png
├── monthly_charges.png
└── README.md
```

## 💻 Installation & Setup

Follow these steps to run the project locally.

**1. Clone the repository**

```bash
git clone https://github.com/Sneha-331/Telecom-Customer-Churn.git
```

**2. Navigate to the project directory**

```bash
cd Telecom-Customer-Churn
```

**3. Install the required dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit application**

```bash
streamlit run app.py
```

**5. Open the dashboard**

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

## 📈 Expected Outcome

The project demonstrates how Machine Learning can support customer retention by identifying potential churners, estimating their risk, and helping businesses prioritize retention efforts.

It combines data analysis, preprocessing, model training, evaluation, and interactive deployment into one end-to-end Machine Learning application.

## 🔮 Future Enhancements

- Integrate real-time customer data.
- Experiment with advanced classification algorithms.
- Improve model performance through cross-validation and threshold optimization.
- Add customer segmentation.
- Develop personalized retention strategies.
- Integrate explainable AI techniques such as SHAP.

## 👩‍💻 Author

**Sneha**

B.Tech Student | Interested in Artificial Intelligence, Machine Learning, and Robotics.

🔗 **GitHub:** [Sneha-331](https://github.com/Sneha-331)

---

⭐ If you find this project interesting, consider giving the repository a star!
