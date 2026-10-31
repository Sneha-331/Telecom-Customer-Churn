
# TELECOM CUSTOMER CHURN PREDICTION
# STEP 7: LOGISTIC REGRESSION AND RANDOM FOREST

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# 1. Load preprocessed data
data = joblib.load("models/preprocessed_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("DATA LOADED SUCCESSFULLY")

# 2. Create Logistic Regression model
log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

# 3. Train Logistic Regression
print("\nTraining Logistic Regression...")

log_model.fit(X_train, y_train)

print("Logistic Regression Training Completed!")

# 4. Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 5. Train Random Forest
print("\nTraining Random Forest...")

rf_model.fit(X_train, y_train)

print("Random Forest Training Completed!")

# 6. Make predictions on test data
log_predictions = log_model.predict(X_test)

rf_predictions = rf_model.predict(X_test)

# 7. Calculate accuracy
log_accuracy = accuracy_score(
    y_test, log_predictions
)

rf_accuracy = accuracy_score(
    y_test, rf_predictions
)

# 8. Display results
print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression Accuracy:",
      round(log_accuracy * 100, 2), "%")

print("Random Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

# 9. Save trained models
joblib.dump(
    log_model,
    "models/logistic_regression.pkl"
)

joblib.dump(
    rf_model,
    "models/random_forest.pkl"
)

print("\nMODELS SAVED SUCCESSFULLY!")

print("\nSTEP 7 COMPLETED!")
