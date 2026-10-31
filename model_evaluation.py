
# ==========================================
# TELECOM CUSTOMER CHURN - MODEL EVALUATION
# ==========================================

import os
import pickle
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# ==========================================
# STEP 1: CREATE MODELS FOLDER
# ==========================================

os.makedirs("models", exist_ok=True)

print("\nLoading saved data and models...")

# ==========================================
# STEP 2: LOAD PREPROCESSED DATA
# ==========================================

#with open("models/preprocessed_data.pkl", "rb") as file:
   # data = pickle.load(file)

data = joblib.load("models/preprocessed_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

feature_names = data.get("feature_names", None)

# ==========================================
# STEP 3: LOAD TRAINED MODELS
# ==========================================

logistic_model = joblib.load("models/logistic_regression.pkl")
random_forest_model = joblib.load("models/random_forest.pkl")

# ==========================================
# STEP 4: EVALUATION FUNCTION
# ==========================================

def evaluate_model(model, model_name):

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save confusion matrix
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )

    display.plot()
    plt.title(model_name + " - Confusion Matrix")
    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")
    plt.savefig(f"models/{filename}_confusion_matrix.png")
    plt.close()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    }, probabilities


# ==========================================
# STEP 5: EVALUATE LOGISTIC REGRESSION
# ==========================================

logistic_results, logistic_probabilities = evaluate_model(
    logistic_model,
    "Logistic Regression"
)

# ==========================================
# STEP 6: EVALUATE RANDOM FOREST
# ==========================================

rf_results, rf_probabilities = evaluate_model(
    random_forest_model,
    "Random Forest"
)

# ==========================================
# STEP 7: GRID SEARCH - RANDOM FOREST
# ==========================================

print("\nStarting Grid Search for Random Forest...")

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

# IMPORTANT: n_jobs=1 disables parallel processing.
# This avoids the loky resource tracker error.

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ),
    param_grid=param_grid,
    cv=3,
    scoring="roc_auc",
    n_jobs=1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation ROC-AUC:")
print(round(grid_search.best_score_, 4))

# Save tuned model
joblib.dump(
    best_rf_model,
    "models/best_random_forest.pkl"
)

print("\nTuned model saved successfully!")

# ==========================================
# STEP 8: EVALUATE TUNED RANDOM FOREST
# ==========================================

tuned_rf_results, tuned_rf_probabilities = evaluate_model(
    best_rf_model,
    "Tuned Random Forest"
)

# ==========================================
# STEP 9: COMPARE ALL MODELS
# ==========================================

comparison = pd.DataFrame([
    logistic_results,
    rf_results,
    tuned_rf_results
])

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))

comparison.to_csv(
    "models/model_comparison.csv",
    index=False
)

# ==========================================
# STEP 10: ROC CURVE COMPARISON
# ==========================================

plt.figure(figsize=(8, 6))

models_data = [
    ("Logistic Regression", logistic_probabilities),
    ("Random Forest", rf_probabilities),
    ("Tuned Random Forest", tuned_rf_probabilities)
]

for name, probabilities in models_data:

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(y_test, probabilities)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("models/roc_curve_comparison.png")
plt.close()

# ==========================================
# STEP 11: FEATURE IMPORTANCE
# ==========================================

importances = best_rf_model.feature_importances_

if feature_names is not None and len(feature_names) == len(importances):
    names = list(feature_names)
else:
    names = [
        f"Feature {i + 1}"
        for i in range(len(importances))
    ]

importance_df = pd.DataFrame({
    "Feature": names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(importance_df.head(10).to_string(index=False))

importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

top_features = importance_df.head(10).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")
plt.tight_layout()

plt.savefig("models/feature_importance.png")
plt.close()

# ==========================================
# STEP 12: COMPLETION
# ==========================================

print("\n" + "=" * 50)
print("STEP 8 COMPLETED SUCCESSFULLY!")
print("=" * 50)

print("\nAll evaluation results and graphs are saved")
print("inside the models folder.")

print("\nFiles generated:")
print("1. best_random_forest.pkl")
print("2. model_comparison.csv")
print("3. feature_importance.csv")
print("4. roc_curve_comparison.png")
print("5. feature_importance.png")
print("6. Confusion matrix images")
