
# TELECOM CUSTOMER CHURN PROJECT
# STEP 6: FEATURE ENCODING, SCALING, TRAIN-TEST SPLIT AND SMOTE

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv(
    "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

print("Original Dataset:", df.shape)

# 2. Clean TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

# 3. Remove unnecessary customer ID
df.drop("customerID", axis=1, inplace=True)

# 4. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

print("\nOriginal Target Distribution:")
print(y.value_counts())

# 5. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("\nNumerical Features:", numeric_features)
print("\nCategorical Features:", categorical_features)

# 6. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# 7. Feature encoding and scaling
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data without fitting again
X_test_processed = preprocessor.transform(X_test)

print("\nAfter Encoding and Scaling:")
print("Training:", X_train_processed.shape)
print("Testing:", X_test_processed.shape)

# 8. Apply SMOTE only to training data
smote = SMOTE(random_state=42)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed, y_train
)

print("\nBefore SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_balanced.value_counts())

# 9. Save preprocessing objects and data
data = {
    "preprocessor": preprocessor,
    "X_train": X_train_balanced,
    "X_test": X_test_processed,
    "y_train": y_train_balanced,
    "y_test": y_test,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(data, "models/preprocessed_data.pkl")

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
