
# TELECOM CUSTOMER CHURN PROJECT
# STEP 5: DATA PREPROCESSING AND EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Original Dataset Shape:", df.shape)

# 2. Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Fill missing TotalCharges using median
df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum().sum())

# 5. Remove customerID
df.drop("customerID", axis=1, inplace=True)

# 6. Dataset statistics
print("\nDataset Statistics:")
print(df.describe())

# 7. Churn distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.savefig("churn_distribution.png")
plt.show()

# 8. Churn based on contract
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.savefig("churn_by_contract.png")
plt.show()

# 9. Monthly charges distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="MonthlyCharges", hue="Churn",
             kde=True, multiple="stack")
plt.title("Monthly Charges Distribution")
plt.savefig("monthly_charges.png")
plt.show()

# 10. Tenure distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="tenure", hue="Churn",
             kde=True, multiple="stack")
plt.title("Customer Tenure Distribution")
plt.savefig("tenure_distribution.png")
plt.show()

print("\nEDA AND DATA CLEANING COMPLETED!")
