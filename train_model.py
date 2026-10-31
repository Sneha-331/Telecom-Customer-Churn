
# TELECOM CUSTOMER CHURN PROJECT

# Step 1: Import Pandas
import pandas as pd

# Step 2: Load the dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Step 3: Display first five records
print("\nFIRST FIVE CUSTOMERS")
print(df.head())

# Step 4: Display dataset dimensions
print("\nDATASET SHAPE")
print(df.shape)

# Step 5: Display column names
print("\nCOLUMN NAMES")
print(df.columns.tolist())

# Step 6: Display dataset information
print("\nDATASET INFORMATION")
df.info()

# Step 7: Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# Step 8: Check churn distribution
print("\nCHURN DISTRIBUTION")
print(df["Churn"].value_counts())
