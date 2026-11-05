import pandas as pd

df = pd.read_csv("fake_job_postings.csv")

print("=" * 55)
print("DATA QUALITY ANALYSIS")
print("=" * 55)

# Dataset shape
print("\n1. Dataset Shape:", df.shape)

# Missing values
print("\n2. Missing Values:")
print(df.isnull().sum())

# Total missing values
print("\n3. Total Missing Values:", df.isnull().sum().sum())

# Duplicate records
print("\n4. Duplicate Records:", df.duplicated().sum())

# Data types
print("\n5. Data Types:")
print(df.dtypes)

# Target distribution
print("\n6. Target Distribution:")
print(df["fraudulent"].value_counts())

print("\n7. Target Percentage:")
print((df["fraudulent"].value_counts(normalize=True) * 100).round(2))

print("\nDATA QUALITY ANALYSIS COMPLETED!")