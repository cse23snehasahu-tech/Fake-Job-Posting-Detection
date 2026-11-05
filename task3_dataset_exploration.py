import pandas as pd

# Load Dataset
df = pd.read_csv("fake_job_postings.csv")

print("=" * 60)
print("FAKE JOB POSTING DETECTION - DATASET EXPLORATION")
print("=" * 60)

# Dataset dimensions
print("\n1. Dataset Shape:")
print(df.shape)

# Column names
print("\n2. Dataset Columns:")
print(df.columns.tolist())

# First five records
print("\n3. First Five Records:")
print(df.head())

# Dataset information
print("\n4. Dataset Information:")
df.info()

# Missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Target distribution
print("\n6. Fake vs Genuine Job Postings:")
print(df["fraudulent"].value_counts())

print("\n7. Target Percentage:")
print(df["fraudulent"].value_counts(normalize=True) * 100)

print("\nDATASET EXPLORATION COMPLETED!")