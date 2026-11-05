import pandas as pd

# Load original dataset
df = pd.read_csv("fake_job_postings.csv")

print("Original Dataset Shape:", df.shape)

# Text columns
text_columns = [
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Categorical columns
categorical_columns = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

# Handle missing text values
for col in text_columns:
    df[col] = df[col].fillna("")

# Handle missing categorical values
for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

# Check missing values
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nTotal Missing Values:", df.isnull().sum().sum())

# Save cleaned dataset
df.to_csv("fake_job_postings_cleaned.csv", index=False)

print("\nCleaned Dataset Shape:", df.shape)
print("\nCleaned dataset saved successfully!")