
import pandas as pd
import re
import html

# Load cleaned dataset
df = pd.read_csv("fake_job_postings_cleaned.csv")

print("=" * 55)
print("TEXT PREPROCESSING")
print("=" * 55)

# Text columns to combine
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Function to clean text
def clean_text(text):
    text = str(text)
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Combine text columns
df["combined_text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

# Apply text cleaning
df["cleaned_text"] = df["combined_text"].apply(clean_text)

# Display results
print("\nOriginal Text:")
print(df["combined_text"].iloc[0][:500])

print("\nCleaned Text:")
print(df["cleaned_text"].iloc[0][:500])

print("\nTotal Records:", len(df))
print("Empty Cleaned Text:", (df["cleaned_text"] == "").sum())

# Save preprocessed dataset
df.to_csv("fake_job_postings_preprocessed.csv", index=False)

print("\nPreprocessed dataset saved successfully!")
print("TEXT PREPROCESSING COMPLETED!")
