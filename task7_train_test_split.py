
import pandas as pd
from sklearn.model_selection import train_test_split

# Load preprocessed dataset
df = pd.read_csv("fake_job_postings_preprocessed.csv")

# Input and target
X = df["cleaned_text"]
y = df["fraudulent"]

print("=" * 50)
print("TRAIN TEST SPLIT")
print("=" * 50)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTotal Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# Save split datasets
train_df = pd.DataFrame({
    "cleaned_text": X_train,
    "fraudulent": y_train
})

test_df = pd.DataFrame({
    "cleaned_text": X_test,
    "fraudulent": y_test
})

train_df.to_csv("train_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("\nTraining and testing datasets saved!")
print("TRAIN TEST SPLIT COMPLETED!")
