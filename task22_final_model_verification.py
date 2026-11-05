
import joblib
import numpy as np

print("=" * 55)
print("FINAL MODEL VERIFICATION")
print("=" * 55)

model = joblib.load("final_model.pkl")
vectorizer = joblib.load("final_vectorizer.pkl")

print("\n1. Model loaded successfully!")
print("2. Vectorizer loaded successfully!")

# Sample job descriptions for testing
samples = [
    "Software Engineer required. Must have Python Java SQL programming skills and 3 years experience.",
    "Work from home and earn thousands daily. No experience needed. Pay registration fee immediately."
]

# Transform sample text
sample_vectors = vectorizer.transform(samples)

# Generate predictions
predictions = model.predict(sample_vectors)

print("\n3. SAMPLE PREDICTIONS")
print("-" * 40)

for i, prediction in enumerate(predictions):
    result = "Fraudulent Job" if prediction == 1 else "Genuine Job"
    print(f"\nSample {i + 1}: {result}")

print("\n4. Model Verification Completed Successfully!")
print("TASK 22 COMPLETED!")
