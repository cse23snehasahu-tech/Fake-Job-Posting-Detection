
import joblib
import pandas as pd

# Load the selected baseline SVM model
final_model = joblib.load("svm_model.pkl")

# Load the TF-IDF vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Save final model and vectorizer
joblib.dump(final_model, "final_model.pkl")
joblib.dump(vectorizer, "final_vectorizer.pkl")

# Save final model information
model_info = {
    "Selected Model": ["Baseline SVM"],
    "Accuracy": [0.986298],
    "Precision": [0.873494],
    "Recall": [0.838150],
    "F1 Score": [0.855457],
    "ROC-AUC": [0.986522]
}

df = pd.DataFrame(model_info)
df.to_csv("final_model_info.csv", index=False)

print("=" * 55)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 55)
print("\nSelected Model: Baseline SVM")
print("Accuracy: 98.63%")
print("Precision: 87.35%")
print("Recall: 83.82%")
print("F1 Score: 85.55%")
print("ROC-AUC: 98.65%")

print("\nCreated Files:")
print("1. final_model.pkl")
print("2. final_vectorizer.pkl")
print("3. final_model_info.csv")

print("\nTASK 21 COMPLETED!")
