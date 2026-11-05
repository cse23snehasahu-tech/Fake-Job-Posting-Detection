
import pandas as pd

print("=" * 60)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 60)

results = {
    "Model": ["Logistic Regression", "Naive Bayes"],
    "Accuracy": [0.9754, 0.9597],
    "Precision": [0.6906, 0.6593],
    "Recall": [0.8902, 0.3468],
    "F1 Score": [0.7778, 0.4545],
    "ROC-AUC": [0.9853, 0.9294]
}

df = pd.DataFrame(results)

print("\nMODEL COMPARISON TABLE")
print("-" * 60)
print(df.to_string(index=False))

print("\nBEST MODEL FOR FRAUD DETECTION")
best_model = df.loc[df["F1 Score"].idxmax(), "Model"]
print("Best Model:", best_model)

df.to_csv("model_comparison_results.csv", index=False)

print("\nComparison results saved successfully!")
print("TASK 11 COMPLETED!")
