
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

print("=" * 65)
print("FINAL MODEL COMPARISON")
print("=" * 65)

X_test = load_npz("X_test_tfidf.npz")
test_df = pd.read_csv("test_data.csv")
y_test = test_df["fraudulent"]

models = {
    "Baseline Logistic Regression": joblib.load("logistic_regression_model.pkl"),
    "Tuned Logistic Regression": joblib.load("tuned_logistic_regression_model.pkl"),
    "Baseline SVM": joblib.load("svm_model.pkl"),
    "Tuned SVM": joblib.load("tuned_svm_model.pkl")
}

results = []

for name, model in models.items():

    print(f"\nEvaluating: {name}")

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score)
    }

    results.append(metrics)

    print("Accuracy:", round(metrics["Accuracy"], 4))
    print("Precision:", round(metrics["Precision"], 4))
    print("Recall:", round(metrics["Recall"], 4))
    print("F1 Score:", round(metrics["F1 Score"], 4))
    print("ROC-AUC:", round(metrics["ROC-AUC"], 4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

print("\n" + "=" * 65)
print("FINAL COMPARISON TABLE")
print("=" * 65)

print(results_df.to_string(index=False))

results_df.to_csv(
    "final_model_comparison.csv", index=False
)

print("\nComparison saved: final_model_comparison.csv")
print("\nTASK 20 COMPLETED!")
