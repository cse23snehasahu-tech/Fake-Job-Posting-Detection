
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

print("=" * 60)
print("RECOVERING PREVIOUS MODEL RESULTS")
print("=" * 60)

X_test = load_npz("X_test_tfidf.npz")
test_df = pd.read_csv("test_data.csv")
y_test = test_df["fraudulent"]

models = {
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
    "KNN": "knn_model.pkl"
}

results = []

for name, filename in models.items():
    print(f"\nEvaluating {name}...")

    model = joblib.load(filename)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = model.decision_function(X_test)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_score)
    }

    results.append(row)

df = pd.DataFrame(results)

print("\nRECOVERED MODEL RESULTS")
print(df.round(4).to_string(index=False))

df.to_csv("recovered_model_results.csv", index=False)

print("\nResults saved successfully!")
