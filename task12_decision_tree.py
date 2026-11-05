
import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    confusion_matrix, roc_auc_score
)

print("=" * 55)
print("DECISION TREE MODEL TRAINING")
print("=" * 55)

X_train = load_npz("X_train_tfidf.npz")
X_test = load_npz("X_test_tfidf.npz")

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

y_train = train_df["fraudulent"]
y_test = test_df["fraudulent"]

model = DecisionTreeClassifier(
    max_depth=30,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nMODEL PERFORMANCE")
print("-" * 40)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "decision_tree_model.pkl")

print("\nModel saved successfully!")
print("DECISION TREE COMPLETED!")
