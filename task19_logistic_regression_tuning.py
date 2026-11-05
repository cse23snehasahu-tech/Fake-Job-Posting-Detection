
import pandas as pd
import joblib

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

print("=" * 60)
print("LOGISTIC REGRESSION HYPERPARAMETER TUNING")
print("=" * 60)

X_train = load_npz("X_train_tfidf.npz")
X_test = load_npz("X_test_tfidf.npz")

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

y_train = train_df["fraudulent"]
y_test = test_df["fraudulent"]

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="liblinear",
    random_state=42
)

parameters = {
    "C": [0.1, 1, 5]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    cv=3,
    scoring="f1",
    n_jobs=1,
    verbose=1
)

print("\nStarting lightweight hyperparameter tuning...")

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print("\nBEST CROSS-VALIDATION F1 SCORE")
print(round(grid_search.best_score_, 4))

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTUNED LOGISTIC REGRESSION PERFORMANCE")
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

joblib.dump(best_model, "tuned_logistic_regression_model.pkl")

print("\nTuned Logistic Regression model saved successfully!")
print("TASK 19 COMPLETED!")
