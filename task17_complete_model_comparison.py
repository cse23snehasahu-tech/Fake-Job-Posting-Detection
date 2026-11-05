
import pandas as pd

print("=" * 65)
print("COMPLETE MACHINE LEARNING MODEL COMPARISON")
print("=" * 65)

results = {
    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "SVM",
        "KNN",
        "SGD Classifier"
    ],

    "Accuracy": [
        0.9754, 0.9597, 0.9460, 0.9692,
        0.9863, 0.9846, 0.9740
    ],

    "Precision": [
        0.6906, 0.6593, 0.4597, 0.6599,
        0.8735, 0.8933, 0.6739
    ],

    "Recall": [
        0.8902, 0.3468, 0.6590, 0.7514,
        0.8382, 0.7746, 0.8960
    ],

    "F1 Score": [
        0.7778, 0.4545, 0.5416, 0.7027,
        0.8555, 0.8297, 0.7692
    ],

    "ROC-AUC": [
        0.9853, 0.9294, 0.8145, 0.9830,
        0.9865, 0.9489, 0.9849
    ]
}

df = pd.DataFrame(results)

print("\nCOMPLETE MODEL COMPARISON")
print("-" * 65)
print(df.round(4).to_string(index=False))

best_model = df.loc[df["F1 Score"].idxmax()]

print("\nBEST MODEL BASED ON F1 SCORE")
print("-" * 40)
print("Model:", best_model["Model"])
print("F1 Score:", round(best_model["F1 Score"], 4))

df.to_csv("complete_model_comparison.csv", index=False)

print("\nComparison CSV saved successfully!")
print("TASK 17 COMPLETED!")
