
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load training and testing data
train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

X_train = train_df["cleaned_text"].fillna("")
X_test = test_df["cleaned_text"].fillna("")

print("=" * 50)
print("TF-IDF VECTORIZATION")
print("=" * 50)

# Initialize TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=2,
    max_df=0.95
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

print("\nTraining Matrix Shape:", X_train_tfidf.shape)
print("Testing Matrix Shape:", X_test_tfidf.shape)

print("\nTotal Vocabulary Features:", len(tfidf.vocabulary_))

print("\nSample Important Vocabulary:")
print(list(tfidf.vocabulary_.keys())[:20])

# Save vectorized matrices and vectorizer
from scipy.sparse import save_npz
import joblib

save_npz("X_train_tfidf.npz", X_train_tfidf)
save_npz("X_test_tfidf.npz", X_test_tfidf)

joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nTF-IDF files saved successfully!")
print("TF-IDF VECTORIZATION COMPLETED!")
