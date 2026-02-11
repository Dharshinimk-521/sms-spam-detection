import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding="latin-1")


# Adjust column names if needed
df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})
df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# Save everything
os.makedirs("../models", exist_ok=True)

with open("../models/saved_models.pkl", "wb") as f:
    pickle.dump({
        "model": model,
        "vectorizer": vectorizer,
        "accuracy": accuracy,
        "model_name": "Multinomial Naive Bayes"
    }, f)


print("✅ Model trained and saved successfully!")
print("Accuracy:", accuracy)
