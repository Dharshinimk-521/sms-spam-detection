"""
SMS Spam Detection System
Enhanced with visualizations and web interface
"""
import pickle
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# -------------------- PATH SETUP --------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")


os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------- FLASK APP --------------------
app = Flask(__name__)

# -------------------- LOAD DATA --------------------
print("📂 Loading dataset from:", DATA_PATH)

data = pd.read_csv(DATA_PATH, encoding="latin-1")
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

train_data = data[:4400]
test_data = data[4400:]

# -------------------- VECTORIZATION --------------------
Vectorizer = TfidfVectorizer(max_features=3000)
X_train = Vectorizer.fit_transform(train_data.message)
y_train = train_data.label

X_test = Vectorizer.transform(test_data.message)
y_test = test_data.label


# -------------------- MODEL TRAINING / LOADING --------------------
if os.path.exists(MODEL_PATH):
    print("📦 Loading saved model...")

    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    Classifier = saved["model"]
    Vectorizer = saved["vectorizer"]
    best_model_name = saved["model_name"]
    best_accuracy = saved["accuracy"]
    model_accuracies = saved["all_accuracies"]

else:
    print("\n🤖 Training models...")

    models = {
        "SVM (Linear)": OneVsRestClassifier(
            SVC(kernel="linear", probability=True)
        ),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }

    model_accuracies = {}
    Classifier = None
    best_model_name = ""
    best_accuracy = 0.0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        model_accuracies[name] = acc

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            Classifier = model

        print(f"✅ {name}: {acc*100:.2f}%")

    print(f"\n🏆 Best Model: {best_model_name}")

    # 💾 SAVE EVERYTHING
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": Classifier,
            "vectorizer": Vectorizer,
            "accuracy": best_accuracy,
            "model_name": best_model_name,
            "all_accuracies": model_accuracies
        }, f)

    print("💾 Model saved successfully")

def extract_important_words(message, vectorizer, classifier, top_n=5):
    try:
        feature_names = np.array(vectorizer.get_feature_names_out())
        vector = vectorizer.transform([message])

        # Works for Linear SVM & Logistic Regression
        if hasattr(classifier, "coef_"):
            coef = classifier.coef_[0]
            scores = vector.toarray()[0] * coef

        # Works for Naive Bayes
        elif hasattr(classifier, "feature_log_prob_"):
            spam_index = list(classifier.classes_).index("spam")
            scores = vector.toarray()[0] * classifier.feature_log_prob_[spam_index]

        else:
            return []

        top_indices = np.argsort(scores)[-top_n:]
        words = feature_names[top_indices]
        words = [w for w in words if w.strip() != ""]

        return words[::-1]

    except Exception as e:
        print("Explainability error:", e)
        return []

# -------------------- VISUALIZATION --------------------
plt.figure(figsize=(8, 5))
plt.bar(model_accuracies.keys(),
        [v * 100 for v in model_accuracies.values()])
plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")
plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, "model_comparison.png"))
plt.close()

#mhahah
import os
print("Template folder:", app.template_folder)
print("Templates found:", os.listdir(app.template_folder))

# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"success": False, "error": "Empty message"})

    vector = Vectorizer.transform([message])
    prediction = Classifier.predict(vector)[0]
    important_words = extract_important_words(
    message,
    Vectorizer,
    Classifier
)


    spam_conf = ham_conf = 0.0

    if hasattr(Classifier, "predict_proba"):
        probs = Classifier.predict_proba(vector)[0]
        classes = Classifier.classes_

        if "spam" in classes:
            spam_conf = probs[list(classes).index("spam")] * 100
        if "ham" in classes:
            ham_conf = probs[list(classes).index("ham")] * 100

    # ---- EXPLAINABILITY (keywords) ----
    feature_names = Vectorizer.get_feature_names_out()
    vector_arr = vector.toarray()[0]

    important_words = [
        feature_names[i]
        for i in vector_arr.argsort()[-5:][::-1]
        if vector_arr[i] > 0
    ]


    return jsonify({
    "success": True,
    "message": message,   # ✅ ADD THIS
    "prediction": prediction,
    "is_spam": prediction == "spam",
    "confidence": {
        "spam": round(spam_conf, 2),
        "ham": round(ham_conf, 2)
    },
    "important_words": important_words

})




@app.route("/stats")
def stats():
    return jsonify({
        "success": True,
        "best_model": best_model_name,
        "best_accuracy": round(best_accuracy * 100, 2),
        "dataset_info": {
            "total_messages": len(data),
            "spam": int((data.label == "spam").sum()),
            "ham": int((data.label == "ham").sum())
        }
    })


# -------------------- RUN --------------------
if __name__ == "__main__":
    print("\n🚀 Server running at http://127.0.0.1:5000\n")
    app.run(debug=True)
