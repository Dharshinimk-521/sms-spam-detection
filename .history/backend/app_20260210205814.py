from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_models.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

# ---------------- LOAD MODEL ----------------
with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
vectorizer = saved["vectorizer"]
best_model_name = saved.get("model_name", "Spam Classifier")
best_accuracy = saved.get("accuracy", 0)

# ---------------- DATASET INFO ----------------
df = pd.read_csv(DATA_PATH, encoding="latin-1")
total_messages = len(df)

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("message", "").strip()

    if not text:
        return jsonify(success=False, error="Empty message")

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    probs = model.predict_proba(vectorized)[0]

    # 🔍 HOW MODEL ANALYSED (TOP WORDS)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = vectorized.toarray()[0]

    top_indices = np.argsort(tfidf_values)[-5:][::-1]
    important_words = [
        feature_names[i] for i in top_indices if tfidf_values[i] > 0
    ]

    return jsonify({
        "success": True,
        "message": text,
        "is_spam": bool(prediction),
        "confidence": {
            "ham": round(probs[0] * 100, 2),
            "spam": round(probs[1] * 100, 2)
        },
        "analysis": {
            "important_words": important_words
        }
    })


@app.route("/stats")
def stats():
    return jsonify({
        "success": True,
        "best_model": best_model_name,
        "best_accuracy": round(best_accuracy * 100, 2),
        "dataset_info": {
            "total_messages": total_messages
        }
    })


if __name__ == "__main__":
    app.run(debug=True)
