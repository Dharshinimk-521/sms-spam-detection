from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Load model and vectorizer
MODEL_PATH = os.path.join("models", "saved_models.pkl")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model = data["model"]
vectorizer = data["vectorizer"]
best_model_name = data.get("model_name", "Spam Classifier")
best_accuracy = data.get("accuracy", 0)

# Load dataset info
df = pd.read_csv(os.path.join("data", "spam.csv"))
total_messages = len(df)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get("message", "")

    if not text.strip():
        return jsonify({"success": False, "error": "Empty message"})

    vectorized = vectorizer.transform([text])
    prediction = model.predict(vectorized)[0]
    probability = max(model.predict_proba(vectorized)[0]) * 100

    # 🔍 Explainability (important words)
    feature_names = vectorizer.get_feature_names_out()
    vector = vectorized.toarray()[0]

    important_words = [
        feature_names[i]
        for i in vector.argsort()[-5:][::-1]
        if vector[i] > 0
    ]

    return jsonify({
        "success": True,
        "message": text,
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(probability, 2),
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
