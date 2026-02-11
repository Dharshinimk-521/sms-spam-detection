"""
SMS Spam Detection System
Enhanced with visualizations and web interface
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

app = Flask(__name__)
app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static"
)

# Global variables
global Classifier
global Vectorizer
global model_accuracies

# Create necessary directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

# -------- LOAD DATASET --------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")

print("Loading dataset from:", DATA_PATH)

data = pd.read_csv(DATA_PATH, encoding="latin-1")

# Clean dataset - keep only v1 (label) and v2 (message)
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Split dataset
train_data = data[:4400]
test_data = data[4400:]

print(f"✅ Dataset loaded: {len(data)} total messages")
print(f"   Training: {len(train_data)}, Testing: {len(test_data)}")

# -------- VECTORIZATION --------
print("\n🔧 Creating TF-IDF vectors...")
Vectorizer = TfidfVectorizer(max_features=3000)
X_train = Vectorizer.fit_transform(train_data.message)
y_train = train_data.label
X_test = Vectorizer.transform(test_data.message)
y_test = test_data.label

# -------- TRAIN MULTIPLE MODELS FOR COMPARISON --------
print("\n Training multiple models...")

models = {
    'SVM (Linear)': OneVsRestClassifier(SVC(kernel='linear', probability=True)),
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

model_accuracies = {}
best_model_name = None
best_accuracy = 0

for name, model in models.items():
    print(f"   Training {name}...")
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    
    print(f"   ✅ {name}: {accuracy*100:.2f}% accuracy")
    
    # Track best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        Classifier = model

print(f"\n🏆 Best Model: {best_model_name} with {best_accuracy*100:.2f}% accuracy")

# -------- GENERATE VISUALIZATIONS --------
print("\n Generating visualizations...")

# 1. Model Accuracy Comparison
plt.figure(figsize=(10, 6))
plt.style.use('dark_background')
colors = ['#667eea', '#4ecdc4', '#f093fb']
bars = plt.bar(model_accuracies.keys(), 
               [acc*100 for acc in model_accuracies.values()],
               color=colors,
               edgecolor='white',
               linewidth=2)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.ylim(0, 105)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Dataset Distribution
plt.figure(figsize=(8, 6))
plt.style.use('dark_background')
distribution = test_data.label.value_counts()
colors_pie = ['#4ecdc4', '#ff6b6b']
explode = (0.05, 0.05)

plt.pie(distribution.values, 
        labels=['Ham (Not Spam)', 'Spam'],
        autopct='%1.1f%%',
        colors=colors_pie,
        explode=explode,
        shadow=True,
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'})

plt.title('Test Dataset Distribution', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()
plt.savefig('visualizations/dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Visualizations saved in 'visualizations/' folder")

# -------- WEB INTERFACE ROUTE --------
@app.route('/')
def home():
    """Render main dashboard"""
    return render_template('index.html')

# -------- API ENDPOINT --------
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict if SMS is spam or ham
    Accepts JSON: {"message": "Your SMS text here"}
    Returns JSON with prediction and confidence
    """
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'success': False,
                'error': 'No message provided'
            })
        
        # Vectorize message
        vectorized = Vectorizer.transform([message])
        
        # Predict
        prediction = Classifier.predict(vectorized)[0]
        
        # Get probability scores
        if hasattr(Classifier, 'predict_proba'):
            proba = Classifier.predict_proba(vectorized)[0]
            # Convert to percentage
            confidence_spam = float(proba[1] * 100) if len(proba) > 1 else 0
            confidence_ham = float(proba[0] * 100)
        else:
            confidence_spam = 0
            confidence_ham = 0
        
        return jsonify({
            'success': True,
            'message': message,
            'prediction': prediction,
            'is_spam': prediction == 'spam',
            'confidence': {
                'spam': round(confidence_spam, 2),
                'ham': round(confidence_ham, 2)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# -------- STATS API ENDPOINT --------
@app.route('/stats', methods=['GET'])
def stats():
    """Return model statistics"""
    return jsonify({
        'success': True,
        'model_accuracies': {k: round(v*100, 2) for k, v in model_accuracies.items()},
        'best_model': best_model_name,
        'best_accuracy': round(best_accuracy*100, 2),
        'dataset_info': {
            'total_messages': len(data),
            'training_size': len(train_data),
            'test_size': len(test_data),
            'spam_count': int((data.label == 'spam').sum()),
            'ham_count': int((data.label == 'ham').sum())
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting SMS Spam Detection System...")
    print(f"📍 Open http://127.0.0.1:{port} in your browser\n")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=True,
        use_reloader=True
    )