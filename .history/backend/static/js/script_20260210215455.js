// Example messages
const examples = {
    spam: "URGENT! You've won $5000! Click here now to claim your prize: http://fakescam.com",
    ham: "Hey, are we still meeting for lunch tomorrow at 1pm? Let me know!"
};

// Verify message function
async function verifyMessage() {
    const message = document.getElementById('sms-input').value.trim();
    
    if (!message) {
        alert('Please enter a message to verify!');
        return;
    }
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
        
    } catch (error) {
        document.getElementById('loading').style.display = 'none';
        alert('Network error: ' + error.message);
    }
}

// Display results
function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block';
    
    // Update prediction badge
    const badge = document.getElementById('prediction-badge');
    const predictionText = document.getElementById('prediction-text');
    
    if (data.is_spam) {
        badge.className = 'prediction-badge spam';
        predictionText.textContent = '⚠️ SPAM DETECTED';
        document.getElementById('result-title').textContent = '⚠️ Warning: Potential Spam';
    } else {
        badge.className = 'prediction-badge ham';
        predictionText.textContent = '✅ SAFE MESSAGE';
        document.getElementById('result-title').textContent = '✅ Message is Safe';
    }
    
    // Update confidence bars
    const hamPercent = data.confidence.ham;
    const spamPercent = data.confidence.spam;
    
    document.getElementById('ham-percent').textContent = hamPercent + '%';
    document.getElementById('spam-percent').textContent = spamPercent + '%';
    
    document.getElementById('ham-bar').style.width = hamPercent + '%';
    document.getElementById('spam-bar').style.width = spamPercent + '%';
    
    // Display analyzed message
    document.getElementById('analyzed-message').textContent = data.message;
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    // Show important words (model explanation)
    const analysisBox = document.getElementById('analysis-box');
    const wordsEl = document.getElementById('important-words');

    if (data.important_words && data.important_words.length > 0) {
        analysisBox.style.display = 'block';
        wordsEl.textContent =
            "Key words influencing decision: " + data.important_words.join(", ");
    } else {
        analysisBox.style.display = 'none';
    }

}

// Try example
function tryExample(type) {
    const message = examples[type];
    document.getElementById('sms-input').value = message;
    
    // Auto-verify after setting example
    setTimeout(() => {
        verifyMessage();
    }, 300);
}

// Load stats on page load
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        if (data.success) {
            document.getElementById('best-model').textContent = data.best_model;
            document.getElementById('best-accuracy').textContent = data.best_accuracy + '%';
            document.getElementById('total-messages').textContent = data.dataset_info.total_messages;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Enter key to verify
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
    
    document.getElementById('sms-input').addEventListener('keypress', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            verifyMessage();
        }
    });
});