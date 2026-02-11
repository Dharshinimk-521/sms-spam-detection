// Verify message
async function verifyMessage() {
    const message = document.getElementById("sms-input").value.trim();

    if (!message) {
        alert("Please enter a message");
        return;
    }

    document.getElementById("loading").style.display = "block";
    document.getElementById("results").style.display = "none";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        document.getElementById("loading").style.display = "none";

        if (!data.success) {
            alert(data.error);
            return;
        }

        displayResults(data);

    } catch (err) {
        document.getElementById("loading").style.display = "none";
        alert("Server error");
    }
}

// Display results
function displayResults(data) {
    document.getElementById("results").style.display = "block";

    const badge = document.getElementById("prediction-badge");
    const text = document.getElementById("prediction-text");

    if (data.is_spam) {
        text.textContent = "⚠️ SPAM DETECTED";
        document.getElementById("result-title").textContent = "⚠️ Warning: Potential Spam";
    } else {
        text.textContent = "✅ SAFE MESSAGE";
        document.getElementById("result-title").textContent = "✅ Message is Safe";
    }

    document.getElementById("ham-percent").textContent = data.confidence.ham;
    document.getElementById("spam-percent").textContent = data.confidence.spam;

    document.getElementById("ham-bar").style.width = data.confidence.ham + "%";
    document.getElementById("spam-bar").style.width = data.confidence.spam + "%";

    document.getElementById("analyzed-message").textContent = data.message;

    // -------- MODEL ANALYSIS --------
    const analysisBox = document.getElementById("analysis-box");
    const wordsList = document.getElementById("analysis-words");

    wordsList.innerHTML = "";

    if (data.analysis && data.analysis.important_words.length > 0) {
        data.analysis.important_words.forEach(word => {
            const span = document.createElement("span");
            span.className = "analysis-word";
            span.textContent = word;
            wordsList.appendChild(span);
        });
        analysisBox.style.display = "block";
    } else {
        analysisBox.style.display = "none";
    }
}

// Load stats
async function loadStats() {
    const res = await fetch("/stats");
    const data = await res.json();

    if (data.success) {
        document.getElementById("best-model").textContent = data.best_model;
        document.getElementById("best-accuracy").textContent = data.best_accuracy;
        document.getElementById("total-messages").textContent = data.dataset_info.total_messages;
    }
}

document.addEventListener("DOMContentLoaded", loadStats);
