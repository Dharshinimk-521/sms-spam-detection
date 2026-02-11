document.addEventListener("DOMContentLoaded", () => {
    loadStats();
});

function loadStats() {
    fetch("/stats")
        .then(res => res.json())
        .then(data => {
            if (!data.success) return;

            document.getElementById("best-model").innerText = data.best_model;
            document.getElementById("best-accuracy").innerText = data.best_accuracy + "%";
            document.getElementById("total-messages").innerText =
                data.dataset_info.total_messages;
        })
        .catch(err => console.error("Stats error:", err));
}

function analyzeMessage() {
    const message = document.getElementById("message-input").value;

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message })
    })
        .then(res => res.json())
        .then(data => {
            if (!data.success) {
                alert(data.error);
                return;
            }

            document.getElementById("result-text").innerText =
                `${data.prediction} (${data.confidence}%)`;

            const analysisBox = document.getElementById("analysis-words");
            analysisBox.innerHTML = "";

            data.analysis.important_words.forEach(word => {
                const span = document.createElement("span");
                span.className = "chip";
                span.innerText = word;
                analysisBox.appendChild(span);
            });
        })
        .catch(err => console.error("Prediction error:", err));
}
