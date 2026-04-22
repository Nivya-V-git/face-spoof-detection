console.log("script.js loaded");

async function updateStatus() {
    try {
        const response = await fetch('/get_status');
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();

        document.getElementById('status').innerText = data.label || "Unknown";
        document.getElementById('confidence').innerText = (data.confidence * 100).toFixed(1) + "%";
        document.getElementById('suggestion').innerText = data.suggestion || "No suggestion";

        const statusEl = document.getElementById('status');
        if (data.label === "REAL") statusEl.style.color = "#4ade80";
        else if (data.label === "SPOOF") statusEl.style.color = "#f87171";
        else statusEl.style.color = "#facc15";
    } catch (err) {
        console.error('Fetch error:', err);
    }
}

setInterval(updateStatus, 500);
updateStatus();