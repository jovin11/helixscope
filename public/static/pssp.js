
function makePrediction() {
    const sequence = document.getElementById('sequenceInput').value;

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input_data: sequence  // Send as a single string
        }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('predictionResult').innerText = `Prediction: ${data.prediction}`;
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('predictionResult').innerText = `Error: Could not get prediction.`;
    });
}