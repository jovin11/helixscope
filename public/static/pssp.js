const form = document.querySelector("#predictionForm");
const sequenceInput = document.querySelector("#sequenceInput");
const sequenceCount = document.querySelector("#sequenceCount");
const predictButton = document.querySelector("#predictButton");
const exampleButton = document.querySelector("#exampleButton");
const clearButton = document.querySelector("#clearButton");
const copyButton = document.querySelector("#copyButton");
const formError = document.querySelector("#formError");
const emptyState = document.querySelector("#emptyState");
const loadingState = document.querySelector("#loadingState");
const predictionOutput = document.querySelector("#predictionOutput");
const metrics = document.querySelector("#metrics");
const structureTrack = document.querySelector("#structureTrack");
const sequenceAlignment = document.querySelector("#sequenceAlignment");

const EXAMPLE_SEQUENCE = "MKTIIALSYIFCLVFADYKDDDDK";
const STRUCTURE_NAMES = { H: "Helix", E: "Sheet", C: "Coil" };

let latestPrediction = null;

function normalizedLength(value) {
    const lines = value.trim().split(/\r?\n/);
    if (lines[0]?.startsWith(">")) lines.shift();
    return lines.join("").replace(/\s/g, "").length;
}

function updateCount() {
    const maximum = sequenceCount.textContent.split("/")[1].trim();
    sequenceCount.textContent = `${normalizedLength(sequenceInput.value)} / ${maximum}`;
}

function setState(state) {
    emptyState.hidden = state !== "empty";
    loadingState.hidden = state !== "loading";
    predictionOutput.hidden = state !== "result";
    predictButton.disabled = state === "loading";
    predictButton.querySelector(".button-label").textContent =
        state === "loading" ? "Predicting…" : "Run prediction";
}

function showError(message) {
    formError.textContent = message;
    formError.hidden = false;
}

function clearError() {
    formError.textContent = "";
    formError.hidden = true;
}

function metric(label, value, suffix = "") {
    return `<div class="metric"><span>${label}</span><strong>${value}${suffix}</strong></div>`;
}

function renderAlignment(sequence, prediction, confidence) {
    sequenceAlignment.replaceChildren();
    const chunkSize = 50;

    for (let offset = 0; offset < sequence.length; offset += chunkSize) {
        const chunk = document.createElement("div");
        chunk.className = "alignment-chunk";

        const end = Math.min(offset + chunkSize, sequence.length);
        const sequenceSlice = sequence.slice(offset, end);
        const predictionSlice = prediction.slice(offset, end);
        const confidenceSlice = confidence.slice(offset, end);

        chunk.innerHTML = `
            <div class="position-row"><span>${offset + 1}</span><span>${end}</span></div>
            <div class="alignment-row"><span class="row-label">SEQ</span><code>${sequenceSlice}</code></div>
            <div class="alignment-row"><span class="row-label">PRED</span><code class="prediction-code"></code></div>
        `;

        const predictionCode = chunk.querySelector(".prediction-code");
        [...predictionSlice].forEach((structure, index) => {
            const residue = document.createElement("span");
            residue.className = `residue residue-${structure.toLowerCase()}`;
            residue.textContent = structure;
            residue.title = `${STRUCTURE_NAMES[structure]} · ${Math.round(confidenceSlice[index] * 100)}% confidence`;
            predictionCode.append(residue);
        });
        sequenceAlignment.append(chunk);
    }
}

function renderResult(data) {
    latestPrediction = data;
    const { length, mean_confidence: meanConfidence, class_counts: counts } = data.summary;

    metrics.innerHTML = [
        metric("Residues", length),
        metric("Mean confidence", Math.round(meanConfidence * 100), "%"),
        metric("Helix", counts.helix),
        metric("Sheet", counts.sheet),
        metric("Coil", counts.coil),
    ].join("");

    structureTrack.replaceChildren();
    [...data.prediction].forEach((structure, index) => {
        const segment = document.createElement("span");
        segment.className = `track-residue residue-${structure.toLowerCase()}`;
        segment.style.opacity = String(0.45 + data.confidence[index] * 0.55);
        segment.title = `${index + 1}: ${STRUCTURE_NAMES[structure]} (${Math.round(data.confidence[index] * 100)}%)`;
        structureTrack.append(segment);
    });

    renderAlignment(data.sequence, data.prediction, data.confidence);
    copyButton.hidden = false;
    setState("result");
}

async function makePrediction(event) {
    event.preventDefault();
    clearError();

    if (!sequenceInput.value.trim()) {
        showError("Enter an amino-acid sequence first.");
        sequenceInput.focus();
        return;
    }

    setState("loading");
    try {
        const response = await fetch("/api/v1/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sequence: sequenceInput.value }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Prediction request failed.");
        renderResult(data);
    } catch (error) {
        showError(error.message || "Prediction failed. Please try again.");
        setState(latestPrediction ? "result" : "empty");
    }
}

function clearPrediction() {
    sequenceInput.value = "";
    latestPrediction = null;
    copyButton.hidden = true;
    clearError();
    updateCount();
    setState("empty");
    sequenceInput.focus();
}

form.addEventListener("submit", makePrediction);
sequenceInput.addEventListener("input", () => {
    clearError();
    updateCount();
});
exampleButton.addEventListener("click", () => {
    sequenceInput.value = EXAMPLE_SEQUENCE;
    updateCount();
    clearError();
    sequenceInput.focus();
});
clearButton.addEventListener("click", clearPrediction);
copyButton.addEventListener("click", async () => {
    if (!latestPrediction) return;
    const text = `Sequence\n${latestPrediction.sequence}\n\nPrediction\n${latestPrediction.prediction}`;
    await navigator.clipboard.writeText(text);
    copyButton.textContent = "Copied";
    setTimeout(() => { copyButton.textContent = "Copy result"; }, 1400);
});

updateCount();
