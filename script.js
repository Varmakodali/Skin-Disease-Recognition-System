const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileNameDisplay = document.getElementById('file-name');
const analyzeBtn = document.getElementById('analyze-btn');
const captureBtn = document.getElementById('capture-btn');
const resultsSection = document.getElementById('results');
const loader = document.getElementById('loader');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const uploadView = document.getElementById('upload-view');
const cameraView = document.getElementById('camera-view');

let selectedFile = null;
let stream = null;

// Tab Switching
async function switchTab(type) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    if (type === 'camera') {
        uploadView.classList.add('hidden');
        cameraView.classList.remove('hidden');
        analyzeBtn.classList.add('hidden');
        captureBtn.classList.remove('hidden');
        await startCamera();
    } else {
        uploadView.classList.remove('hidden');
        cameraView.classList.add('hidden');
        analyzeBtn.classList.remove('hidden');
        captureBtn.classList.add('hidden');
        stopCamera();
    }
}

async function startCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        video.srcObject = stream;
    } catch (err) {
        alert("Camera access denied or not available.");
        switchTab('upload');
    }
}

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
}

// File Selection
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    handleFile(e.target.files[0]);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragging');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (file && file.type.startsWith('image/')) {
        selectedFile = file;
        fileNameDisplay.textContent = `Selected: ${file.name}`;
        analyzeBtn.disabled = false;

        // Show original preview immediately
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('original-preview').src = e.target.result;
        };
        reader.readAsDataURL(file);
    }
}

// Analysis
analyzeBtn.addEventListener('click', () => runAnalysis(selectedFile));

captureBtn.addEventListener('click', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    canvas.toBlob((blob) => {
        const file = new File([blob], "capture.jpg", { type: "image/jpeg" });

        // Show the captured frame in the "Original Image" preview
        const reader = new FileReader();
        reader.onload = (e) => {
            document.getElementById('original-preview').src = e.target.result;
        };
        reader.readAsDataURL(file);

        runAnalysis(file);
    }, 'image/jpeg');
});

async function runAnalysis(file) {
    if (!file) return;

    loader.classList.remove('hidden');

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        alert('Error analyzing image. Please verify the server is running.');
        console.error(err);
    } finally {
        loader.classList.add('hidden');
    }
}

function displayResults(data) {
    // Hide upload, show results
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Update main result
    document.getElementById('top-prediction').textContent = data.prediction;
    document.getElementById('top-confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;

    // Update heatmap
    document.getElementById('heatmap-preview').src = data.heatmap;

    // Update probability list
    const probList = document.getElementById('prob-list');
    probList.innerHTML = '';

    // Sort probabilities by confidence
    const sortedProbs = Object.entries(data.all_probabilities)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 5); // Show top 5

    sortedProbs.forEach(([name, prob]) => {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
            <div class="prob-header">
                <span>${name}</span>
                <span>${(prob * 100).toFixed(1)}%</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-bar" style="width: 0%"></div>
            </div>
        `;
        probList.appendChild(item);

        // Animate the bar
        setTimeout(() => {
            item.querySelector('.prob-bar').style.width = `${prob * 100}%`;
        }, 100);
    });

    // Update Medical Report
    document.getElementById('disease-causes').textContent = data.details.causes;

    const precautionsList = document.getElementById('disease-precautions');
    precautionsList.innerHTML = '';
    data.details.precautions.forEach(text => {
        const li = document.createElement('li');
        li.textContent = text;
        precautionsList.appendChild(li);
    });


}

function resetUI() {
    resultsSection.classList.add('hidden');
    fileNameDisplay.textContent = '';
    analyzeBtn.disabled = true;
    selectedFile = null;
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
