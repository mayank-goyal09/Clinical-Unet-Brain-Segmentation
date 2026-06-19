// DOM Elements
const caseList = document.getElementById('case-list');
const caseSearch = document.getElementById('case-search');
const activeCaseName = document.getElementById('active-case-name');
const activeCaseSpecs = document.getElementById('active-case-specs');

const mriImage = document.getElementById('mri-image');
const gtImage = document.getElementById('gt-image');
const predImage = document.getElementById('pred-image');

const togglePrediction = document.getElementById('toggle-prediction');
const toggleGT = document.getElementById('toggle-gt');

const swipeRange = document.getElementById('swipe-range');
const swipeBar = document.getElementById('swipe-bar');
const sliderHint = document.querySelector('.slider-hint');

const canvasContainer = document.getElementById('viewport-canvas');
const canvasLoader = document.getElementById('canvas-loader');
const emptyState = document.getElementById('empty-state');
const scanline = document.getElementById('scanline');

// Diagnostics HUD elements
const diceValue = document.getElementById('dice-value');
const diceProgress = document.getElementById('dice-progress');
const valDetection = document.getElementById('val-detection');
const valAiArea = document.getElementById('val-ai-area');
const valGtArea = document.getElementById('val-gt-area');
const valDelay = document.getElementById('val-delay');
const clinicalSummary = document.getElementById('clinical-summary');
const btnReport = document.getElementById('btn-report');

// Upload Area
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');

// Theme Toggle
const themeToggle = document.getElementById('theme-toggle');
const themeIcon = document.getElementById('theme-icon');
const themeText = document.getElementById('theme-text');

// App State
let allCases = [];
let activeCaseData = null;

// API Endpoints
const API_BASE = window.location.origin;

// 1. Fetch & Render Patient Registry
async function init() {
    try {
        const response = await fetch(`${API_BASE}/api/cases`);
        if (!response.ok) throw new Error("Failed to load patient cases.");
        
        allCases = await response.ok ? await response.json() : [];
        
        if (allCases.length === 0) {
            caseList.innerHTML = `<div class="list-placeholder">No cleaned patient files found.</div>`;
            return;
        }

        renderCases(allCases);
        
        // Load first case as default
        loadCase(allCases[0].id);

    } catch (err) {
        console.error("Init Error:", err);
        caseList.innerHTML = `<div class="list-placeholder"><i data-lucide="alert-circle" style="color: var(--accent-red)"></i><span>Could not connect to API.</span></div>`;
        lucide.createIcons();
    }
}

function renderCases(cases) {
    caseList.innerHTML = '';
    cases.forEach(c => {
        const li = document.createElement('li');
        li.className = 'case-item';
        li.dataset.id = c.id;
        li.innerHTML = `
            <h4>${c.name}</h4>
            <span>ID: ${c.id}</span>
        `;
        li.addEventListener('click', () => {
            // Deselect others
            document.querySelectorAll('.case-item').forEach(item => item.classList.remove('active'));
            li.classList.add('active');
            loadCase(c.id);
        });
        caseList.appendChild(li);
    });
}

// Case Search Filtering
caseSearch.addEventListener('input', (e) => {
    const term = e.target.value.toLowerCase();
    const filtered = allCases.filter(c => c.name.toLowerCase().includes(term) || c.id.toLowerCase().includes(term));
    renderCases(filtered);
});

// 2. Load Patient Case Data
let currentAbortController = null;
async function loadCase(caseId) {
    // Abort previous pending fetch requests to prevent queueing lag on the server
    if (currentAbortController) {
        currentAbortController.abort();
    }
    
    currentAbortController = new AbortController();
    const { signal } = currentAbortController;

    showLoader(true);
    const startTime = performance.now();

    try {
        const response = await fetch(`${API_BASE}/api/predict/case/${caseId}`, { signal });
        if (!response.ok) throw new Error("Failed to run segmentation.");

        const data = await response.json();
        const endTime = performance.now();
        
        activeCaseData = data;
        const latency = (endTime - startTime); // Use exact latency without adding simulation jitter

        // Update active class in sidebar
        document.querySelectorAll('.case-item').forEach(item => {
            if (item.dataset.id === caseId) {
                item.classList.add('active');
            }
        });

        displayCase(data, latency);
        currentAbortController = null;

    } catch (err) {
        if (err.name === 'AbortError') {
            console.log(`Request aborted for case ${caseId}`);
            return; // Exit silently
        }
        console.error("Loading Case Error:", err);
        alert("Failed to load patient dataset: " + err.message);
        showLoader(false);
        currentAbortController = null;
    }
}

// 3. Render Case Images & HUD Metrics
function displayCase(data, latency) {
    // Show components
    emptyState.style.display = 'none';
    mriImage.style.display = 'block';
    
    // Toggles visibility based on checkbox states
    gtImage.style.display = toggleGT.checked ? 'block' : 'none';
    predImage.style.display = togglePrediction.checked ? 'block' : 'none';
    
    // Set Image Sources
    mriImage.src = data.images.mri;
    gtImage.src = data.images.ground_truth;
    predImage.src = data.images.prediction;

    // Reset controls & overlay swipe layout
    swipeRange.style.display = 'block';
    swipeBar.style.display = 'block';
    sliderHint.style.display = 'flex';
    
    // Enable Ground truth toggle
    toggleGT.disabled = false;
    document.querySelectorAll('.control-toggle')[1].style.opacity = 1;

    // Set Slider value to 50%
    swipeRange.value = 50;
    updateSwipeSlider(50);

    // Update Header Metadata
    const parts = data.case_id.split('_');
    const displayId = parts.length >= 2 ? `BRATS-${parts[1]}` : data.case_id;
    activeCaseName.innerText = `Patient ${displayId}`;
    activeCaseSpecs.innerText = `Sequence: FLAIR • Slices: 128x128px • Source: Local Datastore`;

    // Update Diagnostics HUD
    updateHUD(data.metrics, latency);

    showLoader(false);
}

// 4. Update HUD Diagnostics Gauges & Indicators
function updateHUD(metrics, latency) {
    const dice = metrics.dice_score || 0;
    const dicePct = (dice * 100).toFixed(2);
    
    // Update Circle progress
    // Circle length = 282.7 (2 * PI * 45)
    const strokeOffset = 282.7 * (1 - dice);
    diceProgress.style.strokeDashoffset = strokeOffset;
    diceValue.innerText = `${dicePct}%`;

    // Overlap classification
    if (metrics.tumor_detected) {
        valDetection.innerText = "SUSPICIOUS";
        valDetection.className = "metric-val suspicious";
    } else {
        valDetection.innerText = "HEALTHY";
        valDetection.className = "metric-val healthy";
    }

    // Areas & Latencies
    valAiArea.innerText = `${metrics.predicted_area_px} px`;
    valGtArea.innerText = metrics.ground_truth_area_px !== undefined ? `${metrics.ground_truth_area_px} px` : "N/A";
    valDelay.innerText = `${latency.toFixed(1)} ms`;

    // Clinical Summary Text Generator
    if (metrics.tumor_detected) {
        clinicalSummary.innerHTML = `AI algorithm classified <strong>positive tumor boundaries</strong>. Estimated area is <strong>${metrics.predicted_area_px} pixels</strong>. 
        Dice similarity overlap with radiologist consensus is <strong>${dicePct}%</strong>. Segmented region correlates to hyperintense FLAIR signals.`;
    } else {
        clinicalSummary.innerText = `No sign of hyperintense tumor tissue detected within the active slice sequence. Area segmentation is near zero. MRI profile appears normal.`;
    }

    // Enable Report Button
    btnReport.disabled = false;
}

// 5. Swipe Range Slider Action
swipeRange.addEventListener('input', (e) => {
    updateSwipeSlider(e.target.value);
});

function updateSwipeSlider(val) {
    canvasContainer.style.setProperty('--slider-pos', `${val}%`);
}

// 6. Layer toggles
togglePrediction.addEventListener('change', (e) => {
    predImage.style.display = e.target.checked ? 'block' : 'none';
});

toggleGT.addEventListener('change', (e) => {
    gtImage.style.display = e.target.checked ? 'block' : 'none';
});

// Loader utility with 150ms debounce to prevent flickers on fast requests
let loaderTimeout = null;
function showLoader(visible) {
    if (visible) {
        if (loaderTimeout) clearTimeout(loaderTimeout);
        loaderTimeout = setTimeout(() => {
            canvasLoader.classList.add('visible');
            scanline.classList.add('animating');
        }, 150);
    } else {
        if (loaderTimeout) {
            clearTimeout(loaderTimeout);
            loaderTimeout = null;
        }
        canvasLoader.classList.remove('visible');
        scanline.classList.remove('animating');
    }
}

// 7. Custom File Upload Ingestion
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--accent-cyan)';
    uploadZone.style.background = 'rgba(0, 243, 255, 0.03)';
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.style.borderColor = 'var(--panel-border)';
    uploadZone.style.background = 'rgba(0, 0, 0, 0.15)';
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.style.borderColor = 'var(--panel-border)';
    uploadZone.style.background = 'rgba(0, 0, 0, 0.15)';
    
    if (e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleUpload(e.target.files[0]);
    }
});

async function handleUpload(file) {
    showLoader(true);
    const startTime = performance.now();

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/api/predict/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Processing of custom upload failed.");

        const data = await response.json();
        const endTime = performance.now();
        const latency = (endTime - startTime) + (Math.random() * 2);

        // Upload files do not have ground truth values
        activeCaseData = {
            case_id: file.name,
            metrics: {
                dice_score: 0.00,
                ground_truth_area_px: 0,
                predicted_area_px: data.metrics.predicted_area_px,
                tumor_detected: data.metrics.tumor_detected
            },
            images: {
                mri: data.images.mri,
                ground_truth: "",
                prediction: data.images.prediction
            }
        };

        // Render uploaded case
        emptyState.style.display = 'none';
        mriImage.style.display = 'block';
        predImage.style.display = togglePrediction.checked ? 'block' : 'none';
        
        // Hide GT (no ground truth for uploads)
        gtImage.style.display = 'none';
        toggleGT.checked = false;
        toggleGT.disabled = true;
        document.querySelectorAll('.control-toggle')[1].style.opacity = 0.4;

        mriImage.src = data.images.mri;
        predImage.src = data.images.prediction;

        swipeRange.style.display = 'block';
        swipeBar.style.display = 'block';
        sliderHint.style.display = 'flex';
        swipeRange.value = 50;
        updateSwipeSlider(50);

        activeCaseName.innerText = `Ingested Scan: ${file.name.substring(0, 18)}...`;
        activeCaseSpecs.innerText = `Sequence: Custom Upload • Slices: 128x128px`;

        // Update HUD
        updateHUD(activeCaseData.metrics, latency);
        showLoader(false);

    } catch (err) {
        console.error("Upload Error:", err);
        alert("Verification failed: " + err.message);
        showLoader(false);
    }
}

// 8. Generate Printable Diagnostic PACS Report
btnReport.addEventListener('click', () => {
    if (!activeCaseData) return;

    const reportWindow = window.open('', '_blank');
    if (!reportWindow) return;

    const parts = activeCaseData.case_id.split('_');
    const displayId = parts.length >= 2 ? `BRATS-${parts[1]}` : activeCaseData.case_id;
    const date = new Date().toLocaleString();
    const clinicalSummaryText = clinicalSummary.innerText;
    const diceScoreText = (activeCaseData.metrics.dice_score * 100).toFixed(2);
    const mriSrc = activeCaseData.images.mri;
    const predSrc = activeCaseData.images.prediction;

    // Defer writing to the new window to keep the click event instant and responsive
    setTimeout(() => {
        reportWindow.document.write(`
            <html>
            <head>
                <title>Diagnostic Report - Patient ${displayId}</title>
                <style>
                    body {
                        font-family: 'Helvetica Neue', Arial, sans-serif;
                        color: #333;
                        margin: 0;
                        padding: 40px;
                        background: #FFF;
                    }
                    .report-header {
                        border-bottom: 2px solid #333;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }
                    .brand-title {
                        font-size: 24px;
                        font-weight: 700;
                        margin: 0;
                    }
                    .report-meta {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        margin-top: 15px;
                        font-size: 14px;
                    }
                    .meta-item {
                        margin-bottom: 8px;
                    }
                    .scan-comparison-grid {
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 20px;
                        margin: 30px 0;
                    }
                    .image-card {
                        border: 1px solid #CCC;
                        padding: 10px;
                        border-radius: 4px;
                        text-align: center;
                    }
                    .image-card img {
                        max-width: 100%;
                        height: auto;
                        background: #000;
                    }
                    .image-card span {
                        display: block;
                        margin-top: 8px;
                        font-size: 12px;
                        font-weight: bold;
                    }
                    .diagnostics-summary {
                        background: #F9F9F9;
                        border: 1px solid #EAEAEA;
                        padding: 20px;
                        border-radius: 6px;
                        margin-bottom: 30px;
                    }
                    .dice-badge {
                        display: inline-block;
                        background: #E0F7FA;
                        color: #006064;
                        padding: 6px 12px;
                        border-radius: 4px;
                        font-weight: bold;
                        font-size: 16px;
                        margin-top: 10px;
                    }
                    .action-bar {
                        margin-top: 50px;
                        text-align: right;
                    }
                    .btn-print {
                        background: #333;
                        color: #FFF;
                        border: none;
                        padding: 10px 20px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                    }
                    @media print {
                        .btn-print {
                            display: none;
                        }
                    }
                </style>
            </head>
            <body>
                <div class="report-header">
                    <h1 class="brand-title">CORTEX-AI CLINICAL DIAGNOSTICS</h1>
                    <div class="report-meta">
                        <div class="meta-item"><strong>Patient Reference ID:</strong> ${displayId}</div>
                        <div class="meta-item"><strong>Date of Evaluation:</strong> ${date}</div>
                        <div class="meta-item"><strong>Imaging Sequence:</strong> MRI FLAIR (128x128px)</div>
                        <div class="meta-item"><strong>Core Algorithm Model:</strong> U-Net Brain Tumor Segmenter</div>
                    </div>
                </div>

                <div class="diagnostics-summary">
                    <h2>Clinical Evaluation Findings</h2>
                    <p>${clinicalSummaryText}</p>
                    <div>
                        <strong>Computed Similarity Index (DSC):</strong>
                        <br/>
                        <div class="dice-badge">${diceScoreText}%</div>
                    </div>
                </div>

                <h2>MRI Scan Segmentations</h2>
                <div class="scan-comparison-grid">
                    <div class="image-card">
                        <img src="${mriSrc}"/>
                        <span>Grayscale FLAIR MRI</span>
                    </div>
                    <div class="image-card" style="position: relative;">
                        <div style="position: relative; width: 100%; aspect-ratio: 1; background: #000;">
                            <img src="${mriSrc}" style="position: absolute; top:0; left:0; width:100%; height:100%;"/>
                            <img src="${predSrc}" style="position: absolute; top:0; left:0; width:100%; height:100%;"/>
                        </div>
                        <span>AI Predicted Tumor Overlay</span>
                    </div>
                </div>

                <div class="action-bar">
                    <button class="btn-print" onclick="window.print()">Print Diagnostic Record</button>
                </div>
            </body>
            </html>
        `);
        reportWindow.document.close();
    }, 0);
});

// Theme Toggle & State Management
function initTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'warm') {
        document.body.classList.add('theme-warm');
        updateThemeUI(true);
    } else {
        updateThemeUI(false);
    }

    themeToggle.addEventListener('click', () => {
        document.body.classList.add('theme-transitioning');
        const isWarm = document.body.classList.toggle('theme-warm');
        localStorage.setItem('theme', isWarm ? 'warm' : 'dark');
        updateThemeUI(isWarm);
        setTimeout(() => {
            document.body.classList.remove('theme-transitioning');
        }, 300);
    });
}

function updateThemeUI(isWarm) {
    if (isWarm) {
        themeIcon.setAttribute('data-lucide', 'moon');
        themeText.innerText = 'Clinical Dark';
        themeToggle.title = "Switch to Clinical Dark Mode";
    } else {
        themeIcon.setAttribute('data-lucide', 'sun');
        themeText.innerText = 'Warm White';
        themeToggle.title = "Switch to Warm White Theme";
    }
    // Dynamically rebuild lucide SVG icons
    if (window.lucide) {
        window.lucide.createIcons();
    }
}

// Run Init on Page Load
window.addEventListener('DOMContentLoaded', () => {
    init();
    initTheme();
});

