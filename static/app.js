/* ================================================================
   Sentio — Voice Sentiment Analyzer (frontend)
   ================================================================ */

// Emotion color palette (vibrant for dark theme)
const COLORS = {
  joy:        "#fbbf24",
  anger:      "#f87171",
  sadness:    "#60a5fa",
  fear:       "#a78bfa",
  surprise:   "#fb923c",
  disgust:    "#34d399",
  neutral:    "#9ca3af",
  excitement: "#f472b6",
};

// Ordered bottom → top (negative → positive) for the Y-axis
const EMOTION_ORDER = [
  "anger", "disgust", "fear", "sadness", "neutral", "surprise", "excitement", "joy",
];

// ── DOM refs ──────────────────────────────────────────────────────
const dropZone   = document.getElementById("drop-zone");
const fileInput  = document.getElementById("file-input");
const fileInfo   = document.getElementById("file-info");
const fileName   = document.getElementById("file-name");
const fileSize   = document.getElementById("file-size");
const removeBtn  = document.getElementById("remove-file");
const analyzeBtn = document.getElementById("analyze-btn");
const uploadSec  = document.getElementById("upload-section");
const loadingSec = document.getElementById("loading");
const errorBan   = document.getElementById("error-banner");
const resultsSec = document.getElementById("results");

let selectedFile = null;
let timelineChart = null;
let distChart     = null;

// ── Helpers ───────────────────────────────────────────────────────
function fmt(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function humanSize(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function show(el) { el.classList.remove("hidden"); }
function hide(el) { el.classList.add("hidden"); }

// ── File Selection ────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) selectFile(fileInput.files[0]);
});

removeBtn.addEventListener("click", clearFile);

function selectFile(file) {
  selectedFile = file;
  fileName.textContent = file.name;
  fileSize.textContent = humanSize(file.size);
  show(fileInfo);
  analyzeBtn.disabled = false;
  hide(errorBan);
  hide(resultsSec);
}

function clearFile() {
  selectedFile = null;
  fileInput.value = "";
  hide(fileInfo);
  analyzeBtn.disabled = true;
}

// ── Analyze ───────────────────────────────────────────────────────
analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  hide(errorBan);
  hide(resultsSec);
  hide(uploadSec);
  show(loadingSec);

  const form = new FormData();
  form.append("audio", selectedFile);

  try {
    const res  = await fetch("/analyze", { method: "POST", body: form });
    const data = await res.json();

    hide(loadingSec);
    show(uploadSec);

    if (!res.ok || data.error) {
      showError(data.error || "Analysis failed.");
      return;
    }

    renderResults(data);
  } catch (err) {
    hide(loadingSec);
    show(uploadSec);
    showError("Network error — is the server running?");
  }
});

function showError(msg) {
  errorBan.textContent = msg;
  show(errorBan);
}

// ── Render Results ────────────────────────────────────────────────
function renderResults(data) {
  const { segments, changes, distribution, duration } = data;

  document.getElementById("stat-duration").textContent = fmt(duration);
  document.getElementById("stat-segments").textContent = segments.length;
  document.getElementById("stat-changes").textContent  = changes.length;

  const dominant = Object.entries(distribution).sort((a, b) => b[1] - a[1])[0];
  const domEl = document.getElementById("stat-dominant");
  domEl.textContent = dominant ? dominant[0] : "—";
  if (dominant) domEl.style.color = COLORS[dominant[0]] || "#e5e7eb";

  renderTimeline(segments, duration);
  renderDistribution(distribution);
  renderChanges(changes);
  renderTranscript(segments);

  show(resultsSec);
}

// ── Timeline Chart (stepped line) ─────────────────────────────────
function renderTimeline(segments, duration) {
  const ctx = document.getElementById("timeline-chart");

  const points = segments.map((s) => ({
    x: s.start,
    y: EMOTION_ORDER.indexOf(s.emotion),
  }));

  if (segments.length) {
    const last = segments[segments.length - 1];
    points.push({ x: last.end, y: EMOTION_ORDER.indexOf(last.emotion) });
  }

  const segColors = segments.map((s) => COLORS[s.emotion]);
  const ptColors = [
    ...segments.map((s) => COLORS[s.emotion]),
    segments.length ? COLORS[segments[segments.length - 1].emotion] : "#6b7280",
  ];

  if (timelineChart) timelineChart.destroy();

  timelineChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [{
        data: points,
        stepped: "before",
        borderWidth: 2.5,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBackgroundColor: ptColors,
        pointBorderColor: ptColors,
        fill: false,
        segment: {
          borderColor: (c) => segColors[c.p0DataIndex] || "#6b7280",
        },
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        x: {
          type: "linear",
          min: 0,
          max: duration,
          title: { display: true, text: "Time", font: { size: 11 }, color: "#6b7280" },
          ticks: {
            callback: (v) => fmt(v),
            font: { size: 10 },
            color: "#6b7280",
            maxTicksLimit: 12,
          },
          grid: { color: "rgba(255,255,255,0.06)" },
        },
        y: {
          min: -0.5,
          max: EMOTION_ORDER.length - 0.5,
          ticks: {
            stepSize: 1,
            callback: (v) => EMOTION_ORDER[v] || "",
            font: { size: 11 },
            color: (c) => COLORS[EMOTION_ORDER[c.tick.value]] || "#6b7280",
          },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(15,15,30,0.9)",
          titleColor: "#e5e7eb",
          bodyColor: "#d1d5db",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            title: () => "",
            label: (c) => {
              const idx = c.dataIndex;
              const seg = segments[Math.min(idx, segments.length - 1)];
              const snippet = seg.text.length > 50 ? seg.text.slice(0, 50) + "…" : seg.text;
              return [
                `${fmt(seg.start)} — ${seg.emotion} (${(seg.confidence * 100).toFixed(0)}%)`,
                snippet,
              ];
            },
          },
        },
      },
    },
  });
}

// ── Distribution Doughnut ─────────────────────────────────────────
function renderDistribution(distribution) {
  const ctx = document.getElementById("distribution-chart");
  const labels = Object.keys(distribution);
  const values = Object.values(distribution);
  const colors = labels.map((l) => COLORS[l] || "#6b7280");

  if (distChart) distChart.destroy();

  distChart = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderWidth: 2,
        borderColor: "rgba(10,10,26,0.6)",
      }],
    },
    options: {
      responsive: true,
      cutout: "62%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { font: { size: 11 }, padding: 12, usePointStyle: true, color: "#9ca3af" },
        },
        tooltip: {
          backgroundColor: "rgba(15,15,30,0.9)",
          titleColor: "#e5e7eb",
          bodyColor: "#d1d5db",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          callbacks: {
            label: (c) => `${c.label}: ${c.parsed}%`,
          },
        },
      },
    },
  });
}

// ── Emotion Changes List ──────────────────────────────────────────
function renderChanges(changes) {
  const container = document.getElementById("changes-list");
  container.innerHTML = "";

  if (!changes.length) {
    container.innerHTML = '<p class="no-changes">No emotion changes detected</p>';
    return;
  }

  changes.forEach((ch) => {
    const div = document.createElement("div");
    div.className = "change-item";
    div.innerHTML = `
      <span class="change-time">${ch.time_fmt}</span>
      <span class="change-dot" style="background:${COLORS[ch.from]}"></span>
      <span class="change-label">${ch.from}</span>
      <span class="change-arrow">→</span>
      <span class="change-dot" style="background:${COLORS[ch.to]}"></span>
      <span class="change-label">${ch.to}</span>
    `;
    container.appendChild(div);
  });
}

// ── Transcript ────────────────────────────────────────────────────
function renderTranscript(segments) {
  const container = document.getElementById("transcript");
  container.innerHTML = "";

  segments.forEach((seg) => {
    const div = document.createElement("div");
    div.className = "transcript-seg";
    div.innerHTML = `
      <span class="seg-time">${fmt(seg.start)}</span>
      <span class="seg-badge emo-${seg.emotion}">${seg.emotion}</span>
      <span class="seg-text">${escHtml(seg.text)}</span>
    `;
    container.appendChild(div);
  });
}

function escHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}
