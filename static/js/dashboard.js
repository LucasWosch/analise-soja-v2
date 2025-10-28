// Detecta se está sendo servido via FastAPI ou aberto direto
const API = window.location.origin.includes("localhost") || window.location.protocol === "file:"
  ? "http://localhost:8000"
  : "";

// ==================== UPLOAD ====================
async function upload() {
  const f = document.getElementById("file").files[0];
  if (!f) {
    alert("Selecione um arquivo CSV.");
    return;
  }

  const fd = new FormData();
  fd.append("file", f);

  document.getElementById("uploadStatus").textContent = "⏳ Enviando arquivo...";

  const res = await fetch(API + "/upload_csv", { method: "POST", body: fd });
  const data = await res.json();

  if (!res.ok) {
    alert(data.detail || "Erro no upload");
    return;
  }

  document.getElementById("uploadStatus").textContent = `✅ ${data.rows_saved} registros importados.`;
  renderSummary(data.summary);
  renderCharts(data.images);
}

// ==================== TRAIN ====================
async function train() {
  const target = document.getElementById("target").value;
  const model = document.getElementById("model").value;

  const res = await fetch(API + "/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: target, model_type: model }),
  });

  const data = await res.json();
  document.getElementById("trainout").textContent = JSON.stringify(data, null, 2);
}

// ==================== RETRAIN ====================
async function retrain() {
  const target = document.getElementById("target").value;
  const model = document.getElementById("model").value;

  const res = await fetch(API + "/retrain", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: target, model_type: model }),
  });

  const data = await res.json();
  document.getElementById("trainout").textContent = JSON.stringify(data, null, 2);
}

// ==================== RENDER ====================
function renderSummary(summary) {
  const dash = document.getElementById("dashboard");
  dash.classList.remove("hidden");

  const summaryDiv = document.getElementById("summary");
  summaryDiv.innerHTML = "";

  for (const [k, v] of Object.entries(summary)) {
    summaryDiv.innerHTML += `
      <div class="bg-gray-800/80 p-4 rounded-lg text-center border border-gray-700">
        <p class="text-gray-400 text-sm">${k}</p>
        <p class="text-cyan-400 font-semibold text-lg">${v}</p>
      </div>`;
  }
}

function renderCharts(images) {
  const dash = document.getElementById("dashboard");
  dash.classList.remove("hidden");

  const charts = document.getElementById("charts");
  charts.innerHTML = "";

  for (const [key, b64] of Object.entries(images)) {
    charts.innerHTML += `
      <div class="text-center">
        <img src="data:image/png;base64,${b64}" class="rounded-lg shadow-md mx-auto"/>
        <p class="text-gray-400 mt-2">${key.replace(/_/g, " ")}</p>
      </div>`;
  }
}

// ==================== EVENTOS ====================
document.getElementById("uploadBtn").onclick = upload;
document.getElementById("trainBtn").onclick = train;
document.getElementById("retrainBtn").onclick = retrain;
