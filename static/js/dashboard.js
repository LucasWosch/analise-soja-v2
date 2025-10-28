const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const trainBtn = document.getElementById("trainBtn");
const predictBtn = document.getElementById("predictBtn");
const uploadStatus = document.getElementById("uploadStatus");
const dashboard = document.getElementById("dashboard");

uploadBtn.onclick = async () => {
  const file = document.getElementById("csvfile").files[0];
  if (!file) return alert("Selecione um arquivo CSV!");
  const formData = new FormData();
  formData.append("file", file);
  uploadStatus.textContent = "⏳ Enviando arquivo...";

  const res = await fetch("/upload_csv", { method: "POST", body: formData });
  const json = await res.json();
  uploadStatus.textContent = `✅ ${json.rows_saved} registros importados.`;
  renderDashboard(json);
};

analyzeBtn.onclick = async () => {
  const res = await fetch("/analyze", { method: "POST" });
  const json = await res.json();
  renderDashboard(json);
};

trainBtn.onclick = async () => {
  const res = await fetch("/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target: "yield_kg_ha" }),
  });
  const json = await res.json();
  alert(`Modelo treinado!\nR²: ${json.metrics.r2.toFixed(3)}\nMAE: ${json.metrics.mae.toFixed(2)}`);
};

predictBtn.onclick = async () => {
  const form = document.getElementById("predictForm");
  const record = {};
  [...form.elements].forEach(el => { if (el.name && el.value) record[el.name] = el.value; });

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ record }),
  });
  const json = await res.json();
  document.getElementById("predictResult").textContent =
    `🌾 Previsão: ${json.prediction.toFixed(2)} kg/ha`;
};

// Renderiza dashboard
function renderDashboard(json) {
  dashboard.classList.remove("hidden");
  const summary = document.getElementById("summary");
  const figs = document.getElementById("figures");
  summary.innerHTML = "";
  figs.innerHTML = "";

  if (json.summary) {
    for (const [k, v] of Object.entries(json.summary)) {
      summary.innerHTML += `
        <div class="bg-gray-800/80 p-4 rounded-lg text-center border border-gray-700">
          <p class="text-gray-400 text-sm">${k}</p>
          <p class="text-cyan-400 font-semibold text-lg">${v}</p>
        </div>`;
    }
  }

  if (json.images) {
    for (const [name, b64] of Object.entries(json.images)) {
      figs.innerHTML += `
        <div class="text-center">
          <img src="data:image/png;base64,${b64}" class="rounded-lg mx-auto shadow-md"/>
          <p class="text-gray-400 mt-2">${name.replace(/_/g, " ")}</p>
        </div>`;
    }
  }
}
