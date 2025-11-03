# 🌱 Análise de Sementes — v1

API completa para **análise, exploração e previsão de produtividade agrícola** (soja, milho, trigo etc.)  
Construída com **FastAPI + SQLite + scikit‑learn** — recebe CSV, normaliza, gera gráficos analíticos e permite treinar modelo de regressão (Random Forest / Linear).

---

## 🚀 Funcionalidades principais

| Função | Status |
|---|---|
| Upload de CSV agrícola (colunas flexíveis) | ✅ |
| Normalização automática do dataset | ✅ |
| Geração de painéis gráficos (base64) | ✅ |
| Métricas estatísticas básicas | ✅ |
| Treinar modelo ML | ✅ |
| Salvar artefatos do modelo | ✅ |
| Predizer 1 registro via JSON | ✅ |
| CORS liberado (frontend externo pode consumir) | ✅ |
| Interface gráfica de dashboard | ✅ |
---

## 🧠 Tecnologias usadas

| Componente         | Versão |
|--------------------|--------|
| Python             | 3.12   |
| FastAPI            | 0.115  |
| Uvicorn            | 0.32   |
| Pandas             | 2.2    |
| NumPy              | 2.0    |
| Scikit‑Learn       | 1.5    |
| Joblib             | 1.4    |
| SQLite             | nativo |

> **⚠️ Recomendação:** não usar Python 3.13 ainda — stack científica ainda não está 100% estável, use 3.12.

---

## 📁 Estrutura final do projeto

```
.
├── main.py
├── static/                 # frontend (html/js) opcional
├── data/                   # CSV original (opcional)
├── models/                 # artefatos ML gerados na hora do treino
└── app/
|   ├── __init__.py
|   ├── routes/             # rotas FastAPI
|    │   ├── upload.py
|    │   ├── analyze.py
|    │   ├── train.py
|    │   └── predict.py
|    └── services/           # lógica de negócio
|        ├── analytics.py
|        ├── data_utils.py
|        └── models.py       # pipeline ML
├── static/
    └──index.html # página principal
    └── js/
    └── dashboard.js # lógica de interação do frontend

```

---

## 🔌 Como rodar

```powershell
# ativar ambiente virtual
.\.venv\Scripts\Activate

# instalar dependências
pip install -r requirements.txt

# subir API
uvicorn main:app --reload
```

---

## Endpoints relevantes

| Método | Rota             | Descrição |
|--------|-----------------|-----------|
| POST   | `/upload_csv`   | envia CSV e popula SQLite |
| POST   | `/analyze`      | gera figuras base64 + resumo estatístico |
| POST   | `/train`        | treina modelo ML e grava artefatos |
| POST   | `/predict`      | previsão individual via JSON |

Exemplo previsões:

```json
POST /predict
{
  "record": {
    "crop": "soybean",
    "year": 2022,
    "area": 550,
    "rain_mm": 1120
  }
}
```

---

## 📌 Observações importantes

- `models/` é criado automaticamente no 1º import do módulo de ML
- treino usa `RandomForestRegressor(300 árvores)` por padrão — com `verbose=1` para visualização didática
- qualquer frontend pode consumir — CORS = liberado
- Dashboard: A interface de visualização está disponível em static/index.html e pode ser acessada via navegador. O JavaScript (static/js/dashboard.js) lida com a interação com a API      para exibição de gráficos e dados analíticos.
---

## 📄 Licença

MIT
