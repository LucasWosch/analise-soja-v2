# main.py
# ============================================
# FastAPI + SQLite + ML
# Endpoints:
#   GET  /                      -> entrega static/index.html
#   POST /upload_csv            -> carrega CSV, normaliza, grava no SQLite e devolve métricas/gráficos
#   POST /analyze               -> lê do SQLite e devolve métricas/gráficos
#   POST /train                 -> treina e salva modelo (RF/Linear)
#   POST /retrain               -> re-treina (igual ao /train)
#   POST /predict               -> predição pontual + projeção 10 anos (gráfico base64)
#   GET  /options/crops         -> opções únicas para <select> de culturas
#   GET  /options/seasons       -> opções únicas para <select> de estações
# Executar diretamente: python main.py
# ============================================

from __future__ import annotations

import io
import base64
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# matplotlib para gerar o gráfico da projeção
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ====== Services ======
from services.data_utils import (
    DB_PATH,
    TABLE_NAME,
    load_df_to_sqlite,
    read_whole_table,
    normalize_columns_for_crop_dataset,
)
from services.viz import build_all_figures_base64, compute_basic_summary
from services.ml import train_and_save_model, predict_one, MODEL_DIR

# (Opcional) para start automático sem CLI
import uvicorn


# ============================================
# App / Middleware / Static
# ============================================

app = FastAPI(title="Analytics & Predição – CSV Flexível (SQLite)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


# ============================================
# Página inicial
# ============================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ============================================
# Schemas
# ============================================

class TrainRequest(BaseModel):
    target: str = "yield_kg_ha"          # alvo padrão do dataset
    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "random_forest"    # "random_forest" | "linear"

class PredictRequest(BaseModel):
    record: Dict[str, Any]               # ex.: crop, year, state, area, ...


# ============================================
# Helpers
# ============================================

def _ensure_numeric(v: Any, default: float | int | None = None) -> float | int | None:
    try:
        if v is None:
            return default
        if isinstance(v, (int, float)):
            return v
        s = str(v).strip()
        if not s:
            return default
        # trocar vírgula por ponto se vier "1,23"
        if "," in s:
            s = s.replace(",", ".")
        return float(s) if "." in s else int(s)
    except Exception:
        return default

def _make_line_chart_base64(
    x_values: List[int],
    y_values: List[float],
    title: str = "Projeção de Produção (10 anos)",
    x_label: str = "Ano",
    y_label: str = "Produção (t)",
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_values, y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _unique_non_empty_str_list(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    for col in candidates:
        if col in df.columns:
            vals = (
                df[col]
                .dropna()
                .astype(str)
                .map(lambda s: s.strip())
            )
            vals = [v for v in vals if v]
            return sorted(list(set(vals)))
    return []


# ============================================
# Endpoints
# ============================================

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception:
        df = pd.read_csv(pd.io.common.BytesIO(content), sep=";")

    # normaliza colunas (Crop -> crop, etc.) e cria colunas úteis
    df = normalize_columns_for_crop_dataset(df)

    # grava no SQLite (substitui a tabela)
    n = load_df_to_sqlite(df, DB_PATH, TABLE_NAME, if_exists="replace", add_surrogate_id=True)

    # figuras + sumário
    images = build_all_figures_base64(df)
    summary = compute_basic_summary(df)

    return {"rows_saved": int(n), "images": images, "summary": summary}


@app.post("/analyze")
async def analyze():
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(status_code=400, detail="Tabela vazia. Faça upload primeiro.")
    images = build_all_figures_base64(df)
    summary = compute_basic_summary(df)
    return {"images": images, "summary": summary}


@app.post("/train")
async def train(req: TrainRequest):
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(status_code=400, detail="Tabela vazia. Faça upload primeiro.")
    metrics = train_and_save_model(
        df, target_col=req.target,
        model_type=req.model_type,
        test_size=req.test_size,
        random_state=req.random_state
    )
    return {"model_dir": str(MODEL_DIR), "metrics": metrics}


@app.post("/retrain")
async def retrain(req: TrainRequest):
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(status_code=400, detail="Tabela vazia. Faça upload primeiro.")
    metrics = train_and_save_model(
        df, target_col=req.target,
        model_type=req.model_type,
        test_size=req.test_size,
        random_state=req.random_state
    )
    return {"model_dir": str(MODEL_DIR), "metrics": metrics}


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Retorna:
      - prediction: predição pontual para o 'record' informado
      - forecast: projeção para os próximos 10 anos mantendo demais atributos constantes
                  (years, values e chart_base64 com linha temporal)
    """
    record = req.record.copy()

    # Compatibilidade mínima com pipelines que esperam essas features
    expected_cols = ['id', 'state', 'season_macro']
    for col in expected_cols:
        if col not in record:
            record[col] = 0 if col == 'id' else "dummy"

    # Predição pontual
    yhat = float(predict_one(record))

    # Ano base (tenta 'year', depois 'ano'; se não houver, usa ano corrente)
    year = _ensure_numeric(record.get("year"))
    if year is None:
        year = _ensure_numeric(record.get("ano"))
    if year is None:
        year = datetime.now().year

    # Projeção de 10 anos variando apenas o ano
    horizon = 10
    years = [int(year) + i for i in range(horizon)]
    values: List[float] = []
    for y in years:
        rec = record.copy()
        rec["year"] = y
        if "ano" in rec:
            rec["ano"] = y
        values.append(float(predict_one(rec)))

    chart_b64 = _make_line_chart_base64(
        x_values=years,
        y_values=values,
        title="Projeção de Produção (10 anos)",
        x_label="Ano",
        y_label="Produção (t)",
    )

    return {
        "prediction": yhat,
        "forecast": {
            "years": years,
            "values": values,
            "chart_base64": chart_b64
        }
    }


# ============================================
# Endpoints para popular <select> (Cultura / Estação)
# ============================================

@app.get("/options/crops")
async def options_crops():
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        return {"options": []}
    options = _unique_non_empty_str_list(df, ["crop", "cultura"])
    return {"options": options}

@app.get("/options/seasons")
async def options_seasons():
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        return {"options": []}
    options = _unique_non_empty_str_list(df, ["season", "season_macro", "station", "estacao"])
    return {"options": options}


# ============================================
# Start automático (sem precisar rodar uvicorn na CLI)
# ============================================

if __name__ == "__main__":
    print("🚀 Servidor iniciado em http://localhost:8000")
    # Você pode alterar host/port se precisar
    uvicorn.run(app, host="0.0.0.0", port=8000)
