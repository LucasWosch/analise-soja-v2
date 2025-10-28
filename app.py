# app.py
# Executar: uvicorn app:app --reload
# API:
#   POST /upload_csv           -> salva CSV no SQLite (usa o mesmo caminho da ingestão ou outro)
#   POST /analyze              -> retorna gráficos (base64) + métricas
#   POST /train                -> treina modelo (alvo default: yield_kg_ha)
#   POST /predict              -> recebe JSON e retorna predição
#   POST /retrain              -> re-treina lendo toda a tabela do SQLite
#
# Front-end de teste: abra static/index.html (aponta para estes endpoints)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import pandas as pd

from services.data_utils import (
    DB_PATH, TABLE_NAME, load_df_to_sqlite, read_whole_table, normalize_columns_for_crop_dataset
)
from services.viz import build_all_figures_base64, compute_basic_summary
from services.ml import train_and_save_model, predict_one, MODEL_DIR

app = FastAPI(title="Analytics & Predição – CSV Flexível (SQLite)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ======== Schemas ========
class TrainRequest(BaseModel):
    target: str = "yield_kg_ha"          # alvo padrão para seu dataset
    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "random_forest"    # "random_forest" | "linear"

class PredictRequest(BaseModel):
    record: Dict[str, Any]               # campos de entrada (ex.: crop, year, state, area, ...)

# ======== Endpoints ========

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Envie um arquivo .csv")

    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except Exception:
        df = pd.read_csv(pd.io.common.BytesIO(content), sep=";")

    # normaliza colunas para este dataset (Crop -> crop, etc) e cria colunas úteis
    df = normalize_columns_for_crop_dataset(df)

    # grava no SQLite (substitui a tabela)
    n = load_df_to_sqlite(df, DB_PATH, TABLE_NAME, if_exists="replace", add_surrogate_id=True)

    # figuras + sumário com base no DF recém-enviado
    images = build_all_figures_base64(df)
    summary = compute_basic_summary(df)

    return {"rows_saved": int(n), "images": images, "summary": summary}

@app.post("/analyze")
async def analyze():
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(400, "Tabela vazia. Faça upload primeiro.")
    images = build_all_figures_base64(df)
    summary = compute_basic_summary(df)
    return {"images": images, "summary": summary}

@app.post("/train")
async def train(req: TrainRequest):
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(400, "Tabela vazia. Faça upload primeiro.")
    metrics = train_and_save_model(df, target_col=req.target,
                                   model_type=req.model_type,
                                   test_size=req.test_size,
                                   random_state=req.random_state)
    return {"model_dir": str(MODEL_DIR), "metrics": metrics}

@app.post("/retrain")
async def retrain(req: TrainRequest):
    # igual ao /train – mantido separado só por clareza semântica
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(400, "Tabela vazia. Faça upload primeiro.")
    metrics = train_and_save_model(df, target_col=req.target,
                                   model_type=req.model_type,
                                   test_size=req.test_size,
                                   random_state=req.random_state)
    return {"model_dir": str(MODEL_DIR), "metrics": metrics}

@app.post("/predict")
async def predict(req: PredictRequest):
    # a predição funciona com qualquer schema desde que as colunas sejam compatíveis com o que o modelo foi treinado
    yhat = predict_one(req.record)
    return {"prediction": float(yhat)}
