from fastapi import APIRouter, HTTPException
from app.services.data_utils import DB_PATH, TABLE_NAME, read_whole_table
from app.services.analytics import build_all_figures_base64, compute_basic_summary

router = APIRouter()

@router.post("/analyze")
async def analyze():
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(400, "Tabela vazia. Faça upload primeiro.")
    images = build_all_figures_base64(df)
    summary = compute_basic_summary(df)
    return {"images": images, "summary": summary}
