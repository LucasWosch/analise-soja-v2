from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd

from app.services.data_utils import (
    DB_PATH, TABLE_NAME, load_df_to_sqlite,
    normalize_columns_for_crop_dataset
)
from app.services.analytics import build_all_figures_base64, compute_basic_summary

router = APIRouter()

@router.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Envie CSV")

    content = await file.read()
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content))
    except:
        df = pd.read_csv(pd.io.common.BytesIO(content), sep=';')

    df = normalize_columns_for_crop_dataset(df)
    n = load_df_to_sqlite(df, DB_PATH, TABLE_NAME, if_exists="replace", add_surrogate_id=True)

    images = build_all_figures_base64(df)
    summary = compute_basic_summary(df)
    return {"rows_saved": int(n), "images": images, "summary": summary}
