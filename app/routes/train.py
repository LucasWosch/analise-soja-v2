from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.data_utils import DB_PATH, TABLE_NAME, read_whole_table
from app.services.models import train_and_save_model, MODEL_DIR

router = APIRouter()

class TrainRequest(BaseModel):
    target: str = "yield_kg_ha"
    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "random_forest"

@router.post("/train")
async def train(req: TrainRequest):
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(400, "Tabela vazia. Faça upload primeiro.")
    metrics = train_and_save_model(df, target_col=req.target,
                                   model_type=req.model_type,
                                   test_size=req.test_size,
                                   random_state=req.random_state)
    return {"model_dir": str(MODEL_DIR), "metrics": metrics}

@router.post("/retrain")
async def retrain(req: TrainRequest):
    df = read_whole_table(DB_PATH, TABLE_NAME)
    if df.empty:
        raise HTTPException(400, "Tabela vazia. Faça upload primeiro.")
    metrics = train_and_save_model(df, target_col=req.target,
                                   model_type=req.model_type,
                                   test_size=req.test_size,
                                   random_state=req.random_state)
    return {"model_dir": str(MODEL_DIR), "metrics": metrics}
