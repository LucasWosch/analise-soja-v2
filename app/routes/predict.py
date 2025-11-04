from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from app.services.models import predict_one

router = APIRouter()

class PredictRequest(BaseModel):
    record: Dict[str, Any]

@router.post("/predict")
async def predict(req: PredictRequest):
    record = req.record.copy()

    # Resolvendo problema de compatibilidade de colunas
    expected_cols = ['id', 'state', 'season_macro']  # apenas para compatibilidade interna
    for col in expected_cols:
        if col not in record:
            record[col] = 0 if col == 'id' else "dummy"

    yhat = predict_one(record)
    return {"prediction": float(yhat)}
