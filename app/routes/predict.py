from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any
from app.services.models import predict_one

router = APIRouter()

class PredictRequest(BaseModel):
    record: Dict[str, Any]

@router.post("/predict")
async def predict(req: PredictRequest):
    yhat = predict_one(req.record)
    return {"prediction": float(yhat)}
