from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import upload, analyze, train, predict

app = FastAPI(title="Analytics & Predição – CSV Flexível (SQLite)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# === Rotas da API ===
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(train.router)
app.include_router(predict.router)

# === Frontend ===
# Monta a pasta 'static' servindo o index.html na raiz
app.mount("/", StaticFiles(directory="static", html=True), name="static")
