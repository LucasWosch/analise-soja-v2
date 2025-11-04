# services/models.py
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from typing import Callable, Tuple, Dict, Any, List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "latest_model.pkl"
FEATURES_PATH = MODEL_DIR / "latest_features.pkl"
TARGET_PATH = MODEL_DIR / "latest_target.txt"

DEFAULT_TARGET = "yield_kg_ha"

# -------------------------------------------------------------------------------------------------/

def _default_emit(pct: int, msg: str):
    """Emissor padrão: imprime no console. Substitua por um callback para UI/WebSocket."""
    print(f"[{pct:3d}%] {msg}")

# -------------------------------------------------------------------------------------------------/
def _split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col])
    return X, y

# -------------------------------------------------------------------------------------------------/
def _build_preprocess(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    # detecta numéricas/categóricas automaticamente
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    pre = ColumnTransformer([
        ("num", StandardScaler(with_mean=False), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return pre, num_cols, cat_cols

# -------------------------------------------------------------------------------------------------/
def _build_model(model_type: str, progress_verbose: bool = False):
    if model_type == "linear":
        return LinearRegression()
    # RandomForest com verbose se quiser ver as árvores progredindo
    return RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        verbose=1 if progress_verbose else 0
    )

# -------------------------------------------------------------------------------------------------/
# ========= API =========

def train_and_save_model(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    emit: Callable[[int, str], None] = _default_emit,
) -> Dict[str, Any]:
    """
    Treina e salva o modelo retornando métricas. Mostra progresso via 'emit(pct, msg)'.
    - Para integrar com front, passe um callback 'emit' que envie eventos (ex.: via WebSocket).
    """
    t0 = time.time()
    emit(0, "Iniciando pipeline de treinamento")

    if target_col not in df.columns:
        raise ValueError(f"Alvo '{target_col}' não encontrado no DataFrame.")

    emit(5, f"Verificando dados e removendo linhas sem '{target_col}'")
    df_train = df.dropna(subset=[target_col]).copy()
    if len(df_train) < 20:
        raise ValueError("Poucos dados rotulados para treinar (mínimo ~20).")

    emit(15, "Separando X e y")
    X, y = _split_xy(df_train, target_col)

    emit(25, "Montando pré-processamento (numéricas/categóricas)")
    pre, num_cols, cat_cols = _build_preprocess(X)

    emit(35, f"Instanciando modelo: {model_type}")
    model = _build_model(model_type, progress_verbose=True)

    pipe = Pipeline([("pre", pre), ("model", model)])

    emit(45, f"Train/test split (test_size={test_size})")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

    emit(60, "Treinando modelo (isso pode levar alguns segundos)...")
    pipe.fit(Xtr, ytr)

    emit(85, "Gerando predições e calculando métricas")
    yhat = pipe.predict(Xte)

    r2  = float(r2_score(yte, yhat))
    mae = float(mean_absolute_error(yte, yhat))

    metrics = {
        "r2": round(r2, 4),
        "r2_text": f"{r2:.2%} da variância explicada",

        "mae": round(mae, 3),
        "mae_text": f"{mae:,.1f} {target_col} (erro médio absoluto)",

        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "model_type": model_type,
        "target": target_col,
    }

    emit(92, "Salvando artefatos do modelo")
    joblib.dump(pipe, MODEL_PATH)
    joblib.dump({"num_cols": num_cols, "cat_cols": cat_cols}, FEATURES_PATH)
    TARGET_PATH.write_text(target_col, encoding="utf-8")

    elapsed = time.time() - t0
    emit(100, f"Treino concluído em {elapsed:.1f}s (R²={metrics['r2']:.3f}, MAE={metrics['mae']:.3f})")
    return metrics

# -------------------------------------------------------------------------------------------------/
def _load_model_and_target():
    pipe = joblib.load(MODEL_PATH)
    target_col = TARGET_PATH.read_text(encoding="utf-8").strip()
    feat = joblib.load(FEATURES_PATH)
    return pipe, target_col, feat

# -------------------------------------------------------------------------------------------------/
def predict_one(record: dict) -> float:
    # Corrigido: agora desempacota 3 valores
    pipe, target_col, feat = _load_model_and_target()
    # (feat contém num_cols/cat_cols salvos — o OneHotEncoder já está dentro do pipeline com handle_unknown="ignore")
    df = pd.DataFrame([record])
    yhat = pipe.predict(df)[0]
    return float(yhat)

# -------------------------------------------------------------------------------------------------/