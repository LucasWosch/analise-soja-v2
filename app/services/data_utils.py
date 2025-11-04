# services/db_io.py
from pathlib import Path
import sqlite3
import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------------------------/
# ======== Caminhos fixos ========
DB_PATH = Path("plantio.db")
TABLE_NAME = "plantio_raw"

# ======== Normalização específica do SEU CSV atual ========
# CSV: Crop,Crop_Year,Season,State,Area,Production,Annual_Rainfall,Fertilizer,Pesticide,Yield

COLUMN_RENAME_MAP = {
    "yield": "yield_kg_ha",
    "annual_rainfall": "rain_mm",
    "fertilizer": "fertilizer_kg_ha",
    "pesticide": "pesticide_kg_ha",
    "crop_year": "year",
}

SEASON_CANON = {
    "autumn": "Outono",
    "winter": "Inverno",
    "summer": "Verao",
    "whole year": "Ano todo",
    "kharif": "Chuvosa",
    "rabi": "Inverno",
}

SEASON_MACRO = {
    "autumn": "Chuvosa",
    "kharif": "Chuvosa",
    "rabi": "Seca",
    "winter": "Seca",
    "summer": "Intermediaria",
    "whole year": "Anual",
}


# -------------------------------------------------------------------------------------------------/
def _sanitize_columns(cols):
    cleaned = (
        pd.Series(cols).astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    # evita duplicadas
    seen = {}
    out = []
    for c in cleaned:
        if not c:
            c = "col"
        base = c
        i = 1
        while c in seen:
            i += 1
            c = f"{base}_{i}"
        seen[c] = True
        out.append(c)
    return out


# -------------------------------------------------------------------------------------------------/
def normalize_columns_for_crop_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # 1) renomeia
    norm_map = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
    new_cols = []
    for c in df.columns:
        c_norm = str(c).strip().lower()
        new_cols.append(norm_map.get(c_norm, c))
    df.columns = new_cols

    # 2) snake_case
    df.columns = _sanitize_columns(df.columns)

    # 3) SEASON -> PT + macro
    if "season" in df.columns:
        season_raw = (
            df["season"].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )
        df["season"] = season_raw.map(SEASON_CANON).fillna(df["season"].astype(str).str.strip())
        df["season_macro"] = season_raw.map(SEASON_MACRO).fillna("Desconhecida")

    # 4) coerções
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["area", "production", "rain_mm", "fertilizer_kg_ha", "pesticide_kg_ha", "yield_kg_ha"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5) id surrogate (estável para replace)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    return df

# -------------------------------------------------------------------------------------------------/
# ======== SQLite IO ========

def load_df_to_sqlite(df: pd.DataFrame, db_path, table: str, if_exists="", add_surrogate_id=True) -> int:
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False, chunksize=5000)
        cur = conn.cursor()
        # índices úteis
        for idx_col in ["id", "year", "state", "season", "season_macro", "crop"]:
            if idx_col in df.columns:
                unique = "UNIQUE" if idx_col == "id" else ""
                cur.execute(f'CREATE {unique} INDEX IF NOT EXISTS idx_{table}_{idx_col} ON {table} ({idx_col});')
        conn.commit()
    finally:
        conn.close()
    return len(df)

# -------------------------------------------------------------------------------------------------/
def read_whole_table(db_path, table: str) -> pd.DataFrame:
    db_path = Path(db_path)
    if not db_path.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f'SELECT * FROM {table}', conn)
    finally:
        conn.close()


# -------------------------------------------------------------------------------------------------/