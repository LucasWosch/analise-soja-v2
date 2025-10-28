from pathlib import Path
import sqlite3
import pandas as pd

# =============== CONFIGURAÇÕES FIXAS ===============

CSV_PATH   = r"crop_yield.csv"
DB_PATH    = r"plantio.db"
TABLE_NAME = "plantio_raw"
IF_EXISTS  = "replace"
CHUNKSIZE  = 5000

COLUMN_RENAME_MAP = {
    "yield": "yield_kg_ha",
    "annual_rainfall": "rain_mm",
    "fertilizer": "fertilizer_kg_ha",
    "pesticide": "pesticide_kg_ha",
    "crop_year": "year",
}
# ===================================================


def sanitize_columns(cols):
    cleaned = (
        pd.Series(cols).astype(str).str.strip().str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
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


def read_csv_auto(path, encoding=None, sep=None):
    encodings = [encoding] if encoding else ["utf-8", "latin-1"]
    seps = [sep] if sep else [",", ";"]
    last_exc = None
    for enc in encodings:
        for s in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=s)
                return df, enc, s
            except Exception as e:
                last_exc = e
    raise RuntimeError(f"Falha ao ler CSV. Último erro: {last_exc}")


def load_csv_to_sqlite(csv_path: str, db_path: str, table: str, if_exists: str = "replace", chunksize: int = 5000):
    csv = Path(csv_path)
    if not csv.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv}")

    df, used_enc, used_sep = read_csv_auto(csv)

    # 1) renomeia colunas conhecidas (case-insensitive)
    if COLUMN_RENAME_MAP:
        norm_map = {k.lower(): v for k, v in COLUMN_RENAME_MAP.items()}
        new_cols = []
        for c in df.columns:
            c_norm = str(c).lower()
            new_cols.append(norm_map.get(c_norm, c))
        df.columns = new_cols

    # 2) normaliza nomes -> snake_case
    df.columns = sanitize_columns(df.columns)

    # 3) traduz/normaliza SEASON no PANDAS (sem updates no DB)
    if "season" in df.columns:
        season_raw = (
            df["season"].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
        )

        # tradução PT-BR da estação (coluna 'season' será sobrescrita com PT-BR)
        season_pt_map = {
            "autumn": "Outono",
            "winter": "Inverno",
            "summer": "Verão",
            "whole year": "Ano todo",
            "kharif": "Chuvosa",   # traduzimos Kharif para 'Chuvosa' (época das monções)
            "rabi": "Inverno",     # Rabi ~ inverno (safra de frio/seca)
        }
        df["season"] = season_raw.map(season_pt_map).fillna(df["season"].astype(str).str.strip())

        # macro-estação PT-BR (nova coluna 'season_macro')
        macro_map_pt = {
            "kharif": "Chuvosa",
            "autumn": "Chuvosa",       # Outono ~ prox. estação chuvosa (conforme decisão do projeto)
            "rabi": "Seca",
            "winter": "Seca",
            "summer": "Intermediária",
            "whole year": "Anual",
        }
        df["season_macro"] = season_raw.map(macro_map_pt).fillna("Desconhecida")

    # 4) tenta coerção de campos de data (se existirem)
    for candidate in ["date", "data", "dt", "data_ref", "periodo", "periodo_ref"]:
        if candidate in df.columns:
            df[candidate] = pd.to_datetime(df[candidate], errors="coerce")

    # 5) cria coluna 'id' ANTES do insert (incremental, estável para replace)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    # 6) grava no SQLite
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists=if_exists, index=False, chunksize=chunksize)

        cur = conn.cursor()
        # índices úteis
        for idx_col in ["id", "year", "state", "season", "season_macro"]:
            if idx_col in df.columns:
                unique = "UNIQUE" if idx_col == "id" else ""
                cur.execute(f'CREATE {unique} INDEX IF NOT EXISTS idx_{table}_{idx_col} ON {table} ({idx_col});')

        # Views de agregação por estação (PT-BR) e por macro-estação
        cur.execute("DROP VIEW IF EXISTS v_agg_por_season_pt;")
        cur.execute(f"""
            CREATE VIEW v_agg_por_season_pt AS
            SELECT
                season                               AS estacao_pt,
                COUNT(*)                              AS registros,
                AVG(yield_kg_ha)                      AS media_yield_kg_ha,
                AVG(rain_mm)                          AS media_rain_mm,
                AVG(fertilizer_kg_ha)                 AS media_fertilizer_kg_ha,
                AVG(pesticide_kg_ha)                  AS media_pesticide_kg_ha
            FROM {table}
            GROUP BY season
            ORDER BY registros DESC;
        """)

        cur.execute("DROP VIEW IF EXISTS v_agg_por_season_macro_pt;")
        cur.execute(f"""
            CREATE VIEW v_agg_por_season_macro_pt AS
            SELECT
                season_macro                         AS macro_estacao_pt,
                COUNT(*)                              AS registros,
                AVG(yield_kg_ha)                      AS media_yield_kg_ha,
                AVG(rain_mm)                          AS media_rain_mm,
                AVG(fertilizer_kg_ha)                 AS media_fertilizer_kg_ha,
                AVG(pesticide_kg_ha)                  AS media_pesticide_kg_ha
            FROM {table}
            GROUP BY season_macro
            ORDER BY registros DESC;
        """)
        conn.commit()
    finally:
        conn.close()

    print(f"[OK] CSV importado para '{db_path}' tabela '{table}' ({len(df)} linhas).")
    print(f"[INFO] encoding='{used_enc}', sep='{used_sep}', columns={list(df.columns)}")


def preview_rows(db_path: str, table: str, limit: int = 10):
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {limit}", conn)
        print("\n=== PRÉVIA TABELA ===")
        print(df)
        try:
            v1 = pd.read_sql_query("SELECT * FROM v_agg_por_season_pt", conn)
            print("\n=== v_agg_por_season_pt ===")
            print(v1)
        except Exception:
            pass
        try:
            v2 = pd.read_sql_query("SELECT * FROM v_agg_por_season_macro_pt", conn)
            print("\n=== v_agg_por_season_macro_pt ===")
            print(v2)
        except Exception:
            pass
    finally:
        conn.close()


if __name__ == "__main__":
    load_csv_to_sqlite(CSV_PATH, DB_PATH, TABLE_NAME, if_exists=IF_EXISTS, chunksize=CHUNKSIZE)
    preview_rows(DB_PATH, TABLE_NAME, limit=10)
