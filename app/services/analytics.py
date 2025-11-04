# services/analytics.py
import io, base64, re, unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASIC_NUM_COLS = ["yield_kg_ha","rain_mm","fertilizer_kg_ha","pesticide_kg_ha","area","production"]


# -------------------------------------------------------------------------------------------------/
def _fig_to_b64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# -------------------------------------------------------------------------------------------------/
def _norm_text(s: pd.Series) -> pd.Series:
    # lower + strip + remove accents + collapse spaces
    s = s.astype(str).str.strip().str.lower()
    s = s.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


# -------------------------------------------------------------------------------------------------/
def _crop_mask(df: pd.DataFrame, crop_name: str) -> pd.Series:
    """
    Cria máscara booleana para linhas cuja 'crop' combine com a cultura desejada.
    - Insensível a caixa/acentos
    - Aceita sinônimos comuns (pt/en) para algumas culturas
    """
    if "crop" not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    # normaliza coluna
    crop_norm = _norm_text(df["crop"])

    # dicionário de sinônimos
    synonyms = {
        "soja":  ["soja", "soy", "soya", "soybean", "soyabean"],
        "milho": ["milho", "corn", "maize"],
        "trigo": ["trigo", "wheat"],
        "arroz": ["arroz", "rice", "paddy"],
        "algodao": ["algodao", "cotton"],
        "cana": ["cana", "sugarcane", "sugar cane"],
    }

    # normaliza o nome pedido
    name_norm = _norm_text(pd.Series([crop_name])).iloc[0]
    patterns = synonyms.get(name_norm, [name_norm])

    # monta regex OR seguro (escapando)
    rx = r"(" + "|".join([re.escape(p) for p in patterns]) + r")"
    return crop_norm.str.contains(rx, na=False)


# -------------------------------------------------------------------------------------------------/
def bar_top_crops(df: pd.DataFrame) -> str:
    if "crop" not in df.columns:
        plt.figure(); plt.text(0.1,0.5,"Sem coluna 'crop'")
        return _fig_to_b64()
    s = df["crop"].value_counts().head(10)
    plt.figure(figsize=(7,4))
    s.plot(kind="bar")
    plt.title("Top culturas (contagem)")
    plt.xlabel("Cultura"); plt.ylabel("Registros")
    return _fig_to_b64()


# -------------------------------------------------------------------------------------------------/
def yield_by_state(df: pd.DataFrame) -> str:
    if "state" not in df.columns or "yield_kg_ha" not in df.columns:
        plt.figure(); plt.text(0.1,0.5,"Falta 'state' ou 'yield_kg_ha'")
        return _fig_to_b64()
    s = df.groupby("state")["yield_kg_ha"].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(7,4))
    s.plot(kind="bar")
    plt.title("Média de produtividade por estado")
    plt.xlabel("Estado"); plt.ylabel("kg/ha")
    return _fig_to_b64()


# -------------------------------------------------------------------------------------------------/
def hist_numeric(df: pd.DataFrame) -> str:
    num = [c for c in BASIC_NUM_COLS if c in df.columns]
    if not num:
        plt.figure(); plt.text(0.1,0.5,"Sem colunas numéricas")
        return _fig_to_b64()
    plt.figure(figsize=(7,4))
    df[num].hist(bins=20, figsize=(8,4))
    plt.suptitle("Distribuições numéricas")
    return _fig_to_b64()


# -------------------------------------------------------------------------------------------------/
def corr_matrix(df: pd.DataFrame) -> str:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        plt.figure(); plt.text(0.1,0.5,"Numéricas insuficientes")
        return _fig_to_b64()
    corr = num.corr(numeric_only=True)
    plt.figure(figsize=(6,5))
    plt.imshow(corr, interpolation="nearest")
    plt.title("Matriz de correlação")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.colorbar()
    return _fig_to_b64()

# -------------------------------------------------------------------------------------------------/

def box_by_season_macro(df: pd.DataFrame) -> str:
    if "season_macro" not in df.columns or "yield_kg_ha" not in df.columns:
        plt.figure(); plt.text(0.1,0.5,"Falta 'season_macro' ou 'yield_kg_ha'")
        return _fig_to_b64()
    groups = [g["yield_kg_ha"].dropna().values for _, g in df.groupby("season_macro")]
    labels = [str(k) for k,_ in df.groupby("season_macro")]
    plt.figure(figsize=(7,4))
    plt.boxplot(groups, labels=labels)
    plt.title("Produtividade por macro-estação")
    plt.ylabel("kg/ha")
    return _fig_to_b64()



# -------------------------------------------------------------------------------------------------/
# GENÉRICO: Produção por Ano para QUALQUER cultura

def production_by_year(df: pd.DataFrame, crop_name: str = "soja") -> str:
    """
    Soma 'production' por 'year' apenas para linhas cuja 'crop' combine com `crop_name`.
    Exibe todos (ou mais) anos no eixo X.
    """
    required = {"crop", "year", "production"}
    if not required.issubset(df.columns):
        plt.figure();
        plt.text(0.1, 0.5, "Faltam colunas: 'crop', 'year' ou 'production'")
        return _fig_to_b64()

    mask = _crop_mask(df, crop_name)
    sub = df.loc[mask, ["year", "production"]].copy()
    if sub.empty:
        plt.figure();
        plt.text(0.1, 0.5, f"Sem registros para cultura '{crop_name}'")
        return _fig_to_b64()

    sub["year"] = pd.to_numeric(sub["year"], errors="coerce")
    sub["production"] = pd.to_numeric(sub["production"], errors="coerce")
    sub = sub.dropna(subset=["year", "production"])
    if sub.empty:
        plt.figure();
        plt.text(0.1, 0.5, "Sem dados numéricos válidos")
        return _fig_to_b64()

    s = sub.groupby("year")["production"].sum().sort_index()

    plt.figure(figsize=(9, 4.5))
    plt.plot(s.index, s.values, marker="o", linewidth=2, color="#2c7fb8")
    plt.title(f"Produção por Ano – {crop_name.capitalize()}")
    plt.xlabel("Ano")
    plt.ylabel("Produção (soma)")

    # Mostra mais ticks (todos os anos, se possível)
    years = s.index.tolist()
    if len(years) <= 20:
        plt.xticks(years, rotation=45)
    else:
        # exibe 1 a cada N anos se muitos
        step = max(1, len(years) // 15)
        plt.xticks(years[::step], rotation=45)

    plt.grid(True, linestyle="--", alpha=0.5)
    return _fig_to_b64()



# -------------------------------------------------------------------------------------------------/
def compute_basic_summary(df: pd.DataFrame) -> dict:
    out = {
        "registros": int(len(df)),
        "culturas": int(df["crop"].nunique()) if "crop" in df.columns else None,
        "estados": int(df["state"].nunique()) if "state" in df.columns else None,
    }
    if "year" in df.columns and df["year"].notna().any():
        out["ano_min"] = int(pd.to_numeric(df["year"], errors="coerce").dropna().min())
        out["ano_max"] = int(pd.to_numeric(df["year"], errors="coerce").dropna().max())
    if "yield_kg_ha" in df.columns:
        y = pd.to_numeric(df["yield_kg_ha"], errors="coerce").dropna()
        if len(y):
            out["yield_media"] = float(y.mean())
            out["yield_mediana"] = float(y.median())
    return out



# -------------------------------------------------------------------------------------------------/

def build_all_figures_base64(df: pd.DataFrame, crop_for_production: str = "soyabean") -> dict:
    """
    Gera o dashboard completo. Você pode escolher qual cultura aparecerá
    no gráfico de 'Produção por Ano' passando `crop_for_production`.
    """
    return {
        "bar_top_crops": bar_top_crops(df),
        "yield_by_state": yield_by_state(df),
        "corr_matrix": corr_matrix(df),
        "box_by_season_macro": box_by_season_macro(df),
        "production_by_year": production_by_year(df, crop_name=crop_for_production),
    }

# -------------------------------------------------------------------------------------------------/