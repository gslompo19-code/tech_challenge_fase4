import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go


# =========================================
# Config
# =========================================
st.set_page_config(
    page_title="IBOVESPA — Tech Challenge Fase 4",
    layout="wide"
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ajuste aqui se seu CSV estiver com outro nome dentro de /dados
DEFAULT_CSV_PATHS = [
    os.path.join(ROOT_DIR, "dados", "Dados Ibovespa (2).csv"),
    os.path.join(ROOT_DIR, "dados", "Dados Ibovespa.csv"),
    os.path.join(ROOT_DIR, "Dados Ibovespa (2).csv"),
    os.path.join(ROOT_DIR, "Dados Ibovespa.csv"),
]

METRICAS_PATH = os.path.join(ROOT_DIR, "metricas.json")
LOG_PATH = os.path.join(ROOT_DIR, "logs_previsoes.csv")

# Modelo (você já tem este)
MODELO_PATH = os.path.join(ROOT_DIR, "modelo_ibov.pkl")

# Opcional (recomendado do notebook)
PIPELINE_PATH = os.path.join(ROOT_DIR, "modelo_ibov_pipeline.pkl")
SCALER_PATH = os.path.join(ROOT_DIR, "scaler_ibov.pkl")
FEATURES_PATH = os.path.join(ROOT_DIR, "features.json")


# =========================================
# Utilitários de parsing e features
# =========================================
def _to_float_ptbr(x):
    """Converte números no padrão pt-BR: 123.456,78 -> 123456.78"""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_volume(value):
    """
    Ex.: "115,64M" -> 115_640_000
         "987,2K"  -> 987_200
         "1,25B"   -> 1_250_000_000
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper()

    mult = 1.0
    if s.endswith("K"):
        mult = 1e3
        s = s[:-1]
    elif s.endswith("M"):
        mult = 1e6
        s = s[:-1]
    elif s.endswith("B"):
        mult = 1e9
        s = s[:-1]

    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def compute_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd_components(prices: pd.Series, short=12, long=26, signal=9):
    ema_short = prices.ewm(span=short, adjust=False).mean()
    ema_long = prices.ewm(span=long, adjust=False).mean()
    macd = ema_short - ema_long
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def obv_series(close: pd.Series, vol: pd.Series) -> pd.Series:
    obv = np.zeros(len(close), dtype=float)
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv[i] = obv[i - 1] + (vol.iloc[i] if pd.notna(vol.iloc[i]) else 0.0)
        elif close.iloc[i] < close.iloc[i - 1]:
            obv[i] = obv[i - 1] - (vol.iloc[i] if pd.notna(vol.iloc[i]) else 0.0)
        else:
            obv[i] = obv[i - 1]
    return pd.Series(obv, index=close.index)


def zscore_roll(s: pd.Series, w: int = 20) -> pd.Series:
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return (s - m) / sd


def carregar_dados(csv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói as features de forma compatível com o que você descreveu no notebook:
    - MMs (3, 7, 14, 21, 30)
    - Volatilidade (5, 10, 20)
    - Desvio da mm3
    - Dia da semana
    - RSI
    - MACD (linha, sinal, hist)
    - Bollinger (largura)
    - ATR
    - OBV
    - e um conjunto enxuto de features finais
    """
    df = csv_df.copy()
    df.columns = df.columns.str.strip()

    # Data
    if "Data" not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'Data'.")

    df["Data"] = pd.to_datetime(df["Data"], format="%d.%m.%Y", errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data")

    # Colunas de preço (padrão Investing)
    for col in ["Último", "Abertura", "Máxima", "Mínima"]:
        if col not in df.columns:
            raise ValueError(f"CSV precisa ter a coluna '{col}'.")
        df[col] = df[col].apply(_to_float_ptbr)

    # Volume
    if "Vol." not in df.columns:
        raise ValueError("CSV precisa ter a coluna 'Vol.' (formato Investing, ex.: 115,64M).")
    df["Vol."] = df["Vol."].apply(parse_volume)

    # Features base
    df["ret_1d"] = df["Último"].pct_change()
    df["log_ret"] = np.log(df["Último"]).diff()
    df["ret_5d"] = df["Último"].pct_change(5)
    df["rv_20"] = df["ret_1d"].rolling(20, min_periods=20).std()

    # Médias móveis
    for w in [3, 7, 14, 21, 30]:
        df[f"mm_{w}"] = df["Último"].rolling(w, min_periods=w).mean()

    # Volatilidade
    for w in [5, 10, 20]:
        df[f"vol_{w}"] = df["Último"].rolling(w, min_periods=w).std()

    df["desvio_mm3"] = df["Último"] - df["mm_3"]
    df["desvio_mm3_pct"] = (df["desvio_mm3"] / df["mm_3"]).replace([np.inf, -np.inf], np.nan)

    df["dia"] = df["Data"].dt.weekday

    # RSI
    df["rsi"] = compute_rsi(df["Último"], 14)

    # MACD
    macd, sinal, hist = macd_components(df["Último"], 12, 26, 9)
    df["macd"] = macd
    df["sinal_macd"] = sinal
    df["hist_macd"] = hist

    # Bollinger (20)
    bb_m = df["Último"].rolling(20, min_periods=20).mean()
    bb_s = df["Último"].rolling(20, min_periods=20).std()
    bb_sup = bb_m + 2 * bb_s
    bb_inf = bb_m - 2 * bb_s
    df["bb_largura"] = (bb_sup - bb_inf) / bb_m

    # ATR (14)
    tr1 = df["Máxima"] - df["Mínima"]
    tr2 = (df["Máxima"] - df["Último"].shift(1)).abs()
    tr3 = (df["Mínima"] - df["Último"].shift(1)).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR"] = df["TR"].rolling(14, min_periods=14).mean()
    df["atr_pct"] = df["ATR"] / df["Último"]

    # Volume transform
    df["vol_log"] = np.log(df["Vol."].clip(lower=1))
    df["vol_ret"] = df["Vol."].pct_change().replace([np.inf, -np.inf], np.nan)

    # OBV
    df["obv"] = obv_series(df["Último"], df["Vol."])
    df["obv_diff"] = df["obv"].diff()

    # Z-scores úteis
    df["z_close_20"] = zscore_roll(df["Último"], 20)
    df["z_rsi_20"] = zscore_roll(df["rsi"], 20)
    df["z_macd_20"] = zscore_roll(df["macd"], 20)

    # Alvo (se quiser mostrar backtest; no deploy a previsão é "amanhã")
    df["Alvo"] = (df["Último"].shift(-1) > df["Último"]).astype(int)

    return df


# =========================================
# Carregamento do modelo / scaler / features
# =========================================
@st.cache_resource
def load_assets():
    # 1) Preferir pipeline completo (melhor)
    if os.path.exists(PIPELINE_PATH):
        pipe = joblib.load(PIPELINE_PATH)
        return {"type": "pipeline", "pipe": pipe}

    # 2) Modelo separado (o que você tem hoje)
    if not os.path.exists(MODELO_PATH):
        raise FileNotFoundError(
            f"Não achei '{os.path.basename(MODELO_PATH)}' na raiz do repo. "
            f"Coloque o arquivo ao lado do app.py."
        )

    model = joblib.load(MODELO_PATH)

    scaler = None
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)

    features = None
    if os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            features = json.load(f).get("features")

    return {"type": "separado", "model": model, "scaler": scaler, "features": features}


def predict_proba_class1(assets, X: np.ndarray) -> np.ndarray:
    """
    Retorna probabilidade da classe 1 (ALTA amanhã).
    """
    if assets["type"] == "pipeline":
        pipe = assets["pipe"]
        if hasattr(pipe, "predict_proba"):
            return pipe.predict_proba(X)[:, 1]
        return pipe.predict(X).astype(float)

    model = assets["model"]
    scaler = assets["scaler"]

    X_in = X
    if scaler is not None:
        X_in = scaler.transform(X_in)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_in)[:, 1]
    return model.predict(X_in).astype(float)


def append_log(row: dict):
    df = pd.DataFrame([row])
    if os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


def try_load_default_csv() -> pd.DataFrame | None:
    for p in DEFAULT_CSV_PATHS:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


# =========================================
# UI
# =========================================
st.title("📈 IBOVESPA — Deploy do Modelo (Tech Challenge Fase 4)")

st.markdown(
    """
Este app:
- carrega o modelo treinado na Fase 2 (`.pkl`);
- permite inserir dados via upload de CSV;
- exibe gráfico interativo com histórico + probabilidade de alta;
- mostra métricas do modelo (monitoramento);
- salva logs de uso em `logs_previsoes.csv`.
"""
)

# Sidebar: métricas
st.sidebar.header("📊 Monitoramento (métricas do modelo)")
if os.path.exists(METRICAS_PATH):
    with open(METRICAS_PATH, "r", encoding="utf-8") as f:
        metricas = json.load(f)

    st.sidebar.metric("Acurácia (Treino)", str(metricas.get("acuracia_treino", "—")))
    st.sidebar.metric("Acurácia (Teste)", str(metricas.get("acuracia_teste", "—")))
    st.sidebar.metric("F1 (CV médio)", str(metricas.get("f1_cv_medio", "—")))
    st.sidebar.metric("Overfitting (%)", str(metricas.get("overfitting_percentual", "—")))

    conf = metricas.get("confusao")
else:
    metricas = {}
    conf = None
    st.sidebar.info("Não encontrei `metricas.json` (recomendado para o desafio).")

# Carregar modelo
try:
    assets = load_assets()
except Exception as e:
    st.error(f"Erro ao carregar modelo/arquivos: {e}")
    st.stop()

# Entrada de dados
st.subheader("1) Insira dados (CSV)")
uploaded = st.file_uploader(
    "Envie o CSV do IBOVESPA (formato Investing: Data, Último, Abertura, Máxima, Mínima, Vol., Var%)",
    type=["csv"]
)

df_raw = None
if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = try_load_default_csv()

if df_raw is None:
    st.warning("Envie um CSV para começar (ou coloque o arquivo dentro da pasta `dados/`).")
    st.stop()

# Processar
try:
    df_feat = carregar_dados(df_raw)
except Exception as e:
    st.error(f"Erro ao processar o CSV: {e}")
    st.stop()

# Features: se existir features.json usa; senão usa um padrão compatível
features = assets.get("features")
if not features:
    features = [
        "ret_1d", "log_ret", "ret_5d", "rv_20",
        "atr_pct", "bb_largura", "desvio_mm3_pct",
        "vol_log", "vol_ret", "obv_diff",
        "rsi", "macd", "sinal_macd", "hist_macd",
        "dia", "z_close_20", "z_rsi_20", "z_macd_20"
    ]

# Remover linhas com NaN nas features
df_model = df_feat.dropna(subset=features).copy()
if len(df_model) < 50:
    st.warning("Depois de gerar as features, sobraram poucas linhas válidas. Tente um CSV com mais histórico.")
    st.stop()

# Controles
col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    lookback = st.slider("Janela do gráfico (últimos N dias)", 60, 500, 180, 20)
with col_cfg2:
    threshold = st.slider("Threshold (prob. de ALTA)", 0.30, 0.70, 0.50, 0.01)
with col_cfg3:
    st.caption("Dica: 0.50 é padrão; você pode calibrar no seu vídeo explicando o trade-off.")

# Predição
X = df_model[features].values
proba = predict_proba_class1(assets, X)
df_model["proba_alta"] = proba
df_model["prev_alta"] = (df_model["proba_alta"] >= threshold).astype(int)
df_model["label_prev"] = df_model["prev_alta"].map({1: "⬆️ ALTA amanhã", 0: "⬇️ QUEDA/ESTÁVEL amanhã"})

# Resultado mais recente
st.subheader("2) Previsão mais recente")
last = df_model.iloc[-1]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Última data", str(last["Data"].date()))
c2.metric("Fechamento (Último)", f'{last["Último"]:,.0f}'.replace(",", "."))
c3.metric("Probabilidade de ALTA (amanhã)", f"{100 * last['proba_alta']:.1f}%")
c4.metric("Classe prevista", "ALTA" if last["prev_alta"] == 1 else "QUEDA/ESTÁVEL")

# Log
append_log({
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "data_ultima": str(last["Data"].date()),
    "ultimo": float(last["Último"]),
    "proba_alta": float(last["proba_alta"]),
    "threshold": float(threshold),
    "prev_alta": int(last["prev_alta"]),
})

# Gráfico interativo (Plotly)
st.subheader("3) Gráfico interativo (histórico + probabilidade)")
df_plot = df_model.tail(lookback).copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_plot["Data"],
    y=df_plot["Último"],
    name="Fechamento (Último)",
    mode="lines"
))
fig.add_trace(go.Scatter(
    x=df_plot["Data"],
    y=df_plot["proba_alta"],
    name="Probabilidade de ALTA (amanhã)",
    mode="lines",
    yaxis="y2"
))

fig.update_layout(
    height=520,
    xaxis=dict(title="Data"),
    yaxis=dict(title="IBOVESPA (pontos)"),
    yaxis2=dict(title="Probabilidade", overlaying="y", side="right", range=[0, 1]),
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

with st.expander("🔎 Ver tabela (últimos 30 dias válidos)"):
    st.dataframe(df_model[["Data", "Último", "proba_alta", "label_prev"]].tail(30), use_container_width=True)

# Matriz de confusão (se existir no metricas.json)
if conf is not None:
    st.subheader("4) Matriz de confusão (monitoramento)")
    conf_arr = np.array(conf)
    fig_cm = go.Figure(data=go.Heatmap(
        z=conf_arr,
        x=["Prev 0", "Prev 1"],
        y=["Real 0", "Real 1"]
    ))
    fig_cm.update_layout(height=380)
    st.plotly_chart(fig_cm, use_container_width=True)

st.caption("✅ Log de uso salvo em `logs_previsoes.csv` (na raiz do app).")
