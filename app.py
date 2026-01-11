# ============================================================
# STREAMLIT APP — IBOVESPA TREND PREDICTOR
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import os

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Ibovespa – Previsão de Tendência",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Previsão de Tendência do Ibovespa")
st.markdown("Modelo baseado em **CatBoost + Indicadores Técnicos**")

# ============================================================
# FUNÇÕES AUXILIARES (IGUAIS AO COLAB)
# ============================================================

def zscore_roll(s, w=20):
    m = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std()
    return (s - m) / sd


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd_components(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sinal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sinal
    return macd, sinal, hist


def obv_series(df):
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    return obv


# ============================================================
# FEATURE ENGINEERING — PIPELINE COMPLETO
# ============================================================

@st.cache_data
def carregar_dados_yf(ticker="^BVSP", periodo="10y"):
    df = yf.download(ticker, period=periodo)
    df = df.reset_index()

    df["var_pct"] = df["Close"].pct_change()

    for d in [3, 7, 14, 21, 30]:
        df[f"mm_{d}"] = df["Close"].rolling(d).mean()

    for d in [5, 10, 20]:
        df[f"vol_{d}"] = df["Close"].rolling(d).std()

    df["desvio_mm3"] = df["Close"] - df["mm_3"]
    df["dia"] = df["Date"].dt.weekday

    df["rsi"] = calculate_rsi(df["Close"])
    macd, sinal, hist = macd_components(df["Close"])
    df["macd"], df["sinal_macd"], df["hist_macd"] = macd, sinal, hist

    bb_media = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["bb_largura"] = (bb_media + 2 * bb_std - (bb_media - 2 * bb_std)) / bb_media

    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)

    df["ATR"] = tr.rolling(14).mean()
    df["atr_pct"] = df["ATR"] / df["Close"]

    df["obv"] = obv_series(df)
    df["obv_diff"] = pd.Series(df["obv"]).diff()

    df["ret_1d"] = df["Close"].pct_change()
    df["log_ret"] = np.log(df["Close"]).diff()
    df["ret_5d"] = df["Close"].pct_change(5)
    df["rv_20"] = df["ret_1d"].rolling(20).std()

    df["desvio_mm3_pct"] = df["desvio_mm3"] / df["mm_3"]
    df["vol_log"] = np.log(df["Volume"].clip(lower=1))
    df["vol_ret"] = df["Volume"].pct_change()

    df["z_close_20"] = zscore_roll(df["Close"])
    df["z_rsi_20"] = zscore_roll(df["rsi"])
    df["z_macd_20"] = zscore_roll(df["macd"])

    features = [
        "ret_1d","log_ret","ret_5d","rv_20",
        "atr_pct","bb_largura","desvio_mm3_pct",
        "vol_log","vol_ret","obv_diff",
        "rsi","macd","sinal_macd","hist_macd",
        "dia","z_close_20","z_rsi_20","z_macd_20"
    ]

    df = df.dropna(subset=features)

    return df, features


# ============================================================
# CARREGAR MODELO E DADOS
# ============================================================

modelo = joblib.load("modelo_catboost.pkl")
scaler = joblib.load("scaler.pkl")

dados, features = carregar_dados_yf()

X = scaler.transform(dados[features])
dados["Previsão"] = modelo.predict(X)
dados["Prob_Alta"] = modelo.predict_proba(X)[:, 1]
dados["Prob_Queda"] = modelo.predict_proba(X)[:, 0]

# ============================================================
# INTERFACE
# ============================================================

st.subheader("📈 Backtest – Histórico Completo")

st.line_chart(
    dados.set_index("Date")[["Close"]]
)

st.subheader("📊 Previsões do Modelo")

tabela = dados[["Date", "Close", "Previsão", "Prob_Alta", "Prob_Queda"]].tail(60)
tabela["Tendência"] = tabela["Previsão"].map({1: "📈 Alta", 0: "📉 Queda"})

st.dataframe(tabela, use_container_width=True)

# ============================================================
# LOG DE USO (BÔNUS)
# ============================================================

if st.button("💾 Salvar Log da Última Previsão"):
    os.makedirs("logs", exist_ok=True)

    tabela.tail(1).to_csv(
        "logs/uso_app.csv",
        mode="a",
        header=not os.path.exists("logs/uso_app.csv"),
        index=False
    )
    st.success("Log salvo com sucesso!")
