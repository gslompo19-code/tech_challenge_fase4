# =====================================================
# SISTEMA PREDITIVO IBOVESPA — STREAMLIT + YFINANCE
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import yfinance as yf
import os

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="Sistema Preditivo IBOVESPA",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# ESTILO VISUAL (mais profissional)
# =====================================================
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
[data-testid="metric-container"] {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# CARREGAR MODELO
# =====================================================
@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_ibov.pkl")

modelo = carregar_modelo()

# =====================================================
# CARREGAR DADOS DO IBOVESPA (YFINANCE)
# =====================================================
@st.cache_data
def carregar_ibov():
    df = yf.download("^BVSP", start="2013-01-01")
    df.reset_index(inplace=True)
    return df

# =====================================================
# FEATURE ENGINEERING (IGUAL AO TREINO)
# =====================================================
def criar_features(df):
    df = df.copy()

    df["retorno"] = df["Close"].pct_change()
    df["media_5"] = df["Close"].rolling(5).mean()
    df["media_21"] = df["Close"].rolling(21).mean()
    df["volatilidade"] = df["retorno"].rolling(21).std()

    df.dropna(inplace=True)
    return df

# =====================================================
# TÍTULO
# =====================================================
st.title("📊 Sistema Preditivo de Tendência do IBOVESPA")

st.markdown("""
Este sistema utiliza **Machine Learning (CatBoost)** para prever a  
**tendência de ALTA ou QUEDA do IBOVESPA**, utilizando dados históricos reais  
obtidos automaticamente do mercado.
""")

# =====================================================
# ABAS
# =====================================================
aba1, aba2, aba3 = st.tabs([
    "🔮 Previsão Atual",
    "📉 Backtest Histórico",
    "ℹ️ Sobre o Modelo"
])

# =====================================================
# ABA 1 — PREVISÃO ATUAL
# =====================================================
with aba1:
    st.subheader("🔮 Previsão da Próxima Tendência")

    dados_ibov = carregar_ibov()
    dados_feat = criar_features(dados_ibov)

    X = dados_feat[modelo.feature_names_]

    ultima_linha = X.iloc[[-1]]
    data_ref = dados_feat["Date"].iloc[-1].date()

    proba = modelo.predict_proba(ultima_linha)[0]
    prob_queda = proba[0]
    prob_alta = proba[1]

    col1, col2 = st.columns(2)

    col1.metric("📈 Probabilidade de Alta", f"{prob_alta*100:.2f}%")
    col2.metric("📉 Probabilidade de Queda", f"{prob_queda*100:.2f}%")

    st.progress(int(prob_alta * 100))

    st.markdown(f"📅 **Base da previsão:** {data_ref}")

    if prob_alta >= 0.6:
        st.success("📈 **TENDÊNCIA DE ALTA DO IBOVESPA**")
    elif prob_queda >= 0.6:
        st.error("📉 **TENDÊNCIA DE QUEDA DO IBOVESPA**")
    else:
        st.warning("⚖️ **TENDÊNCIA NEUTRA / INCERTA**")

# =====================================================
# ABA 2 — BACKTEST HISTÓRICO COMPLETO
# =====================================================
with aba2:
    st.subheader("📉 Backtest – Histórico Completo")

    dados_ibov = carregar_ibov()
    dados_feat = criar_features(dados_ibov)

    X_full = dados_feat[modelo.feature_names_]

    dados_feat["Previsao"] = modelo.predict(X_full)
    dados_feat["Classe"] = dados_feat["Previsao"].map({
        0: "Previsto Queda",
        1: "Previsto Alta"
    })

    qtd = st.slider(
        "Quantidade de dias para visualização:",
        min_value=30,
        max_value=len(dados_feat),
        value=252
    )

    dados_plot = dados_feat.tail(qtd)

    fig = px.scatter(
        dados_plot,
        x="Date",
        y="Close",
        color="Classe",
        title="Backtest – Valor do IBOVESPA com Previsão do Modelo",
        labels={
            "Close": "IBOVESPA",
            "Date": "Data",
            "Classe": "Previsão"
        }
    )

    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        dados_plot[["Date", "Close", "Classe"]],
        use_container_width=True
    )

# =====================================================
# ABA 3 — SOBRE O MODELO
# =====================================================
with aba3:
    st.subheader("ℹ️ Informações do Modelo")

    st.markdown("""
**Modelo:** CatBoostClassifier  
**Tipo:** Classificação Binária (Alta / Queda)  
**Dados:** IBOVESPA (^BVSP – Yahoo Finance)  
**Horizonte:** Próximo período  
**Validação:** Temporal (TimeSeriesSplit)
""")

    st.markdown("""
### 🎯 Objetivo do Sistema
Apoiar a **análise de tendência do mercado acionário brasileiro**,  
utilizando aprendizado de máquina aplicado a séries temporais financeiras.

### ⚠️ Aviso
Este sistema possui **finalidade educacional e analítica**,  
não constituindo recomendação de investimento.
""")
