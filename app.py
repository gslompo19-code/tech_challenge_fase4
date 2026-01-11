import streamlit as st
import pandas as pd
import joblib
import json
import plotly.express as px

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="Sistema Preditivo IBOVESPA",
    layout="wide"
)

# =====================================================
# CARREGAMENTO DE ARQUIVOS
# =====================================================
modelo = joblib.load("modelo_ibov.pkl")
dados = pd.read_csv("dados/historico_ibov.csv")
metricas = json.load(open("metricas.json"))

# Backtest salvo no notebook
backtest = pd.read_csv("dados/backtest_catboost.csv", parse_dates=["Data"])

# =====================================================
# TÍTULO
# =====================================================
st.title("📊 Sistema Preditivo de Tendência do IBOVESPA")

st.markdown("""
Este sistema utiliza **Machine Learning (CatBoost)** para prever a **tendência do IBOVESPA**
com base em dados históricos.
""")

# =====================================================
# ABAS
# =====================================================
aba1, aba2, aba3 = st.tabs([
    "🔮 Previsão",
    "📉 Backtest",
    "ℹ️ Sobre o Modelo"
])

# =====================================================
# ABA 1 — PREVISÃO (PRODUTO)
# =====================================================
with aba1:
    st.subheader("🔮 Previsão de Tendência")

    st.markdown("""
    Preencha os valores abaixo e clique em **Prever** para obter a tendência esperada
    do IBOVESPA para o próximo período.
    """)

    features = dados.drop(columns=["target"], errors="ignore")

    entrada = {}
    cols = st.columns(3)

    for i, col in enumerate(features.columns):
        with cols[i % 3]:
            entrada[col] = st.number_input(
                label=col,
                value=float(dados[col].mean())
            )

    entrada_df = pd.DataFrame([entrada])

    if st.button("📈 Prever Tendência"):
        pred = modelo.predict(entrada_df)[0]

        if pred == 1:
            st.success("📈 **TENDÊNCIA DE ALTA do IBOVESPA**")
        else:
            st.error("📉 **TENDÊNCIA DE QUEDA do IBOVESPA**")

# =====================================================
# ABA 2 — BACKTEST
# =====================================================
with aba2:
    st.subheader("📉 Backtest – Valor Real vs Previsão")

    qtd = st.slider(
        "Quantidade de períodos para visualização:",
        min_value=10,
        max_value=100,
        value=30
    )

    dados_bt = backtest.tail(qtd)

    fig = px.line(
        dados_bt,
        x="Data",
        y=["Valor Real", "Previsão"],
        markers=True,
        title="Comparação entre Valor Real e Previsão do Modelo"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(dados_bt, use_container_width=True)

# =====================================================
# ABA 3 — SOBRE O MODELO
# =====================================================
with aba3:
    st.subheader("ℹ️ Informações do Modelo")

    st.markdown("""
    **Modelo utilizado:** CatBoostClassifier  
    **Tipo:** Classificação binária (Alta / Queda)  
    **Validação:** Temporal (TimeSeriesSplit)  
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Acurácia Treino", f"{metricas['acuracia_treino']*100:.2f}%")
    col2.metric("Acurácia Teste", f"{metricas['acuracia_teste']*100:.2f}%")
    col3.metric("F1-score (CV)", f"{metricas['f1_cv_medio']:.3f}")
    col4.metric("Overfitting", f"{metricas['overfitting_percentual']:.2f}%")

    st.markdown("""
    ### 🎯 Objetivo do Modelo
    Antecipar a **tendência do IBOVESPA**, auxiliando na análise de mercado e tomada
    de decisão baseada em dados.
    """)
